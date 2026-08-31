"""
The experiment runner — captures the reproducibility envelope automatically.

Wrap the work in ``experiment(...)`` and the seed, code version, hardware,
runtime, and peak memory are recorded for you; you fill in the discovered edges
and metrics. On exit the record is complete and (optionally) saved to JSON.

    from scarcity.experiment import experiment, DatasetSpec, build_edge_provenance

    ds = DatasetSpec(name="synthetic_xy", version="1", n_observations=6000,
                     n_variables=3, variables=["x", "y", "z"])
    with experiment("recovery", dataset=ds, config={"vectorized": False},
                    seed=0, out_dir="artifacts/experiments") as rec:
        eng = OnlineDiscoveryEngine(vectorized=False)
        eng.initialize({"fields": [{"name": v} for v in ds.variables]})
        for row in rows:
            eng.process_row(row)
        rec.edges = build_edge_provenance(eng, dataset=ds)
        rec.metrics = {"f1": 1.0, "null_fpr": 0.0}
    # rec.runtime_seconds, peak_memory_mb, env, hardware are already filled;
    # a JSON record has been written to artifacts/experiments/.
"""
from __future__ import annotations

import os
import random
import time
import tracemalloc
import traceback
import uuid
import warnings as _warnings
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from scarcity.experiment.record import (
    DatasetSpec, EnvInfo, HardwareInfo, ExperimentRecord,
)


def set_seeds(seed: int) -> None:
    """Seed every RNG the engine might touch (stdlib, numpy, torch)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


@contextmanager
def experiment(
    name: str,
    *,
    dataset: Optional[DatasetSpec] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    out_dir: Optional[str] = None,
    repo_hint: Optional[str] = None,
) -> Iterator[ExperimentRecord]:
    """Context manager that captures the reproducibility envelope.

    Seeds RNGs, records code version + hardware, times the block, tracks peak
    memory, captures warnings and any exception, and (if ``out_dir`` is given)
    saves the JSON record on exit. The caller fills ``rec.edges`` / ``rec.metrics``
    inside the block.
    """
    set_seeds(seed)
    rec = ExperimentRecord(
        name=name,
        experiment_id=uuid.uuid4().hex[:12],
        seed=seed,
        dataset=dataset,
        config=dict(config or {}),
        env=EnvInfo.capture(repo_hint),
        hardware=HardwareInfo.capture(),
    )
    # GPU *availability* lives in hardware.gpu; gpu_used reflects whether this
    # run actually used the vectorized/GPU path (the caller may also set it).
    rec.gpu_used = bool((config or {}).get("vectorized", False))

    tracemalloc.start()
    t0 = time.perf_counter()
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        try:
            yield rec
        except Exception as exc:
            rec.status = "failed"
            rec.failures.append(f"{type(exc).__name__}: {exc}")
            rec.failures.append(traceback.format_exc().strip().splitlines()[-1])
            raise
        finally:
            rec.runtime_seconds = round(time.perf_counter() - t0, 4)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rec.peak_memory_mb = round(peak / (1024 ** 2), 2)
            for w in caught:
                msg = f"{w.category.__name__}: {w.message}"
                if msg not in rec.warnings:
                    rec.warnings.append(msg)
            if out_dir:
                path = os.path.join(out_dir, f"{name}_{rec.experiment_id}.json")
                try:
                    rec.save(path)
                except Exception as exc:  # never let saving mask the result
                    rec.warnings.append(f"record save failed: {exc}")
