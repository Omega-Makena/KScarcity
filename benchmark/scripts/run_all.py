"""Reproduce the headline results — one command, one report.

Runs the hardening benchmark suite and writes benchmark/RESULTS.md with each
benchmark's table plus a provenance header (git commit, timestamp, Python/torch
versions). A reviewer can clone, install, and run:

    python benchmark/scripts/run_all.py            # full configs
    python benchmark/scripts/run_all.py --quick     # fast smoke (small n/seeds)
    python benchmark/scripts/run_all.py --only scaling drift

Each benchmark is a standalone script; this orchestrates them as subprocesses,
captures their printed tables, and records exactly which commit produced them.
"""
import argparse
import datetime as _dt
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent

# name -> (script, full_args, quick_args). Ordered from cheapest to most expensive.
BENCHMARKS = {
    "scaling":            ("scaling.py",                ["--n_rows", "250", "--vars", "5", "10", "20", "34"],
                                                        ["--n_rows", "120", "--vars", "5", "10", "20"]),
    "drift":              ("drift.py",                  ["--n_half", "800", "--windows", "0", "100", "300", "--seeds", "7", "8", "9"],
                                                        ["--n_half", "400", "--windows", "0", "200", "--seeds", "7"]),
    "dirty_data":         ("dirty_data.py",             ["--n", "600", "--seeds", "0", "1", "2"],
                                                        ["--n", "400", "--seeds", "0"]),
    "online_fdr":         ("online_fdr_characterize.py",["--n", "200", "--streams", "600"],
                                                        ["--n", "150", "--streams", "200"]),
    "ablation":           ("ablation.py",               ["--n_samples", "3000", "--seeds", "42", "43", "44", "--B_perm", "100"],
                                                        ["--n_samples", "800", "--seeds", "42", "--B_perm", "30"]),
    "ablation_components":("ablation_components.py",     ["--n", "1500", "--seeds", "0", "1", "2"],
                                                        ["--n", "800", "--seeds", "0", "--B_perm", "30", "--stream_n", "400"]),
    "threshold":          ("threshold_sensitivity.py",  ["--n", "1200", "--seeds", "0", "1"],
                                                        ["--n", "700", "--seeds", "0"]),
    "discovery_baselines":("discovery_baselines.py",    ["--n", "1500", "--seeds", "0", "1", "2"],
                                                        ["--n", "600", "--seeds", "0"]),
    "baselines_failure":  ("baselines_failure_modes.py",["--seeds", "0", "1", "2"],
                                                        ["--seeds", "0"]),
    "online_recovery":    ("online_recovery.py",        ["--n", "1500", "--seeds", "0", "1", "2"],
                                                        ["--n", "800", "--seeds", "0"]),
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _versions() -> str:
    import platform
    v = f"python {platform.python_version()}"
    try:
        import torch
        v += f", torch {torch.__version__}"
    except Exception:
        pass
    return v


def _run_one(name: str, script: str, args, timeout: int) -> str:
    cmd = [sys.executable, str(_HERE / script), *args]
    try:
        out = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True,
                             timeout=timeout)
        body = out.stdout.strip()
        # keep only the table region (drop the trailing JSON dump and noisy warns)
        lines = [ln for ln in body.splitlines()
                 if not ln.startswith(("[{", "{", "  Generat", "Generat"))
                 and "Warning" not in ln and "pynvml" not in ln]
        tail = "\n".join(lines[-40:]) if lines else "(no output)"
        status = "ok" if out.returncode == 0 else f"exit {out.returncode}"
        return f"### {name}  ({status})\n\n```\n{tail}\n```\n"
    except subprocess.TimeoutExpired:
        return f"### {name}  (TIMEOUT after {timeout}s)\n"
    except Exception as ex:                                    # noqa: BLE001
        return f"### {name}  (ERROR: {type(ex).__name__}: {ex})\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke configs")
    ap.add_argument("--only", nargs="+", default=None, help="run only these benchmarks")
    ap.add_argument("--timeout", type=int, default=1800, help="per-benchmark timeout (s)")
    ap.add_argument("--out", default=str(_ROOT / "benchmark" / "RESULTS.md"))
    args = ap.parse_args()

    names = args.only or list(BENCHMARKS)
    sections = []
    for name in names:
        if name not in BENCHMARKS:
            print(f"unknown benchmark: {name}", flush=True)
            continue
        script, full_args, quick_args = BENCHMARKS[name]
        run_args = quick_args if args.quick else full_args
        print(f"[run_all] {name} ...", flush=True)
        sections.append(_run_one(name, script, run_args,
                                 timeout=min(args.timeout, 300 if args.quick else args.timeout)))

    header = (
        f"# Scarcity benchmark results\n\n"
        f"- commit: `{_git_commit()}`\n"
        f"- generated: {_dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"- environment: {_versions()}\n"
        f"- mode: {'quick' if args.quick else 'full'}\n\n"
        f"Regenerate with `python benchmark/scripts/run_all.py"
        f"{' --quick' if args.quick else ''}`.\n\n"
    )
    Path(args.out).write_text(header + "\n".join(sections), encoding="utf-8")
    print(f"[run_all] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
