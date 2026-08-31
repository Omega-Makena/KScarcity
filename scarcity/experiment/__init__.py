"""
Experiment and provenance layer.

A first-class, reproducible record of what an experiment ran on, with what code,
on what hardware, at what cost, and with what evidence behind every discovered
relationship — so a result can be reconstructed and audited later.

    from scarcity.experiment import experiment, DatasetSpec, build_edge_provenance
"""
from scarcity.experiment.record import (
    DatasetSpec,
    EnvInfo,
    HardwareInfo,
    EdgeProvenance,
    ExperimentRecord,
)
from scarcity.experiment.runner import experiment, set_seeds
from scarcity.experiment.provenance import build_edge_provenance, explain_edge

__all__ = [
    "DatasetSpec",
    "EnvInfo",
    "HardwareInfo",
    "EdgeProvenance",
    "ExperimentRecord",
    "experiment",
    "set_seeds",
    "build_edge_provenance",
    "explain_edge",
]
