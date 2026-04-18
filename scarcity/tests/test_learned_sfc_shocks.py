import numpy as np

from scarcity.simulation.learned_sfc import LearnedSFCEconomy
from scarcity.simulation.sfc import SFCConfig


class _DummyBridge:
    def get_simulator(self):
        return None


class _DummySim:
    def __init__(self):
        self.state = {
            "gdp_growth": 10.0,
            "current_account": -5.0,
        }
        self.calls = []

    def perturb(self, variable: str, value: float):
        self.calls.append((variable, value))


def _make_economy(config: SFCConfig) -> LearnedSFCEconomy:
    return LearnedSFCEconomy(bridge=_DummyBridge(), sfc_config=config)


def test_shock_dict_at_time_supports_canonical_sfc_format():
    cfg = SFCConfig(
        shock_vectors={
            "demand_shock": np.array([0.0, 0.10, 0.20]),
            "fx_shock": np.array([0.0, -0.05, -0.10]),
        }
    )
    econ = _make_economy(cfg)

    t1 = econ._shock_dict_at_time(1)
    assert t1["demand_shock"] == 0.10
    assert t1["fx_shock"] == -0.05


def test_shock_dict_at_time_supports_legacy_sequence_format():
    cfg = SFCConfig()
    # Legacy format used by learned_sfc internals in older flows.
    cfg.shock_vectors = [
        {"demand": 0.03},
        {"fx": -0.02, "trade": 0.01},
    ]
    econ = _make_economy(cfg)

    t1 = econ._shock_dict_at_time(1)
    assert t1 == {"fx": -0.02, "trade": 0.01}


def test_apply_shocks_at_maps_canonical_keys_and_perturbs_simulator():
    cfg = SFCConfig(
        shock_vectors={
            "demand_shock": np.array([0.0, 0.10]),
            "fx_shock": np.array([0.0, 0.20]),
        }
    )
    econ = _make_economy(cfg)
    econ._sim = _DummySim()

    econ._apply_shocks_at(1)

    assert ("gdp_growth", 11.0) in econ._sim.calls
    assert ("current_account", -6.0) in econ._sim.calls


def test_extract_shock_vector_uses_normalized_accessor():
    cfg = SFCConfig(
        shock_vectors={
            "demand_shock": np.array([0.0, 0.04]),
        }
    )
    econ = _make_economy(cfg)
    econ.time = 1

    shock_vector = econ._extract_shock_vector()
    assert shock_vector == {"demand_shock": 0.04}
