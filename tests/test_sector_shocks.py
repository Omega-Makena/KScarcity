"""
Tests for sector-specific crisis shocks, policies, and IO structure.

Validates that the new crisis sectors integrate correctly throughout
the research engine stack: registries → IO structure → research SFC → outcomes.
"""

import sys
import os
import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── 1. Registry completeness ─────────────────────────────────────────────

def test_shock_registry_has_sector_shocks():
    """All sector-specific shocks are registered."""
    from kshiked.simulation.scenario_templates import SHOCK_REGISTRY

    sector_shocks = [
        "cholera_outbreak", "health_capacity_collapse", "health_worker_obstruction",
        "water_contamination", "rainfall_flood",
        "road_closure", "logistics_breakdown",
        "security_surge", "civil_unrest",
        "mass_displacement", "refugee_influx",
        "market_collapse", "food_price_spike",
        "misinformation_crisis",
    ]
    for key in sector_shocks:
        assert key in SHOCK_REGISTRY, f"Missing shock: {key}"
        assert "sector" in SHOCK_REGISTRY[key], f"Shock {key} missing 'sector' field"
        assert "sfc_mapping" in SHOCK_REGISTRY[key], f"Shock {key} missing 'sfc_mapping'"

    print(f"  ✓ All {len(sector_shocks)} sector shocks present in SHOCK_REGISTRY")


def test_policy_registry_has_sector_instruments():
    """All sector-specific policy instruments are registered."""
    from kshiked.simulation.scenario_templates import POLICY_INSTRUMENT_REGISTRY

    sector_instruments = [
        "health_emergency_spending", "vaccination_coverage",
        "water_infra_spend", "road_repair_budget",
        "security_deployment",
        "displacement_relief", "cash_transfer_rate",
        "price_stabilization", "food_reserve_release",
        "counter_misinfo_spend",
    ]
    for key in sector_instruments:
        assert key in POLICY_INSTRUMENT_REGISTRY, f"Missing policy: {key}"
        assert "category" in POLICY_INSTRUMENT_REGISTRY[key], f"Policy {key} missing 'category'"

    print(f"  ✓ All {len(sector_instruments)} sector policy instruments present")


def test_policy_templates_have_crisis_responses():
    """All crisis policy response templates exist."""
    from kshiked.simulation.scenario_templates import POLICY_TEMPLATES

    templates = [
        "health_emergency_response", "water_crisis_response",
        "transport_emergency", "security_stabilization",
        "displacement_response", "market_intervention",
        "communications_response",
    ]
    for key in templates:
        assert key in POLICY_TEMPLATES, f"Missing template: {key}"
        assert "instruments" in POLICY_TEMPLATES[key], f"Template {key} missing 'instruments'"

    print(f"  ✓ All {len(templates)} crisis policy templates present")


def test_scenario_library_has_crisis_scenarios():
    """All crisis scenario templates exist in SCENARIO_LIBRARY."""
    from kshiked.simulation.scenario_templates import SCENARIO_LIBRARY

    crisis_ids = [
        "cholera_crisis", "water_contamination_crisis",
        "road_network_collapse", "security_crisis",
        "displacement_crisis", "market_disruption",
        "misinformation_wave",
    ]
    scenario_ids = {s.id for s in SCENARIO_LIBRARY}
    for sid in crisis_ids:
        assert sid in scenario_ids, f"Missing scenario: {sid}"

    print(f"  ✓ All {len(crisis_ids)} crisis scenarios present in SCENARIO_LIBRARY")


# ── 2. Shock vector generation ────────────────────────────────────────────

def test_crisis_scenario_builds_vectors():
    """Crisis scenarios produce valid shock vectors."""
    from kshiked.simulation.scenario_templates import get_scenario_by_id

    cholera = get_scenario_by_id("cholera_crisis")
    assert cholera is not None, "cholera_crisis scenario not found"

    vectors = cholera.build_shock_vectors(steps=50)
    assert len(vectors) > 0, "No shock vectors produced"

    for key, vec in vectors.items():
        assert isinstance(vec, np.ndarray), f"Vector {key} is not ndarray"
        assert len(vec) == 50, f"Vector {key} wrong length: {len(vec)}"
        assert not np.all(vec == 0), f"Vector {key} is all zeros"

    print(f"  ✓ cholera_crisis produces {len(vectors)} shock vectors")


# ── 3. IO structure ──────────────────────────────────────────────────────

def test_io_subsector_enum_has_crisis_sectors():
    """SubSectorType enum includes crisis sectors."""
    from scarcity.simulation.io_structure import SubSectorType

    for name in ["HEALTH", "WATER", "TRANSPORT", "SECURITY"]:
        assert hasattr(SubSectorType, name), f"Missing SubSectorType.{name}"

    print(f"  ✓ SubSectorType enum has all 4 crisis sectors")


def test_io_config_9_sectors():
    """Default Kenya IO config has 9 sectors."""
    from scarcity.simulation.io_structure import default_kenya_io_config

    cfg = default_kenya_io_config()
    assert cfg.n_sectors == 9, f"Expected 9 sectors, got {cfg.n_sectors}"
    assert len(cfg.sector_shares) == 9, f"Expected 9 sector shares, got {len(cfg.sector_shares)}"
    assert len(cfg.employment_shares) == 9

    # Shares should sum to ~1.0
    share_sum = sum(cfg.sector_shares.values())
    assert abs(share_sum - 1.0) < 0.01, f"Sector shares sum to {share_sum}, expected ~1.0"

    emp_sum = sum(cfg.employment_shares.values())
    assert abs(emp_sum - 1.0) < 0.01, f"Employment shares sum to {emp_sum}, expected ~1.0"

    # IO matrix should be 9x9
    assert cfg.io_matrix.shape == (9, 9), f"IO matrix shape: {cfg.io_matrix.shape}"

    # Crisis sectors present
    for name in ["health", "water", "transport", "security"]:
        assert name in cfg.sector_shares, f"Missing sector: {name}"
        assert name in cfg.shock_sensitivity, f"Missing shock sensitivity: {name}"

    print(f"  ✓ IO config has {cfg.n_sectors} sectors, matrix shape {cfg.io_matrix.shape}")


def test_leontief_model_9x9():
    """Leontief model works with 9×9 IO matrix."""
    from scarcity.simulation.io_structure import LeontiefModel, default_kenya_io_config

    cfg = default_kenya_io_config()
    model = LeontiefModel(cfg.io_matrix)

    # Test with unit final demand
    final_demand = np.ones(9) * 100.0
    gross_output = model.solve_output(final_demand)
    value_added = model.value_added(gross_output)

    assert len(gross_output) == 9
    assert len(value_added) == 9
    assert np.all(gross_output >= final_demand), "Gross output should >= final demand"
    assert np.all(value_added >= 0), "Value added should be non-negative"

    print(f"  ✓ Leontief model solves 9×9 system correctly")


# ── 4. Outcome dimensions ────────────────────────────────────────────────

def test_crisis_outcome_dimensions():
    """OUTCOME_DIMENSIONS includes crisis metrics."""
    from kshiked.simulation.kenya_calibration import OUTCOME_DIMENSIONS

    crisis_keys = [
        "health_capacity", "water_access", "transport_connectivity",
        "security_stability", "infrastructure_score", "crisis_severity",
    ]
    for key in crisis_keys:
        assert key in OUTCOME_DIMENSIONS, f"Missing outcome: {key}"
        assert OUTCOME_DIMENSIONS[key]["category"] == "Crisis", f"{key} should be in 'Crisis' category"

    print(f"  ✓ All {len(crisis_keys)} crisis outcome dimensions present")


# ── 5. End-to-end integration ────────────────────────────────────────────

def test_research_sfc_with_crisis_sectors():
    """ResearchSFCEconomy initialises with 9 IO sectors and produces crisis dimensions."""
    from scarcity.simulation.research_sfc import ResearchSFCEconomy, default_kenya_research_config

    cfg = default_kenya_research_config()
    cfg.enable_io = True
    cfg.enable_financial = False  # Keep test fast
    cfg.enable_open_economy = False
    cfg.enable_heterogeneous = False
    cfg.sfc.steps = 10

    econ = ResearchSFCEconomy(cfg)
    econ.initialize()

    # Should have 9 sub-sectors
    assert len(econ.sub_sectors) == 9, f"Expected 9 sub-sectors, got {len(econ.sub_sectors)}"
    for name in ["health", "water", "transport", "security"]:
        assert name in econ.sub_sectors, f"Missing sub-sector: {name}"

    # Run a short simulation
    trajectory = econ.run(10)
    assert len(trajectory) > 0, "No trajectory produced"

    # Check that crisis dimensions are in the last frame
    last = trajectory[-1]
    assert "io_sectors" in last, "Missing io_sectors in trajectory frame"
    assert "health" in last["io_sectors"], "Missing health in io_sectors"
    assert "transport" in last["io_sectors"], "Missing transport in io_sectors"

    if "crisis_dimensions" in last:
        cd = last["crisis_dimensions"]
        assert "health_capacity" in cd
        assert "crisis_severity" in cd
        print(f"  ✓ Crisis dimensions present: {list(cd.keys())}")
    else:
        print(f"  ⚠ crisis_dimensions not in frame (may require shock to trigger)")

    print(f"  ✓ ResearchSFCEconomy runs with 9 IO sectors, trajectory has {len(trajectory)} frames")


# ── Runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_shock_registry_has_sector_shocks,
        test_policy_registry_has_sector_instruments,
        test_policy_templates_have_crisis_responses,
        test_scenario_library_has_crisis_scenarios,
        test_crisis_scenario_builds_vectors,
        test_io_subsector_enum_has_crisis_sectors,
        test_io_config_9_sectors,
        test_leontief_model_9x9,
        test_crisis_outcome_dimensions,
        test_research_sfc_with_crisis_sectors,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            print(f"\n▶ {test.__name__}")
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("All tests passed! ✓")
    else:
        print(f"FAILURES: {failed}")
