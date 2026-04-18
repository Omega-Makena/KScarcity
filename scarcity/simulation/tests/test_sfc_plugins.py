"""
Tests for SFC plugin modules added in Items 6 and 7:
  - Item 6: attach_quintile_agents() — quintile-disaggregated MPC + Gini tracking
  - Item 7: _detect_crisis_regime() — nonlinear crisis regime switches
"""

from __future__ import annotations

import numpy as np
import pytest

from scarcity.simulation.sfc import SFCConfig, SFCEconomy


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_economy(config: SFCConfig | None = None, quintile: bool = True,
                  financial: bool = True, open_econ: bool = False) -> SFCEconomy:
    """Build and initialize a test economy with optional plugins."""
    econ = SFCEconomy(config or SFCConfig())
    if financial:
        econ.attach_financial_accelerator()
    if open_econ:
        econ.attach_open_economy()
    if quintile:
        econ.attach_quintile_agents()
    econ.initialize(gdp=100.0)
    return econ


# ─────────────────────────────────────────────────────────────────────────────
# Item 6: Quintile plugin — attachment and basic properties
# ─────────────────────────────────────────────────────────────────────────────

class TestQuintilePlugin:

    def test_attach_sets_internal_state(self):
        """attach_quintile_agents() populates the three plugin attributes."""
        econ = SFCEconomy()
        assert econ._het_cfg is None
        assert econ._quintile_income_shares is None
        assert econ._InequalityMetrics is None

        econ.attach_quintile_agents()
        econ.initialize(gdp=100.0)

        assert econ._het_cfg is not None
        assert econ._quintile_income_shares is not None
        assert econ._InequalityMetrics is not None

    def test_income_shares_sum_to_one(self):
        """Quintile income shares must always sum to 1."""
        econ = _make_economy()
        econ.step()
        total = sum(econ._quintile_income_shares)
        assert abs(total - 1.0) < 1e-9, f"Shares sum to {total}, expected 1.0"

    def test_effective_mpc_in_unit_interval(self):
        """Effective MPC must stay in (0, 1) at every step."""
        econ = _make_economy()
        for _ in range(20):
            frame = econ.step()
            mpc = frame["outcomes"]["effective_mpc"]
            assert 0.0 < mpc < 1.0, f"effective_mpc={mpc} out of (0,1)"

    def test_effective_mpc_below_single_propensity(self):
        """Quintile-weighted MPC should differ from the flat propensity when plugin is active."""
        econ_plain = SFCEconomy()
        econ_plain.initialize(gdp=100.0)
        econ_plain.step()
        plain_mpc = econ_plain.config.consumption_propensity  # default 0.8

        econ_q = _make_economy()
        frame = econ_q.step()
        q_mpc = frame["outcomes"]["effective_mpc"]

        # Kenya's income distribution is very unequal; the weighted MPC should be
        # below the flat default (0.8) because the high-MPC bottom quintile has
        # a small income share (4%), while the low-MPC top quintile has 56%.
        assert q_mpc < plain_mpc, (
            f"Quintile MPC ({q_mpc:.4f}) should be below flat MPC ({plain_mpc:.4f}) "
            f"given Kenya's unequal distribution"
        )

    def test_gini_in_unit_interval(self):
        """Gini coefficient must be in [0, 1]."""
        econ = _make_economy()
        for _ in range(10):
            frame = econ.step()
            gini = frame["outcomes"]["gini"]
            assert 0.0 <= gini <= 1.0, f"Gini={gini} out of [0,1]"

    def test_gini_reflects_initial_inequality(self):
        """Initial Gini (Kenya default shares: 0.04/0.08/0.12/0.20/0.56) should be material."""
        econ = _make_economy()
        frame = econ.step()
        gini = frame["outcomes"]["gini"]
        # Kenya's Gini ≈ 0.38-0.45; our shares imply similar level
        assert gini > 0.30, f"Gini ({gini:.3f}) suspiciously low for Kenya distribution"

    def test_q1_share_plus_q5_share_in_outcomes(self):
        """Both q1_share and q5_share must appear in every frame's outcomes."""
        econ = _make_economy()
        frame = econ.step()
        assert "q1_share" in frame["outcomes"]
        assert "q5_share" in frame["outcomes"]
        assert frame["outcomes"]["q1_share"] > 0.0
        assert frame["outcomes"]["q5_share"] > 0.0

    def test_unemployment_shock_redistributes_against_q1(self):
        """When unemployment rises above NAIRU, Q1's income share must fall."""
        econ = _make_economy()
        econ.step()  # warm up
        shares_before = list(econ._quintile_income_shares)

        # Force unemployment well above NAIRU (5%) by lowering it directly and
        # running a demand-shock step
        econ.unemployment = 0.18  # 13pp above NAIRU=5% → large gap
        cfg_over = SFCConfig(
            shock_vectors={"demand_shock": np.full(5, -0.15)},
        )
        econ.config = cfg_over
        econ.step()

        q1_after = econ._quintile_income_shares[0]
        q5_after = econ._quintile_income_shares[4]
        assert q1_after < shares_before[0], "Q1 share should fall when unemployment rises"
        assert q5_after > shares_before[4], "Q5 share should rise relative to Q1 in recessions"

    def test_no_quintile_plugin_uses_flat_mpc(self):
        """Without the plugin, effective_mpc in outcomes equals consumption_propensity."""
        cfg = SFCConfig(consumption_propensity=0.75)
        econ = SFCEconomy(cfg)
        econ.initialize(gdp=100.0)
        frame = econ.step()
        assert abs(frame["outcomes"]["effective_mpc"] - 0.75) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Item 7: Crisis regime detection
# ─────────────────────────────────────────────────────────────────────────────

class TestCrisisRegimeDetection:

    def test_no_crisis_at_baseline(self):
        """At initialization (GDP=100, no shocks), all regime flags should be 0."""
        econ = _make_economy()
        frame = econ.step()
        outcomes = frame["outcomes"]
        assert outcomes["regime_sudden_stop"] == 0.0
        assert outcomes["regime_bank_run"] == 0.0
        assert outcomes["regime_debt_crisis"] == 0.0

    def test_sudden_stop_fires_below_threshold(self):
        """regime_sudden_stop = 1 when output_gap ≤ crisis_output_gap_threshold."""
        cfg = SFCConfig(crisis_output_gap_threshold=-0.05)  # low threshold for easy trigger
        econ = _make_economy(config=cfg)
        # Force a very negative output_gap directly
        econ.output_gap = -0.10  # below threshold of -0.05
        regimes = econ._detect_crisis_regime()
        assert regimes["sudden_stop"] is True

    def test_sudden_stop_off_above_threshold(self):
        """regime_sudden_stop = 0 when output_gap > threshold."""
        cfg = SFCConfig(crisis_output_gap_threshold=-0.12)
        econ = _make_economy(config=cfg)
        econ.output_gap = -0.05  # above threshold
        regimes = econ._detect_crisis_regime()
        assert regimes["sudden_stop"] is False

    def test_bank_run_fires_above_npl_threshold(self):
        """regime_bank_run = 1 when NPL ratio ≥ crisis_npl_threshold."""
        cfg = SFCConfig(crisis_npl_threshold=0.15)
        econ = _make_economy(config=cfg, financial=True)
        econ._bank_state.npl_ratio = 0.20  # above threshold
        regimes = econ._detect_crisis_regime()
        assert regimes["bank_run"] is True

    def test_bank_run_off_below_npl_threshold(self):
        """regime_bank_run = 0 when NPL ratio < threshold."""
        cfg = SFCConfig(crisis_npl_threshold=0.20)
        econ = _make_economy(config=cfg, financial=True)
        econ._bank_state.npl_ratio = 0.10  # below threshold
        regimes = econ._detect_crisis_regime()
        assert regimes["bank_run"] is False

    def test_bank_run_off_without_financial_plugin(self):
        """Without the BGG plugin, NPL is 0 so bank_run never fires."""
        econ = _make_economy(financial=False)
        econ.output_gap = 0.0
        regimes = econ._detect_crisis_regime()
        assert regimes["bank_run"] is False

    def test_debt_crisis_fires_above_threshold(self):
        """regime_debt_crisis = 1 when govt debt/GDP ≥ crisis_debt_gdp_threshold."""
        cfg = SFCConfig(crisis_debt_gdp_threshold=0.30)  # low threshold
        econ = _make_economy(config=cfg)
        # Default initialization: bonds = 40% of GDP; threshold = 30% → should fire
        regimes = econ._detect_crisis_regime()
        assert regimes["debt_crisis"] is True

    def test_debt_crisis_off_below_threshold(self):
        """regime_debt_crisis = 0 when debt/GDP < threshold."""
        cfg = SFCConfig(crisis_debt_gdp_threshold=1.50)  # very high threshold
        econ = _make_economy(config=cfg)
        regimes = econ._detect_crisis_regime()
        assert regimes["debt_crisis"] is False

    def test_regime_flags_in_trajectory_outcomes(self):
        """All three regime flags appear in every trajectory frame's outcomes dict."""
        econ = _make_economy()
        trajectory = econ.run(5)
        for frame in trajectory[1:]:  # skip t=0 (initial frame)
            outcomes = frame["outcomes"]
            assert "regime_sudden_stop" in outcomes, "regime_sudden_stop missing from outcomes"
            assert "regime_bank_run" in outcomes, "regime_bank_run missing from outcomes"
            assert "regime_debt_crisis" in outcomes, "regime_debt_crisis missing from outcomes"

    def test_sudden_stop_collapses_investment(self):
        """During sudden_stop, investment should be multiplied by crisis_investment_collapse."""
        # Use a threshold that fires immediately: output_gap at init is ~0
        # We'll use a very high threshold to confirm investment collapse is half
        cfg = SFCConfig(
            crisis_output_gap_threshold=0.99,   # always fires (output_gap is always < 0.99)
            crisis_investment_collapse=0.50,
        )
        econ_crisis = SFCEconomy(cfg)
        econ_crisis.attach_quintile_agents()
        econ_crisis.initialize(gdp=100.0)
        frame_crisis = econ_crisis.step()

        cfg_normal = SFCConfig(crisis_output_gap_threshold=-10.0)  # never fires
        econ_normal = SFCEconomy(cfg_normal)
        econ_normal.attach_quintile_agents()
        econ_normal.initialize(gdp=100.0)
        frame_normal = econ_normal.step()

        inv_crisis = frame_crisis["flows"]["investment"]
        inv_normal = frame_normal["flows"]["investment"]
        # Crisis investment should be substantially lower (roughly half, accounting for
        # other differences that make it not exactly 0.5×)
        assert inv_crisis < inv_normal * 0.75, (
            f"Crisis investment ({inv_crisis:.2f}) should be well below "
            f"normal ({inv_normal:.2f}) with collapse_mult=0.5"
        )
