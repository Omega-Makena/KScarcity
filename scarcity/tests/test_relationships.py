"""
Test: Relationship Hypothesis Classes

Tests that each hypothesis type correctly identifies its relationship type
using the synthetic data generators.
"""

import numpy as np
from scarcity.tests.fixtures import (
    generate_causal,
    generate_correlational,
    generate_temporal,
    generate_functional,
    generate_equilibrium,
    generate_compositional,
    generate_competitive,
    generate_synergistic,
    generate_probabilistic,
    generate_structural,
    generate_mediating,
    generate_moderating,
    generate_graph,
    generate_similarity,
    generate_logical,
)

from scarcity.engine.relationships import (
    CausalHypothesis,
    CorrelationalHypothesis,
    TemporalHypothesis,
    FunctionalHypothesis,
    EquilibriumHypothesis,
    CompositionalHypothesis,
    CompetitiveHypothesis,
    SynergisticHypothesis,
    ProbabilisticHypothesis,
    StructuralHypothesis,
)

from scarcity.engine.relationships_extended import (
    MediatingHypothesis,
    ModeratingHypothesis,
    GraphHypothesis,
    SimilarityHypothesis,
    LogicalHypothesis,
)


class TestCausalHypothesis:
    def test_detects_causal_relationship(self):
        """Should detect X → Y causality."""
        dataset = generate_causal(n=200, lag=2, strength=0.8)
        hyp = CausalHypothesis('X', 'Y', lag=2)

        # update() — not fit_step() — is the API that owns confidence: it fits,
        # then advances the Bayesian accumulator. evaluate() is read-only and
        # deliberately does not compute confidence.
        for i in range(len(dataset.data['X'])):
            row = {'X': dataset.data['X'][i], 'Y': dataset.data['Y'][i]}
            result = hyp.update(row)

        assert result['direction'] == 1, "Should detect X→Y direction"
        assert result['confidence'] > 0.5, f"Confidence too low: {result['confidence']}"


class TestCorrelationalHypothesis:
    def test_detects_correlation(self):
        """Should detect correlation."""
        dataset = generate_correlational(n=200)
        hyp = CorrelationalHypothesis('X', 'Y')
        
        for i in range(len(dataset.data['X'])):
            row = {'X': dataset.data['X'][i], 'Y': dataset.data['Y'][i]}
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert abs(result.get('correlation', 0)) > 0.5, "Should detect correlation"

    def test_avoids_spurious_correlation_on_independent_noise(self):
        """Independent noise should not produce a strong correlation signal."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 600)
        y = rng.normal(0, 1, 600)
        hyp = CorrelationalHypothesis('X', 'Y')

        for i in range(len(x)):
            hyp.fit_step({'X': float(x[i]), 'Y': float(y[i])})

        result = hyp.evaluate({})
        assert abs(result.get('correlation', 0.0)) < 0.2, f"Unexpected correlation: {result}"
        assert result.get('fit_score', 0.0) < 0.3, f"Unexpected fit score: {result}"


class TestTemporalHypothesis:
    def test_detects_autocorrelation(self):
        """Should detect autoregressive structure."""
        dataset = generate_temporal(n=200)
        hyp = TemporalHypothesis('Y', lag=3)
        
        for i in range(len(dataset.data['Y'])):
            row = {'Y': dataset.data['Y'][i]}
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        # Should have learned some coefficients
        assert len(result.get('coefficients', [])) > 0


class TestFunctionalHypothesis:
    def test_detects_functional_relationship(self):
        """Should detect deterministic Y = f(X)."""
        dataset = generate_functional(n=200)
        hyp = FunctionalHypothesis('X', 'Y', degree=2)
        
        for i in range(len(dataset.data['X'])):
            row = {'X': dataset.data['X'][i], 'Y': dataset.data['Y'][i]}
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert result['fit_score'] > 0.9, "Should fit deterministic function well"
        assert result.get('deterministic', False), "Should detect deterministic relationship"

    def test_predict_value_beats_holdout_mean_baseline(self):
        """Functional predictor should beat a naive mean baseline on holdout."""
        dataset = generate_functional(n=260)
        hyp = FunctionalHypothesis('X', 'Y', degree=2)

        train_n = 200
        x_train = dataset.data['X'][:train_n]
        y_train = dataset.data['Y'][:train_n]
        x_holdout = dataset.data['X'][train_n:]
        y_holdout = dataset.data['Y'][train_n:]

        for i in range(len(x_train)):
            hyp.fit_step({'X': x_train[i], 'Y': y_train[i]})

        mean_baseline = float(np.mean(y_train))
        model_abs_err = []
        baseline_abs_err = []
        for i in range(len(x_holdout)):
            pred = hyp.predict_value({'X': x_holdout[i]})
            assert pred is not None, "Predictor should be available after training"
            _, y_hat = pred
            model_abs_err.append(abs(y_hat - y_holdout[i]))
            baseline_abs_err.append(abs(mean_baseline - y_holdout[i]))

        model_mae = float(np.mean(model_abs_err))
        baseline_mae = float(np.mean(baseline_abs_err))
        assert model_mae < baseline_mae * 0.75, (
            f"Model MAE {model_mae:.4f} should clearly beat baseline MAE {baseline_mae:.4f}"
        )


class TestCausalFalsePositiveResistance:
    def test_independent_noise_does_not_create_directional_causality(self):
        """Directional causality should stay weak on independent random noise."""
        rng = np.random.default_rng(7)
        x = rng.normal(0, 1, 500)
        y = rng.normal(0, 1, 500)
        hyp = CausalHypothesis('X', 'Y', lag=2)

        # Drive through update() so confidence is actually accumulated. Reading
        # it off evaluate() with a .get(..., 0.0) default made this assertion
        # vacuous — it passed on the missing key, not on a low value.
        for i in range(len(x)):
            result = hyp.update({'X': float(x[i]), 'Y': float(y[i])})

        assert abs(result.get('gain_forward', 0.0)) < 0.1, f"Unexpected directional gain: {result}"
        assert 'confidence' in result, "update() must report accumulated confidence"
        assert result['confidence'] < 0.35, f"Unexpected confidence: {result}"


class TestEquilibriumHypothesis:
    def test_detects_mean_reversion(self):
        """Should detect mean-reverting process."""
        dataset = generate_equilibrium(n=300, mean=0.0, reversion=0.2)
        hyp = EquilibriumHypothesis('Y')
        
        for i in range(len(dataset.data['Y'])):
            row = {'Y': dataset.data['Y'][i]}
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert result.get('is_reverting', False), "Should detect mean reversion"
        assert abs(result.get('equilibrium', 1.0)) < 1.0, "Equilibrium should be near 0"


class TestCompositionalHypothesis:
    def test_detects_sum_constraint(self):
        """Should detect A + B + C = Total."""
        dataset = generate_compositional(n=100)
        hyp = CompositionalHypothesis(['A', 'B', 'C'], 'Total')
        
        for i in range(len(dataset.data['A'])):
            row = {
                'A': dataset.data['A'][i],
                'B': dataset.data['B'][i],
                'C': dataset.data['C'][i],
                'Total': dataset.data['Total'][i]
            }
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert result.get('constraint_holds', False), "Should detect sum constraint"


class TestCompetitiveHypothesis:
    def test_detects_trade_off(self):
        """Should detect X + Y = constant."""
        dataset = generate_competitive(n=100)
        hyp = CompetitiveHypothesis('X', 'Y')
        
        for i in range(len(dataset.data['X'])):
            row = {'X': dataset.data['X'][i], 'Y': dataset.data['Y'][i]}
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert result.get('is_competitive', False), "Should detect competitive relationship"


class TestSynergisticHypothesis:
    def test_detects_interaction(self):
        """Should detect significant X1*X2 interaction."""
        dataset = generate_synergistic(n=200)
        hyp = SynergisticHypothesis('X1', 'X2', 'Y')
        
        for i in range(len(dataset.data['X1'])):
            row = {
                'X1': dataset.data['X1'][i],
                'X2': dataset.data['X2'][i],
                'Y': dataset.data['Y'][i]
            }
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        # Interaction coefficient should be significant
        assert abs(result.get('interaction_coefficient', 0)) > 0.5


class TestProbabilisticHypothesis:
    def test_detects_distribution_shift(self):
        """Should detect X shifts distribution of Y."""
        dataset = generate_probabilistic(n=500)
        hyp = ProbabilisticHypothesis('X', 'Y')
        
        for i in range(len(dataset.data['X'])):
            row = {'X': dataset.data['X'][i], 'Y': dataset.data['Y'][i]}
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert result.get('effect_size', 0) > 0.5, f"Effect size too low: {result}"


class TestStructuralHypothesis:
    def test_detects_hierarchy(self):
        """Should detect group structure."""
        dataset = generate_structural(n=200)
        hyp = StructuralHypothesis('Group', 'Y')
        
        for i in range(len(dataset.data['Group'])):
            row = {
                'Group': dataset.data['Group'][i],
                'Y': dataset.data['Y'][i]
            }
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert result.get('icc', 0) > 0.1, f"ICC too low: {result}"


class TestMediatingHypothesis:
    def test_detects_mediation(self):
        """Should detect X → M → Y path."""
        dataset = generate_mediating(n=200)
        hyp = MediatingHypothesis('X', 'M', 'Y')
        
        for i in range(len(dataset.data['X'])):
            row = {
                'X': dataset.data['X'][i],
                'M': dataset.data['M'][i],
                'Y': dataset.data['Y'][i]
            }
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert result.get('has_mediation', False), f"Should detect mediation: {result}"

    def test_detects_indirect_only_mediation_with_cancelling_total_effect(self):
        """Regression: indirect-only mediation must fire even when the total
        effect is ~0.

        The Baron-Kenny causal-steps gate (|c'| < |c|) rejected mediation
        whenever the direct and indirect paths cancel, which is common in
        feedback systems (e.g. glucose->insulin->beta). The modern criterion
        (Zhao/Lynch/Chen 2010) drops that requirement: a significant indirect
        effect a*b is sufficient. Here the direct path is built to cancel the
        indirect one, so total effect c ~ 0 while a*b is strongly significant.
        Under the old gate this returned has_mediation=False.
        """
        rng = np.random.default_rng(3)
        n = 300
        x = rng.normal(0, 1, n)
        m = 0.9 * x + 0.1 * rng.normal(0, 1, n)          # a ~ 0.9
        y = 0.8 * m - 0.6 * x + 0.1 * rng.normal(0, 1, n)  # b ~ 0.8, c' ~ -0.6
        # indirect a*b ~ 0.72, direct c' ~ -0.6  =>  total c ~ 0.12  (|c'| > |c|)

        hyp = MediatingHypothesis('X', 'M', 'Y')
        for i in range(n):
            hyp.fit_step({'X': float(x[i]), 'M': float(m[i]), 'Y': float(y[i])})

        result = hyp.evaluate({})
        assert abs(result['c_prime']) > abs(result['c_path']), (
            f"fixture must have cancelling paths (|c'|>|c|): {result}")
        assert result['has_mediation'], f"Indirect-only mediation should fire: {result}"
        assert result['sobel_p'] < 0.05, f"Indirect effect should be significant: {result}"

    def test_sobel_se_is_scaled_by_residual_variance(self):
        """Regression: Sobel SE must use sigma^2 * P, not the raw RLS P matrix.

        P is the inverse information matrix, not the coefficient covariance.
        Using it unscaled assumes sigma^2 = 1 and inflated SE by ~20x on this
        fixture, driving sobel_p to 0.63 and suppressing every mediation signal.
        The online estimator uses lambda=0.98 forgetting (n_eff ~ 50), so its SE
        stays a bounded factor above the OLS full-sample reference of ~0.057.
        """
        dataset = generate_mediating(n=200)
        hyp = MediatingHypothesis('X', 'M', 'Y')
        for i in range(len(dataset.data['X'])):
            hyp.fit_step({
                'X': dataset.data['X'][i],
                'M': dataset.data['M'][i],
                'Y': dataset.data['Y'][i],
            })

        result = hyp.evaluate({})
        sobel_se = abs(result['indirect_effect'] / result['sobel_z'])
        assert sobel_se < 0.30, f"Sobel SE not residual-scaled: {sobel_se:.4f}"
        assert result['sobel_z'] > 3.0, f"Indirect effect should be significant: {result}"
        assert result['sobel_p'] < 0.01, f"Sobel p too weak: {result}"

    def test_no_mediation_on_direct_effect_only(self):
        """Null control — X→Y direct, M irrelevant, so no indirect path."""
        rng = np.random.default_rng(11)
        n = 200
        x = rng.normal(0, 1, n)
        m = rng.normal(0, 1, n)          # independent of both X and Y
        y = 0.8 * x + 0.1 * rng.normal(0, 1, n)

        hyp = MediatingHypothesis('X', 'M', 'Y')
        for i in range(n):
            hyp.fit_step({'X': float(x[i]), 'M': float(m[i]), 'Y': float(y[i])})

        result = hyp.evaluate({})
        assert not result['has_mediation'], f"False mediation: {result}"


class TestModeratingHypothesis:
    def test_detects_moderation(self):
        """Should detect Z moderates X→Y."""
        dataset = generate_moderating(n=200)
        hyp = ModeratingHypothesis('X', 'Z', 'Y')
        
        for i in range(len(dataset.data['X'])):
            row = {
                'X': dataset.data['X'][i],
                'Z': dataset.data['Z'][i],
                'Y': dataset.data['Y'][i]
            }
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        # Interaction should be detected
        assert abs(result.get('interaction', 0)) > 0.1


class TestGraphHypothesis:
    def test_detects_graph_structure(self):
        """Should detect non-linear coupling that Pearson misses."""
        dataset = generate_graph(n=200)
        hyp = GraphHypothesis('Source', 'Target')

        for i in range(len(dataset.data['Source'])):
            row = {
                'Source': dataset.data['Source'][i],
                'Target': dataset.data['Target'][i]
            }
            hyp.fit_step(row)

        result = hyp.evaluate({})
        assert result.get('has_graph_structure', False), f"Should detect structure: {result}"
        assert result['normalized_mi'] > 0.3, f"MI too low: {result}"
        # The whole point of this type: MI sees what Pearson cannot.
        assert result['nonlinear_excess'] > 0.1, f"No non-linear excess: {result}"
        assert abs(result['pearson_r']) < 0.3, f"Coupling should be near-invisible to Pearson: {result}"

    def test_does_not_fire_on_independent_noise(self):
        """Null control — independent variables are not graph-coupled.

        Note: the histogram MI estimator is only calibrated once n is large
        relative to n_bins^2. At n=200 with the default 10 bins the null rate is
        0%; at n=30 (the class's own evaluate() threshold) it is ~95%.
        """
        rng = np.random.default_rng(5)
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        hyp = GraphHypothesis('Source', 'Target')

        for i in range(len(x)):
            hyp.fit_step({'Source': float(x[i]), 'Target': float(y[i])})

        result = hyp.evaluate({})
        assert not result.get('has_graph_structure', False), (
            f"False positive on independent noise: {result}")


class TestSimilarityHypothesis:
    def test_detects_clusters(self):
        """Should detect cluster structure."""
        dataset = generate_similarity(n=300, n_clusters=3)
        hyp = SimilarityHypothesis(['X', 'Y'], n_clusters=3)
        
        for i in range(len(dataset.data['X'])):
            row = {
                'X': dataset.data['X'][i],
                'Y': dataset.data['Y'][i]
            }
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert result.get('explained_variance', 0) > 0.2


class TestLogicalHypothesis:
    def test_detects_boolean_rule(self):
        """Should detect Z = X AND Y."""
        dataset = generate_logical(n=200)
        hyp = LogicalHypothesis('X', 'Y', 'Z')
        
        for i in range(len(dataset.data['X'])):
            row = {
                'X': dataset.data['X'][i],
                'Y': dataset.data['Y'][i],
                'Z': dataset.data['Z'][i]
            }
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert result.get('best_rule') == 'AND', f"Should detect AND rule: {result}"
        # 'rule_accuracy' was split into two distinct measures: the online EMA
        # ('best_accuracy_ema') and full-buffer verification with the current
        # thresholds ('verified_accuracy'). The latter is the honest one.
        assert result.get('verified_accuracy', 0) > 0.9, f"Rule accuracy too low: {result}"


class TestRLSNumericalStability:
    """Regression tests for RLS covariance windup.

    The naive update ``P_new = (P - outer(K, Px)) / lam`` divides by lam every
    step, amplifying rounding by (1/lam)**n. On an ill-conditioned design that
    destroyed the covariance: P lost positive-definiteness around step 1520 and
    reached -4.8e14 by step 3840, turning every variance read off its diagonal
    into a NaN and silently disabling MediatingHypothesis. _rls_step now uses
    the Joseph form, which is algebraically identical but PSD by construction.
    """

    @staticmethod
    def _ill_conditioned_stream(n, seed=0):
        # Constant column vs a mean-114/sd-21 regressor: condition number ~4e5,
        # the same shape as [1, glucose] in the biological trajectories.
        rng = np.random.default_rng(seed)
        g = 114.0 + 20.8 * rng.normal(size=n)
        y = 0.115 * g + 0.5 * rng.normal(size=n)
        return g, y

    def test_covariance_stays_positive_definite_over_long_stream(self):
        from scarcity.engine.relationships import _rls_step
        g, y = self._ill_conditioned_stream(6000)
        P = np.eye(2) * 100.0
        coef = np.zeros(2)
        for i in range(len(g)):
            P, coef, _ = _rls_step(P, coef, np.array([1.0, g[i]]), y[i], 0.98)
            if (i + 1) % 500 == 0:
                eig = np.linalg.eigvalsh(P)
                assert np.all(eig > 0), f"P lost positive-definiteness at step {i+1}: {eig}"
                assert np.all(np.isfinite(P)), f"P diverged at step {i+1}"

    def test_long_stream_estimate_stays_accurate(self):
        """Windup also biased the coefficients — naive drifted ~10% by n=6000."""
        from scarcity.engine.relationships import _rls_step
        g, y = self._ill_conditioned_stream(6000)
        P = np.eye(2) * 100.0
        coef = np.zeros(2)
        for i in range(len(g)):
            P, coef, _ = _rls_step(P, coef, np.array([1.0, g[i]]), y[i], 0.98)
        assert abs(coef[1] - 0.115) < 0.005, f"slope drifted: {coef[1]}"

    def test_joseph_form_matches_naive_update_algebraically(self):
        """The fix must not change the mathematics, only its conditioning."""
        from scarcity.engine.relationships import _rls_step
        rng = np.random.default_rng(3)
        for _ in range(5):
            k = int(rng.integers(2, 5))
            A = rng.normal(size=(k, k))
            P = A @ A.T + np.eye(k) * 0.5
            x = rng.normal(size=k)
            coef = rng.normal(size=k)
            lam = 0.98
            Px = P @ x
            K = Px / (lam + float(x @ Px))
            naive_P = (P - np.outer(K, Px)) / lam
            got_P, _, _ = _rls_step(P, coef, x, float(rng.normal()), lam)
            assert np.abs(got_P - naive_P).max() < 1e-9

    def test_mediation_survives_a_long_stream(self):
        """MediatingHypothesis read variances off P, so windup silenced it."""
        rng = np.random.default_rng(5)
        n = 4000
        x = rng.normal(0, 1, n)
        m = 0.8 * x + 0.1 * rng.normal(0, 1, n)
        y = 0.8 * m + 0.1 * rng.normal(0, 1, n)

        hyp = MediatingHypothesis('X', 'M', 'Y')
        for i in range(n):
            hyp.fit_step({'X': float(x[i]), 'M': float(m[i]), 'Y': float(y[i])})

        result = hyp.evaluate({})
        # Under the naive update this was NaN -> z=0 -> p=1, so mediation could
        # never fire past ~1520 steps. The forgetting factor holds the effective
        # sample near 1/(1-lam) ~ 50, so z does not grow with stream length —
        # significance, not a large z, is the thing to assert.
        assert np.isfinite(result['sobel_z']), f"Sobel z is not finite: {result}"
        assert result['sobel_p'] < 0.05, f"mediation lost on a long stream: {result}"
        assert result['has_mediation'], f"mediation not detected: {result}"


def test_correlational_windowed_forgetting_decays_stale_edge():
    """A windowed correlation forgets a relationship that dies mid-stream;
    the default cumulative estimator does not (documents the drift-decay fix)."""
    import numpy as np
    from scarcity.engine.relationships import CorrelationalHypothesis
    from scarcity.engine.relationship_config import CorrelationalConfig
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=1500); y1 = 0.9 * x1 + rng.normal(scale=0.2, size=1500)
    x2 = rng.normal(size=1500); y2 = rng.normal(size=1500)      # regime 2: independent

    def run(window):
        h = CorrelationalHypothesis("x", "y", buffer_size=200,
                                    config=CorrelationalConfig(window=window))
        for t in range(1500):
            h.fit_step({"x": float(x1[t]), "y": float(y1[t])})
        for t in range(1500):
            h.fit_step({"x": float(x2[t]), "y": float(y2[t])})
        return h.evaluate({})

    windowed = run(150)
    cumulative = run(0)
    # windowed forgets: correlation collapses and is no longer significant
    assert abs(windowed["correlation"]) < 0.2 and windowed["p_value"] > 0.05
    # cumulative stays stale: still significant despite the relationship being gone
    assert cumulative["p_value"] < 0.05
