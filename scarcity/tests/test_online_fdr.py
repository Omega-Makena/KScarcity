"""LORD++ online FDR controller — guarantee and power tests."""
import numpy as np
import pytest

from scarcity.engine.online_fdr import LordOnlineFDR, _gamma_sequence

ALPHA = 0.05


def test_gamma_sequence_valid():
    """Spending weights are nonnegative, decreasing, and sum ~1."""
    g = _gamma_sequence(5000, s=1.6)
    assert all(x >= 0 for x in g)
    assert all(g[i] >= g[i + 1] for i in range(len(g) - 1))
    assert 0.95 <= sum(g) <= 1.0


def test_rejects_a_clear_signal_but_not_a_lone_null():
    lord = LordOnlineFDR(alpha=ALPHA)
    # a stream of nulls then one strong signal
    assert lord.test(0.8) is False
    assert lord.test(0.4) is False
    assert lord.test(1e-9) is True          # unmistakable signal is rejected
    assert lord.n_rejected == 1


def test_fdr_controlled_under_the_global_null():
    """Under all-null uniform p-values, FDR == P(any rejection) must be <= alpha,
    while an uncorrected p<alpha rule false-fires on essentially every stream."""
    rng = np.random.default_rng(0)
    n, M = 200, 600
    lord_any = unc_any = 0
    for _ in range(M):
        p = rng.random(n)
        lord = LordOnlineFDR(alpha=ALPHA)
        if sum(lord.test(pi) for pi in p) > 0:
            lord_any += 1
        unc_any += int(np.any(p < ALPHA))
    lord_fdr = lord_any / M
    assert lord_fdr <= ALPHA + 0.02, f"LORD FDR {lord_fdr:.3f} exceeds target"
    assert unc_any / M > 0.9, "sanity: uncorrected testing should false-fire almost always"


def test_fdr_controlled_with_power_under_signal():
    """With planted signals among nulls, FDR stays <= alpha and power is real."""
    rng = np.random.default_rng(1)
    n, n_sig, M = 200, 20, 600
    fdps, powers = [], []
    for _ in range(M):
        is_sig = np.zeros(n, bool)
        is_sig[:n_sig] = True
        rng.shuffle(is_sig)
        p = np.where(is_sig, rng.beta(0.1, 30, n), rng.random(n))
        lord = LordOnlineFDR(alpha=ALPHA)
        dec = np.array([lord.test(pi) for pi in p])
        R = int(dec.sum())
        fdps.append(int((dec & ~is_sig).sum()) / max(R, 1))
        powers.append(int((dec & is_sig).sum()) / n_sig)
    assert np.mean(fdps) <= ALPHA + 0.02
    assert np.mean(powers) > 0.5


def test_w0_must_be_between_zero_and_alpha():
    with pytest.raises(ValueError):
        LordOnlineFDR(alpha=0.05, w0=0.05)
    with pytest.raises(ValueError):
        LordOnlineFDR(alpha=0.05, w0=0.0)
