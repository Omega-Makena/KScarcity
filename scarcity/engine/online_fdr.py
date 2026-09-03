"""Online false-discovery-rate control for sequential hypothesis decisions.

The batch calibrator applies Benjamini-Hochberg once over a fixed pool. The
streaming engine, however, tests hypotheses *as data arrives* — an unbounded
sequence of decisions — where a fixed threshold (p < alpha) inflates the
false-discovery rate without limit. LORD++ (Ramdas, Yang, Wainwright, Jordan
2017) controls the FDR of such a sequence with a provable guarantee under
independence: it spends an error budget that is replenished on each rejection,
so the expected fraction of false discoveries stays <= alpha.

This module is standalone and deterministic given its p-value stream; wire it
into an engine by feeding it each hypothesis's calibrated p-value in arrival
order and accepting an edge only when ``test()`` returns True.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


def _gamma_sequence(n: int, s: float = 1.6) -> List[float]:
    """Normalized decreasing spending sequence gamma_j = (1/j^s) / Z, j=1..n,
    with sum -> 1 as n -> inf (Z = zeta(s), s > 1). A valid LORD weight sequence:
    nonnegative, decreasing, summable."""
    raw = [1.0 / (j ** s) for j in range(1, n + 1)]
    z = sum(raw)
    return [g / z for g in raw]


@dataclass
class LordOnlineFDR:
    """LORD++ online FDR controller.

    Parameters
    ----------
    alpha : target FDR level (e.g. 0.05).
    w0    : initial error-budget wealth, 0 < w0 < alpha (default alpha/2).
    gamma_s : exponent of the 1/j^s spending sequence (s > 1).
    max_steps : horizon used to normalize the spending sequence.
    """
    alpha: float = 0.05
    w0: float = None  # type: ignore[assignment]
    gamma_s: float = 1.6
    max_steps: int = 100_000

    _t: int = field(default=0, init=False)          # 1-indexed step count
    _reject_times: List[int] = field(default_factory=list, init=False)
    _n_reject: int = field(default=0, init=False)

    def __post_init__(self):
        if self.w0 is None:
            self.w0 = self.alpha / 2.0
        if not (0.0 < self.w0 < self.alpha):
            raise ValueError("require 0 < w0 < alpha")
        self._gamma = _gamma_sequence(self.max_steps, self.gamma_s)

    def _g(self, i: int) -> float:
        """gamma_i for i >= 1, else 0 (gamma has no mass at or before a rejection)."""
        if i < 1 or i > len(self._gamma):
            return 0.0
        return self._gamma[i - 1]

    def alpha_t(self) -> float:
        """Test level for the *next* hypothesis (the LORD++ recursion)."""
        t = self._t + 1
        rt = self._reject_times
        val = self._g(t) * self.w0
        if rt:
            val += (self.alpha - self.w0) * self._g(t - rt[0])
            for tau in rt[1:]:
                val += self.alpha * self._g(t - tau)
        return val

    def test(self, p_value: float) -> bool:
        """Feed the next p-value in arrival order; return True to reject (accept
        the discovery). Updates internal state."""
        a_t = self.alpha_t()
        self._t += 1
        reject = p_value <= a_t
        if reject:
            self._reject_times.append(self._t)
            self._n_reject += 1
        return reject

    @property
    def n_tested(self) -> int:
        return self._t

    @property
    def n_rejected(self) -> int:
        return self._n_reject
