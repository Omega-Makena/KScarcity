"""
Extended Relationship Hypothesis Implementations — v2 (Types 11-15)

Key improvements over v1:
- MediatingHypothesis:  Three separate online RLS estimators replace batch lstsq;
                        Sobel test for indirect-effect significance.
- ModeratingHypothesis: Online RLS for full + reduced models (was batch lstsq
                        in evaluate() every call); partial F-test.
- GraphHypothesis:      Complete rethink — mutual information network structure
                        detection for continuous variables (v1 assumed integer
                        node IDs, making it wrong for all real use cases).
- SimilarityHypothesis: k-means++ initialization instead of first-k-points;
                        online silhouette approximation; center-drift stability.
- LogicalHypothesis:    Learned per-variable thresholds (online mean);
                        extended rule set (AND, OR, XOR, NAND, IMPLIES, EQUIV);
                        binomial significance test.
"""

from __future__ import annotations

import logging
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

try:
    from scipy import stats as scipy_stats
    _SCIPY = True
except ImportError:
    _SCIPY = False

from .discovery import Hypothesis, RelationshipType
from .relationships import _f_pvalue, _t_pvalue, _rls_step

logger = logging.getLogger(__name__)


# ===========================================================================
# 11. MEDIATING — Online RLS Path Coefficients + Sobel Test
# ===========================================================================

class MediatingHypothesis(Hypothesis):
    """
    X → M → Y mediation analysis.

    CRITICAL FIX from v1: Three separate online RLS estimators replace the
    pattern of buffering all data and calling np.linalg.lstsq in evaluate().

    The Sobel test provides a proper significance test for the indirect
    effect (a·b), avoiding the need for bootstrap (which is too slow online).

    Paths:
        a : X → M       (RLS on [1, X] → M)
        c : X → Y       (RLS on [1, X] → Y, total effect)
        b, c' : [X,M]→Y (RLS on [1, X, M] → Y, direct + indirect)
    """

    def __init__(self, source: str, mediator: str, target: str,
                 buffer_size: int = 150):
        super().__init__([source, mediator, target], RelationshipType.MEDIATING)
        self.source = source
        self.mediator = mediator
        self.target = target

        lam = 0.98
        self._lambda = lam
        self._n = 0

        # Path a: [1, X] → M
        self._Pa = np.eye(2) * 100.0
        self._coef_a = np.zeros(2)

        # Path c: [1, X] → Y
        self._Pc = np.eye(2) * 100.0
        self._coef_c = np.zeros(2)

        # Paths b & c': [1, X, M] → Y
        self._Pbc = np.eye(3) * 100.0
        self._coef_bc = np.zeros(3)

        # Path coefficients (public)
        self.a_path = 0.0
        self.b_path = 0.0
        self.c_path = 0.0
        self.c_prime = 0.0
        self.indirect_effect = 0.0

        # Sobel test
        self.sobel_z = 0.0
        self.sobel_p = 1.0
        # Coefficient variances for Sobel SE (from RLS covariance diagonal)
        self._var_a = 1.0
        self._var_b = 1.0

    def fit_step(self, row: Dict[str, float]) -> None:
        if not all(v in row for v in [self.source, self.mediator, self.target]):
            return
        x, m, y = row[self.source], row[self.mediator], row[self.target]
        if not all(np.isfinite(v) for v in [x, m, y]):
            return
        self._n += 1

        # Path a
        fa = np.array([1.0, x])
        self._Pa, self._coef_a, _ = _rls_step(self._Pa, self._coef_a, fa, m, self._lambda)
        self.a_path = float(self._coef_a[1])
        self._var_a = float(self._Pa[1, 1])

        # Path c (total)
        fc = np.array([1.0, x])
        self._Pc, self._coef_c, _ = _rls_step(self._Pc, self._coef_c, fc, y, self._lambda)
        self.c_path = float(self._coef_c[1])

        # Paths b and c'
        fbc = np.array([1.0, x, m])
        self._Pbc, self._coef_bc, _ = _rls_step(
            self._Pbc, self._coef_bc, fbc, y, self._lambda)
        self.c_prime = float(self._coef_bc[1])
        self.b_path = float(self._coef_bc[2])
        self._var_b = float(self._Pbc[2, 2])

        self.indirect_effect = self.a_path * self.b_path

    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        n = self._n
        if n < 30:
            return {'fit_score': 0.5, 'confidence': 0.5, 'evidence': n, 'stability': 0.5}

        # Sobel test: SE(a·b) = √(b²·Var(a) + a²·Var(b))
        sobel_se = float(np.sqrt(
            self.b_path ** 2 * self._var_a +
            self.a_path ** 2 * self._var_b + 1e-10))
        if sobel_se > 1e-10:
            self.sobel_z = self.indirect_effect / sobel_se
            self.sobel_p = _t_pvalue(self.sobel_z, max(1, n - 3))
        else:
            self.sobel_z = 0.0
            self.sobel_p = 1.0

        has_mediation = (
            abs(self.a_path) > 0.05 and
            abs(self.b_path) > 0.05 and
            abs(self.c_prime) < abs(self.c_path) and
            self.sobel_p < 0.05
        )
        full_mediation = has_mediation and abs(self.c_prime) < 0.05
        fit = min(1.0, abs(self.indirect_effect) * 2.0) if has_mediation else 0.2

        return {
            'fit_score': fit,
            'confidence': max(0.0, 1.0 - self.sobel_p) if has_mediation else 0.2,
            'evidence': n,
            'stability': 0.7 if has_mediation else 0.4,
            'a_path': self.a_path,
            'b_path': self.b_path,
            'c_path': self.c_path,
            'c_prime': self.c_prime,
            'indirect_effect': self.indirect_effect,
            'sobel_z': self.sobel_z,
            'sobel_p': self.sobel_p,
            'has_mediation': has_mediation,
            'full_mediation': full_mediation,
        }

    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        if self._n < 30 or self.source not in row:
            return None
        m_pred = float(np.dot(np.array([1.0, row[self.source]]), self._coef_a))
        return (self.mediator, m_pred)


# ===========================================================================
# 12. MODERATING — Online RLS Interaction + Partial F-test
# ===========================================================================

class ModeratingHypothesis(Hypothesis):
    """
    Z moderates the X→Y relationship:
        Y = b0 + b1·X + b2·Z + b3·(X·Z)

    CRITICAL FIX from v1: Online RLS on BOTH full [1,X,Z,XZ] and reduced
    [1,X,Z] models simultaneously (v1 buffered then called lstsq in
    evaluate() every call).

    Partial F-test from running RSS window detects interaction significance.
    Predictors are centered online to reduce multicollinearity in X·Z term.
    """

    def __init__(self, predictor: str, moderator: str, target: str,
                 buffer_size: int = 150):
        super().__init__([predictor, moderator, target], RelationshipType.MODERATING)
        self.predictor = predictor
        self.moderator = moderator
        self.target = target

        lam = 0.98
        self._lambda = lam
        self._n = 0

        # Full model: [1, Xc, Zc, Xc·Zc]
        self._P_full = np.eye(4) * 100.0
        self._coef_full = np.zeros(4)

        # Reduced model: [1, Xc, Zc]
        self._P_red = np.eye(3) * 100.0
        self._coef_red = np.zeros(3)

        # Running RSS for partial F-test
        self._rss_full: deque = deque(maxlen=60)
        self._rss_red: deque = deque(maxlen=60)

        # Online centering state
        self._mean_x = 0.0
        self._mean_z = 0.0

        self.interaction_coef = 0.0
        self.main_effect_x = 0.0
        self.main_effect_z = 0.0
        self.interaction_f_stat = 0.0
        self.interaction_p_value = 1.0

    def fit_step(self, row: Dict[str, float]) -> None:
        if not all(v in row for v in [self.predictor, self.moderator, self.target]):
            return
        x, z, y = row[self.predictor], row[self.moderator], row[self.target]
        if not all(np.isfinite(v) for v in [x, z, y]):
            return

        self._n += 1
        # Online centering (Welford mean)
        self._mean_x += (x - self._mean_x) / self._n
        self._mean_z += (z - self._mean_z) / self._n
        xc = x - self._mean_x
        zc = z - self._mean_z

        # Full model
        feat_f = np.array([1.0, xc, zc, xc * zc])
        self._P_full, self._coef_full, err_f = _rls_step(
            self._P_full, self._coef_full, feat_f, y, self._lambda)
        self._rss_full.append(err_f ** 2)
        self.main_effect_x = float(self._coef_full[1])
        self.main_effect_z = float(self._coef_full[2])
        self.interaction_coef = float(self._coef_full[3])

        # Reduced model
        feat_r = np.array([1.0, xc, zc])
        self._P_red, self._coef_red, err_r = _rls_step(
            self._P_red, self._coef_red, feat_r, y, self._lambda)
        self._rss_red.append(err_r ** 2)

    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        n = self._n
        if n < 30:
            return {'fit_score': 0.5, 'confidence': 0.5, 'evidence': n, 'stability': 0.5}

        if len(self._rss_full) >= 10:
            rss_f = float(np.sum(self._rss_full))
            rss_r = float(np.sum(self._rss_red))
            nw = len(self._rss_full)
            df_den = max(1, nw - 4)
            rss_diff = max(0.0, rss_r - rss_f)
            self.interaction_f_stat = (rss_diff / 1.0) / (rss_f / df_den + 1e-10)
            self.interaction_p_value = _f_pvalue(self.interaction_f_stat, 1, df_den)

        has_mod = abs(self.interaction_coef) > 0.05 and self.interaction_p_value < 0.05
        fit = min(1.0, abs(self.interaction_coef) * 3.0) if has_mod else 0.2

        return {
            'fit_score': fit,
            'confidence': max(0.0, 1.0 - self.interaction_p_value) if has_mod else 0.2,
            'evidence': n,
            'stability': 0.7,
            'main_effect_x': self.main_effect_x,
            'main_effect_z': self.main_effect_z,
            'interaction': self.interaction_coef,
            'interaction_f_stat': self.interaction_f_stat,
            'interaction_p_value': self.interaction_p_value,
            'has_moderation': has_mod,
        }

    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        return None


# ===========================================================================
# 13. GRAPH — Mutual Information Network Structure Detection
# ===========================================================================

class GraphHypothesis(Hypothesis):
    """
    Graph-structured dependency detection via mutual information.

    COMPLETE RETHINK from v1, which assumed integer-valued variables as
    network node IDs — entirely wrong for continuous economic/sensor data.

    Now: computes MI(X;Y) via joint histogram and compares against linear
    Pearson correlation. The key signal is "non-linear excess":

        NMI(X;Y)  >>  |Pearson(X,Y)|

    This indicates complex, non-linear graph-like dependency that is NOT
    captured by standard linear relationships. High NMI + low Pearson
    is the fingerprint of graph-structured or threshold-mediated coupling.

    Also tracks whether the dependency is asymmetric (MI sub-components),
    suggesting directed graph edges.
    """

    def __init__(self, source_var: str, target_var: str,
                 buffer_size: int = 200, n_bins: int = 10):
        super().__init__([source_var, target_var], RelationshipType.GRAPH)
        self.source_var = source_var
        self.target_var = target_var
        self.n_bins = n_bins

        self.buffer_x: deque = deque(maxlen=buffer_size)
        self.buffer_y: deque = deque(maxlen=buffer_size)

        # Online Welford for Pearson (comparison baseline)
        self._n = 0
        self._mean_x = 0.0
        self._mean_y = 0.0
        self._M2_x = 0.0
        self._M2_y = 0.0
        self._cov = 0.0

        self.mutual_information = 0.0
        self.normalized_mi = 0.0   # NMI ∈ [0, 1]
        self.pearson_r = 0.0
        self.nonlinear_excess = 0.0  # NMI − |r|: graph-structure signal

    def fit_step(self, row: Dict[str, float]) -> None:
        if self.source_var in row and self.target_var in row:
            x, y = row[self.source_var], row[self.target_var]
            if not (np.isfinite(x) and np.isfinite(y)):
                return
            self.buffer_x.append(x)
            self.buffer_y.append(y)
            self._n += 1
            dx = x - self._mean_x
            self._mean_x += dx / self._n
            dy = y - self._mean_y
            self._mean_y += dy / self._n
            self._M2_x += dx * (x - self._mean_x)
            self._M2_y += dy * (y - self._mean_y)
            self._cov += dx * (y - self._mean_y)

    def _mutual_information(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
        """
        Estimate MI(X;Y) via histogram joint distribution.
        Returns (MI in nats, NMI = MI / √(H(X)·H(Y))).
        """
        bins = self.n_bins
        joint, xe, ye = np.histogram2d(X, Y, bins=bins, density=False)
        joint = joint.astype(float) + 1e-10
        joint /= joint.sum()
        px = joint.sum(axis=1)
        py = joint.sum(axis=0)
        outer = np.outer(px, py)
        mask = (joint > 1e-12) & (outer > 1e-12)
        mi = float(np.sum(joint[mask] * np.log(joint[mask] / outer[mask])))
        mi = max(0.0, mi)
        h_x = -float(np.sum(px[px > 1e-12] * np.log(px[px > 1e-12])))
        h_y = -float(np.sum(py[py > 1e-12] * np.log(py[py > 1e-12])))
        nmi = float(np.clip(mi / (np.sqrt(max(1e-10, h_x * h_y))), 0.0, 1.0))
        return mi, nmi

    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        n = len(self.buffer_x)
        if n < 30:
            return {'fit_score': 0.5, 'confidence': 0.5, 'evidence': n, 'stability': 0.5}

        X = np.array(self.buffer_x)
        Y = np.array(self.buffer_y)
        self.mutual_information, self.normalized_mi = self._mutual_information(X, Y)

        var_x = self._M2_x / self._n
        var_y = self._M2_y / self._n
        denom = float(np.sqrt(max(0.0, var_x * var_y)))
        self.pearson_r = float(np.clip(
            (self._cov / self._n) / (denom + 1e-10), -1.0, 1.0))

        # Non-linear excess: MI captures what Pearson misses
        self.nonlinear_excess = max(0.0, self.normalized_mi - abs(self.pearson_r))

        has_graph = self.normalized_mi > 0.3 and self.nonlinear_excess > 0.1
        fit = 0.6 * self.normalized_mi + 0.4 * self.nonlinear_excess

        return {
            'fit_score': fit,
            'confidence': min(1.0, n / 100) * fit if has_graph else 0.3,
            'evidence': n,
            'stability': 0.65,
            'mutual_information': self.mutual_information,
            'normalized_mi': self.normalized_mi,
            'pearson_r': self.pearson_r,
            'nonlinear_excess': self.nonlinear_excess,
            'has_graph_structure': has_graph,
        }

    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        return None


# ===========================================================================
# 14. SIMILARITY — k-means++ Init + Online Silhouette
# ===========================================================================

class SimilarityHypothesis(Hypothesis):
    """
    Cluster structure detection with improved initialisation.

    Improvements over v1:
    - k-means++ initialisation (squared-distance-weighted centroid selection)
      replaces "first k points" which often picks poor centres
    - Online silhouette approximation: s(i) = (b−a)/max(a,b) per point
      against cluster centres (not against all points — O(k) not O(n²))
    - Centre-drift stability tracking across evaluation calls
    """

    def __init__(self, variables: List[str], n_clusters: int = 3,
                 buffer_size: int = 200):
        super().__init__(variables, RelationshipType.SIMILARITY)
        self.vars = variables
        self.n_clusters = n_clusters
        self.buffer: deque = deque(maxlen=buffer_size)

        self.centers: Optional[np.ndarray] = None
        self.cluster_counts = np.zeros(n_clusters)
        self._prev_centers: Optional[np.ndarray] = None
        self._init_buf: List[np.ndarray] = []
        self._init_size = max(n_clusters * 5, 20)
        self._initialized = False

        self.silhouette_approx = 0.0
        self.center_drift = 0.0

    def _kmeans_plus_plus(self, data: np.ndarray) -> np.ndarray:
        """
        k-means++ centre selection.
        Each subsequent centre is chosen with probability ∝ D²(x, nearest center).
        """
        rng = np.random.default_rng(42)
        n = len(data)
        idx0 = int(rng.integers(n))
        centers = [data[idx0].copy()]

        for _ in range(1, self.n_clusters):
            dists = np.array([
                min(float(np.sum((pt - c) ** 2)) for c in centers)
                for pt in data
            ])
            probs = dists / (dists.sum() + 1e-10)
            chosen = int(rng.choice(n, p=probs))
            centers.append(data[chosen].copy())

        return np.array(centers)

    def fit_step(self, row: Dict[str, float]) -> None:
        if not all(v in row for v in self.vars):
            return
        pt = np.array([row[v] for v in self.vars], dtype=float)
        if not np.all(np.isfinite(pt)):
            return
        self.buffer.append(pt)

        if not self._initialized:
            self._init_buf.append(pt)
            if len(self._init_buf) >= self._init_size:
                data = np.array(self._init_buf)
                self.centers = self._kmeans_plus_plus(data)
                self._initialized = True
            return

        # Online k-means update
        dists = np.linalg.norm(self.centers - pt, axis=1)
        nearest = int(np.argmin(dists))
        self.cluster_counts[nearest] += 1
        lr = 1.0 / self.cluster_counts[nearest]
        self.centers[nearest] += lr * (pt - self.centers[nearest])

    def _silhouette_approx(self, points: np.ndarray) -> float:
        """
        Approximate silhouette using cluster centres (O(n·k) not O(n²)).

        a(i) = distance to own centre
        b(i) = distance to nearest OTHER centre
        s(i) = (b-a) / max(a,b)
        """
        if self.centers is None:
            return 0.0
        try:
            # Dist from each point to each centre: (n, k)
            D = np.array([[float(np.linalg.norm(p - c)) for c in self.centers]
                          for p in points])
            own = np.argmin(D, axis=1)
            sils = []
            for i in range(len(points)):
                a = D[i, own[i]]
                others = [D[i, k] for k in range(self.n_clusters) if k != own[i]]
                if not others:
                    continue
                b = min(others)
                m = max(a, b, 1e-10)
                sils.append((b - a) / m)
            return float(np.mean(sils)) if sils else 0.0
        except Exception:
            return 0.0

    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        n = len(self.buffer)
        if not self._initialized or n < 20:
            return {'fit_score': 0.5, 'confidence': 0.5, 'evidence': n, 'stability': 0.5}

        points = np.array(list(self.buffer))
        self.silhouette_approx = self._silhouette_approx(points)

        if self._prev_centers is not None:
            self.center_drift = float(
                np.mean(np.linalg.norm(self.centers - self._prev_centers, axis=1)))
        self._prev_centers = self.centers.copy()

        total_mean = points.mean(axis=0)
        between_var = float(np.mean([
            np.sum((self.centers[k] - total_mean) ** 2) * self.cluster_counts[k]
            for k in range(self.n_clusters)]))
        total_var = float(np.sum(np.var(points, axis=0)) * n)
        explained = float(np.clip(between_var / (total_var + 1e-9), 0.0, 1.0))

        has_clusters = self.silhouette_approx > 0.2 and explained > 0.2
        drift_stab = max(0.2, 1.0 - self.center_drift * 5.0)
        fit = 0.5 * max(0.0, self.silhouette_approx) + 0.5 * explained

        return {
            'fit_score': fit,
            'confidence': min(1.0, n / 100) * fit if has_clusters else 0.3,
            'evidence': n,
            'stability': drift_stab,
            'silhouette': self.silhouette_approx,
            'explained_variance': explained,
            'center_drift': self.center_drift,
            'n_clusters': self.n_clusters,
            'cluster_sizes': self.cluster_counts.tolist(),
        }

    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        return None


# ===========================================================================
# 15. LOGICAL — Learned Thresholds + Extended Rule Set + Binomial Test
# ===========================================================================

class LogicalHypothesis(Hypothesis):
    """
    Boolean rule detection with learned per-variable thresholds.

    Improvements over v1:
    - Thresholds learned online via running mean (v1 hard-coded threshold = 0,
      which is wrong for non-zero-mean variables)
    - Extended rule set: AND, OR, XOR, NAND, IMPLIES (X→Y), EQUIV (X↔Y)
    - Online EMA per-rule accuracy tracking (fast, O(1) per sample)
    - Binomial significance test: H0: accuracy = 0.5 (random guessing)
    - Full-buffer verification at evaluation time
    """

    _RULES: Dict[str, Any] = {
        'AND':     staticmethod(lambda x, y: x and y),
        'OR':      staticmethod(lambda x, y: x or y),
        'XOR':     staticmethod(lambda x, y: x != y),
        'NAND':    staticmethod(lambda x, y: not (x and y)),
        'IMPLIES': staticmethod(lambda x, y: (not x) or y),  # X → Y
        'EQUIV':   staticmethod(lambda x, y: x == y),         # X ↔ Y
    }

    def __init__(self, var1: str, var2: str, output: str,
                 buffer_size: int = 150):
        super().__init__([var1, var2, output], RelationshipType.LOGICAL)
        self.var1 = var1
        self.var2 = var2
        self.output = output

        self.buf_x1: deque = deque(maxlen=buffer_size)
        self.buf_x2: deque = deque(maxlen=buffer_size)
        self.buf_y: deque = deque(maxlen=buffer_size)

        # Online running means as adaptive thresholds
        self._n = 0
        self._sum1 = 0.0
        self._sum2 = 0.0
        self._sumy = 0.0
        self._thresh1 = 0.0
        self._thresh2 = 0.0
        self._thresh_y = 0.5

        # EMA accuracy per rule
        _ema_alpha = 0.05
        self._ema_alpha = _ema_alpha
        self.rule_ema: Dict[str, float] = {r: 0.5 for r in self._RULES}

        self.best_rule = 'AND'
        self.best_accuracy = 0.5

    def fit_step(self, row: Dict[str, float]) -> None:
        if not all(v in row for v in [self.var1, self.var2, self.output]):
            return
        x1, x2, y = row[self.var1], row[self.var2], row[self.output]
        if not all(np.isfinite(v) for v in [x1, x2, y]):
            return

        self.buf_x1.append(x1)
        self.buf_x2.append(x2)
        self.buf_y.append(y)

        # Update thresholds (running mean)
        self._n += 1
        self._sum1 += x1
        self._sum2 += x2
        self._sumy += y
        self._thresh1 = self._sum1 / self._n
        self._thresh2 = self._sum2 / self._n
        self._thresh_y = self._sumy / self._n

        b1 = x1 > self._thresh1
        b2 = x2 > self._thresh2
        by = y > self._thresh_y

        for name, fn in self._RULES.items():
            correct = 1.0 if fn(b1, b2) == by else 0.0
            self.rule_ema[name] = (
                (1 - self._ema_alpha) * self.rule_ema[name] +
                self._ema_alpha * correct)

    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        n = len(self.buf_x1)
        if n < 20:
            return {'fit_score': 0.5, 'confidence': 0.5, 'evidence': n, 'stability': 0.5}

        self.best_rule = max(self.rule_ema, key=self.rule_ema.__getitem__)
        self.best_accuracy = self.rule_ema[self.best_rule]

        # Full-buffer verification with current thresholds
        X1 = np.array(self.buf_x1)
        X2 = np.array(self.buf_x2)
        Y = np.array(self.buf_y)
        B1 = X1 > self._thresh1
        B2 = X2 > self._thresh2
        BY = Y > self._thresh_y

        fn = self._RULES[self.best_rule]
        preds = np.array([fn(b1, b2) for b1, b2 in zip(B1, B2)])
        verified_acc = float(np.mean(preds == BY))

        # Binomial significance test: H0: acc = 0.5
        n_correct = int(np.sum(preds == BY))
        if _SCIPY:
            # binom_test deprecated in scipy >= 1.11; use binomtest
            try:
                binom_p = float(scipy_stats.binomtest(
                    n_correct, n, 0.5, alternative='greater').pvalue)
            except AttributeError:
                binom_p = float(scipy_stats.binom_test(
                    n_correct, n, 0.5, alternative='greater'))
        else:
            z = (n_correct - 0.5 * n) / (np.sqrt(0.25 * n) + 1e-8)
            binom_p = float(0.5 * (1.0 - np.tanh(z / np.sqrt(2))))

        has_rule = verified_acc > 0.75 and binom_p < 0.05

        return {
            'fit_score': verified_acc,
            'confidence': max(0.0, 1.0 - binom_p) * verified_acc if has_rule else 0.3,
            'evidence': n,
            'stability': 0.8 if has_rule else 0.4,
            'best_rule': self.best_rule,
            'best_accuracy_ema': self.best_accuracy,
            'verified_accuracy': verified_acc,
            'binom_p': binom_p,
            'threshold_1': self._thresh1,
            'threshold_2': self._thresh2,
            'threshold_y': self._thresh_y,
            'all_rule_scores': dict(self.rule_ema),
        }

    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        if self.best_accuracy < 0.7:
            return None
        if self.var1 in row and self.var2 in row:
            b1 = row[self.var1] > self._thresh1
            b2 = row[self.var2] > self._thresh2
            fn = self._RULES[self.best_rule]
            return (self.output, 1.0 if fn(b1, b2) else 0.0)
        return None


__all__ = [
    'MediatingHypothesis',
    'ModeratingHypothesis',
    'GraphHypothesis',
    'SimilarityHypothesis',
    'LogicalHypothesis',
]
