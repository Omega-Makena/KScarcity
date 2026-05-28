import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import Ridge
except ImportError:
    pass


class AnomalyDetectionEvaluator:
    def __init__(self, data: pd.DataFrame, injected_anomalies: pd.DataFrame):
        """
        injected_anomalies is a bool DataFrame matching `data` shape
        indicating where anomalies were injected.
        """
        self.data = data
        self.anomalies = injected_anomalies

    def evaluate_zscore(self, threshold: float = 3.0) -> Dict[str, float]:
        preds = pd.DataFrame(False, index=self.data.index, columns=self.data.columns)
        for col in self.data.columns:
            z = (self.data[col] - self.data[col].mean()) / (self.data[col].std() + 1e-9)
            preds[col] = np.abs(z) > threshold
        return self._calc_metrics(preds)

    def evaluate_isolation_forest(self) -> Dict[str, float]:
        try:
            iso = IsolationForest(contamination=0.05, random_state=42)
            pred_arr = iso.fit_predict(self.data.fillna(self.data.mean()).values)
            is_anom = pred_arr == -1
            preds = pd.DataFrame(False, index=self.data.index, columns=self.data.columns)
            for col in self.data.columns:
                preds[col] = is_anom
            return self._calc_metrics(preds)
        except Exception:
            return {'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'auc': np.nan}

    def evaluate_scarcity_residuals(self, graph_predictions: pd.DataFrame,
                                    threshold: float = 3.0) -> Dict[str, float]:
        """Legacy interface: takes pre-computed graph predictions, flags large residuals."""
        preds = pd.DataFrame(False, index=self.data.index, columns=self.data.columns)
        for col in self.data.columns:
            if col in graph_predictions.columns:
                res = self.data[col] - graph_predictions[col]
                z_res = (res - res.mean()) / (res.std() + 1e-9)
                preds[col] = np.abs(z_res) > threshold
        return self._calc_metrics(preds)

    def evaluate_scarcity_graph_anomaly(self, graph: Dict[str, List[str]],
                                        threshold: float = 2.5) -> Dict[str, float]:
        """
        Graph-conditioned residual anomaly detection.

        For each variable with discovered parents, fits a lag-1 regression
        (parents_{t-1} → target_t) and flags timesteps where the residual
        Z-score exceeds `threshold`. Detects relationship-breaking anomalies
        that blind detectors miss: when a parent moves normally but the child
        fails to follow its established pattern.

        Handles all 15 Scarcity relationship types — causal, correlational,
        functional, temporal, equilibrium, synergistic, mediating, moderating,
        compositional, competitive, probabilistic, structural, graph, similarity,
        logical — because all types contribute edges of the form parent → target.

        Falls back to univariate Z-score for variables without discovered parents.
        """
        preds = pd.DataFrame(False, index=self.data.index, columns=self.data.columns)

        for col in self.data.columns:
            parents = [p for p in graph.get(col, [])
                       if p in self.data.columns and p != col]

            if not parents:
                z = (self.data[col] - self.data[col].mean()) / (self.data[col].std() + 1e-9)
                preds[col] = np.abs(z) > threshold
                continue

            df = self.data[[col] + parents].copy()
            for p in parents:
                df[p] = df[p].fillna(df[p].mean())
            df = df.dropna(subset=[col])

            if len(df) < 5:
                z = (self.data[col] - self.data[col].mean()) / (self.data[col].std() + 1e-9)
                preds[col] = np.abs(z) > threshold
                continue

            # Lag-1: X[t-1] → y[t], no contemporaneous leakage
            X = df[parents].values[:-1]
            y = df[col].values[1:]
            idx = df.index[1:]

            model = Ridge(alpha=1.0)
            model.fit(X, y)
            residuals = y - model.predict(X)

            res_mean = residuals.mean()
            res_std = residuals.std() + 1e-9
            z_res = np.abs((residuals - res_mean) / res_std)

            flags = pd.Series(False, index=self.data.index)
            flags.loc[idx] = z_res > threshold
            preds[col] = flags

        return self._calc_metrics(preds)

    def evaluate_rrcf_graph_conditioned(self, graph: Dict[str, List[str]],
                                        contamination: float = 0.05) -> Dict[str, float]:
        """
        Graph-conditioned IsolationForest.

        Transforms the raw variable space into a graph-residual space before
        scoring. Each variable's value is replaced by its residual from a
        lag-1 regression on its discovered parents. IsolationForest then
        runs on residuals instead of raw values.

        Effect: correlated normal movements (macro shocks that propagate through
        the causal graph as expected) produce small residuals and are NOT flagged.
        Only truly anomalous deviations from the discovered structural relationships
        trigger alerts. Reduces false positives from correlated noise while
        retaining sensitivity to genuine structural anomalies.
        """
        try:
            residual_df = self.data.copy().astype(float)

            for col in self.data.columns:
                parents = [p for p in graph.get(col, [])
                           if p in self.data.columns and p != col]
                if not parents:
                    continue

                df = self.data[[col] + parents].copy()
                for p in parents:
                    df[p] = df[p].fillna(df[p].mean())
                df = df.dropna(subset=[col])

                if len(df) < 5:
                    continue

                X = df[parents].values[:-1]
                y = df[col].values[1:]
                idx = df.index[1:]

                model = Ridge(alpha=1.0)
                model.fit(X, y)
                residuals = y - model.predict(X)

                # Standardize residuals so all dims are comparably scaled
                res_std = residuals.std() + 1e-9
                residual_df.loc[idx, col] = (residuals - residuals.mean()) / res_std

            iso = IsolationForest(contamination=contamination, random_state=42)
            pred_arr = iso.fit_predict(residual_df.fillna(0).values)
            is_anom = pred_arr == -1

            preds = pd.DataFrame(False, index=self.data.index, columns=self.data.columns)
            for col in self.data.columns:
                preds[col] = is_anom
            return self._calc_metrics(preds)
        except Exception:
            return {'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'auc': np.nan}

    def _rolling_rrcf_scores(self, data_arr: np.ndarray, window_size: int,
                              num_trees: int) -> np.ndarray:
        """Roll the production RRCF over data_arr rows, return per-row anomaly scores."""
        from scarcity.engine.anomaly import _compute_rrcf_codispersion
        n, d = data_arr.shape
        scores = np.zeros(n, dtype=np.float32)
        history = np.zeros((window_size, d), dtype=np.float32)
        h_idx = 0
        h_filled = False
        for t in range(n):
            query = data_arr[t].astype(np.float32)
            valid_len = window_size if h_filled else h_idx
            if valid_len >= 10:
                scores[t] = _compute_rrcf_codispersion(history[:valid_len], query, num_trees)
            history[h_idx] = query
            h_idx += 1
            if h_idx >= window_size:
                h_idx = 0
                h_filled = True
        return scores

    def evaluate_rrcf_engine(self, window_size: int = 50, num_trees: int = 50,
                              threshold: float = 6.0) -> Dict[str, float]:
        """
        Blind RRCF using the production Numba-compiled _compute_rrcf_codispersion
        from scarcity.engine.anomaly. Rolls a sliding window of `window_size` rows
        over the raw variable space and scores each row against its preceding history
        — identical algorithm to the streaming OnlineAnomalyDetector, applied
        statically to a DataFrame.

        threshold=6.0 matches the default score_threshold in OnlineAnomalyDetector.
        The score is scaled 0–10 (higher = more anomalous).
        """
        try:
            data_arr = self.data.fillna(self.data.mean()).values.astype(np.float32)
            scores = self._rolling_rrcf_scores(data_arr, window_size, num_trees)
            is_anom = scores > threshold
            preds = pd.DataFrame(False, index=self.data.index, columns=self.data.columns)
            for col in self.data.columns:
                preds[col] = is_anom
            return self._calc_metrics(preds)
        except Exception as e:
            return {'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'fpr': np.nan,
                    'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'auc': np.nan}

    def evaluate_rrcf_graph_conditioned_engine(self, graph: Dict[str, List[str]],
                                                window_size: int = 50,
                                                num_trees: int = 50,
                                                threshold: float = 6.0) -> Dict[str, float]:
        """
        Graph-conditioned RRCF using the production Numba kernel.

        Transforms each variable to its lag-1 Ridge residual (same transformation as
        evaluate_rrcf_graph_conditioned) then runs _compute_rrcf_codispersion on the
        residual space. This is the most direct apples-to-apples comparison: the same
        RRCF algorithm used in production, with and without Scarcity's graph knowledge.

        Expected causal propagations (normal macro shocks) produce near-zero residuals
        and score low. Structural decoupling anomalies produce large residuals and score
        high, even when the raw variable values are within their normal range.
        """
        try:
            residual_df = self.data.copy().astype(float)
            for col in self.data.columns:
                parents = [p for p in graph.get(col, [])
                           if p in self.data.columns and p != col]
                if not parents:
                    continue
                df = self.data[[col] + parents].copy()
                for p in parents:
                    df[p] = df[p].fillna(df[p].mean())
                df = df.dropna(subset=[col])
                if len(df) < 5:
                    continue
                X = df[parents].values[:-1]
                y = df[col].values[1:]
                idx = df.index[1:]
                model = Ridge(alpha=1.0)
                model.fit(X, y)
                residuals = y - model.predict(X)
                res_std = residuals.std() + 1e-9
                residual_df.loc[idx, col] = (residuals - residuals.mean()) / res_std

            data_arr = residual_df.fillna(0).values.astype(np.float32)
            scores = self._rolling_rrcf_scores(data_arr, window_size, num_trees)
            is_anom = scores > threshold
            preds = pd.DataFrame(False, index=self.data.index, columns=self.data.columns)
            for col in self.data.columns:
                preds[col] = is_anom
            return self._calc_metrics(preds)
        except Exception:
            return {'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'fpr': np.nan,
                    'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'auc': np.nan}

    def _calc_metrics(self, predictions: pd.DataFrame) -> Dict[str, float]:
        y_true = self.anomalies.fillna(False).values.flatten().astype(bool)
        y_pred = predictions.fillna(False).values.flatten().astype(bool)

        tp = int(np.sum(y_true & y_pred))
        fp = int(np.sum(~y_true & y_pred))
        fn = int(np.sum(y_true & ~y_pred))
        tn = int(np.sum(~y_true & ~y_pred))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'fpr': float(fpr),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'auc': np.nan,
        }
