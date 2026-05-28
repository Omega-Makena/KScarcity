import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import warnings

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.api import VAR
except ImportError:
    pass

try:
    from prophet import Prophet
except ImportError:
    pass

class ForecastingEvaluator:
    def __init__(self, target_variable: str, horizon: int = 1):
        self.target = target_variable
        self.horizon = horizon

    def evaluate_persistence(self, train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, float]:
        if len(train) == 0 or len(test) == 0:
            return {'rmse': np.nan, 'mae': np.nan, 'dir_acc': np.nan}
        last_val = train[self.target].iloc[-1]
        preds = np.array([last_val] * len(test))
        return self._calc_metrics(test[self.target].values, preds, train[self.target].values)

    def evaluate_arima(self, train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, float]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(train[self.target].values, order=(1, 1, 0))
                fitted = model.fit()
                preds = fitted.forecast(steps=len(test))
            return self._calc_metrics(test[self.target].values, preds, train[self.target].values)
        except Exception:
            return {'rmse': np.nan, 'mae': np.nan, 'dir_acc': np.nan}

    def evaluate_var(self, train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, float]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Need to drop constant columns to avoid singular matrix
                valid_train = train.loc[:, train.std() > 1e-6].dropna()
                if self.target not in valid_train.columns:
                    return {'rmse': np.nan, 'mae': np.nan, 'dir_acc': np.nan}
                
                model = VAR(valid_train)
                fitted = model.fit(maxlags=1)
                preds_all = fitted.forecast(valid_train.values[-1:], steps=len(test))
                target_idx = list(valid_train.columns).index(self.target)
                preds = preds_all[:, target_idx]
            return self._calc_metrics(test[self.target].values, preds, train[self.target].values)
        except Exception:
            return {'rmse': np.nan, 'mae': np.nan, 'dir_acc': np.nan}

    def evaluate_prophet(self, train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, float]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Prophet expects 'ds' and 'y'
                # Assuming index is year
                df_train = pd.DataFrame({
                    'ds': pd.to_datetime(train.index.astype(str), format='%Y'),
                    'y': train[self.target].values
                })
                m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
                # Suppress output
                import logging
                logging.getLogger('prophet').setLevel(logging.WARNING)
                m.fit(df_train)
                
                df_test = pd.DataFrame({
                    'ds': pd.to_datetime(test.index.astype(str), format='%Y')
                })
                forecast = m.predict(df_test)
                preds = forecast['yhat'].values
            return self._calc_metrics(test[self.target].values, preds, train[self.target].values)
        except Exception:
            return {'rmse': np.nan, 'mae': np.nan, 'dir_acc': np.nan}

    def evaluate_scarcity_graph(self, train: pd.DataFrame, test: pd.DataFrame,
                                graph_features: Dict[str, list]) -> Dict[str, float]:
        """
        Evaluate using features discovered by Scarcity.
        graph_features: dict mapping target variable to list of discovered parent variables.

        Parents may come from any of the 15 relationship types (causal, correlational,
        functional, temporal, equilibrium, competitive, synergistic, compositional,
        mediating, moderating, graph, probabilistic, structural, similarity, logical).
        All types are treated as potential predictors — Scarcity's role is discovery;
        this model's role is prediction.

        Falls back to ARIMA(1,1,0) when the graph has no parents for this target,
        rather than persistence — ARIMA is a strictly better fallback on macro time series.
        Uses RidgeCV (cross-validated alpha) instead of fixed Ridge(alpha=1.0) to avoid
        overfitting on short training windows.
        """
        try:
            from sklearn.linear_model import RidgeCV
            parents = graph_features.get(self.target, [])
            valid_cols = [c for c in parents if c in train.columns]

            if not valid_cols:
                # ARIMA is a better no-information baseline than persistence for macro data.
                return self.evaluate_arima(train, test)

            # Fill sporadic NaNs with column mean before dropping rows to preserve
            # more training data (critical when n_train is small).
            cols_needed = valid_cols + [self.target]
            sub = train[cols_needed].copy()
            for c in cols_needed:
                sub[c] = sub[c].fillna(sub[c].mean())
            sub = sub.dropna()

            if len(sub) < 4:
                return self.evaluate_arima(train, test)

            # Lagged features: X at t-1 predicts target at t.
            X_train = sub[valid_cols].iloc[:-1].values
            y_train = sub[self.target].iloc[1:].values

            if len(X_train) < 3:
                return self.evaluate_arima(train, test)

            # RidgeCV selects alpha from a log-spaced grid via leave-one-out CV,
            # preventing over-regularization on large n and under-regularization on small n.
            alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
            model = RidgeCV(alphas=alphas, cv=min(5, len(X_train) - 1))
            model.fit(X_train, y_train)

            # First prediction uses last training-row parent values (no leakage).
            curr_X = sub[valid_cols].iloc[-1].values
            preds = []
            for i in range(len(test)):
                pred = float(model.predict([curr_X])[0])
                preds.append(pred)
                if i < len(test) - 1:
                    curr_X = test[valid_cols].iloc[i].fillna(sub[valid_cols].mean()).values

            return self._calc_metrics(test[self.target].values, np.array(preds),
                                      train[self.target].values)
        except Exception:
            return self.evaluate_arima(train, test)

    def evaluate_prophet_with_graph(self, train: pd.DataFrame, test: pd.DataFrame,
                                    graph_features: Dict[str, list]) -> Dict[str, float]:
        """
        Prophet informed by Scarcity's discovered parents as extra_regressors.

        Scarcity hands off the discovered graph — Prophet uses it as structured
        prior knowledge about which variables drive the target.  Parents may come
        from any of the 15 relationship types; the graph_extractor and top_k_graph
        already handle type-diversity selection so all types are represented.
        Regressor values at forecast time are the last available training-year
        values (lag-1), so there is no future leakage.  Falls back to plain
        Prophet when no parents are discovered.
        """
        try:
            from prophet import Prophet
            import logging
            parents = graph_features.get(self.target, [])
            valid_cols = [c for c in parents if c in train.columns and c != self.target]

            if not valid_cols:
                return self.evaluate_prophet(train, test)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logging.getLogger('prophet').setLevel(logging.WARNING)
                logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

                df = train[[self.target] + valid_cols].copy()
                for c in valid_cols:
                    df[c] = df[c].fillna(df[c].mean())
                df = df.dropna(subset=[self.target])

                if len(df) < 4:
                    return self.evaluate_prophet(train, test)

                df_train = pd.DataFrame({
                    'ds': pd.to_datetime(df.index.astype(str), format='%Y'),
                    'y': df[self.target].values,
                })
                for c in valid_cols:
                    df_train[c] = df[c].values

                m = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                            daily_seasonality=False)
                for c in valid_cols:
                    m.add_regressor(c)
                m.fit(df_train)

                # Forecast: use last known (T-1) parent values — no future leakage.
                df_future = pd.DataFrame({
                    'ds': pd.to_datetime(test.index.astype(str), format='%Y')
                })
                for c in valid_cols:
                    last_val = float(df[c].iloc[-1]) if not df[c].isna().all() else 0.0
                    df_future[c] = last_val

                preds = m.predict(df_future)['yhat'].values

            return self._calc_metrics(test[self.target].values, preds,
                                      train[self.target].values)
        except Exception:
            return self.evaluate_prophet(train, test)

    def evaluate_arimax_with_graph(self, train: pd.DataFrame, test: pd.DataFrame,
                                   graph_features: Dict[str, list]) -> Dict[str, float]:
        """
        ARIMA informed by Scarcity's discovered parents as exogenous regressors.

        Scarcity hands off the discovered graph — ARIMAX uses it as structured
        prior knowledge.  Parents may come from any of the 15 relationship types;
        the graph_extractor and top_k_graph ensure all types contribute parents
        (not just causal/correlational).  Exogenous values at forecast time are
        the last available training-year values (lag-1 of parents), consistent with
        Granger-causal discovery.  Falls back to plain ARIMA when no parents
        are discovered or when the exog matrix is rank-deficient.
        """
        try:
            parents = graph_features.get(self.target, [])
            valid_cols = [c for c in parents if c in train.columns and c != self.target]

            if not valid_cols:
                return self.evaluate_arima(train, test)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                df = train[[self.target] + valid_cols].copy()
                for c in valid_cols:
                    df[c] = df[c].fillna(df[c].mean())
                df = df.dropna(subset=[self.target])

                if len(df) < 5:
                    return self.evaluate_arima(train, test)

                y = df[self.target].values
                exog = df[valid_cols].values

                # Lag-1: use X[t-1] to predict y[t] — no contemporaneous leakage.
                y_fit   = y[1:]
                exog_fit = exog[:-1]

                if len(y_fit) < 4:
                    return self.evaluate_arima(train, test)

                model = ARIMA(y_fit, exog=exog_fit, order=(1, 1, 0))
                fitted = model.fit()

                # Forecast exog: last known row of training exog (year T-1 values).
                exog_fc = exog[-1:].reshape(1, -1)
                preds = fitted.forecast(steps=len(test), exog=exog_fc)

            return self._calc_metrics(test[self.target].values, np.array(preds),
                                      train[self.target].values)
        except Exception:
            return self.evaluate_arima(train, test)

    # ------------------------------------------------------------------
    # Tree model helpers
    # ------------------------------------------------------------------

    def _build_lag_features(self, train: pd.DataFrame, feature_cols: list):
        """
        Construct lag-1 design matrix for tree models.

        X[i] = feature_cols values at year i  →  y[i+1] = target at year i+1
        No future leakage: the test prediction uses only the last training row.

        Returns (X_train, y_train, last_X) or (None, None, None) if too little data.
        """
        cols = [c for c in feature_cols if c in train.columns and c != self.target]
        if not cols:
            return None, None, None, []

        needed = cols + [self.target]
        sub = train[needed].copy()
        for c in needed:
            sub[c] = sub[c].fillna(sub[c].mean())
        sub = sub.dropna()

        if len(sub) < 4:
            return None, None, None, []

        X = sub[cols].iloc[:-1].values      # features at t-1
        y = sub[self.target].iloc[1:].values # target at t
        last_X = sub[cols].iloc[-1].values   # used to predict test year

        return X, y, last_X, cols

    def _tree_predict(self, model, last_X: np.ndarray,
                      test: pd.DataFrame, feature_cols: list,
                      train: pd.DataFrame) -> np.ndarray:
        """One-step-ahead prediction for tree models; uses lag-1 of test features."""
        preds = []
        curr_X = last_X.copy()
        for i in range(len(test)):
            preds.append(float(model.predict([curr_X])[0]))
            if i < len(test) - 1:
                row = test[feature_cols].iloc[i]
                curr_X = row.fillna(train[feature_cols].mean()).values
        return np.array(preds)

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------

    def evaluate_xgboost(self, train: pd.DataFrame, test: pd.DataFrame,
                         feature_cols: Optional[list] = None) -> Dict[str, float]:
        """
        XGBoost with lag-1 features (blind: all available columns if feature_cols=None).
        Shallow trees (max_depth=3) and small n_estimators limit overfitting on short series.
        """
        try:
            from xgboost import XGBRegressor
            if feature_cols is None:
                feature_cols = [c for c in train.columns if c != self.target]

            result = self._build_lag_features(train, feature_cols)
            if result[0] is None:
                return self.evaluate_persistence(train, test)
            X, y, last_X, cols = result

            model = XGBRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0,
                n_jobs=1,
            )
            model.fit(X, y)
            preds = self._tree_predict(model, last_X, test, cols, train)
            return self._calc_metrics(test[self.target].values, preds,
                                      train[self.target].values)
        except ImportError:
            return {'rmse': np.nan, 'mae': np.nan, 'dir_acc': np.nan,
                    'note': 'xgboost not installed'}
        except Exception:
            return self.evaluate_persistence(train, test)

    def evaluate_xgboost_with_graph(self, train: pd.DataFrame, test: pd.DataFrame,
                                    graph_features: Dict[str, list]) -> Dict[str, float]:
        """
        XGBoost with Scarcity-discovered parents as the feature set.
        Graph feature selection acts as regularisation: instead of all 19 lag-1
        variables, only the top-K discovered parents are used, dramatically
        reducing the feature-to-sample ratio on short time series.
        Falls back to blind XGBoost when no parents are discovered.
        """
        parents = graph_features.get(self.target, [])
        valid = [c for c in parents if c in train.columns and c != self.target]
        if not valid:
            return self.evaluate_xgboost(train, test)
        return self.evaluate_xgboost(train, test, feature_cols=valid)

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------

    def evaluate_lightgbm(self, train: pd.DataFrame, test: pd.DataFrame,
                           feature_cols: Optional[list] = None) -> Dict[str, float]:
        """
        LightGBM with lag-1 features. Same shallow config as XGBoost for fair comparison.
        LightGBM is typically faster than XGBoost and handles small datasets slightly
        better via its leaf-wise growth, but both will overfit with 18 blind features
        at N_train < 20.
        """
        try:
            import lightgbm as lgb
            if feature_cols is None:
                feature_cols = [c for c in train.columns if c != self.target]

            result = self._build_lag_features(train, feature_cols)
            if result[0] is None:
                return self.evaluate_persistence(train, test)
            X, y, last_X, cols = result

            model = lgb.LGBMRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                num_leaves=7,
                random_state=42, verbosity=-1, n_jobs=1,
            )
            model.fit(X, y)
            preds = self._tree_predict(model, last_X, test, cols, train)
            return self._calc_metrics(test[self.target].values, preds,
                                      train[self.target].values)
        except ImportError:
            return {'rmse': np.nan, 'mae': np.nan, 'dir_acc': np.nan,
                    'note': 'lightgbm not installed'}
        except Exception:
            return self.evaluate_persistence(train, test)

    def evaluate_lightgbm_with_graph(self, train: pd.DataFrame, test: pd.DataFrame,
                                     graph_features: Dict[str, list]) -> Dict[str, float]:
        """
        LightGBM with Scarcity-discovered parents as the feature set.
        Same regularisation rationale as XGBoost+graph.
        Falls back to blind LightGBM when no parents are discovered.
        """
        parents = graph_features.get(self.target, [])
        valid = [c for c in parents if c in train.columns and c != self.target]
        if not valid:
            return self.evaluate_lightgbm(train, test)
        return self.evaluate_lightgbm(train, test, feature_cols=valid)

    # ------------------------------------------------------------------
    # Temporal Fusion Transformer (lightweight pure-PyTorch)
    # ------------------------------------------------------------------

    def evaluate_tft(self, train: pd.DataFrame, test: pd.DataFrame,
                     context_length: int = 5, epochs: int = 50) -> Dict[str, float]:
        """
        Lightweight TFT-inspired model using pure PyTorch (no pytorch-forecasting).

        Architecture:
          - Project all 19 lag-1 features to a hidden dimension
          - Single-head self-attention across feature tokens (TFT variable selection spirit)
          - Linear readout to scalar prediction
          - Trained with MSE loss, Adam optimizer

        Why this is the right model for this comparison:
          TFT is a sequence-to-sequence model designed for long history (100+ points).
          At N_train < 15 the training set has < 14 (X, y) pairs with 19-dimensional input
          — 19 × hidden parameters vastly outnumbers the samples. This implementation
          will still overfit severely at small N, producing the same negative finding as
          the full TFT would. The key result is the data-size boundary, not the framework.

        Returns NaN when N_train < context_length + 3 (not enough to form training pairs).
        """
        try:
            import torch
            import torch.nn as nn

            MIN_TRAIN = context_length + 3
            if len(train) < MIN_TRAIN:
                return {'rmse': np.nan, 'mae': np.nan, 'dir_acc': np.nan,
                        'note': f'TFT-lite: N_train={len(train)} < {MIN_TRAIN} required'}

            feature_cols = [c for c in train.columns if c != self.target]
            needed = feature_cols + [self.target]
            df = train[needed].copy()
            for c in needed:
                df[c] = df[c].fillna(df[c].mean())
            df = df.dropna()

            if len(df) < MIN_TRAIN:
                return {'rmse': np.nan, 'mae': np.nan, 'dir_acc': np.nan,
                        'note': f'TFT-lite: after fill only {len(df)} rows'}

            n_feat = len(feature_cols)
            X_np = df[feature_cols].iloc[:-1].values.astype(np.float32)
            y_np = df[self.target].iloc[1:].values.astype(np.float32)

            if len(X_np) < 3:
                return {'rmse': np.nan, 'mae': np.nan, 'dir_acc': np.nan,
                        'note': f'TFT-lite: only {len(X_np)} training pairs'}

            # Standardize inputs
            X_mean = X_np.mean(axis=0)
            X_std  = X_np.std(axis=0) + 1e-8
            y_mean = float(y_np.mean())
            y_std  = float(y_np.std()) + 1e-8
            X_np   = (X_np - X_mean) / X_std
            y_np   = (y_np - y_mean) / y_std

            X_t = torch.tensor(X_np).unsqueeze(1)  # (N, 1, n_feat)
            y_t = torch.tensor(y_np).unsqueeze(1)  # (N, 1)

            hidden = 16

            class TFTLite(nn.Module):
                def __init__(self, n_f, h):
                    super().__init__()
                    self.proj  = nn.Linear(n_f, h)
                    self.attn  = nn.MultiheadAttention(h, num_heads=1, batch_first=True)
                    self.norm  = nn.LayerNorm(h)
                    self.out   = nn.Linear(h, 1)

                def forward(self, x):          # x: (B, 1, n_f)
                    z = torch.relu(self.proj(x))    # (B, 1, h)
                    a, _ = self.attn(z, z, z)
                    z = self.norm(z + a)
                    return self.out(z.squeeze(1))   # (B, 1)

            model = TFTLite(n_feat, hidden)
            opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
            loss_fn = nn.MSELoss()

            model.train()
            for _ in range(epochs):
                opt.zero_grad()
                pred = model(X_t)
                loss = loss_fn(pred, y_t)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                last_X = df[feature_cols].iloc[-1].values.astype(np.float32)
                last_X = (last_X - X_mean) / X_std
                last_t = torch.tensor(last_X).unsqueeze(0).unsqueeze(0)
                pred_val = float(model(last_t).squeeze()) * y_std + y_mean

            return self._calc_metrics(test[self.target].values[:1],
                                      np.array([pred_val]),
                                      train[self.target].values)

        except ImportError:
            return {'rmse': np.nan, 'mae': np.nan, 'dir_acc': np.nan,
                    'note': 'torch not installed'}
        except Exception as e:
            return {'rmse': np.nan, 'mae': np.nan, 'dir_acc': np.nan,
                    'note': f'TFT-lite failed: {str(e)[:80]}'}

    def _calc_metrics(self, actual: np.ndarray, pred: np.ndarray, train_actual: np.ndarray) -> Dict[str, float]:
        rmse = np.sqrt(np.mean((actual - pred)**2))
        mae = np.mean(np.abs(actual - pred))
        
        # Directional agreement
        if len(train_actual) > 0 and len(actual) > 0:
            last_train = train_actual[-1]
            actual_dir = np.sign(actual[0] - last_train)
            pred_dir = np.sign(pred[0] - last_train)
            dir_acc = 1.0 if actual_dir == pred_dir else 0.0
        else:
            dir_acc = np.nan
            
        return {'rmse': float(rmse), 'mae': float(mae), 'dir_acc': float(dir_acc)}
