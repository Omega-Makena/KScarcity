import pandas as pd
import numpy as np
import itertools
from pathlib import Path

# Scarcity imports
import sys
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.real_data.world_bank_loader import prepare_multi_country_data
from benchmark.evaluation.forecasting import ForecastingEvaluator
from benchmark.real_data.rolling_backtest import RollingOriginBacktest

def tune_prophet():
    print("Loading data...")
    data_dict = prepare_multi_country_data(['KEN'])
    kenya_data = data_dict['KEN']
    
    targets = ['gdp_growth', 'inflation_cpi']
    
    # Prophet hyperparameters to tune
    changepoint_prior_scales = [0.001, 0.05, 0.5]
    n_changepoints = [5, 10, 25]
    seasonalities = [False, True] # yearly_seasonality
    
    param_grid = list(itertools.product(changepoint_prior_scales, n_changepoints, seasonalities))
    
    results = []
    
    print(f"Starting tuning over {len(param_grid)} parameter combinations...")
    
    # We will temporarily mock the Prophet evaluation to accept parameters
    for cps, nc, ys in param_grid:
        print(f"Testing: changepoint_prior_scale={cps}, n_changepoints={nc}, yearly_seasonality={ys}")
        
        # Monkey patch ForecastingEvaluator.evaluate_prophet
        def evaluate_prophet_custom(self, train: pd.DataFrame, test: pd.DataFrame) -> dict:
            try:
                import warnings
                from prophet import Prophet
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df_train = pd.DataFrame({
                        'ds': pd.to_datetime(train.index.astype(str), format='%Y'),
                        'y': train[self.target].values
                    })
                    m = Prophet(
                        yearly_seasonality=ys, 
                        weekly_seasonality=False, 
                        daily_seasonality=False,
                        changepoint_prior_scale=cps,
                        n_changepoints=nc
                    )
                    import logging
                    logging.getLogger('prophet').setLevel(logging.WARNING)
                    m.fit(df_train)
                    
                    df_test = pd.DataFrame({
                        'ds': pd.to_datetime(test.index.astype(str), format='%Y')
                    })
                    forecast = m.predict(df_test)
                    preds = forecast['yhat'].values
                return self._calc_metrics(test[self.target].values, preds, train[self.target].values)
            except Exception as e:
                return {'rmse': np.nan, 'mae': np.nan, 'dir_acc': np.nan}
                
        # Apply patch
        ForecastingEvaluator.evaluate_prophet = evaluate_prophet_custom
        
        backtest = RollingOriginBacktest(kenya_data, target_variables=targets, initial_train_years=15)
        res = backtest.run_backtest({})
        
        mean_res = res.groupby('target')[['prophet_mae']].mean().to_dict()['prophet_mae']
        
        results.append({
            'changepoint_prior_scale': cps,
            'n_changepoints': nc,
            'yearly_seasonality': ys,
            'gdp_growth_mae': mean_res.get('gdp_growth', np.nan),
            'inflation_cpi_mae': mean_res.get('inflation_cpi', np.nan)
        })
        
    df_res = pd.DataFrame(results)
    print("\n--- Tuning Results ---")
    print(df_res.sort_values('gdp_growth_mae').head())
    
    # Write to a file
    out_path = Path("benchmark/reports/outputs/prophet_tuning.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    tune_prophet()
