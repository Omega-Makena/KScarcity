# Prophet Tuning Results for Macroeconomic Forecasting

We conducted a grid search over Prophet hyperparameters using rolling-origin backtesting (1-step ahead forecasting) on Kenya's historical macroeconomic data (1990-2023).

## Parameter Grid

- **`changepoint_prior_scale`**: [0.001, 0.05, 0.5]
- **`n_changepoints`**: [5, 10, 25]
- **`yearly_seasonality`**: [False, True]

The models were evaluated using Mean Absolute Error (MAE) for `gdp_growth` and `inflation_cpi`.

## Findings

| changepoint_prior_scale | n_changepoints | yearly_seasonality | gdp_growth_mae | inflation_cpi_mae |
|-------------------------|----------------|--------------------|----------------|-------------------|
| 0.001                   | 5              | False              | 1.7888         | 4.7215            |
| 0.001                   | 10             | False              | 1.7830         | 5.1014            |
| 0.001                   | 25             | False              | **1.7796**     | 5.5572            |
| 0.05                    | 5              | False              | 1.7957         | 4.6131            |
| 0.05                    | 25             | False              | 1.7947         | 4.6133            |
| 0.5                     | 5              | False              | 1.8630         | 4.5143            |
| 0.5                     | 10             | False              | 1.8721         | 4.5059            |
| 0.5                     | 25             | False              | 1.8760         | **4.4273**        |

### Analysis
1. **Target Specific Tuning**: There is a clear divergence in optimal hyperparameters. 
   - **GDP Growth**: Benefits from very low flexibility (`changepoint_prior_scale = 0.001`). This suggests the GDP series in Kenya follows a relatively stable trend with few sudden structural shifts that the model should react to.
   - **Inflation CPI**: Performs significantly better with high flexibility (`changepoint_prior_scale = 0.5`). This is consistent with the higher volatility and frequent policy or external shocks (e.g., oil prices, drought) that impact inflation.
2. **Seasonality**: In almost all cases, `yearly_seasonality=False` outperformed `yearly_seasonality=True`. Since we are using annual data (1990-2023), Prophet's "yearly" seasonality (which is meant for sub-annual daily/monthly data to capture cycles within a year) is likely just adding noise or overparameterizing the model.
3. **Changepoint Count**: For Inflation, increasing `n_changepoints` to 25 alongside high prior scale led to the best recovery.

## Recommendations

1. **Implement Per-Target Configs**: Do not use a global Prophet config. 
   - Use `cps=0.001` for GDP-like variables.
   - Use `cps=0.5` for Inflation-like variables.
2. **Disable Seasonality for Annual Data**: Explicitly set `yearly_seasonality=False` when using annual macro series to prevent overfitting on 1-step ahead forecasts.
3. **Dynamic Prior Selection**: Future iterations of the benchmark should allow Scarcity to "select" the Prophet prior based on the identified variable type (e.g., Equilibrium processes might benefit from lower priors, while Structural breaks require higher ones).
