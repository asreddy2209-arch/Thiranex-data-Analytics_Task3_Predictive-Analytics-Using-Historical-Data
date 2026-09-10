# Forecasting measures
```DAX
Historical Revenue = SUM(historical_daily_powerbi[Revenue])
Forecast Revenue = SUM(future_forecast_powerbi[Forecast_Revenue])
Actual Revenue = SUM(actual_vs_predicted_powerbi[Revenue])
Predicted Revenue = SUM(actual_vs_predicted_powerbi[Predicted_Revenue])
Forecast Variance = [Actual Revenue] - [Predicted Revenue]
```
