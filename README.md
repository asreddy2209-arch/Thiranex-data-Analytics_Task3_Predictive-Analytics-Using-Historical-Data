# Predictive Analytics Using Historical Sales Data

Run `python main.py`. The pipeline creates reproducible daily synthetic sales data, aggregates it chronologically, reserves the final 20% as an untouched test period, and fits a Random Forest regression model with calendar and lag features. Metrics are calculated from that held-out period.

Power BI: import `historical_daily_powerbi.csv`, `actual_vs_predicted_powerbi.csv`, `future_forecast_powerbi.csv`, and `model_metrics.csv`. Build historical and forecast trend lines, actual/predicted comparison, forecast-total and RMSE cards, and Date slicers. Data is synthetic and the exact measured metrics and 30-day total are in `outputs/reports/model_report.txt`.
