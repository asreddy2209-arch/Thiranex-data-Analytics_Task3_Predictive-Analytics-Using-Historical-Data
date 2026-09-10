from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
R=np.random.default_rng(7)
def generate(path):
    dates=pd.date_range('2023-01-01','2025-12-31'); rows=[]
    for d in dates:
        season=1+0.22*np.sin(2*np.pi*d.dayofyear/365)+(.25 if d.month in [11,12] else 0); qty=max(1,int(R.normal(42*season,8))); revenue=round(qty*float(R.normal(115,8)),2); profit=round(revenue*float(R.uniform(.22,.35)),2); rows.append([d,R.choice(['Laptop','Headphones','Desk','Air Fryer']),R.choice(['Electronics','Furniture','Home']),R.choice(['North','South','East','West']),qty,revenue,profit])
    pd.DataFrame(rows,columns='Date Product Category Region Quantity Revenue Profit'.split()).to_csv(path,index=False)
def features(df):
    z=df.copy(); z['Day_Index']=(z.Date-z.Date.min()).dt.days; z['Month_Num']=z.Date.dt.month; z['DayOfWeek']=z.Date.dt.dayofweek; z['Lag_7']=z.Revenue.shift(7); z['Rolling_7']=z.Revenue.shift(1).rolling(7).mean(); return z
def run(root):
    raw=root/'data/raw/historical_sales_synthetic.csv'; proc=root/'data/processed'; charts=root/'outputs/charts'; reports=root/'outputs/reports'
    if not raw.exists(): print('Generating synthetic historical sales data...'); generate(raw)
    df=pd.read_csv(raw); df.Date=pd.to_datetime(df.Date,errors='coerce'); df=df.dropna(subset=['Date']).drop_duplicates(); daily=df.groupby('Date',as_index=False).agg(Revenue=('Revenue','sum'),Quantity=('Quantity','sum'),Profit=('Profit','sum')).sort_values('Date'); data=features(daily).dropna().reset_index(drop=True); split=int(len(data)*.8); train,test=data.iloc[:split],data.iloc[split:]; cols=['Day_Index','Month_Num','DayOfWeek','Lag_7','Rolling_7']; model=RandomForestRegressor(n_estimators=250,random_state=42,min_samples_leaf=2); model.fit(train[cols],train.Revenue); test=test.copy(); test['Predicted_Revenue']=model.predict(test[cols]); mae=mean_absolute_error(test.Revenue,test.Predicted_Revenue); mse=mean_squared_error(test.Revenue,test.Predicted_Revenue); rmse=np.sqrt(mse); r2=r2_score(test.Revenue,test.Predicted_Revenue); mape=np.mean(np.abs((test.Revenue-test.Predicted_Revenue)/test.Revenue))*100
    # Iterative 30-day forecast with the last observed/predicted values as lag history.
    history=daily[['Date','Revenue']].copy(); forecasts=[]
    for date in pd.date_range(daily.Date.max()+pd.Timedelta(days=1),periods=30):
        temp=pd.concat([history,pd.DataFrame([[date,np.nan]],columns=['Date','Revenue'])],ignore_index=True); f=features(temp).iloc[-1]; pred=float(model.predict(pd.DataFrame([f[cols].values],columns=cols))[0]); forecasts.append([date,pred]); history.loc[len(history)]=[date,pred]
    forecast=pd.DataFrame(forecasts,columns=['Date','Forecast_Revenue']); test[['Date','Revenue','Predicted_Revenue']].to_csv(proc/'actual_vs_predicted_powerbi.csv',index=False); forecast.to_csv(proc/'future_forecast_powerbi.csv',index=False); daily.to_csv(proc/'historical_daily_powerbi.csv',index=False); pd.DataFrame([['MAE',mae],['MSE',mse],['RMSE',rmse],['R2',r2],['MAPE_percent',mape]],columns=['Metric','Value']).to_csv(proc/'model_metrics.csv',index=False)
    sns.set_theme(style='whitegrid'); plt.figure(figsize=(11,5)); plt.plot(daily.Date,daily.Revenue,label='Historical'); plt.plot(forecast.Date,forecast.Forecast_Revenue,label='30-day forecast'); plt.legend(); plt.tight_layout(); plt.savefig(charts/'historical_and_forecast.png',dpi=150); plt.close(); plt.figure(figsize=(11,5)); plt.plot(test.Date,test.Revenue,label='Actual'); plt.plot(test.Date,test.Predicted_Revenue,label='Predicted'); plt.legend(); plt.tight_layout(); plt.savefig(charts/'actual_vs_predicted.png',dpi=150); plt.close(); plt.figure(figsize=(10,4)); sns.histplot(test.Revenue-test.Predicted_Revenue,kde=True); plt.tight_layout(); plt.savefig(charts/'prediction_error.png',dpi=150); plt.close()
    text=f'Predictive Sales Analytics (synthetic)\nTraining records: {len(train)}; test records: {len(test)}\nModel: RandomForestRegressor using time index, calendar, lag-7 and rolling-7 features.\nMAE: ${mae:,.2f}\nMSE: {mse:,.2f}\nRMSE: ${rmse:,.2f}\nR²: {r2:.3f}\nMAPE: {mape:.2f}%\nNext 30 days forecast total: ${forecast.Forecast_Revenue.sum():,.2f}\n'; (reports/'model_report.txt').write_text(text); print(text)
