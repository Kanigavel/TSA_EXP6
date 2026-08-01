# Ex.No: 6               HOLT WINTERS METHOD
### Date: 01.08.2026



### AIM:
To create and implement Holt Winter's Method Model using Python.
### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
~~~

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import numpy as np


if 'walmart_sales_ts' not in locals():
    df = pd.read_csv('/content/Walmart_Sales.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    walmart_sales_aggregated = df.groupby('Date')['Weekly_Sales'].mean().reset_index()
    walmart_sales_ts = walmart_sales_aggregated.set_index('Date')['Weekly_Sales']
    walmart_sales_ts = walmart_sales_ts.asfreq('W-FRI')

walmart_sales_ts = walmart_sales_ts.dropna()

print("Head of walmart_sales_ts for Holt-Winters:")
display(walmart_sales_ts.head())

plt.figure(figsize=(14, 6))
walmart_sales_ts.plot(title='Walmart Mean Weekly Sales', color='blue')
plt.xlabel('Date')
plt.ylabel('Mean Weekly Sales')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

train_size_weekly = int(len(walmart_sales_ts) * 0.8)
train_data_weekly = walmart_sales_ts[:train_size_weekly]
test_data_weekly = walmart_sales_ts[train_size_weekly:]
model_hw_weekly = ExponentialSmoothing(train_data_weekly, trend='add', seasonal='mul', seasonal_periods=52, initialization_method="estimated")
fit_hw_weekly = model_hw_weekly.fit()


test_predictions_weekly = fit_hw_weekly.forecast(len(test_data_weekly))


plt.figure(figsize=(16, 8))
train_data_weekly.plot(label='Train Data', color='blue')
test_data_weekly.plot(label='Actual Test Data', color='green')
test_predictions_weekly.plot(label='Holt-Winters Predictions', color='red', linestyle='--')
plt.title('Holt-Winters Exponential Smoothing (Weekly): Train, Test, and Predictions')
plt.xlabel('Date')
plt.ylabel('Mean Weekly Sales')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

rmse_weekly = np.sqrt(mean_squared_error(test_data_weekly, test_predictions_weekly))
print(f"RMSE for Holt-Winters Exponential Smoothing (Weekly): {rmse_weekly:.2f}")

final_model_hw_weekly = ExponentialSmoothing(walmart_sales_ts, trend='add', seasonal='mul', seasonal_periods=52, initialization_method="estimated")
final_fit_hw_weekly = final_model_hw_weekly.fit()


future_forecast_steps_weekly = 52 
future_predictions_weekly = final_fit_hw_weekly.forecast(future_forecast_steps_weekly)


plt.figure(figsize=(16, 8))
walmart_sales_ts.plot(label='Original Weekly Sales', color='blue')
future_predictions_weekly.plot(label='Future Forecast (Holt-Winters)', color='purple', linestyle='--')
plt.title('Holt-Winters Exponential Smoothing (Weekly): Original Data and Future Forecast')
plt.xlabel('Date')
plt.ylabel('Mean Weekly Sales')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
~~~
### OUTPUT:
<img width="835" height="591" alt="image" src="https://github.com/user-attachments/assets/267b86a8-b4b2-4598-9ba4-57dee96173a9" />
<img width="691" height="403" alt="image" src="https://github.com/user-attachments/assets/2f93297b-7f8a-48c9-b5be-a02fda2907f2" />
<img width="697" height="425" alt="image" src="https://github.com/user-attachments/assets/192d8197-c3c9-4b52-a9dc-ed42a1d6f96b" />
<img width="697" height="400" alt="image" src="https://github.com/user-attachments/assets/a32dfc5e-28e2-43de-b660-b916f0f4add1" />


### RESULT:
Thus, the program run successfully based on the Holt-Winters model.
