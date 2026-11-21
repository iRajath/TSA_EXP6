# Ex.No: 6               HOLT WINTERS METHOD
### Date: 21.11.2026



### AIM: To implement the Holt-Winters Exponential Smoothing Method using Python on the Hourly Temperature dataset to analyze temperature trends, seasonal variations, and predict future temperature values.

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

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google.colab import files
from io import BytesIO

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

uploaded = files.upload()

file_name = next(iter(uploaded))
df = pd.read_csv(BytesIO(uploaded[file_name]))

df.columns = df.columns.str.strip().str.lower()

df['time period'] = pd.to_datetime(
    df['time period'], format='mixed', dayfirst=True
)

df['subscribers'] = pd.to_numeric(
    df['subscribers'], errors='coerce'
)

df = df.dropna(subset=['time period', 'subscribers'])

df = df.sort_values('time period')
df = df.set_index('time period')

df = df.resample('QS').mean()
df['subscribers'] = df['subscribers'].interpolate()

print("\nFIRST FIVE ROWS:\n")
print(df.head())

plt.figure(figsize=(12, 5))
plt.plot(df['subscribers'])
plt.title('Netflix Subscribers Time Series')
plt.xlabel('Date')
plt.ylabel('Subscribers')
plt.grid(True)
plt.show()

decomposition = seasonal_decompose(
    df['subscribers'], model='additive', period=4
)

decomposition.plot()
plt.tight_layout()
plt.show()

train = df['subscribers'][:-8]
test = df['subscribers'][-8:]

model = ExponentialSmoothing(
    train,
    trend='add',
    seasonal='add',
    seasonal_periods=4,
    initialization_method='estimated'
)

fit = model.fit()

predictions = fit.forecast(len(test))

rmse = np.sqrt(mean_squared_error(test, predictions))

print("\nRMSE:", rmse)

mean = df['subscribers'].mean()
std = df['subscribers'].std()

print("\nMean:", mean)
print("Standard Deviation:", std)

final_model = ExponentialSmoothing(
    df['subscribers'],
    trend='add',
    seasonal='add',
    seasonal_periods=4,
    initialization_method='estimated'
)

final_fit = final_model.fit()

future = final_fit.forecast(8)

plt.figure(figsize=(12, 6))

plt.plot(df['subscribers'], label='Original Data')
plt.plot(future, label='Future Predictions', color='red')

plt.title('Holt-Winters Netflix Subscribers Forecast')
plt.xlabel('Date')
plt.ylabel('Subscribers')
plt.legend()
plt.grid(True)
plt.show()

print("\nFuture Predictions:\n")
print(future)

```
### OUTPUT:


<img width="2008" height="1286" alt="image" src="https://github.com/user-attachments/assets/50b944a0-7450-4273-8981-1a17f8ac3f31" />

<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/f0d049a0-4dea-4508-b459-602ea20a9bb2" />

<img width="2012" height="1698" alt="image" src="https://github.com/user-attachments/assets/d7c819e1-2ff1-43b3-a137-7b5a900f3c4e" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
