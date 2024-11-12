### DEVELOPED BY: Safeeq Fazil A
### REG NO : 212222240086
### Date: 16.10.2024
# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
 


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:

1. Load and preprocess the dataset by converting 'Date' to datetime, filtering the commodity, resampling to monthly averages, and handling missing values.

2. Plot the Autocorrelation Function (ACF) to check for correlation patterns in the time series.

3. Plot the Partial Autocorrelation Function (PACF) to determine the order of the Moving Average (MA) model.

4. Fit a Moving Average (MA) model using ARIMA, and make predictions for the specified period.

5. Apply a log transformation to the dataset and plot the transformed data to observe trends.

6. Fit an Exponential Smoothing model with both trend and seasonal components, and generate forecasts.

7. Visualize and compare the original data, MA model predictions, and Exponential Smoothing forecasts.

### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the dataset
file_path = '/content/vegetable.csv'
data = pd.read_csv(file_path)

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Select data for a specific commodity, e.g., 'Tomato Big(Nepali)'
commodity_data = data[data['Commodity'] == 'Tomato Big(Nepali)']

# Set 'Date' as the index
commodity_data.set_index('Date', inplace=True)

# Resample the data to monthly averages
monthly_data = commodity_data['Average'].resample('M').mean()

# Drop any NaN values
monthly_data = monthly_data.dropna()

# --- 1. ACF and PACF Plots ---

# Plot ACF
plt.figure(figsize=(10, 6))
plot_acf(monthly_data, lags=20)
plt.title('Autocorrelation Function (ACF)')
plt.show()

# Plot PACF
plt.figure(figsize=(10, 6))
plot_pacf(monthly_data, lags=20)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# --- 2. Moving Average Model (MA) ---

# Fit the MA model (order q=2, for instance)
ma_model = ARIMA(monthly_data, order=(0, 0, 2)).fit()

# Make predictions for the last 12 months
ma_predictions = ma_model.predict(start=len(monthly_data) - 12, end=len(monthly_data) - 1)

# Plot the Moving Average Model predictions
plt.figure(figsize=(10, 6))
plt.plot(monthly_data, label='Original Data', color='blue')
plt.plot(ma_predictions, label='MA Model Predictions', color='green')
plt.title('Moving Average (MA) Model Predictions')
plt.legend(loc='best')
plt.show()

# --- 3. Plot Transformed Dataset ---

# Apply log transformation to the dataset (to make trends more visible)
transformed_data = np.log(monthly_data)

# Plot the transformed dataset
plt.figure(figsize=(10, 6))
plt.plot(transformed_data, label='Log-Transformed Data', color='purple')
plt.title('Log-Transformed Monthly Average Prices')
plt.xlabel('Date')
plt.ylabel('Log(Average Price)')
plt.legend(loc='best')
plt.show()

# --- 4. Exponential Smoothing ---

# Fit the Exponential Smoothing model (with trend and seasonal components)
exp_smoothing_model = ExponentialSmoothing(monthly_data, trend='add', seasonal='add', seasonal_periods=12).fit()

# Make predictions for the next 12 months
exp_smoothing_predictions = exp_smoothing_model.forecast(12)

# Plot the Exponential Smoothing predictions
plt.figure(figsize=(10, 6))
plt.plot(monthly_data, label='Original Data', color='blue')
plt.plot(exp_smoothing_predictions, label='Exponential Smoothing Forecast', color='red')
plt.title('Exponential Smoothing Forecast')
plt.legend(loc='best')
plt.show()

```

### OUTPUT:

### Moving Average
![image](https://github.com/user-attachments/assets/83bf6b5c-ae16-4f42-ae61-5677fe52fc43)

### ACF - PACF
![image](https://github.com/user-attachments/assets/3f8ae04c-ef52-4e57-9b52-ea60a6c4d739)
![image](https://github.com/user-attachments/assets/f1b66a99-f72f-4cde-89f1-b6e1982bb15e)


### Exponential Smoothing
![image](https://github.com/user-attachments/assets/d553bc96-11d7-43b5-8f3d-766006b102d5)



### RESULT:
Thus the python program for implemented the Moving Average Model and Exponential smoothing was executed successfully.
