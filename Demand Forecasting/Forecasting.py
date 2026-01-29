# Look for trends
# Look for seasonality ----> weekly/monthly/yearly patterns
# Look for anomolies  ------> why they occured
# Build multiple models.    --------> why one model performs better than other
# How did it impact the business
# How often would you retrain the model      How would you trigger an alert
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/flights.csv")
print(df.head())

# Create a proper DateTime column
df['date'] = pd.to_datetime(
    df[['year', 'month', 'day']].rename(
        columns={'year':'year', 'month': 'month', 'day':'day'}
    )
)

# Aggregate
monthly_demand = (
    df
    .groupby(pd.Grouper(key='date', freq='MS'))
    .size()
    .to_frame(name='flights')
)

# Print
print(monthly_demand.head())
print(monthly_demand.index.is_unique)
print(monthly_demand.index.freq)

# Plot demand to observe the trends
plt.figure(figsize = (10,5))
plt.plot(monthly_demand.index, monthly_demand['flights'])
plt.title("Montly flight demand")
plt.xlabel("Year")
plt.ylabel("Number of years")
plt.show()

# Data Cleaning
# Is my time index reliable, continuous and trustwworthy for modeling
# No missing time periods
# No duplicate time stamps
# No weird values, proper frequency alignment

print(monthly_demand.index)
print(monthly_demand.asfreq('MS'))
print(monthly_demand.index.duplicated().sum())

# Look for broken trends/ detect outliers/anomolies visually
monthly_demand.plot(figsize=(12,5))

# Statistical check
Q1 = monthly_demand['flights'].quantile(0.25)
Q3 = monthly_demand['flights'].quantile(0.75)
IQR = Q3-Q1

outliers = monthly_demand[
    (monthly_demand['flights'] < Q1 - 1.5 * IQR) |
    (monthly_demand['flights'] > Q3 + 1.5 * IQR)
]                       # These months correspond to any holidays, strikes, pandemics or demand surges

print(outliers)

print(monthly_demand.describe())

# Exploratory Data Analysis
# To check if there is a trend/seasonality, anomolities and also model change
print(monthly_demand.head())
print(monthly_demand.tail())
print(monthly_demand.shape)

# Time series plot
monthly_demand.plot(figsize=(12,5), title="Monthly Flight demand")
plt.xlabel("Date")
plt.ylabel("Number of flights")
plt.show()

# Yearly seasonality check(month-wise pattern)
monthly_demand['month'] = monthly_demand.index.month

monthly_demand.boxplot(
    column = 'flights',
    by = 'month',
    figsize=(12,5)
)

plt.title("Monthly Seasonality")
plt.suptitle("")
plt.xlabel("Month")
plt.ylabel("flights")
plt.show()                  # expected  --------> summer = high demand winter = low demand pattern repeats every year

# Rolling stats -----> confirming the trend
monthly_demand['rolling_mean_12'] = (
    monthly_demand['flights'].rolling(window=12).mean()
)

monthly_demand[['flights', 'rolling_mean_12']].plot(figsize=(12,5))
plt.title("12-Month Rolling Mean")
plt.show()

# month-on-month growth
monthly_demand['mom_growth'] = (
    monthly_demand['flights'].pct_change(1)*100
)

monthly_demand['mom_growth'].plot(figsize=(12,4))
plt.title("Month-on-month growth (%)")
plt.ylabel("Growth %")
plt.show()

# Distribution check
monthly_demand['flights'].hist(bins = 20)
plt.title("Distribution of monthly flights")
plt.show()              # expected ------> right-skewed, increasing center over time

# Time-series decomposition
# Is there a trend/seasonality and also what's left over(noise/anomolies)
monthly_demand = monthly_demand.asfreq('MS')

# Use moving averages instead of decomposition because decomposition needs atleast 2 cycles
monthly_demand['rolling_3'] = monthly_demand['flights'].rolling(3).mean()

plt.figure(figsize=(12,5))
plt.plot(monthly_demand['flights'], label = 'Actual', marker = 'o')
plt.plot(monthly_demand['rolling_3'], label = '3-month MA')

plt.title("Monthly Flight Demand with Rolling averages")
plt.ylabel("Number of flights")
plt.legend()
plt.show()

# Anomaly detection (z-score)
monthly_demand['z-score'] = zscore(monthly_demand['flights'])

anomalies = monthly_demand[abs(monthly_demand['z-score']) > 2]
print(anomalies)                # high-spikes ------> holidays, strikes, weather.  low dips -----> disruptions

# Train-test split
train = monthly_demand.iloc[:9]         # Jan-Sep
test = monthly_demand.iloc[9:]          # Oct-Dec

# Baseline forecasting
test['naive_forecast'] = train['flights'].iloc[-1]
test['ma_forecast'] = train['flights'].rolling(3).mean().iloc[-1]

# Evaluation
mae_naive = mean_absolute_error(test['flights'], test['naive_forecast'])
mae_ma = mean_absolute_error(test['flights'], test['ma_forecast'])

print("Naive MAE:", mae_naive)
print("MA(3) MAE:", mae_ma)

# Show actual demand, forecasted demand, where the model starts predicting, how confident we should be (qualitatively)
plt.figure(figsize=(12,5))

# Actuals
plt.plot(monthly_demand.index, monthly_demand['flights'], label = "Actual", marker = 'o')

# Naive forecast
plt.plot(test.index, test['naive_forecast'], label = 'Naive Forecast', linestyle = '--')

# Moving average forecast
plt.plot(test.index, test['ma_forecast'], label = 'MA(3) Forecast', linestyle = '--')

# Vertical line: forecast start
plt.axvline(test.index[0], linestyle = ":", label = "Forecast start")

plt.title("Flight Demand Forecast vs Actuals")
plt.xlabel("Date")
plt.ylabel("Number of flights")
plt.legend()
plt.show()





































