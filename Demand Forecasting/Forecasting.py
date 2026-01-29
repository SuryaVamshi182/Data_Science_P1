# Look for trends
# Look for seasonality ----> weekly/monthly/yearly patterns
# Look for anomolies  ------> why they occured
# Build multiple models.    --------> why one model performs better than other
# How did it impact the business
# How often would you retrain the model      How would you trigger an alert
import pandas as pd 
import matplotlib.pyplot as plt

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













