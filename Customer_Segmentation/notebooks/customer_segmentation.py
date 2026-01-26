import pandas as pd 
import numpy as np

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.shape)
df.info()
print(df.head())

print(df.columns)

churn_counts=df['Churn'].value_counts()
print(churn_counts)
print(df['Churn'].value_counts(normalize=True))

df.isnull().sum()
print(df[df['TotalCharges'] == ' '].shape)

pd.crosstab(df['Contract'], df['Churn'], normalize='index')
print(df.groupby('Churn')['tenure'].describe())

#Data Cleaning
# Make sure that colums contain values of same datatype
# There should be no hidden missing values
# The data should be model-ready
# Understand why each data cleaning data step exists

df.drop(columns=['customerID'], inplace=True)#Drop the customerID column as it is identifier and not predictor
#Fix TotalCharges  --------- Strings are stored as objects, contains " " blank strings(hidden missing values)
# Convert blanks to NaN
df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan)
#Convert to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df.isnull().sum()

print(df['SeniorCitizen'].unique())

df['SeniorCitizen'] = df['SeniorCitizen'].map({0:'No', 1:'Yes'})

#Variable is numeric if you can perform mathematical operations on it 0->Not a senior citizen, 1-->Senior citizen which are only labels
df.info()

print(df.shape)
print(df['Churn'].value_counts(normalize=True))


