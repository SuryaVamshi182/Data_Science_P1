import pandas as pd 
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

#EDA - Exploratory Data Analysis
# Who is more likely to churn?
# Which features strongly influence churn?
# Are there any obvious patterns before ML?

#Check target variable(Churn) distribution
df['Churn'].value_counts                # To see if the dataset is imbalanced and also chrun problems are often more 'No' than 'Yes'
# No >>> Yes class imbalance effects model choice and evaluation metrics

# Churn percentage
df['Churn'].value_counts(normalize = True)*100

#Churn vs categorical features
pd.crosstab(df['SeniorCitizen'], df['Churn'], normalize = 'index')*100

#Churn vs Contract type
pd.crosstab(df['Contract'], df['Churn'], normalize='index')*100     #month-to-month -> higher churn year/two-years -> lower churn

#Churn vs Payment method
pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index')*100    # EC->higher churn auto-payments -> lower churn

#Churn vs neumeric features
df.groupby('Churn')['MonthlyCharges'].mean()          # avg values for churned vs non-churned users churned users have higher montlhy charges

# Tenure vs Churn
df.groupby('Churn')['tenure'].mean()            # Low-tenure -> high churn

# Contract type is a strong churn driver

# Data Preprocessing
# ML models cannot undeerstand text like "Yes" or "No"
# Separate features(X) and target(Y) and convert categorical -> numeric
# Scale numeric values and split into train & test sets

# Define target and features
X = df.drop('Churn', axis = 1)           # dropping churn left are all customer attributes
y = df['Churn']                         # target variable

# Drop customerID because identifier and no business use
print(X.columns)
X = X.drop('customerID', axis=1, errors='ignore')

#Categorical vs numerical columns
categorical_cols = X.select_dtypes(include = 'object').columns
numerical_cols = X.select_dtypes(exclude = 'object').columns

print(categorical_cols)
print(numerical_cols)

# Encode target variable because models need numbers not strings
y = y.map({'No' : 0, 'Yes' : 1})

#One-Hot Encode Categorical features
X = pd.get_dummies(X, drop_first=True)           # each category becomes a column and dataset becomes fully numeric

#Scale the numeric features --- some models like logistic regression and knn depend on scale
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Test-train split  ---- to test the model on unseen data









