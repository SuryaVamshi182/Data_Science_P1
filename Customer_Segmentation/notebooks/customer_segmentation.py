import pandas as pd 

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


