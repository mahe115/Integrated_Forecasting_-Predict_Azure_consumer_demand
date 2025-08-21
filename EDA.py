import pandas as pd


#loading of csv datsets using pandas

azure_df = pd.read_csv('AZURE_BACKEND_TEAM-B/datasets/azure_usage.csv')
external_df = pd.read_csv('AZURE_BACKEND_TEAM-B/datasets/external_factors.csv')


#performaing EDA in the dataset

print(azure_df.info())
print(azure_df.describe())
print(azure_df.isnull().sum())
