import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#loading of csv datsets using pandas

azure_df = pd.read_csv('datasets/azure_usage.csv')
external_df = pd.read_csv('datasets/external_factors.csv')


# Create output folder if it doesn't exist
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)



# Convert 'date' column to datetime in both dataframes
azure_df['date'] = pd.to_datetime(azure_df['date'])
external_df['date'] = pd.to_datetime(external_df['date'])

#performaing EDA in the dataset

print(azure_df.info())
print(azure_df.describe())
print(azure_df.isnull().sum())



# 2. Total CPU Usage Over Time
plt.figure(figsize=(10, 4))
azure_df.groupby('date')['usage_cpu'].sum().plot()
plt.title("Total CPU Usage Over Time")
plt.xlabel("Date")
plt.ylabel("Total CPU Usage")
plt.tight_layout()
plt.savefig(f'{output_dir}/total_cpu_usage_over_time.png')
plt.close()

# 3. Region-wise Total CPU Usage
plt.figure(figsize=(8, 5))
region_usage = azure_df.groupby('region')['usage_cpu'].sum().reset_index()
sns.barplot(data=region_usage, x='region', y='usage_cpu')
plt.title("Total CPU Usage by Region")
plt.xlabel("Region")
plt.ylabel("CPU Usage")
plt.tight_layout()
plt.savefig(f'{output_dir}/cpu_usage_by_region.png')
plt.close()

#4. CPU Usage Distribution by Resource Type
plt.figure(figsize=(8, 5))
sns.boxplot(data=azure_df, x='resource_type', y='usage_cpu')
plt.title("CPU Usage Distribution by Resource Type")
plt.xlabel("Resource Type")
plt.ylabel("CPU Usage")
plt.tight_layout()
plt.savefig(f'{output_dir}/cpu_usage_by_resource_type.png')
plt.close()

# 5. Correlation Heatmap (including external factors)
# Merge dataframes on date
merged_df = pd.merge(azure_df, external_df, on='date', how='left')
corr_cols = ['usage_cpu', 'usage_storage', 'users_active', 'economic_index', 'cloud_market_demand']
plt.figure(figsize=(8, 6))
sns.heatmap(merged_df[corr_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f'{output_dir}/correlation_heatmap.png')
plt.close()