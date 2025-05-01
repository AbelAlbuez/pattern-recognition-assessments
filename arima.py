import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Crear carpeta de salida
output_dir = 'arima_assets'
os.makedirs(output_dir, exist_ok=True)

# Load and preprocess dataset
df = pd.read_csv('./datasets/smart_mobility_dataset.csv', parse_dates=['Timestamp'])

# Encode categorical columns
label_cols = ['Traffic_Light_State', 'Weather_Condition', 'Traffic_Condition']
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Drop timestamp for PCA
features = df.drop(columns=['Timestamp'])

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply PCA
pca = PCA(n_components=None)
pca_result = pca.fit_transform(scaled_features)

# Plot cumulative explained variance
explained_variance = pca.explained_variance_ratio_.cumsum()
plt.figure(figsize=(10, 5))
plt.plot(explained_variance, marker='o')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'pca_variance.png'))

# Interpret PCA components: feature importance for each PC
component_df = pd.DataFrame(pca.components_, columns=features.columns)
component_df.index = [f'PC{i+1}' for i in range(len(component_df))]
component_df.to_csv(os.path.join(output_dir, 'pca_component_weights.csv'))
print("PCA Component Weights:\n", component_df)

# Visual heatmap of feature contributions
plt.figure(figsize=(12, 8))
sns.heatmap(component_df, cmap="coolwarm", center=0)
plt.title("Feature Weights per Principal Component")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pca_component_heatmap.png"))

# Prepare dataframe with all PCn and timestamps
df_pca = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
df_pca['Timestamp'] = df['Timestamp']
df_pca.set_index('Timestamp', inplace=True)
df_pca = df_pca.asfreq('5min')
df_pca = df_pca.interpolate()

df_pca.to_csv(os.path.join(output_dir, 'pca_transformed_dataset.csv'))


# Visualize PC1
df_pca['PC1'].plot(title='PC1 Time Series (Traffic Trend)', figsize=(12, 4))
plt.savefig(os.path.join(output_dir, 'pc1_time_series.png'))

# Comparison of ARIMA on all PCn
mse_results = []

for col in df_pca.columns:
    if col.startswith('PC'):
        print(f"Training ARIMA on {col}...")
        train_size = int(len(df_pca) * 0.8)
        train, test = df_pca[col][:train_size], df_pca[col][train_size:]

        try:
            model = ARIMA(train, order=(5, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(test))
            mse = mean_squared_error(test, forecast)
            mse_results.append({'Component': col, 'MSE': mse})

            # Plot forecast
            plt.figure(figsize=(10, 3))
            plt.plot(test.index, test, label='Actual')
            plt.plot(test.index, forecast, label='Forecast', color='orange')
            plt.title(f'ARIMA Forecast - {col} (MSE: {mse:.2f})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'forecast_{col}.png'))
            plt.close()
        except Exception as e:
            print(f"Failed on {col}: {e}")
            mse_results.append({'Component': col, 'MSE': float('inf')})

# Create summary table of MSE
mse_df = pd.DataFrame(mse_results)
mse_df = mse_df.sort_values(by='MSE')
mse_df.to_csv(os.path.join(output_dir, 'mse_summary.csv'), index=False)
print("\nMSE Summary:\n", mse_df)

# Plot MSEs
plt.figure(figsize=(10, 5))
sns.barplot(data=mse_df, x='Component', y='MSE', palette='Blues_d')
plt.xticks(rotation=45)
plt.title("MSE Comparison of ARIMA Models by PCA Component")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mse_comparison.png"))
