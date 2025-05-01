import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Crear carpeta de salida
output_dir = './dataset-pca'
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

# Prepare dataframe with all PCn and timestamps
df_pca = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
df_pca['Timestamp'] = df['Timestamp']
df_pca.set_index('Timestamp', inplace=True)
df_pca = df_pca.asfreq('5min')
df_pca = df_pca.interpolate()

df_pca.to_csv(os.path.join(output_dir, 'pca_transformed_dataset.csv'))

