# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Cargar el dataset
folder_route = './datasets/'
file_path = folder_route + 'smart_mobility_dataset.csv'  # Cambia el path si es necesario
df = pd.read_csv(file_path)

# 2. Convertir Timestamp a tipo datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# 3. Seleccionar las columnas relevantes para tráfico
features = [
    'Vehicle_Count',
    'Traffic_Speed_kmh',
    'Road_Occupancy_%',
    'Accident_Report',
    'Sentiment_Score',
    'Ride_Sharing_Demand',
    'Parking_Availability',
    'Emission_Levels_g_km',
    'Energy_Consumption_L_h'
]
traffic_data = df[features]

# ---------------------------
# Visualización 1: Valores Faltantes
# ---------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(traffic_data.isnull(), cbar=False, cmap='viridis')
plt.title('Valores Faltantes en el Dataset')
plt.show()

# 4. Limpieza de datos
if traffic_data.isnull().values.any():
    traffic_data = traffic_data.fillna(traffic_data.median())

# ---------------------------
# Visualización 2: Matriz de Correlación
# ---------------------------
plt.figure(figsize=(12, 8))
sns.heatmap(traffic_data.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación de Variables de Tráfico')
plt.show()

# 5. Normalización de los datos
scaler = StandardScaler()
traffic_data_normalized = scaler.fit_transform(traffic_data)

# 6. Aplicar PCA para reducción de dimensionalidad
pca = PCA(n_components=0.95)  # Mantener 95% de la varianza
traffic_data_pca = pca.fit_transform(traffic_data_normalized)

# ---------------------------
# Visualización 3: Varianza Explicada Acumulada
# ---------------------------
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Acumulada')
plt.title('Varianza Explicada por PCA')
plt.grid(True)
plt.show()

# ---------------------------
# Visualización 4: Distribución en PC1 vs PC2
# ---------------------------
pca_columns = [f'PC{i+1}' for i in range(traffic_data_pca.shape[1])]
pca_df = pd.DataFrame(traffic_data_pca, columns=pca_columns)

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', data=pca_df)
plt.title('Distribución de Datos en PC1 vs PC2')
plt.xlabel('Componente Principal 1 (PC1)')
plt.ylabel('Componente Principal 2 (PC2)')
plt.grid(True)
plt.show()

# ---------------------------
# NUEVO: Visualización 5: Variables que componen PC1 y PC2
# ---------------------------
# Obtener las cargas de las variables (loadings)
loadings = pd.DataFrame(pca.components_.T, columns=pca_columns, index=features)

# Solo mostrar PC1 y PC2
pc1_pc2_loadings = loadings[['PC1', 'PC2']]

# Crear gráfico
plt.figure(figsize=(14, 8))

# Gráfico para PC1
plt.subplot(1, 2, 1)
pc1_pc2_loadings['PC1'].sort_values().plot(kind='barh', color='skyblue')
plt.title('Variables que Componen PC1')
plt.xlabel('Peso (Importancia)')
plt.grid(True)

# Gráfico para PC2
plt.subplot(1, 2, 2)
pc1_pc2_loadings['PC2'].sort_values().plot(kind='barh', color='lightcoral')
plt.title('Variables que Componen PC2')
plt.xlabel('Peso (Importancia)')
plt.grid(True)

plt.tight_layout()
plt.show()

# 7. Agregar el Timestamp de nuevo
pca_df['Timestamp'] = df['Timestamp'].values

# 8. Guardar el dataset preprocesado
pca_df.to_csv(folder_route + 'traffic_data_preprocessed.csv', index=False)

print("✅ Proceso completo: Dataset limpio, reducido, normalizado y visualizaciones generadas.")
