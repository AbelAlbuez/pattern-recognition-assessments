import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import seaborn as sns

# Crear carpeta para guardar los resultados
output_dir = "lstm_forecasts"
os.makedirs(output_dir, exist_ok=True)

# Cargar el dataset transformado por PCA
df = pd.read_csv("pca_transformed_dataset.csv", parse_dates=['Timestamp'])
df.set_index('Timestamp', inplace=True)
df = df.asfreq('5min').interpolate()

# Componentes a procesar (PC1 - PC14)
components = [f"PC{i}" for i in range(1, 15) if f"PC{i}" in df.columns]

# Diccionario para guardar el MSE real de cada componente
mse_dict = {}

# Función para preparar las secuencias de entrada para LSTM
def create_sequences(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# Loop por cada componente PCA
for comp in components:
    print(f"Entrenando LSTM para {comp}...")
    series = df[comp].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series)

    X, y = create_sequences(series_scaled)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(32, activation='relu', input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_real = scaler.inverse_transform(y_test)

    mse = mean_squared_error(y_real, y_pred)
    mse_dict[comp] = mse

    # Guardar el gráfico
    plt.figure(figsize=(10, 4))
    plt.plot(y_real, label="Real")
    plt.plot(y_pred, label="Predicted", linestyle="--")
    plt.title(f"LSTM Forecast - {comp} (MSE: {mse:.4f})")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"forecast_{comp}.png"))
    plt.close()

# Guardar el MSE de todos los componentes en un CSV
mse_df = pd.DataFrame(list(mse_dict.items()), columns=["Component", "MSE"])
mse_df.to_csv(os.path.join(output_dir, "mse_summary.csv"), index=False)
print("✅ mse_summary.csv generado con éxito.")

# Leer el archivo generado por lstm.py
output_dir = "lstm_forecasts"
mse_df = pd.read_csv(os.path.join(output_dir, "mse_summary.csv"))

# Ordenar los componentes por menor MSE
mse_df_sorted = mse_df.sort_values("MSE", ascending=True)

# Crear gráfico de barras
plt.figure(figsize=(12, 5))
sns.barplot(data=mse_df_sorted, x="Component", y="MSE", color="skyblue")
plt.title("MSE Comparison of LSTM Models by PCA Component")
plt.xlabel("Component")
plt.ylabel("MSE")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mse_comparison.png"))
plt.close()

print("✅ mse_comparison.png generado con éxito.")