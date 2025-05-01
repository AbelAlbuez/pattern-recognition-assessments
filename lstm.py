import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from keras_tuner.tuners import RandomSearch

# Configurar carpeta de salida
output_dir = "lstm_forecasts"
os.makedirs(output_dir, exist_ok=True)

# Cargar dataset PCA
df = pd.read_csv("pca_transformed_dataset.csv", parse_dates=['Timestamp'])
df.set_index('Timestamp', inplace=True)
df = df.asfreq('5min').interpolate()

components = [f"PC{i}" for i in range(1, 15) if f"PC{i}" in df.columns]
mse_dict = {}

# ----------------------------
# Definir función de secuencias
# ----------------------------
def create_sequences(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# ----------------------------
# 1. Tuning solo para PC1
# ----------------------------
print("🔍 Buscando mejores hiperparámetros para PC1...")

series = df["PC1"].values.reshape(-1, 1)
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series)
X, y = create_sequences(series_scaled)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

def build_model(hp):
    model = Sequential()
    model.add(LSTM(
        units=hp.Int('units', 32, 128, step=32),
        activation=hp.Choice('activation', ['relu', 'tanh']),
        input_shape=(X_train.shape[1], 1)
    ))
    model.add(Dense(1))
    model.compile(
        optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mse'
    )
    return model

tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=1,
    directory='lstm_tuning',
    project_name='pc1_tuning'
)
tuner.search(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2, verbose=0)
best_hp = tuner.get_best_hyperparameters(1)[0]

# Guardar hiperparámetros óptimos
with open(os.path.join(output_dir, "best_hyperparameters.txt"), "w") as f:
    for key, val in best_hp.values.items():
        f.write(f"{key}: {val}\n")

print("✅ Mejores hiperparámetros encontrados y guardados.")

# ----------------------------
# 2. Entrenamiento final con esos hiperparámetros
# ----------------------------
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
        LSTM(units=best_hp['units'], activation=best_hp['activation'], input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=best_hp['learning_rate']), loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_real = scaler.inverse_transform(y_test)

    mse = mean_squared_error(y_real, y_pred)
    mse_dict[comp] = mse

    # Graficar resultados
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

# ----------------------------
# 3. Guardar resumen de errores
# ----------------------------
mse_df = pd.DataFrame(list(mse_dict.items()), columns=["Component", "MSE"])
mse_df.to_csv(os.path.join(output_dir, "mse_summary.csv"), index=False)

# ----------------------------
# 4. Gráfico comparativo de MSE
# ----------------------------
mse_df_sorted = mse_df.sort_values("MSE", ascending=True)
plt.figure(figsize=(12, 5))
sns.barplot(data=mse_df_sorted, x="Component", y="MSE", color="skyblue")
plt.title("MSE Comparison of LSTM Models by PCA Component")
plt.xlabel("Component")
plt.ylabel("MSE")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mse_comparison.png"))
plt.close()

# ----------------------------
# 5. Gráfico de hiperparámetros óptimos (numéricos)
# ----------------------------
numeric_hp = {k: v for k, v in best_hp.values.items() if isinstance(v, (int, float))}
hp_df = pd.DataFrame.from_dict(numeric_hp, orient='index', columns=['Value'])
hp_df.reset_index(inplace=True)
hp_df.columns = ['Hyperparameter', 'Value']

plt.figure(figsize=(8, 4))
sns.barplot(x='Hyperparameter', y='Value', data=hp_df, palette='Blues_d')
plt.title('Best Numeric Hyperparameters for LSTM (Tuned on PC1)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "best_hyperparameters_chart.png"))
plt.close()

print("✅ mse_summary.csv, mse_comparison.png y best_hyperparameters_chart.png generados con éxito.")
