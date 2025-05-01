import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Cargar datos
df = pd.read_csv("traffic_data_preprocessed.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp')

# 2. Normalizar PC1
pc1_series = df['PC1'].values.reshape(-1, 1)
scaler = MinMaxScaler()
pc1_scaled = scaler.fit_transform(pc1_series)

# 3. Crear secuencias
def create_sequences(data, sequence_length=5):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

X, y = create_sequences(pc1_scaled, sequence_length=5)

# 4. Dividir en entrenamiento y prueba
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Definir y entrenar modelo
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# 6. Predicciones
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_real = scaler.inverse_transform(y_test)

# 7. Visualización
plt.figure(figsize=(14, 6))
plt.plot(y_real, label='Real', linewidth=2)
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.title('LSTM Prediction vs Real PC1')
plt.xlabel('Time Step')
plt.ylabel('PC1 Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lstm_predictions.png")  # Opcional: guarda la imagen
plt.show()
