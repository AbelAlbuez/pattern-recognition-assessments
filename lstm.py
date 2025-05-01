#!/usr/bin/env python3
"""
Traffic-level prediction with PCA + LSTM
---------------------------------------

• Lee smart_mobility_dataset.csv
• Preprocesa y codifica variables
• Normaliza, aplica PCA (95 % varianza explicada)
• Crea secuencias temporales (look-back = 10 pasos)
• Entrena un LSTM multicategoría
"""

# ------------ 1. Librerías ------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# ------------ 2. Parámetros ------------
CSV_PATH       = "./datasets/smart_mobility_dataset.csv"
LOOK_BACK      = 10        # nº de pasos que miramos hacia atrás
TEST_SIZE      = 0.2
VAL_SIZE       = 0.2
PCA_VAR_THRES  = 0.95      # % de varianza a preservar
EPOCHS         = 50
BATCH_SIZE     = 64

# ------------ 3. Cargar datos ------------
df = pd.read_csv(CSV_PATH)

# 3.1. Convertimos la marca de tiempo y extraemos atributos cíclicos hora/día
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["hour"]      = df["Timestamp"].dt.hour
df["dayofweek"] = df["Timestamp"].dt.dayofweek
df.drop(columns=["Timestamp"], inplace=True)

# ------------ 4. Separar objetivo ------------
y_raw = df.pop("Traffic_Condition").values            # High / Medium / Low
X_raw = df                                            # resto de columnas

# ------------ 5. Codificar categóricas ------------
cat_cols = X_raw.select_dtypes(include="object").columns.tolist()
num_cols = X_raw.select_dtypes(exclude="object").columns.tolist()

ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
X_cat = ohe.fit_transform(X_raw[cat_cols]) if cat_cols else np.empty((len(df),0))
X_num = X_raw[num_cols].values

# Concatenamos categóricas y numéricas
X_full = np.hstack([X_num, X_cat])

# Guardamos el nº de features antes de PCA (solo informativo)
print(f"Features antes de PCA: {X_full.shape[1]}")

# ------------ 6. Escalado y PCA ------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_full)

pca = PCA(n_components=PCA_VAR_THRES, svd_solver="full")
X_pca  = pca.fit_transform(X_scaled)
print(f"Features después de PCA (95% var.): {X_pca.shape[1]}")

# ------------ 7. Secuencias para LSTM ------------
def make_sequences(X, y, look_back):
    """Convierte X,y en pares (seq_x, y) con ventana deslizante."""
    seq_X, seq_y = [], []
    for i in range(look_back, len(X)):
        seq_X.append(X[i-look_back:i, :])
        seq_y.append(y[i])
    return np.array(seq_X), np.array(seq_y)

# Etiquetado: High/Medium/Low  ->  0/1/2
lbl = LabelEncoder()
y_encoded = lbl.fit_transform(y_raw)

# Generar secuencias
X_seq, y_seq = make_sequences(X_pca, y_encoded, LOOK_BACK)

# One-hot para salida softmax
num_classes = len(lbl.classes_)
y_seq_oh = np.eye(num_classes)[y_seq]

# ------------ 8. Train / Val / Test split ------------
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq_oh, test_size=TEST_SIZE, shuffle=False)

# dividir train en train+val (manteniendo orden temporal)
val_len = int(len(X_train) * VAL_SIZE)
X_val, y_val   = X_train[-val_len:], y_train[-val_len:]
X_train, y_train = X_train[:-val_len], y_train[:-val_len]

# ------------ 9. Definir LSTM ------------
model = Sequential([
    LSTM(64, input_shape=(LOOK_BACK, X_seq.shape[2]), return_sequences=False),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# ------------ 10. Entrenamiento ------------
early = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early],
    verbose=2
)

# ------------ 11. Evaluación ------------
print("\n=== Test set ===")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"accuracy: {test_acc:.3f} | loss: {test_loss:.3f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=lbl.classes_))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))
