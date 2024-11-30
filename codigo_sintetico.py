# Importar las bibliotecas necesarias
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt

# Configuración
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Cargar los datos originales
data = pd.read_csv('variables_fintech_normalizado.csv', sep=';', decimal='.')

# Separar características y la variable objetivo
data_caracteristicas = data.drop('Valorizacion', axis=1)
Resultado_nota = data['Valorizacion']

# Generar datos sintéticos usando re-muestreo (duplicando datos con reemplazo)
data_sintetico = resample(data, n_samples=100, replace=True, random_state=42)

# Separar características y variable objetivo de los datos sintéticos
data_caracteristicas_sintetico = data_sintetico.drop('Valorizacion', axis=1)
Resultado_nota_sintetico = data_sintetico['Valorizacion']

# Normalizar características
scaler = StandardScaler()
data_caracteristicas_sintetico_scaled = scaler.fit_transform(data_caracteristicas_sintetico)

# Normalizar la variable objetivo
val_min = Resultado_nota_sintetico.min()
val_max = Resultado_nota_sintetico.max()
Resultado_nota_sintetico_normalizada = (Resultado_nota_sintetico - val_min) / (val_max - val_min) * 2 - 1

# Dividir en conjuntos de entrenamiento y prueba
data_caracteristicas_train, data_caracteristicas_test, Resultado_nota_train, Resultado_nota_test = train_test_split(
    data_caracteristicas_sintetico_scaled, Resultado_nota_sintetico_normalizada, test_size=0.2, random_state=42)

# Definir el modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(data_caracteristicas_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Capa de salida
])

# Compilar el modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Entrenar el modelo con EarlyStopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
historial = model.fit(
    data_caracteristicas_train, Resultado_nota_train,
    epochs=200,
    validation_data=(data_caracteristicas_test, Resultado_nota_test),
    batch_size=16,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluar el modelo
loss, mae = model.evaluate(data_caracteristicas_test, Resultado_nota_test)

# Graficar la pérdida
plt.figure(figsize=(10, 6))
plt.plot(historial.history["loss"], label="Pérdida de entrenamiento")
plt.plot(historial.history["val_loss"], label="Pérdida de validación")
plt.xlabel("Época")
plt.ylabel("Pérdida (MSE)")
plt.title("Evolución de la pérdida durante el entrenamiento (Con Datos Sintéticos)")
plt.legend()
plt.show()

# Realizar predicciones
predicciones_normalizadas = model.predict(data_caracteristicas_test)

# Desnormalizar las predicciones
predicciones_desescaladas = (predicciones_normalizadas + 1) / 2 * (val_max - val_min) + val_min
Resultado_nota_test_desescalado = (Resultado_nota_test + 1) / 2 * (val_max - val_min) + val_min

# Comparar valores reales vs predicciones
for i in range(len(data_caracteristicas_test)):
    real = Resultado_nota_test_desescalado.iloc[i]  # Usamos iloc para acceder a un valor escalar
    pred = predicciones_desescaladas[i][0]
    print(f"Datos reales: {real:.2f}, Predicción: {pred:.2f}")