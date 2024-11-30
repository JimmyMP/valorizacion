import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import os

# Configuración para evitar errores de codificación y mensajes informativos
sys.stdout.reconfigure(encoding='utf-8')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Cargar los datos normalizados
data = pd.read_csv('variables_fintech_normalizado.csv', sep=';', decimal='.')

# Separar características y la variable objetivo
data_caracteristicas = data.drop('Valorizacion', axis=1)  # 'Valorización' es la variable objetivo
Resultado_nota = data['Valorizacion']

# Normalizar la variable objetivo a un rango de -1 a 1
val_min = Resultado_nota.min()
val_max = Resultado_nota.max()
Resultado_nota_normalizada = (Resultado_nota - val_min) / (val_max - val_min) * 2 - 1

# Dividir en conjuntos de entrenamiento y prueba
data_caracteristicas_train, data_caracteristicas_test, Resultado_nota_train, Resultado_nota_test = train_test_split(
    data_caracteristicas, Resultado_nota_normalizada, test_size=0.2, random_state=42)

# Definir el modelo de red neuronal
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(data_caracteristicas.shape[1],)),
    tf.keras.layers.Dropout(0.3),  # Regularización para evitar sobreajuste
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Capa de salida para regresión
])

# Compilar el modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Entrenar el modelo
historial = model.fit(
    data_caracteristicas_train, Resultado_nota_train,
    epochs=500,  # Más épocas para un mejor ajuste
    validation_data=(data_caracteristicas_test, Resultado_nota_test),
    batch_size=32,  # Tamaño del lote
    verbose=1  # Mostrar progreso del entrenamiento
)

# Evaluar el modelo
loss, mae = model.evaluate(data_caracteristicas_test, Resultado_nota_test)

# Graficar la pérdida
plt.figure(figsize=(10, 6))
plt.plot(historial.history["loss"], label="Pérdida de entrenamiento")
plt.plot(historial.history["val_loss"], label="Pérdida de validación")
plt.xlabel("Época")
plt.ylabel("Pérdida (MSE)")
plt.title("Evolución de la pérdida durante el entrenamiento")
plt.legend()
plt.show()

# Imprimir la evaluación del modelo
print(f"La pérdida final del modelo es {loss:.4f} y el error absoluto medio (MAE) es {mae:.4f}")

# Realizar predicciones
predicciones_normalizadas = model.predict(data_caracteristicas_test)

# Desnormalizar las predicciones y los valores reales
predicciones_desescaladas = (predicciones_normalizadas + 1) / 2 * (val_max - val_min) + val_min
print(predicciones_desescaladas)
Resultado_nota_test_desescalado = (Resultado_nota_test + 1) / 2 * (val_max - val_min) + val_min

# Imprimir los datos reales y las predicciones
for i in range(len(data_caracteristicas_test)):
    real = Resultado_nota_test_desescalado.iloc[i]
    pred = predicciones_desescaladas[i][0]
    print(f"Datos reales: {real:.2f}, Predicción: {pred:.2f}")
