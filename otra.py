import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generar datos sintéticos con combinaciones no lineales
def generar_datos_sinteticos_no_lineales(X, y, n_samples=100, ruido=0.05):
    """
    Genera datos sintéticos para regresión mediante combinaciones no lineales y agrega ruido.
    """
    X = X.values  # Convertir a NumPy
    y = y.values
    n_original = len(X)
    indices = np.random.randint(0, n_original, size=(n_samples, 2))  # Pares de puntos aleatorios
    
    # Crear combinaciones no lineales
    X_sintetico = np.array([
        np.sqrt(X[i] * X[j]) * np.random.uniform(0.8, 1.2) for i, j in indices
    ])
    y_sintetico = np.array([
        (y[i] + y[j]) / 2 * np.random.uniform(0.8, 1.2) for i, j in indices
    ])
    
    # Agregar ruido
    X_sintetico += np.random.normal(0, ruido, X_sintetico.shape)
    y_sintetico += np.random.normal(0, ruido, y_sintetico.shape)
    
    # Combinar datos originales y sintéticos
    X_total = np.vstack([X, X_sintetico])
    y_total = np.concatenate([y, y_sintetico])
    return X_total, y_total

# Cargar datos originales
data = pd.read_csv('variables_fintech_normalizado.csv', sep=';', decimal='.')
data_caracteristicas = data.drop('Valorizacion', axis=1)
Resultado_nota = data['Valorizacion']

# Generar datos sintéticos
data_caracteristicas_sintetico, Resultado_nota_sintetico = generar_datos_sinteticos_no_lineales(
    data_caracteristicas, Resultado_nota, n_samples=200, ruido=0.05)

# Normalizar características
scaler = StandardScaler()
data_caracteristicas_sintetico_scaled = scaler.fit_transform(data_caracteristicas_sintetico)

# Normalizar la variable objetivo
val_min = Resultado_nota_sintetico.min()
val_max = Resultado_nota_sintetico.max()
Resultado_nota_sintetico_normalizada = (Resultado_nota_sintetico - val_min) / (val_max - val_min) * 2 - 1

# Dividir datos en entrenamiento y prueba
data_caracteristicas_train, data_caracteristicas_test, Resultado_nota_train, Resultado_nota_test = train_test_split(
    data_caracteristicas_sintetico_scaled, Resultado_nota_sintetico_normalizada, test_size=0.2, random_state=42)

# Modelo con más capas y regularización
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(data_caracteristicas_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),  # Dropout para evitar sobreajuste
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Capa de salida
])

# Compilar el modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

# Entrenar el modelo
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
historial = model.fit(
    data_caracteristicas_train, Resultado_nota_train,
    epochs=500,
    validation_data=(data_caracteristicas_test, Resultado_nota_test),
    batch_size=32,
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
plt.title("Evolución de la pérdida durante el entrenamiento")
plt.legend()
plt.show()

# Predicciones
predicciones_normalizadas = model.predict(data_caracteristicas_test)

# Desnormalizar las predicciones
predicciones_desescaladas = (predicciones_normalizadas + 1) / 2 * (val_max - val_min) + val_min
Resultado_nota_test_desescalado = (Resultado_nota_test + 1) / 2 * (val_max - val_min) + val_min

# Comparar valores reales vs predicciones
for i in range(len(data_caracteristicas_test)):
    real = Resultado_nota_test_desescalado[i]
    pred = predicciones_desescaladas[i][0]
    print(f"Datos reales: {real:.2f}, Predicción: {pred:.2f}")
