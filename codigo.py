import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
import random
# Configurar codificación para caracteres especiales
import sys
sys.stdout.reconfigure(encoding='utf-8')
random.seed(64)
np.random.seed(64)
tf.random.set_seed(64)
# Cargar los datos normalizados
data = pd.read_csv('variables_fintech_normalizado.csv', sep=';')

# Transformar la variable objetivo (log) y limitar valores extremos
data['Capitalizacion_de_mercado_log'] = np.log1p(data['Capitalizacion_de_mercado'])
percentil_5 = np.percentile(data['Capitalizacion_de_mercado_log'], 5)
percentil_95 = np.percentile(data['Capitalizacion_de_mercado_log'], 95)
data['Capitalizacion_de_mercado_log'] = data['Capitalizacion_de_mercado_log'].clip(lower=percentil_5, upper=percentil_95)

# Determinar características y variable objetivo
caracteristicas = data.drop(['Capitalizacion_de_mercado', 'Capitalizacion_de_mercado_log'], axis=1)
objetivo = data['Capitalizacion_de_mercado_log']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(caracteristicas, objetivo, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir el modelo de red neuronal optimizado
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1)
])

# Configurar una tasa de aprendizaje dinámica
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compilar el modelo
model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])

# Configurar EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# Entrenar el modelo
historial = model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test), callbacks=[early_stop])

# Evaluar el modelo
loss, mse, mae = model.evaluate(X_test, y_test)
print(f"Pérdida: {loss}, Error cuadrático medio: {mse}, Error absoluto medio: {mae}")

# Graficar la pérdida
plt.plot(historial.history['loss'], label='Pérdida de entrenamiento')
plt.plot(historial.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Realizar predicciones
predicciones_log = model.predict(X_test)
predicciones = np.expm1(predicciones_log)  # Destransformar predicciones

# Comparar resultados
for i in range(len(X_test)):
    real = np.expm1(y_test.iloc[i])  # Destransformar valor real
    print(f"Datos reales: {real}, Predicción: {predicciones[i][0]}")

plt.boxplot(data['Capitalizacion_de_mercado_log'])
plt.title("Distribución de la Capitalización de mercado")
plt.show()