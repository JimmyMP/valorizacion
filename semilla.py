import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import random

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

# Definir una función para construir el modelo
def build_model():
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
    return model

# Configurar múltiples semillas y almacenar resultados
seeds = [42, 64, 123, 256, 512]
resultados = []

for seed in seeds:
    print(f"\nEntrenando con semilla: {seed}")
    
    # Configurar las semillas para garantizar reproducibilidad
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Construir y compilar el modelo
    model = build_model()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])
    
    # Configurar EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    
    # Entrenar el modelo
    historial = model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=0)
    
    # Evaluar el modelo
    loss, mse, mae = model.evaluate(X_test, y_test, verbose=0)
    resultados.append({'semilla': seed, 'loss': loss, 'mse': mse, 'mae': mae})
    print(f"Semilla {seed} - Pérdida: {loss}, MSE: {mse}, MAE: {mae}")

# Calcular promedios y desviación estándar
losses = [r['loss'] for r in resultados]
mses = [r['mse'] for r in resultados]
maes = [r['mae'] for r in resultados]

print("\nResultados promediados:")
print(f"Pérdida promedio: {np.mean(losses):.4f}, Desviación estándar: {np.std(losses):.4f}")
print(f"MSE promedio: {np.mean(mses):.4f}, Desviación estándar: {np.std(mses):.4f}")
print(f"MAE promedio: {np.mean(maes):.4f}, Desviación estándar: {np.std(maes):.4f}")
