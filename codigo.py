# Importar módulos necesarios
# Pandas se utiliza para la manipulación y análisis de datos
import pandas as pd

# TensorFlow es utilizado para crear y entrenar modelos de redes neuronales
import tensorflow as tf

# sklearn.model_selection proporciona herramientas para dividir los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split

# Matplotlib se usa para graficar los resultados del modelo
import matplotlib.pyplot as plt

# sys y os permiten ajustes de configuración para evitar errores de codificación y optimización
import sys
import os

# Configuración para evitar problemas de codificación en la consola y deshabilitar algunas optimizaciones de TensorFlow
sys.stdout.reconfigure(encoding='utf-8')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Cargar los datos normalizados
# Los datos se cargan desde un archivo CSV utilizando pandas. 
# El archivo contiene las características y la variable objetivo previamente normalizadas.
data = pd.read_csv('variables_fintech_normalizado.csv', sep=';', decimal='.')

# Separar características y la variable objetivo
# Separamos las características (variables independientes) de la variable objetivo (dependiente).
data_caracteristicas = data.drop('Valorizacion', axis=1)  # 'Valorización' es la variable objetivo
Resultado_nota = data['Valorizacion']

# Normalizar la variable objetivo
# La normalización de la variable objetivo a un rango de -1 a 1 ayuda a estabilizar el entrenamiento del modelo
# y asegura que todas las variables tengan una escala similar.
val_min = Resultado_nota.min()
val_max = Resultado_nota.max()
Resultado_nota_normalizada = (Resultado_nota - val_min) / (val_max - val_min) * 2 - 1
from sklearn.preprocessing import StandardScaler

# Normalizar las características
scaler = StandardScaler()
data_caracteristicas_scaled = scaler.fit_transform(data_caracteristicas)

# Dividir los datos en entrenamiento y prueba
# Dividimos los datos para evaluar el rendimiento del modelo con datos que no ha visto previamente.
# Usamos un 80% para entrenamiento y 20% para prueba.
data_caracteristicas_train, data_caracteristicas_test, Resultado_nota_train, Resultado_nota_test = train_test_split(
    data_caracteristicas_scaled, Resultado_nota_normalizada, test_size=0.2, random_state=42)


# Definir el modelo de red neuronal
# Se elige un modelo de red neuronal porque es flexible para modelar relaciones no lineales complejas
# entre las características y la variable objetivo.
model = tf.keras.models.Sequential([
    # Primera capa oculta con 128 neuronas y activación ReLU. ReLU es adecuada para redes profundas ya que evita problemas de gradientes desaparecidos.
    tf.keras.layers.Dense(64, activation='relu', input_shape=(data_caracteristicas.shape[1],)),
    # Segunda capa oculta con 64 neuronas.
    tf.keras.layers.Dense(32, activation='relu'),
    # Capa de salida con 1 neurona porque el objetivo es predecir un único valor numérico.
    tf.keras.layers.Dense(1)
])

# Compilar el modelo
# Se utiliza el optimizador Adam, que es eficiente y adapta la tasa de aprendizaje durante el entrenamiento.
# La pérdida se mide con error cuadrático medio (MSE), que penaliza más los errores grandes.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenar el modelo
# Entrenamos el modelo con un tamaño de lote de 32 y un número elevado de épocas (500) para asegurar un buen ajuste.
historial = model.fit(
    data_caracteristicas_train, Resultado_nota_train,
    epochs=200,  # Más épocas permiten que el modelo se ajuste mejor a los datos.
    validation_data=(data_caracteristicas_test, Resultado_nota_test),  # Usamos datos de validación para evaluar el progreso.
    batch_size=16,  # Tamaño del lote pequeño para un entrenamiento más detallado.
    callbacks=[early_stopping],
    verbose=1  # Mostrar progreso del entrenamiento en la consola.
)

# Evaluar el modelo
# La evaluación del modelo en el conjunto de prueba nos da una idea de cómo se comportará con datos nuevos.
loss, mae = model.evaluate(data_caracteristicas_test, Resultado_nota_test)

# Graficar la pérdida
# Se grafican las pérdidas (entrenamiento y validación) para observar si el modelo está sobreajustado o subajustado.
plt.figure(figsize=(10, 6))
plt.plot(historial.history["loss"], label="Pérdida de entrenamiento")
plt.plot(historial.history["val_loss"], label="Pérdida de validación")
plt.xlabel("Época")
plt.ylabel("Pérdida (MSE)")
plt.title("Evolución de la pérdida durante el entrenamiento")
plt.legend()
plt.show()

# Imprimir la evaluación del modelo
# Los valores finales de pérdida y MAE nos indican la precisión del modelo.
print(f"La pérdida final del modelo es {loss:.4f} y el error absoluto medio (MAE) es {mae:.4f}")

# Realizar predicciones
# Predicciones normalizadas se obtienen para los datos de prueba.
predicciones_normalizadas = model.predict(data_caracteristicas_test)

# Desnormalizar las predicciones y los valores reales
# Convertimos los valores de vuelta a su escala original para interpretar los resultados.
predicciones_desescaladas = (predicciones_normalizadas + 1) / 2 * (val_max - val_min) + val_min
Resultado_nota_test_desescalado = (Resultado_nota_test + 1) / 2 * (val_max - val_min) + val_min

# Imprimir los datos reales y las predicciones
# Se comparan los valores reales y predichos para evaluar la precisión.
for i in range(len(data_caracteristicas_test)):
    real = Resultado_nota_test_desescalado.iloc[i]
    pred = predicciones_desescaladas[i][0]
    print(f"Datos reales: {real:.2f}, Predicción: {pred:.2f}")
