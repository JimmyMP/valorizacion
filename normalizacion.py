import pandas as pd
import numpy as np

# Leer el archivo CSV con las variables financieras
df = pd.read_csv('variables_fintech.csv', sep=';', encoding='utf-8')

# Reemplazar espacios en los nombres de las columnas por guiones bajos
df.columns = df.columns.str.replace(' ', '_')

# Función genérica de normalización
def normalizar(columna):
    min_val = columna.min()
    max_val = columna.max()
    return ((columna - min_val) / (max_val - min_val)) * 2 - 1

# Identificar las columnas a normalizar (excluyendo la variable objetivo "Capitalizacion_de_mercado")
columnas_a_normalizar = [
    "Ingreso_neto", "%_Crec_Ingreso_neto",
    "Beneficio_antes_prov", "%_Margen_Beneficio_antes_prov",
    "Ingreso_operacional", "%_Margen_Ingreso_operacional",
    "Beneficio_neto", "%_Margen_Beneficio_neto", "BPA",
    "%_Crec_BPA", "%_capital_comun_nivel_1", "%_Ratio_de_capital_nivel_1",
    "%_Ratio_de_capital"
]

# Aplicar la normalización a las columnas
for col in columnas_a_normalizar:
    if col in df.columns:  # Verificar que la columna exista
        df[col] = normalizar(df[col])
    else:
        print(f"Columna '{col}' no encontrada en el DataFrame.")

# Guardar los datos normalizados en formato UTF-8
df.to_csv('variables_fintech_normalizado.csv', index=False, sep=';', encoding='utf-8')
