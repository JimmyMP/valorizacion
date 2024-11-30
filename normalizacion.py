import pandas as pd
import numpy as np

# Leer el archivo CSV con las variables financieras
df = pd.read_csv('variables_fintech.csv', sep=';', encoding='utf-8')

# Función genérica de normalización
def normalizar(columna):
    min_val = columna.min()
    max_val = columna.max()
    return ((columna - min_val) / (max_val - min_val)) * 2 - 1

# Identificar las columnas a normalizar
columnas_a_normalizar = [
    "Capitalizacion de mercado", "Valor contable por accion", "Depositos totales",
    "Prestamos totales", "Activos totales", "Ingreso neto, ajustado (aj)",
    "% Crecimiento (YoY)", "Beneficio antes de provisiones, ajustado (aj)", "% Margen",
    "Ingreso operacional, ajustado (aj)", "% Margen", "Beneficio neto, ajustado (aj)",
    "% Margen", "BPA (Beneficio por accion), ajustado (aj)", "% Crecimiento (YoY)",
    "% Capital comun nivel 1", "Ratio de capital nivel 1", "% Ratio de capital"
]

# Aplicar la normalización a las columnas
for col in columnas_a_normalizar:
    if col in df.columns:  # Verifica si la columna existe en el DataFrame
        df[col] = normalizar(df[col])

# Diccionario para renombrar las columnas
# Diccionario para renombrar las columnas
# Diccionario para renombrar las columnas
nuevo_nombre_columnas = {
    "Capitalizacion de mercado": "Capitalizacion_de_Mercado",
    "Valor contable por accion": "Valor_Contable_Por_Accion",
    "Depositos totales": "Depositos_Totales",
    "Prestamos totales": "Prestamos_Totales",
    "Activos totales": "Activos_Totales",
    "Ingreso neto, ajustado (aj)": "Ingreso_Neto_Ajustado",
    "% Crecimiento (YoY)": "Crecimiento_Interanual",
    "Beneficio antes de provisiones, ajustado (aj)": "Beneficio_Antes_Provisiones_Ajustado",
    "% Margen": "Porcentaje_Margen",
    "Ingreso operacional, ajustado (aj)": "Ingreso_Operacional_Ajustado",
    "Beneficio neto, ajustado (aj)": "Beneficio_Neto_Ajustado",
    "BPA (Beneficio por accion), ajustado (aj)": "BPA_Ajustado",
    "% Crecimiento (YoY)": "Crecimiento_BPA_Interanual",
    "% Capital comun nivel 1": "Porcentaje_Capital_Comun_Nivel_1",
    "Ratio de capital nivel 1": "Ratio_Capital_Nivel_1",
    "% Ratio de capital": "Porcentaje_Ratio_Capital"
}


# Renombrar las columnas
df.rename(columns=nuevo_nombre_columnas, inplace=True)

# Guardar los datos normalizados con los nuevos nombres
df.to_csv('variables_fintech_normalizado.csv', index=False, sep=';', encoding='utf-8')
