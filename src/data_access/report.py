import pandas as pd
import os
from src.data_access.config import FILE_APUESTAS


def guardar_prediccion(tickets: list):
    """
    Guarda la lista de tickets generados en 'Mis_Apuestas.csv'
    para su posterior validación.
    """
    try:
        # Definir columnas F1..F6
        cols = [f"F{i}" for i in range(1, 7)]

        # Crear DataFrame
        df = pd.DataFrame(tickets, columns=cols)

        # Asegurar que el directorio data/ exista
        os.makedirs(os.path.dirname(FILE_APUESTAS), exist_ok=True)

        # Guardar (sobrescribiendo el anterior)
        df.to_csv(FILE_APUESTAS, index=False)
        print(f"📁 Archivo guardado correctamente: {FILE_APUESTAS}")

    except Exception as e:
        print(f"❌ Error al guardar reporte de apuestas: {e}")
