import pandas as pd
from typing import List
from datetime import datetime
from src.domain.dtos import DrawHistoryDTO

class MelateLoader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load_data(self) -> DrawHistoryDTO:
        try:
            # Cargar CSV
            df = pd.read_csv(self.csv_path)
            
            # Limpieza básica de nombres de columnas
            df.columns = df.columns.str.strip().str.upper()
            
            # 1. Procesar Fechas (Columna 'FECHA')
            # Formato detectado: dd/mm/yyyy (ej: 06/01/2026)
            dates = pd.to_datetime(df['FECHA'], dayfirst=True).dt.date.tolist()
            
            # 2. Procesar Números Ganadores (Columnas F1 a F6)
            # F7 suele ser el adicional, por ahora tomamos los 6 naturales para la predicción
            cols_juego = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']
            
            # Validar que existan las columnas
            if not all(col in df.columns for col in cols_juego):
                raise ValueError(f"El CSV no tiene las columnas esperadas: {cols_juego}")

            # Extraer valores y convertir a lista de listas
            winning_numbers = df[cols_juego].values.tolist()
            
            # (Opcional) Ordenar los números de cada sorteo de menor a mayor
            winning_numbers = [sorted(nums) for nums in winning_numbers]
            
            print(f"✅ Historico MRPRO cargado: {len(dates)} sorteos (Estructura F1-F6 detectada).")
            return DrawHistoryDTO(dates=dates, winning_numbers=winning_numbers)

        except FileNotFoundError:
            print(f"❌ Archivo no encontrado: {self.csv_path}")
            return DrawHistoryDTO(dates=[], winning_numbers=[])
        except Exception as e:
            print(f"❌ Error crítico leyendo histórico: {e}")
            return DrawHistoryDTO(dates=[], winning_numbers=[])