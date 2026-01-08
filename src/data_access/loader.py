import pandas as pd
from typing import List
from datetime import datetime
from src.domain.dtos import DrawHistoryDTO

class MelateLoader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load_data(self) -> DrawHistoryDTO:
        try:
            # 1. Cargar CSV
            df = pd.read_csv(self.csv_path)
            
            # Limpieza de nombres de columnas (quitar espacios y poner en mayúsculas)
            df.columns = df.columns.str.strip().str.upper()
            
            # 2. Procesar Columna 'CONCURSO' (Número oficial del sorteo)
            if 'CONCURSO' in df.columns:
                concursos = df['CONCURSO'].astype(int).tolist()
            else:
                # Si no existe la columna, generamos una secuencia como fallback
                print("⚠️ Columna 'CONCURSO' no detectada. Generando numeración automática.")
                concursos = list(range(1, len(df) + 1))
            
            # 3. Procesar Fechas (Columna 'FECHA')
            # dayfirst=True para formato dd/mm/yyyy común en México
            dates = pd.to_datetime(df['FECHA'], dayfirst=True).dt.date.tolist()
            
            # 4. Procesar Números Ganadores (Columnas F1 a F6)
            cols_juego = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']
            
            # Validar que existan las columnas de los números
            if not all(col in df.columns for col in cols_juego):
                raise ValueError(f"El CSV no tiene las columnas esperadas: {cols_juego}")

            # Extraer valores y convertir a lista de listas
            winning_numbers = df[cols_juego].values.tolist()
            
            # Ordenar los números de cada sorteo de menor a mayor para consistencia
            winning_numbers = [sorted([int(n) for n in nums]) for nums in winning_numbers]
            
            print(f"✅ Histórico MRPRO cargado: {len(dates)} sorteos.")
            print(f"📊 Rango de concursos detectado: {concursos[0]} al {concursos[-1]}")
            
            # 5. Retornar el DTO con los tres campos necesarios
            return DrawHistoryDTO(
                dates=dates, 
                winning_numbers=winning_numbers, 
                concursos=concursos
            )

        except FileNotFoundError:
            print(f"❌ Archivo no encontrado: {self.csv_path}")
            return DrawHistoryDTO(dates=[], winning_numbers=[], concursos=[])
        except Exception as e:
            print(f"❌ Error crítico leyendo histórico: {e}")
            return DrawHistoryDTO(dates=[], winning_numbers=[], concursos=[])