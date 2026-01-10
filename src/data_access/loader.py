import pandas as pd
from typing import List
from datetime import datetime
from src.domain.dtos import DrawHistoryDTO


class MelateLoader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load_data(self) -> DrawHistoryDTO:
        """
        Carga el histórico de Melate Retro.
        Estructura esperada: 6 Naturales + 1 Adicional.
        """
        try:
            # 1. Cargar CSV
            df = pd.read_csv(self.csv_path)

            # Limpieza de nombres de columnas
            df.columns = df.columns.str.strip().str.upper()

            # 2. Procesar Columna 'CONCURSO'
            if "CONCURSO" in df.columns:
                concursos = df["CONCURSO"].astype(int).tolist()
            else:
                print(
                    "⚠️ Columna 'CONCURSO' no detectada. Generando numeración automática."
                )
                concursos = list(range(1, len(df) + 1))

            # 3. Procesar Fechas
            dates = pd.to_datetime(df["FECHA"], dayfirst=True).dt.date.tolist()

            # 4. Procesar Números Ganadores
            # Buscamos columnas F1...F6
            cols_juego = [f"F{i}" for i in range(1, 7)]

            # Verificación de columnas naturales
            if not all(col in df.columns for col in cols_juego):
                raise ValueError("El CSV no contiene las columnas F1...F6")

            # Buscamos la columna del Adicional (F7 o ADICIONAL)
            col_adicional = None
            if "F7" in df.columns:
                col_adicional = "F7"
            elif "ADICIONAL" in df.columns:
                col_adicional = "ADICIONAL"
            else:
                raise ValueError("Falta columna de número adicional (F7 o ADICIONAL)")

            # Extraemos todo junto primero
            cols_totales = cols_juego + [col_adicional]
            raw_numbers = df[cols_totales].values.tolist()

            # --- CORRECCIÓN CRÍTICA DE ORDENAMIENTO ---
            # Ordenamos SOLO los primeros 6 (Naturales) y dejamos el 7º (Adicional) fijo al final.
            winning_numbers = []
            for row in raw_numbers:
                ints = [int(n) for n in row]
                naturales = sorted(ints[:6])  # Ordenar solo naturales
                adicional = ints[6]  # El adicional se respeta tal cual
                winning_numbers.append(naturales + [adicional])

            print(f"✅ Histórico MRPRO cargado: {len(dates)} sorteos.")
            print(f"📊 Rango de concursos: {concursos[0]} al {concursos[-1]}")

            return DrawHistoryDTO(
                dates=dates, winning_numbers=winning_numbers, concursos=concursos
            )

        except FileNotFoundError:
            print(f"❌ Archivo no encontrado: {self.csv_path}")
            return DrawHistoryDTO(dates=[], winning_numbers=[], concursos=[])
        except Exception as e:
            print(f"❌ Error crítico leyendo histórico: {e}")
            return DrawHistoryDTO(dates=[], winning_numbers=[], concursos=[])

    def load_history(self) -> DrawHistoryDTO:
        """Alias para mantener compatibilidad con main.py"""
        return self.load_data()
