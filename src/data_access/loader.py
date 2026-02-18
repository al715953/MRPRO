import pandas as pd
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
            df = pd.read_csv(self.csv_path)
            df.columns = df.columns.str.strip().str.upper()

            if "CONCURSO" in df.columns:
                concursos = df["CONCURSO"].astype(int).tolist()
            else:
                print("⚠️ Columna 'CONCURSO' no detectada. Generando numeración automática.")
                concursos = list(range(1, len(df) + 1))

            dates = pd.to_datetime(df["FECHA"], dayfirst=True).dt.date.tolist()

            cols_juego = [f"F{i}" for i in range(1, 7)]
            if not all(col in df.columns for col in cols_juego):
                raise ValueError("El CSV no contiene las columnas F1...F6")

            if "F7" in df.columns:
                col_adicional = "F7"
            elif "ADICIONAL" in df.columns:
                col_adicional = "ADICIONAL"
            else:
                raise ValueError("Falta columna de número adicional (F7 o ADICIONAL)")

            raw_numbers = df[cols_juego + [col_adicional]].values.tolist()
            winning_numbers = []
            for row in raw_numbers:
                ints = [int(n) for n in row]
                winning_numbers.append(sorted(ints[:6]) + [ints[6]])

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


class TrisMultiplicadorLoader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load_data(self) -> DrawHistoryDTO:
        """
        Carga histórico de Tris con Multiplicador.
        Soporta columnas DIGITO1..DIGITO5, D1..D5 o F1..F5.
        """
        try:
            df = pd.read_csv(self.csv_path)
            df.columns = df.columns.str.strip().str.upper()

            if "CONCURSO" in df.columns:
                concursos = df["CONCURSO"].astype(int).tolist()
            else:
                concursos = list(range(1, len(df) + 1))

            if "FECHA" in df.columns:
                dates = pd.to_datetime(df["FECHA"], dayfirst=True).dt.date.tolist()
            else:
                dates = [None] * len(df)

            column_candidates = [
                [f"DIGITO{i}" for i in range(1, 6)],
                [f"D{i}" for i in range(1, 6)],
                [f"F{i}" for i in range(1, 6)],
            ]
            digit_columns = next(
                (cols for cols in column_candidates if all(c in df.columns for c in cols)),
                None,
            )
            if not digit_columns:
                raise ValueError(
                    "No se encontraron columnas de dígitos válidas (DIGITO1..5 / D1..5 / F1..5)."
                )

            winning_numbers = [
                [int(v) for v in row]
                for row in df[digit_columns].astype(int).values.tolist()
            ]

            return DrawHistoryDTO(
                dates=dates, winning_numbers=winning_numbers, concursos=concursos
            )
        except FileNotFoundError:
            print(f"❌ Archivo no encontrado: {self.csv_path}")
            return DrawHistoryDTO(dates=[], winning_numbers=[], concursos=[])
        except Exception as e:
            print(f"❌ Error crítico leyendo histórico Tris: {e}")
            return DrawHistoryDTO(dates=[], winning_numbers=[], concursos=[])

    def load_history(self) -> DrawHistoryDTO:
        return self.load_data()
