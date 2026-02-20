# src/data_access/loader.py

import pandas as pd
import os
from datetime import datetime
from src.domain.dtos import DrawHistoryDTO
from src.data_access.config import DATA_FOLDER


class LotteryLoader:
    """
    Loader universal para MRPRO.
    Capaz de procesar Melate (Combinación) y Tris (Permutación).
    """

    def __init__(self, profile):
        self.profile = profile
        self.csv_path = os.path.join(DATA_FOLDER, profile.csv_filename)

    def load_data(self) -> DrawHistoryDTO:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"No se encontró el archivo: {self.csv_path}")

        # Cargamos el CSV
        df = pd.read_csv(self.csv_path)

        # 1. Normalización de Fechas
        # Intentamos varios formatos comunes en los reportes de Pronósticos
        for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
            try:
                df["FECHA"] = pd.to_datetime(df["FECHA"], format=fmt)
                break
            except (ValueError, KeyError):
                continue

        # 2. Extracción de Números según el Perfil
        if self.profile.code == "melate_retro":
            return self._process_melate(df)
        elif "tris" in self.profile.code:
            return self._process_tris(df)
        else:
            raise ValueError(
                f"Lógica de carga no implementada para: {self.profile.code}"
            )

    def _process_melate(self, df: pd.DataFrame) -> DrawHistoryDTO:
        """Lógica para Melate: 6 números naturales + 1 adicional. Ordenamiento activo."""
        concursos = df["CONCURSO"].tolist()
        fechas = df["FECHA"].tolist()

        # En Melate, el orden de aparición no importa, el modelo entrena mejor con datos ordenados
        numeros_raw = df[["F1", "F2", "F3", "F4", "F5", "F6"]].values
        naturales = [sorted(x.tolist()) for x in numeros_raw]
        additional_col = next(
            (col for col in ("ADICIONAL", "F7") if col in df.columns),
            None,
        )
        if additional_col is None:
            adicionales = [0] * len(concursos)
        else:
            adicionales = (
                pd.to_numeric(df[additional_col], errors="coerce")
                .fillna(0)
                .astype(int)
                .tolist()
            )

        winning_numbers = [naturales[i] + [adicionales[i]] for i in range(len(naturales))]

        return DrawHistoryDTO(
            concursos=concursos,
            dates=fechas,
            winning_numbers=winning_numbers,
        )

    def _process_tris(self, df: pd.DataFrame) -> DrawHistoryDTO:
        """Lógica para Tris: 5 dígitos independientes (0-9). PROHIBIDO ORDENAR."""
        concursos = df["CONCURSO"].tolist()
        fechas = df["FECHA"].tolist()

        # Tris usa R1 a R5. Mantenemos la posición exacta (es una permutación)
        # Si tu CSV de Tris usa nombres de columnas distintos, cámbialos aquí:
        cols_tris = ["R1", "R2", "R3", "R4", "R5"]

        # Verificamos que existan las columnas, si no, intentamos extraer de una cadena
        if set(cols_tris).issubset(df.columns):
            numeros = df[cols_tris].values.tolist()
        else:
            # Si el CSV de Tris trae los números juntos como "12345"
            numeros = [list(map(int, list(str(x).zfill(5)))) for x in df["NUMEROS"]]

        return DrawHistoryDTO(
            concursos=concursos,
            dates=fechas,
            winning_numbers=numeros,
        )
