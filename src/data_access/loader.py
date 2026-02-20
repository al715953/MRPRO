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
        # Tris usa R1 a R5. Mantenemos la posicion exacta (es una permutacion).
        cols_tris = ["R1", "R2", "R3", "R4", "R5"]
        mult_col = next(
            (
                c
                for c in df.columns
                if c.strip().upper() in ("MULTIPLICADOR", "MULTIPLIER")
            ),
            None,
        )

        # Verificamos columnas; si no existen, usamos columna compacta NUMEROS.
        if set(cols_tris).issubset(df.columns):
            digits_df = df[cols_tris].apply(pd.to_numeric, errors="coerce")
            valid_mask = ~digits_df.isna().any(axis=1)
            digits_df = digits_df.loc[valid_mask].astype(int)

            concursos = df.loc[valid_mask, "CONCURSO"].tolist()
            fechas = df.loc[valid_mask, "FECHA"].tolist()
            if mult_col:
                mult_series = (
                    df.loc[valid_mask, mult_col]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .map(
                        {
                            "SI": 1,
                            "SÍ": 1,
                            "YES": 1,
                            "Y": 1,
                            "TRUE": 1,
                            "1": 1,
                        }
                    )
                    .fillna(0)
                    .astype(int)
                    .tolist()
                )
            else:
                mult_series = [0] * len(digits_df)

            digit_rows = digits_df.values.tolist()
            numeros = [digit_rows[i] + [mult_series[i]] for i in range(len(digit_rows))]
        else:
            concursos, fechas, numeros = [], [], []
            for _, row in df.iterrows():
                raw = row.get("NUMEROS")
                digits = "".join(ch for ch in str(raw) if ch.isdigit())
                if not digits:
                    continue
                digits = digits.zfill(5)[-5:]
                mult_raw = str(row.get(mult_col, "")).strip().upper() if mult_col else ""
                has_multiplier = int(
                    mult_raw in ("SI", "SÍ", "YES", "Y", "TRUE", "1")
                )
                concursos.append(row["CONCURSO"])
                fechas.append(row["FECHA"])
                numeros.append([int(ch) for ch in digits] + [has_multiplier])

        return DrawHistoryDTO(
            concursos=concursos,
            dates=fechas,
            winning_numbers=numeros,
        )


class TrisMultiplicadorLoader:
    """
    Adaptador legacy para pruebas/consumidores antiguos.
    Carga historicos Tris desde una ruta de CSV directa.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load_data(self) -> DrawHistoryDTO:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"No se encontró el archivo: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
            try:
                df["FECHA"] = pd.to_datetime(df["FECHA"], format=fmt)
                break
            except (ValueError, KeyError):
                continue

        digit_candidates = [
            [f"DIGITO{i}" for i in range(1, 6)],
            [f"D{i}" for i in range(1, 6)],
            [f"F{i}" for i in range(1, 6)],
            [f"R{i}" for i in range(1, 6)],
        ]

        digit_cols = next(
            (cols for cols in digit_candidates if set(cols).issubset(df.columns)),
            None,
        )

        if digit_cols:
            digits_df = df[digit_cols].apply(pd.to_numeric, errors="coerce")
            valid_mask = ~digits_df.isna().any(axis=1)
            digits_df = digits_df.loc[valid_mask].astype(int)
            concursos = df.loc[valid_mask, "CONCURSO"].tolist()
            fechas = df.loc[valid_mask, "FECHA"].tolist()
            numeros = digits_df.values.tolist()
        elif "NUMEROS" in df.columns:
            concursos, fechas, numeros = [], [], []
            for _, row in df.iterrows():
                digits = "".join(ch for ch in str(row["NUMEROS"]) if ch.isdigit())
                if not digits:
                    continue
                digits = digits.zfill(5)[-5:]
                concursos.append(row["CONCURSO"])
                fechas.append(row["FECHA"])
                numeros.append([int(ch) for ch in digits])
        else:
            raise ValueError(
                "CSV Tris sin columnas válidas: requiere DIGITO1..5 (o D/F/R1..5) o NUMEROS."
            )

        return DrawHistoryDTO(
            concursos=concursos,
            dates=fechas,
            winning_numbers=numeros,
        )
