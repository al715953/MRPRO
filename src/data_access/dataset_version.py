import hashlib
import os
from typing import Any, Dict

import pandas as pd


def compute_dataset_version(csv_path: str) -> Dict[str, Any]:
    """
    Calcula una huella mínima del dataset para trazabilidad de corridas.
    """
    info: Dict[str, Any] = {
        "dataset_hash": "",
        "row_count": 0,
        "max_concurso": None,
    }

    if not csv_path or not os.path.exists(csv_path):
        return info

    with open(csv_path, "rb") as f:
        file_bytes = f.read()
    info["dataset_hash"] = hashlib.sha256(file_bytes).hexdigest()

    df = pd.read_csv(csv_path)
    info["row_count"] = int(len(df))

    if "CONCURSO" in df.columns:
        conc = pd.to_numeric(df["CONCURSO"], errors="coerce").dropna()
        if not conc.empty:
            info["max_concurso"] = int(conc.max())

    return info
