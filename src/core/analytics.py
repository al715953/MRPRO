# src/core/analytics.py
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from src.data_access.config import DATA_FOLDER

# Intentar importar cupy para soporte de hardware acelerado
try:
    import cupy as cp
except ImportError:
    cp = None


class SniperJSONEncoder(json.JSONEncoder):
    """Codificador de élite para asegurar compatibilidad total GPU/CPU."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if cp and isinstance(obj, cp.ndarray):
            return obj.get().tolist()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super(SniperJSONEncoder, self).default(obj)


class PerformanceTracker:
    """Sistema de Registro y Telemetría V15 (Full Spectrum Log + JSON)."""

    def __init__(self):
        self.log_path = os.path.join(DATA_FOLDER, "detailed_forensic_log.csv")
        self.json_path = os.path.join(DATA_FOLDER, "backtest_results.json")
        self._ensure_log_exists()

    def _ensure_log_exists(self):
        if not os.path.exists(self.log_path):
            df = pd.DataFrame(
                columns=[
                    "timestamp",
                    "tag",
                    "draw_id",
                    "hits",
                    "univ_size",
                    "rank",
                    "proximity",
                    "ai_score",
                    "geo_score",
                    "sniper_log",
                ]
            )
            df.to_csv(self.log_path, index=False)

    def log_run(self, result_dto, tag, forensic_data):
        """
        Punto de Persistencia Dual: Guarda en CSV para histórico y JSON para el modelo estático.
        """
        if not forensic_data:
            return

        # --- FASE 1: PERSISTENCIA CSV (Forense) ---
        self._save_to_csv(tag, forensic_data)

        # --- FASE 2: PERSISTENCIA JSON (Resumen de Misión) ---
        self._save_to_json(result_dto, tag, forensic_data)

    def _save_to_csv(self, tag, forensic_data):
        try:
            new_rows = pd.DataFrame(forensic_data)
            new_rows["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            new_rows["tag"] = tag

            columns_order = [
                "timestamp",
                "tag",
                "draw_id",
                "hits",
                "univ_size",
                "rank",
                "proximity",
                "ai_score",
                "geo_score",
                "sniper_log",
            ]

            for col in columns_order:
                if col not in new_rows.columns:
                    new_rows[col] = ""

            new_rows = new_rows[columns_order]

            if os.path.exists(self.log_path):
                existing = pd.read_csv(self.log_path)
                combined = pd.concat([existing, new_rows], ignore_index=True)
            else:
                combined = new_rows

            combined.to_csv(self.log_path, index=False)
        except Exception as e:
            print(f"⚠️ Error guardando CSV: {e}")

    def _save_to_json(self, result_dto, tag, forensic_data):
        """Genera el reporte de inteligencia JSON necesario para el sistema."""
        try:
            os.makedirs(os.path.dirname(self.json_path), exist_ok=True)

            payload = {
                "timestamp": datetime.now().isoformat(),
                "version": tag,
                "summary": result_dto,  # El codificador maneja el DTO
                "forensic_details": forensic_data,
            }

            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, cls=SniperJSONEncoder, indent=4)

            # print(f"✅ Inteligencia JSON generada: {self.json_path}")
        except Exception as e:
            print(f"❌ FALLO CRÍTICO EN SERIALIZACIÓN JSON: {e}")

    def get_summary(self):
        if os.path.exists(self.log_path):
            return pd.read_csv(self.log_path)
        return pd.DataFrame()
