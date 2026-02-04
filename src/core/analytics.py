# src/core/analytics.py
import os
import pandas as pd
import json
from datetime import datetime
# Importamos la infraestructura de rutas centralizada
from src.data_access.config import DATA_FOLDER, MASTER_LOG_PATH

class PerformanceTracker:
    def __init__(
        self,
        master_file=None,
        detail_file=None,
    ):
        # Sincronizamos con las rutas definidas en config.py para evitar errores de ruta en Mac
        self.master_file = master_file or MASTER_LOG_PATH
        self.detail_file = detail_file or os.path.join(DATA_FOLDER, "detailed_forensic_log.csv")
        # Definimos la ruta del JSON para el visualizador usando la carpeta de datos centralizada
        self.json_file = os.path.join(DATA_FOLDER, "backtest_results.json")
        
        # Verificación de seguridad: Asegurar que la carpeta exista antes de guardar
        if not os.path.exists(DATA_FOLDER):
            os.makedirs(DATA_FOLDER, exist_ok=True)

    def log_run(self, result, tag="v1.0", audit_history=None):
        """Registra el resumen de la corrida y genera el JSON para el Plot."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 1. Registro Maestro (CSV)
        roi = (
            (result.net_balance / result.investment) * 100
            if result.investment > 0
            else 0
        )
        master_data = {
            "timestamp": timestamp,
            "tag": tag,
            "draws": result.total_draws_tested,
            "ROI": round(roi, 2),
            "balance": round(result.net_balance, 2),
            "hits_4": result.hit_distribution.get(4, 0),
            "hits_5": result.hit_distribution.get(5, 0),
            "hits_6": result.hit_distribution.get(6, 0),
        }
        
        # Guardar usando la ruta validada
        pd.DataFrame([master_data]).to_csv(
            self.master_file,
            mode="a",
            index=False,
            header=not os.path.exists(self.master_file),
        )

        # 2. Registro Detallado (JSON y CSV)
        if audit_history:
            try:
                with open(self.json_file, "w") as f:
                    json.dump(audit_history, f, indent=4)
                print(f"📡 Estación de diagnóstico: {self.json_file} actualizado.")
            except Exception as e:
                print(f"⚠️ Error al escribir JSON en Mac: {e}")

            detailed_rows = []
            for entry in audit_history:
                detailed_rows.append(
                    {
                        "timestamp": timestamp,
                        "tag": tag,
                        "draw_id": entry.get("draw_id"),
                        "hits": entry.get("hits"),
                        "rank": entry.get("rank"),
                        "proximity": entry.get("proximity"),
                        "ai_score": round(entry.get("ai_score", 0), 4),
                        "geo_score": round(entry.get("geo_score", 0), 4),
                    }
                )
            df_detail = pd.DataFrame(detailed_rows)
            df_detail.to_csv(
                self.detail_file,
                mode="a",
                index=False,
                header=not os.path.exists(self.detail_file),
            )