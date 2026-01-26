import os
import pandas as pd
import json  # Inyectamos la librería para el JSON
from datetime import datetime


class PerformanceTracker:
    def __init__(
        self,
        master_file="data/master_performance.csv",
        detail_file="data/detailed_forensic_log.csv",
    ):
        self.master_file = master_file
        self.detail_file = detail_file
        # Definimos la ruta del JSON para el visualizador
        self.json_file = "data/backtest_results.json"

    def log_run(self, result, tag="v1.0", audit_history=None):
        """Registra el resumen de la corrida y genera el JSON para el Plot."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 1. Registro Maestro (CSV) - Se mantiene igual
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
        pd.DataFrame([master_data]).to_csv(
            self.master_file,
            mode="a",
            index=False,
            header=not os.path.exists(self.master_file),
        )

        # 2. Registro Detallado (JSON y CSV)
        if audit_history:
            # --- NUEVO BLOQUE: ESCRITURA DEL JSON PARA EL VISUALIZADOR ---
            try:
                with open(self.json_file, "w") as f:
                    json.dump(audit_history, f, indent=4)
                print(f"📡 Estación de diagnóstico: {self.json_file} actualizado.")
            except Exception as e:
                print(f"⚠️ Error al escribir JSON: {e}")

            # Registro CSV (Se mantiene igual)
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
