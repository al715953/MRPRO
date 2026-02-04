# src/core/analytics.py
import os
import pandas as pd
from datetime import datetime
from src.data_access.config import MASTER_LOG_PATH, DATA_FOLDER

class PerformanceTracker:
    """Sistema de Registro y Telemetría V7.19 (Full Spectrum Log)."""

    def __init__(self):
        self.log_path = os.path.join(DATA_FOLDER, "detailed_forensic_log.csv")
        self._ensure_log_exists()

    def _ensure_log_exists(self):
        if not os.path.exists(self.log_path):
            # Creamos el CSV con las nuevas columnas si no existe
            df = pd.DataFrame(columns=[
                "timestamp", "tag", "draw_id", "hits", 
                "univ_size", "rank", "proximity", 
                "ai_score", "geo_score", "sniper_log"
            ])
            df.to_csv(self.log_path, index=False)

    def log_run(self, result_dto, tag, forensic_data):
        """
        Guarda los resultados detallados de cada sorteo en el CSV.
        Ahora incluye: Tamaño del Universo y Mensajes del Sniper.
        """
        if not forensic_data:
            return

        # 1. Convertimos la lista de diccionarios a DataFrame
        new_rows = pd.DataFrame(forensic_data)

        # 2. Agregamos Metadatos Globales
        new_rows["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        new_rows["tag"] = tag

        # 3. Definimos el Orden de Columnas (La Lista VIP Actualizada)
        # Usamos .get para rellenar con NaN/Empty si alguna columna falta por error
        columns_order = [
            "timestamp", 
            "tag", 
            "draw_id", 
            "hits", 
            "univ_size",   # <--- NUEVO: Para ver si el filtro Titanium actuó
            "rank", 
            "proximity", 
            "ai_score", 
            "geo_score", 
            "sniper_log"   # <--- NUEVO: Para ver qué eliminó el Sniper
        ]
        
        # Aseguramos que existan las columnas, si no, las creamos vacías
        for col in columns_order:
            if col not in new_rows.columns:
                new_rows[col] = ""

        # Reordenamos y filtramos basura
        new_rows = new_rows[columns_order]

        # 4. Guardamos (Append Mode)
        try:
            # Leemos el existente para concatenar (o usamos mode='a' header=False)
            # Para evitar problemas de headers, cargamos, pegamos y guardamos.
            if os.path.exists(self.log_path):
                existing = pd.read_csv(self.log_path)
                combined = pd.concat([existing, new_rows], ignore_index=True)
            else:
                combined = new_rows
            
            combined.to_csv(self.log_path, index=False)
            # print(f"📝 Log Forense actualizado: {len(new_rows)} registros nuevos.")
            
        except Exception as e:
            print(f"⚠️ Error guardando log: {e}")

    def get_summary(self):
        if os.path.exists(self.log_path):
            return pd.read_csv(self.log_path)
        return pd.DataFrame()