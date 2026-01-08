import requests
import os
import urllib3
import time
from datetime import datetime
from src.data_access.config import URL_MELATE

# Desactivar advertencias de SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def descargar_datos(filepath):
    """
    Descarga el CSV oficial solo si no está actualizado al día de hoy.
    """
    # 1. Verificar si el archivo ya existe y fue modificado hoy
    if os.path.exists(filepath):
        timestamp = os.path.getmtime(filepath)
        fecha_archivo = datetime.fromtimestamp(timestamp).date()
        fecha_hoy = datetime.today().date()

        if fecha_archivo == fecha_hoy:
            print("⚡ Archivo local actualizado hoy. Omitiendo descarga.")
            return True

    print("📡 Conectando con servidor de Lotería Nacional...")
    try:
        response = requests.get(URL_MELATE, timeout=20, stream=True, verify=False)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Base de datos actualizada exitosamente.")
        return True
    except Exception as e:
        print(f"⚠️ No se pudo descargar (Usando versión Offline si existe): {e}")
        return False
