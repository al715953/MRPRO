# src/core/health.py
import os
from datetime import datetime

# from src.data_access.config import DATA_FOLDER
from src.data_access.config import MODEL_FILE_PATH


def get_model_status():
    """
    Calcula la antigüedad del modelo basándose en la carpeta de datos centralizada.
    """
    # Usamos la ruta oficial definida en config.py
    #    model_path = os.path.join(DATA_FOLDER, "mrpro_model_v8_static.json")
    model_path = MODEL_FILE_PATH
    # Diagnóstico de ruta (Opcional: puedes descomentar la siguiente línea para ver dónde busca)
    # print(f"DEBUG: Buscando modelo en {model_path}")

    if not os.path.exists(model_path):
        return "SIN MODELO", "bold red"

    # Obtener timestamp de última modificación
    mtime = os.path.getmtime(model_path)
    ultimo_update = datetime.fromtimestamp(mtime)
    dias_sin_actualizar = (datetime.now() - ultimo_update).days

    # Lógica de criticidad V15
    if dias_sin_actualizar < 7:
        color = "green"
    elif dias_sin_actualizar < 15:
        color = "yellow"
    else:
        color = "bold red"

    return f"{dias_sin_actualizar}d", color
