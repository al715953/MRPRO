import os
import sys

# Detectar si estamos en modo Ejecutable (.exe) o Script (.py)
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Carpetas
DATA_FOLDER = os.path.join(BASE_DIR, "data")

# Archivos
CSV_FILE_PATH = os.path.join(DATA_FOLDER, "Melate-Retro.csv")
FILE_APUESTAS = os.path.join(DATA_FOLDER, "Mis_Apuestas.csv")
TOTAL_BALLS = 39
TICKET_SIZE = 6

# URL Oficial
URL_MELATE = "https://www.loterianacional.gob.mx/Home/Historicos?ARHP=TQBlAGwAYQB0AGUALQBSAGUAdAByAG8A"

# Constantes del Juego
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}
UNIVERSE_MAX = 39

# --- NUEVA CONFIGURACIÓN MONTECARLO ---
NUM_SIMULACIONES = 200000  # 200k sorteos teóricos para calibrar
NUM_CANDIDATOS = 10000000  # 1M Cuántas combinaciones probamos en cada generación

# Colores
CYAN = "\033[0;36m"
RESET = "\033[0m"
VERDE = "\033[0;32m"
BLANCO_B = "\033[1;37m"

# --- CONFIGURACIÓN "CAMPEONA" (OPTIMIZADA) ---
# Copia aquí el resultado de la Opción 5 (Optimizer)
# Valores iniciales conservadores:
BEST_SETTINGS = {
    # 1. Filtros Topológicos (Fase 1)
    "sum_min": 112,
    "sum_max": 128,
    "ac_min": 7,
    "even_min": 2,
    "even_max": 4,
    "prime_min": 2,
    "prime_max": 3,
    # 2. Pesos de Decisión (Fase 2)
    # Define qué tanto caso le hacemos a cada experto
    "w_cluster": 0.55,  # Importancia de la estructura geométrica
    "w_hotness": 0.05,  # Importancia de la frecuencia reciente
    "w_ai": 0.40,  # Importancia del modelo XGBoost
    # 3. Alineación Táctica (Fase 3 - Cuotas)
    # Cuántos tickets seleccionamos de cada estrato (Total 15)
    "quota_elite": 3,  # Tickets con Score > 0.70 (Pocos, zona muerta reciente)
    "quota_mid": 10,  # Tickets con Score 0.60-0.70 (Zona Calidad)
    "quota_low": 2,  # Tickets con Score 0.50-0.60 (Zona Volumen/Guerra)
    # --- NUEVO: UMBRALES DINÁMICOS ---
    "threshold_elite": 0.85,  # Bajamos de 0.70 a 0.65 para capturar mejores candidatos
    "threshold_mid": 0.45,  # Bajamos de 0.60 a 0.55 para ampliar la red
    # Deep Dive
    "threshold_ai_override": 0.72,  # Bajado de 0.85 para capturar casos límite
    "geo_floor_percentile": 50.0,  # Bajado de 40.0 para incluir el P35 (~0.58)
    "quota_stars": 15,  # Incrementamos un cupo para Super Stars por el umbral más bajo
    # General
    "verbose": True,  # Logs detallados
}
