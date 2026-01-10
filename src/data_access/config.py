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
    "sum_min": 100,
    "sum_max": 180,
    "ac_min": 4,
    "inertia_min": 0,
    "even_min": 2,
    "even_max": 4,
}
