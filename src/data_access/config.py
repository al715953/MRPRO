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
FILE_MELATE = os.path.join(DATA_FOLDER, "Melate-Retro.csv")
FILE_APUESTAS = os.path.join(DATA_FOLDER, "Mis_Apuestas.csv")

# URL Oficial
URL_MELATE = "https://www.loterianacional.gob.mx/Home/Historicos?ARHP=TQBlAGwAYQB0AGUALQBSAGUAdAByAG8A"

# Constantes del Juego
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}
UNIVERSE_MAX = 39

# --- NUEVA CONFIGURACIÓN MONTECARLO ---
NUM_SIMULACIONES = 200000  # 200k sorteos teóricos para calibrar
NUM_CANDIDATOS = 10000000  # 1M Cuántas combinaciones probamos en cada generación
