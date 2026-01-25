import os
import sys

# --- INFRAESTRUCTURA DE RUTAS ---
# Detectar si estamos en modo Ejecutable (.exe) o Script (.py) para persistencia en Windows
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Carpetas de Proyecto
DATA_FOLDER = os.path.join(BASE_DIR, "data")
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# Archivos de Datos
CSV_FILE_PATH = os.path.join(DATA_FOLDER, "Melate-Retro.csv")
FILE_APUESTAS = os.path.join(DATA_FOLDER, "Mis_Apuestas.csv")
MASTER_LOG_PATH = os.path.join(DATA_FOLDER, "master_performance.csv")

# --- IDENTIFICACIÓN DE MISIÓN ---
# Etiqueta para la bitácora de experimentos
VERSION_TAG = "V6.9.3 - Consistencia 5/6"

# --- CONSTANTES DE MELATE RETRO ---
TOTAL_BALLS = 39
TICKET_SIZE = 6
URL_MELATE = "https://www.loterianacional.gob.mx/Home/Historicos?ARHP=TQBlAGwAYQB0AGUALQBSAGUAdAByAG8A"

# --- CONFIGURACIÓN DE HARDWARE (UHPC) ---
# Forzar uso de núcleos CUDA en RTX 4070 Ti
GPU_ENABLED = True
NUM_SIMULACIONES = 200000
NUM_CANDIDATOS_MONTECARLO = 10000000

# --- CONFIGURACIÓN "SNIPER" (OPTIMIZADA V10.5) ---
# Estos valores alimentan al Scorer, Selector y Backtester
BEST_SETTINGS = {
    # 1. Filtros Topológicos (Fase 1: Harmony Engine)
    "sum_min": 112,
    "sum_max": 128,
    "f1_max": 9,
    "f6_min": 31,
    "ac_min": 8,
    "even_min": 2,
    "even_max": 4,
    "max_per_decade": 3,
    "prime_min": 1,
    "prime_max": 3,
    "max_delta": 15,  # Saltos más cortos (más realista)
    "std_min": 8.2,  # Dispersión más controlada
    "std_max": 12.4,
    "entropy_min": 2.15,  # Punto dulce detectado en V13
    "entropy_max": 2.45,  # Filtra el ruido estético excesivo
    "sdr_min": 20,  # Suma de Raíces Digitales mínima
    "sdr_max": 42,  # Suma de Raíces Digitales máxima
    "max_same_last_digit": 3,  # Solo máximo 2 números con misma terminación
    # 2. IA Scorer (Fase 2: Sugerencia 3 de Resonancia)
    # scale_pos_weight: Eleva agresivamente los ganadores en el ranking
    "scale_pos_weight": 7.5,
    "n_estimators": 2000,
    "learning_rate": 0.01,
    # 3. Malla Cuántica (Fase 3: Genetic Selector)
    # alpha_core_size: Asegura que el Top 5 de la IA sea inamovible
    "alpha_core_size": 6,
    # repulsion_strength: Controla el colapso de la malla (8.0 es el punto dulce)
    "repulsion_strength": 5.0,
    "sampling_top": 2,  # Tickets de cobertura aleatoria en el Top 100
    # Umbrales de Calidad
    "threshold_ai_override": 0.72,
    "geo_floor_percentile": 50.0,
    # General
    "verbose": True,
    # 2. Poda por Perfiles de Décadas (Topología)
    # Solo permitimos perfiles que representen la mayor frecuencia histórica
    "valid_decade_profiles": [
        "2-1-2-1",
        "1-2-2-1",
        "2-2-1-1",
        "1-1-2-2",
        "2-1-1-2",
        "1-2-1-2",
    ],
}

# --- ESTÉTICA DE CONSOLA (ANSI) ---
CYAN = "\033[0;36m"
VERDE = "\033[0;32m"
BLANCO_B = "\033[1;37m"
RESET = "\033[0m"


# --- REJILLA DE BÚSQUEDA (OPTIMIZER V8.11) ---
# Define los ejes para la Calibración Forense Exhaustiva
SEARCH_GRID = {
    "e_min": [2.00, 2.05, 2.10, 2.15],
    "e_max": [2.45, 2.50, 2.55, 2.60],
    "s_min": [20, 22, 24],
    "s_max": [42, 44, 46],
    "ac": [7, 8],
    "std_min": [7.8, 8.0, 8.2],
    "std_max": [12.4, 12.6, 12.8],
}

# Ejemplo de Rejilla Rápida (Para validación de 5 minutos)
# SEARCH_GRID = {
#    "e_min": [2.10],
#    "e_max": [2.50],
#    "s_min": [24],
#    "s_max": [42],
#    "ac": [7, 8],
#    "std_min": [8.2],
#    "std_max": [12.4],
# }
