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
    "sum_min": 108,
    "sum_max": 132,
    "ac_min": 7,
    "even_min": 2,
    "even_max": 4,
    "max_per_decade":3,
    "prime_min": 1,
    "prime_max": 3,
    "max_delta": 12,          # Saltos más cortos (más realista)
    "std_min": 7.8,           # Dispersión más controlada
    "std_max": 12.8,
    "max_same_last_digit": 3, # Solo máximo 2 números con misma terminación



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
        "2-1-2-1", "1-2-2-1", "2-2-1-1", "1-1-2-2", "2-1-1-2", "1-2-1-2"
    ],}

# --- ESTÉTICA DE CONSOLA (ANSI) ---
CYAN = "\033[0;36m"
VERDE = "\033[0;32m"
BLANCO_B = "\033[1;37m"
RESET = "\033[0m"
