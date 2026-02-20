import os
import sys
from src.domain.lottery_profile import LotteryProfile

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
MODEL_FILE_PATH = os.path.join(DATA_FOLDER, "mrpro_model_v8_static.json")

# --- IDENTIFICACIÓN DE MISIÓN ---
# Etiqueta para la bitácora de experimentos
VERSION_TAG = (
    "V16_SEL_fitnessV1_covTop200_t24_anchor5_subbkt_4-3_2-3_3-3_1_n500_20260215"
)

# --- CONSTANTES DE MELATE RETRO ---
TOTAL_BALLS = 39
TICKET_SIZE = 6
URL_MELATE = "https://www.loterianacional.gob.mx/Home/Historicos?ARHP=TQBlAGwAYQB0AGUALQBSAGUAdAByAG8A"

# --- CONFIGURACIÓN: TRIS CON MULTIPLICADOR ---
# Nota: Ajustar URL final cuando la fuente oficial quede confirmada.
URL_TRIS_MULTIPLICADOR = (
    "https://www.loterianacional.gob.mx/Home/Historicos?ARHP=VHJpcw=="
)
TRIS_DIGITS = 5


LOTTERY_PROFILES = {
    "melate_retro": LotteryProfile(
        code="melate_retro",
        display_name="Melate Retro",
        csv_filename="Melate-Retro.csv",
        source_url=URL_MELATE,
        total_balls=TOTAL_BALLS,
        ticket_size=TICKET_SIZE,
        includes_additional_ball=True,
    ),
    "tris_multiplicador": LotteryProfile(
        code="tris_multiplicador",
        display_name="Tris con Multiplicador",
        csv_filename="Tris-Multiplicador.csv",
        source_url=URL_TRIS_MULTIPLICADOR,
        total_balls=10,
        ticket_size=TRIS_DIGITS,
        includes_additional_ball=False,
    ),
}


def get_lottery_profile(game_code: str) -> LotteryProfile:
    """Devuelve el perfil del juego o Melate Retro como fallback seguro."""
    return LOTTERY_PROFILES.get(game_code, LOTTERY_PROFILES["melate_retro"])


# --- CONFIGURACIÓN DE HARDWARE (UHPC) ---
# Forzar uso de núcleos CUDA en RTX 4070 Ti

GPU_ENABLED = False


NUM_SIMULACIONES = 250000


# --- CONFIGURACIÓN "SNIPER" (OPTIMIZADA V10.5) ---
# Estos valores alimentan al Scorer, Selector y Backtester
BEST_SETTINGS = {
    "dynamic_exclude_count": 1,  # Números a eliminar por inercia térmica
    "anchor_nexus_size": 3,  # Cuántos números 'ancla' compartirán los tickets
    "nexus_density": 0.90,  # 80% de los tickets tendrán las anclas
    "shadow_risk_threshold": 0.08,  # Umbral para que el Shadow Model descarte un ticket
    # 1. Filtros Topológicos (Fase 1: Harmony Engine)
    "sum_min": 95,  # 112
    "sum_max": 115,  # 128
    "f1_max": 11,
    "f6_min": 29,
    "ac_min": 7,
    "even_min": 2,
    "even_max": 4,
    "max_per_decade": 3,
    "prime_min": 1,
    "prime_max": 3,
    "max_delta": 15,  # Saltos más cortos (más realista)
    "max_contig": 1,  # Alias operativo para filtro de consecutivos
    "std_min": 7.5,  # Dispersión más controlada
    "std_max": 13.2,
    "entropy_min": 2.15,  # Punto dulce detectado en V13
    "entropy_max": 2.45,  # Filtra el ruido estético excesivo
    "sdr_min": 22,  # Suma de Raíces Digitales mínima
    "sdr_max": 38,  # Suma de Raíces Digitales máxima
    "max_same_last_digit": 3,  # Solo máximo 2 números con misma terminación
    # 2. IA Scorer (Fase 2: Sugerencia 3 de Resonancia)
    "scale_pos_weight": 4,
    "n_estimators": 3200,
    "learning_rate": 0.012,
    "subsample": 0.85,
    "colsample_bytree": 0.82,
    "max_depth": 10,
    "gamma": 4.0,
    # Malla Híbrida
    "hybrid_alpha": 0.5,  # Peso para AI_Score
    "hybrid_beta": 0.5,  # Peso para Geo_Resonance
    "threshold_ai_override": 0.85,  # Elevamos la vara para el Top 20
    "geo_floor_percentile": 50.0,
    # 3. Malla Cuántica (Fase 3: Genetic Selector)
    # alpha_core_size: Asegura que el Top 5 de la IA sea inamovible
    "alpha_core_size": 3,
    "repulsion_strength": 2.8,
    "sampling_top": 2,  # Tickets de cobertura aleatoria en el Top 100
    # Umbrales de Calidad
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
    # 4. Pesos del Protocolo Sniper E1 (NUEVO)
    "w_gap": 0.25,
    "w_term": 0.10,
    "w_freq": 0.60,
    "sniper_threshold": 0.9,
    "sniper_conservative": False,  # Activa modo conservador (menos exclusiones).
    "sniper_threshold_boost": 0.08,  # Incremento de umbral en modo conservador.
    "auto_std_compensation": False,  # Ajusta std para sostener tamaño objetivo.
    "target_universe_size": 0,  # Si >0, objetivo de tamaño para compensación std.
    # Estamos agregando una trifecta de asesemblers, 3 IA´s
    # ESTRUCTURA CRÍTICA: Cada experto requiere su propio objetivo de entrenamiento
    "ensemble_config": {
        "alpha_ancla": {
            "depth": 6,
            "weight": 0.01,  # ANULACIÓN CASI TOTAL: Evita que tickets conservadores bloqueen el podio
            "objective": "reg:pseudohubererror",
        },
        "beta_sniper": {"depth": 12, "weight": 0.15, "objective": "reg:absoluteerror"},
        "omega_hunter": {
            "depth": 20,
            "weight": 0.84,  # PODER ABSOLUTO: El modelo de profundidad 16 ahora tiene el control total
            "objective": "reg:squaredlogerror",
        },
    },
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
