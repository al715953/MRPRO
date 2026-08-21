import os
from pathlib import Path
import shutil
import sys
from src.domain.lottery_profile import LotteryProfile

# --- INFRAESTRUCTURA DE RUTAS ---
# En desarrollo, config.py vive en src/data_access y la raíz está dos niveles arriba.
# En ejecutables, los datos persistentes viven junto al ejecutable cuando esa ubicación
# es escribible; en instalaciones protegidas se usa la carpeta de datos del usuario.
IS_FROZEN = bool(getattr(sys, "frozen", False))
PROJECT_ROOT = (
    Path(sys.executable).resolve().parent
    if IS_FROZEN
    else Path(__file__).resolve().parents[2]
)
BASE_DIR = str(PROJECT_ROOT)


def _user_data_directory() -> Path:
    if sys.platform.startswith("win"):
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform in {"darwin", "ios"}:
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return root / "MRPRO" / "data"


def _prepare_data_directory(preferred: Path) -> Path:
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        if os.access(preferred, os.W_OK):
            return preferred
    except OSError:
        pass

    fallback = _user_data_directory()
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def migrate_legacy_data(legacy: Path, destination: Path) -> list[str]:
    """Move missing files from the former ``src/data`` without overwriting data."""
    moved = []
    if not legacy.is_dir() or legacy.resolve() == destination.resolve():
        return moved
    destination.mkdir(parents=True, exist_ok=True)
    for source in legacy.iterdir():
        if not source.is_file():
            continue
        target = destination / source.name
        if target.exists():
            continue
        try:
            shutil.move(str(source), str(target))
            moved.append(source.name)
        except OSError:
            # Una instalación empaquetada puede exponer el origen como solo lectura.
            try:
                shutil.copy2(source, target)
                moved.append(source.name)
            except OSError:
                continue
    try:
        legacy.rmdir()
    except OSError:
        pass
    return moved


def _seed_bundled_data(bundle: Path, destination: Path) -> None:
    """Copy packaged seed files once; never replace persistent user data."""
    if not bundle.is_dir() or bundle.resolve() == destination.resolve():
        return
    for source in bundle.iterdir():
        if not source.is_file():
            continue
        target = destination / source.name
        if not target.exists():
            try:
                shutil.copy2(source, target)
            except OSError:
                continue


DATA_FOLDER_PATH = _prepare_data_directory(PROJECT_ROOT / "data")
migrate_legacy_data(PROJECT_ROOT / "src" / "data", DATA_FOLDER_PATH)
if IS_FROZEN and hasattr(sys, "_MEIPASS"):
    _seed_bundled_data(Path(sys._MEIPASS) / "data", DATA_FOLDER_PATH)

# Se conserva como ``str`` para compatibilidad con consumidores que usan os.path.
DATA_FOLDER = str(DATA_FOLDER_PATH)

# Archivos de Datos
CSV_FILE_PATH = str(DATA_FOLDER_PATH / "Melate-Retro.csv")
FILE_APUESTAS = str(DATA_FOLDER_PATH / "Mis_Apuestas.csv")
FILE_CARTERAS_SOMBRA = str(DATA_FOLDER_PATH / "Carteras_Sombra.json")
FILE_TABLERO_SOMBRA = str(DATA_FOLDER_PATH / "Tablero_Sombra.json")
MASTER_LOG_PATH = str(DATA_FOLDER_PATH / "master_performance.csv")
MODEL_FILE_PATH = str(DATA_FOLDER_PATH / "mrpro_model_v8_static.json")
BACKTEST_MODEL_FILE_PATH = str(DATA_FOLDER_PATH / "mrpro_model_v8_temporal_backtest.json")
NUMBER_MODEL_FILE_PATH = str(DATA_FOLDER_PATH / "mrpro_number_model.json")
BACKTEST_NUMBER_MODEL_FILE_PATH = str(
    DATA_FOLDER_PATH / "mrpro_number_model_temporal_backtest.json"
)
BACKTEST_MODEL_CACHE_PATH = DATA_FOLDER_PATH / "backtest_models"

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
# El endpoint de Historicos usa ARHP en Base64 UTF-16LE.
# "Tris" => "VAByAGkAcwA="
URL_TRIS_MULTIPLICADOR = (
    "https://www.loterianacional.gob.mx/Home/Historicos?ARHP=VAByAGkAcwA="
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
    # El modelo por número queda en observación hasta superar el baseline temporal.
    "ai_context_weight": 1.00,
    "ai_number_weight": 0.00,
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

# --- CONFIGURACION BASE: TRIS V1-A ---
BEST_SETTINGS_TRISbk = {
    # --- ejecución ---
    "tris_backtest_mode": "universe_strategy",  # ya lo tienes
    "compare_models": True,  # activa LR vs RandomTopK
    "compare_models_random_seeds": 30,  # o 10 para empezar
    # --- universo / estrategia ---
    "universe_mode": "topk_scored_universe",
    "score_model": "layered_mesh_v1",  # o "ticket_ngram" / "positional_logp" /random_topk
    "universe_topk_k": 1000,  # prueba 2000, 5000, 10000, 20000
    "num_tickets": 200,
    "backtest_size": 500,
    "topk_k": 2000,
    "beam_width": 2500,
    "per_pos_topm": 6,
    "diversity_min_hamming": 2,
    "short_window": 200,
    "long_window": 2000,
    "alpha_bayes": 2.0,
    "mix_lambda": 0.7,
    "markov_window": 2000,
    "alpha_markov": 0.5,
    "blend_markov": 0.10,
    "uniform_mix": 0.20,
    "temperature": 1.2,
    "gate_calib_size": 300,
    "gate_margin": 0.0005,
    "structural_enabled": True,
    "structural_sum_min": 15,
    "structural_sum_max": 30,
    "structural_allowed_even_counts": [2, 3],  # parity off
    "structural_enable_global_sum_filter": False,
    "structural_enable_global_parity_filter": False,
    "structural_min_unique_digits": 3,
    "structural_max_consecutive_run": 3,
    "structural_max_positional_repeats_vs_prev": 2,
    "structural_immediate_repeat_mode": "global_count",
    "structural_immediate_repeat_disallow_positions": [
        False,
        False,
        False,
        False,
        False,
    ],
    "structural_positional_limits": None,
    "structural_camera_entropy_rules": None,
    # Camera-aware experimental knobs (legacy-safe by default)
    "camera_masked_universe": True,
    "camera_topm_per_position": 4,
    "camera_alpha": 1.0,
    "camera_short_window": 100,
    "camera_long_window": 1000,
    "camera_mix_lambda": 0.3,
    "camera_latency_boost": 0.0,
    "camera_immediate_repeat_penalty": 0.20,
    "camera_parity_bias_strength": 0.0,
    "camera_mech_blend_with_v1a": 0.5,
    "camera_use_slot_context": False,
    # FeatureLR params
    "feature_lr_alpha": 2.0,
    "feature_lr_short_window": 200,
    "feature_lr_long_window": 2000,
    "feature_lr_mix_lambda": 0.2,
    "feature_lr_use_mirror": False,
    "feature_lr_shrink_c": 20000,
    "run_context_verbose": True,
}
BEST_SETTINGS_TRIS = {
    # --- ejecución ---
    "tris_backtest_mode": "universe_strategy",
    "num_tickets": 200,
    "backtest_size": 500,
    "compare_models": True,
    "compare_models_random_seeds": 30,
    # --- universo / estrategia ---
    "universe_mode": "topk_scored_universe",
    "score_model": "layered_mesh_v1",
    "universe_topk_k": 10000,
    # --- cámara / PMF ---
    "camera_masked_universe": True,
    "camera_debug_strict": False,
    "camera_alpha": 1.0,
    "camera_short_window": 100,
    "camera_long_window": 1000,
    "camera_mix_lambda": 0.30,
    "camera_latency_boost": 0.00,
    "camera_immediate_repeat_penalty": 0.15,
    "camera_parity_bias_strength": 0.00,
    "camera_mech_blend_with_v1a": 0.50,
    "camera_use_slot_context": False,
    # --- guardrails mínimos ---
    "structural_enabled": True,
    "structural_enable_global_sum_filter": False,
    "structural_enable_global_parity_filter": False,
    "structural_min_unique_digits": 3,
    "structural_max_consecutive_run": 3,
    "structural_immediate_repeat_mode": "per_position",
    "structural_immediate_repeat_disallow_positions": [
        False,
        False,
        False,
        False,
        False,
    ],
    # Legacy explícitas (evitar confusión)
    "structural_sum_min": 15,
    "structural_sum_max": 30,
    "structural_allowed_even_counts": [2, 3],
    "structural_max_positional_repeats_vs_prev": 2,
    # --- scoring por capas (defaults) ---
    "layered_use_hamming_memory": True,
    "layered_use_cross_turbulence": True,
    "layered_use_camera_repeat_penalty": True,
    "layered_w_positional_logp": 1.00,
    "layered_w_hamming_memory": 0.20,
    "layered_w_cross_turbulence": 0.10,
    "layered_w_camera_repeat_penalty": 0.15,
}

# --- PERFIL EXPERIMENTAL: TRIS CAMERA LAB ---
BEST_SETTINGS_TRIS_CAMERA_LAB = {
    **BEST_SETTINGS_TRIS,
    "score_model": "camera_mech_v1",
    "universe_mode": "topk_scored_universe",
    "camera_masked_universe": True,
    "structural_enable_global_sum_filter": False,
    "structural_enable_global_parity_filter": False,
    # Guardrails minimos
    "structural_min_unique_digits": 3,
    "structural_max_consecutive_run": 3,
    "structural_immediate_repeat_mode": "per_position",
    "structural_immediate_repeat_disallow_positions": [
        False,
        False,
        True,
        False,
        False,
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
