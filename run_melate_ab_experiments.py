"""Ejecuta experimentos A/B reproducibles del scorer de Melate Retro."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from src.core.backtester import BacktestEngine
from src.core.analytics import PerformanceTracker
from src.core.fixed_origin_training import prepare_fixed_origin_models
from src.data_access.config import (
    BEST_SETTINGS,
    DATA_FOLDER_PATH,
    LOTTERY_PROFILES,
    TICKET_SIZE,
    TOTAL_BALLS,
)
from src.data_access.loader import LotteryLoader
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, sort_history_chronologically
from src.strategies.genetic_selector import GeneticSelectorStrategy
from src.strategies.universe.hybrid import HybridUniverseReductionStrategy
from src.strategies.universe.shadow import (
    LEGACY_HARD_FILTER_OVERRIDES,
    PROFILE_OOS_SET,
)
from src.strategies.universe_reduction import UniverseReductionStrategy


CORE_VARIANTS = (
    {
        "name": "A_contextual_geo_adaptive",
        "description": "IA contextual + Geo con safety net adaptativo actual",
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "adaptive",
            "fitness_selector_mode": "native",
            "sniper_soft_reserve_fraction": 0.10,
        },
    },
    {
        "name": "B_geo_only",
        "description": "Control Geo puro; la IA no participa en radar ni ranking",
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "fixed",
            "hybrid_alpha": 0.0,
            "hybrid_beta": 1.0,
            "fitness_selector_mode": "native",
            "sniper_soft_reserve_fraction": 0.10,
        },
    },
    {
        "name": "C_contextual_number15_geo_adaptive",
        "description": "IA contextual 85% + IA por número 15%, combinada con Geo",
        "overrides": {
            "ai_context_weight": 0.85,
            "ai_number_weight": 0.15,
            "resonance_blend_mode": "adaptive",
            "fitness_selector_mode": "native",
            "sniper_soft_reserve_fraction": 0.10,
        },
    },
)

BLEND_SWEEP_VARIANTS = (
    CORE_VARIANTS[0],
    CORE_VARIANTS[1],
    {
        "name": "D_ai10_geo90_fixed",
        "description": "IA contextual 10% + Geo 90% con mezcla fija",
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "fixed",
            "hybrid_alpha": 0.10,
            "hybrid_beta": 0.90,
            "fitness_selector_mode": "native",
            "sniper_soft_reserve_fraction": 0.10,
        },
    },
    {
        "name": "E_ai25_geo75_fixed",
        "description": "IA contextual 25% + Geo 75% con mezcla fija",
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "fixed",
            "hybrid_alpha": 0.25,
            "hybrid_beta": 0.75,
            "fitness_selector_mode": "native",
            "sniper_soft_reserve_fraction": 0.10,
        },
    },
    {
        "name": "F_ai40_geo60_fixed",
        "description": "IA contextual 40% + Geo 60% con mezcla fija",
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "fixed",
            "hybrid_alpha": 0.40,
            "hybrid_beta": 0.60,
            "fitness_selector_mode": "native",
            "sniper_soft_reserve_fraction": 0.10,
        },
    },
)

SELECTOR_SHADOW_VARIANTS = (
    CORE_VARIANTS[0],
    {
        "name": "G_context50_number50_adaptive",
        "description": "IA contextual 50% + IA por número 50%, con Geo adaptativo",
        "overrides": {
            "ai_context_weight": 0.50,
            "ai_number_weight": 0.50,
            "resonance_blend_mode": "adaptive",
            "fitness_selector_mode": "native",
            "sniper_soft_reserve_fraction": 0.10,
        },
    },
    {
        "name": "H_deep_rank_5000_same_budget",
        "description": "Selector estratificado hasta rank 5000 con igual presupuesto",
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "adaptive",
            "fitness_selector_mode": "native",
            "sniper_soft_reserve_fraction": 0.10,
            "fitness_focus_max_rank": 5000,
            "fitness_candidate_max_rank": 5000,
            "fitness_rank_edges": [5, 20, 100, 300, 750, 1500, 3000, 5000],
            "fitness_bucket_plan": [
                [6, 20, 2],
                [21, 100, 3],
                [101, 300, 3],
                [301, 750, 3],
                [751, 1500, 3],
                [1501, 3000, 3],
                [3001, 5000, 2],
            ],
        },
    },
)

CONTROLLED_WEIGHT_VARIANTS = (
    CORE_VARIANTS[0],
    CORE_VARIANTS[2],
    SELECTOR_SHADOW_VARIANTS[1],
)

DEEP_DISPERSION_VARIANTS = (
    {
        "name": "I_native_30_reference",
        "description": "MRPRO nativo con 30 tickets como control de presupuesto",
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "adaptive",
            "fitness_selector_mode": "native",
            "sniper_soft_reserve_fraction": 0.10,
        },
    },
    {
        "name": "J_core20_deep10_equal_population",
        "description": (
            "20 tickets del núcleo nativo + 10 estratos profundos de igual población"
        ),
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "adaptive",
            "fitness_selector_mode": "core_plus_deep",
            "sniper_soft_reserve_fraction": 0.10,
            "deep_dispersion_core_tickets": 20,
            "deep_dispersion_tickets": 10,
            "deep_dispersion_min_rank": 501,
            "deep_dispersion_max_overlap": 3,
            "deep_dispersion_pair_novelty_weight": 0.40,
            "deep_dispersion_number_rarity_weight": 0.25,
            "deep_dispersion_dissimilarity_weight": 0.20,
            "deep_dispersion_local_quality_weight": 0.15,
        },
    },
)

ELITE_COVERAGE_DEEP_VARIANTS = (
    DEEP_DISPERSION_VARIANTS[1],
    {
        "name": "K_elite10_cover10_deep10",
        "description": (
            "10 ranks élite exactos + 10 de cobertura 2/3/4 + 10 profundos"
        ),
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "adaptive",
            "fitness_selector_mode": "elite_coverage_deep",
            "sniper_soft_reserve_fraction": 0.10,
            "portfolio_elite_tickets": 10,
            "portfolio_coverage_tickets": 10,
            "portfolio_deep_tickets": 10,
            "portfolio_coverage_max_rank": 500,
            "portfolio_min_deep_rank": 501,
            "portfolio_max_overlap": 3,
            "portfolio_pair_novelty_weight": 0.15,
            "portfolio_triple_novelty_weight": 0.30,
            "portfolio_quad_novelty_weight": 0.30,
            "portfolio_number_rarity_weight": 0.05,
            "portfolio_dissimilarity_weight": 0.05,
            "portfolio_local_quality_weight": 0.15,
        },
    },
    {
        "name": "L_elite15_cover10_deep5",
        "description": (
            "15 ranks élite exactos + 10 de cobertura 2/3/4 + 5 profundos"
        ),
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "adaptive",
            "fitness_selector_mode": "elite_coverage_deep",
            "sniper_soft_reserve_fraction": 0.10,
            "portfolio_elite_tickets": 15,
            "portfolio_coverage_tickets": 10,
            "portfolio_deep_tickets": 5,
            "portfolio_coverage_max_rank": 500,
            "portfolio_min_deep_rank": 501,
            "portfolio_max_overlap": 3,
            "portfolio_pair_novelty_weight": 0.15,
            "portfolio_triple_novelty_weight": 0.30,
            "portfolio_quad_novelty_weight": 0.30,
            "portfolio_number_rarity_weight": 0.05,
            "portfolio_dissimilarity_weight": 0.05,
            "portfolio_local_quality_weight": 0.15,
        },
    },
    {
        "name": "M_elite10_cover15_deep5",
        "description": (
            "10 ranks élite exactos + 15 de cobertura 2/3/4 + 5 profundos"
        ),
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "adaptive",
            "fitness_selector_mode": "elite_coverage_deep",
            "sniper_soft_reserve_fraction": 0.10,
            "portfolio_elite_tickets": 10,
            "portfolio_coverage_tickets": 15,
            "portfolio_deep_tickets": 5,
            "portfolio_coverage_max_rank": 500,
            "portfolio_min_deep_rank": 501,
            "portfolio_max_overlap": 3,
            "portfolio_pair_novelty_weight": 0.15,
            "portfolio_triple_novelty_weight": 0.30,
            "portfolio_quad_novelty_weight": 0.30,
            "portfolio_number_rarity_weight": 0.05,
            "portfolio_dissimilarity_weight": 0.05,
            "portfolio_local_quality_weight": 0.15,
        },
    },
)

FIVE_HIT_SELECTOR_VARIANTS = (
    {
        "name": "N_native24_reference",
        "description": "Referencia productiva V17.1 nativa con 24 tickets",
        "overrides": {
            "resonance_blend_mode": "adaptive",
            "fitness_selector_mode": "native",
            "sniper_soft_reserve_fraction": 0.10,
        },
    },
    {
        "name": "O_rank500_overlap3_elite1",
        "description": "Mejor rank hasta 500 con cobertura 5/6 disjunta",
        "overrides": {
            "fitness_selector_mode": "five_hit_coverage",
            "resonance_blend_mode": "adaptive",
            "sniper_soft_reserve_fraction": 0.10,
            "five_hit_elite_tickets": 1,
            "five_hit_candidate_max_rank": 500,
            "five_hit_max_overlap": 3,
        },
    },
    {
        "name": "P_rank2000_overlap3_elite1",
        "description": "Mejor rank hasta 2000 con cobertura 5/6 disjunta",
        "overrides": {
            "fitness_selector_mode": "five_hit_coverage",
            "resonance_blend_mode": "adaptive",
            "sniper_soft_reserve_fraction": 0.10,
            "five_hit_elite_tickets": 1,
            "five_hit_candidate_max_rank": 2000,
            "five_hit_max_overlap": 3,
        },
    },
    {
        "name": "Q_rank5000_overlap2_elite1",
        "description": "Distancia más agresiva hasta rank 5000",
        "overrides": {
            "fitness_selector_mode": "five_hit_coverage",
            "resonance_blend_mode": "adaptive",
            "sniper_soft_reserve_fraction": 0.10,
            "five_hit_elite_tickets": 1,
            "five_hit_candidate_max_rank": 5000,
            "five_hit_max_overlap": 2,
        },
    },
    {
        "name": "R_core20_deep4",
        "description": "20 tickets nativos + 4 estratos profundos",
        "overrides": {
            "resonance_blend_mode": "adaptive",
            "sniper_soft_reserve_fraction": 0.10,
            "fitness_selector_mode": "core_plus_deep",
            "resonance_blend_mode": "adaptive",
            "sniper_soft_reserve_fraction": 0.10,
            "deep_dispersion_core_tickets": 20,
            "deep_dispersion_tickets": 4,
            "deep_dispersion_min_rank": 501,
            "deep_dispersion_max_overlap": 3,
        },
    },
    {
        "name": "S_core16_deep8",
        "description": "16 tickets nativos + 8 estratos profundos",
        "overrides": {
            "fitness_selector_mode": "core_plus_deep",
            "resonance_blend_mode": "adaptive",
            "sniper_soft_reserve_fraction": 0.10,
            "deep_dispersion_core_tickets": 16,
            "deep_dispersion_tickets": 8,
            "deep_dispersion_min_rank": 501,
            "deep_dispersion_max_overlap": 3,
        },
    },
    {
        "name": "T_core12_deep12",
        "description": "12 tickets nativos + 12 estratos profundos",
        "overrides": {
            "fitness_selector_mode": "core_plus_deep",
            "resonance_blend_mode": "adaptive",
            "sniper_soft_reserve_fraction": 0.10,
            "deep_dispersion_core_tickets": 12,
            "deep_dispersion_tickets": 12,
            "deep_dispersion_min_rank": 501,
            "deep_dispersion_max_overlap": 3,
        },
    },
    {
        "name": "U_elite4_cover16_deep4",
        "description": "4 élite + 16 cobertura + 4 profundidad",
        "overrides": {
            "fitness_selector_mode": "elite_coverage_deep",
            "resonance_blend_mode": "adaptive",
            "sniper_soft_reserve_fraction": 0.10,
            "portfolio_elite_tickets": 4,
            "portfolio_coverage_tickets": 16,
            "portfolio_deep_tickets": 4,
            "portfolio_coverage_max_rank": 1000,
            "portfolio_min_deep_rank": 1001,
            "portfolio_max_overlap": 3,
        },
    },
    {
        "name": "V_elite4_cover12_deep8",
        "description": "4 élite + 12 cobertura + 8 profundidad",
        "overrides": {
            "fitness_selector_mode": "elite_coverage_deep",
            "resonance_blend_mode": "adaptive",
            "sniper_soft_reserve_fraction": 0.10,
            "portfolio_elite_tickets": 4,
            "portfolio_coverage_tickets": 12,
            "portfolio_deep_tickets": 8,
            "portfolio_coverage_max_rank": 1000,
            "portfolio_min_deep_rank": 1001,
            "portfolio_max_overlap": 3,
        },
    },
)

FIVE_HIT_SCORER_VARIANTS = (
    {
        "name": "W_adaptive_context_core16_deep8",
        "description": "Mezcla adaptativa actual, IA contextual y cartera 16+8",
        "overrides": {
            "resonance_blend_mode": "adaptive",
            "sniper_soft_reserve_fraction": 0.10,
            "fitness_selector_mode": "core_plus_deep",
            "deep_dispersion_core_tickets": 16,
            "deep_dispersion_tickets": 8,
            "deep_dispersion_min_rank": 501,
        },
    },
    {
        "name": "X_ai_only_core16_deep8",
        "description": "IA contextual pura y cartera 16+8",
        "overrides": {
            "resonance_blend_mode": "fixed",
            "hybrid_alpha": 1.0,
            "hybrid_beta": 0.0,
            "sniper_soft_reserve_fraction": 0.10,
            "fitness_selector_mode": "core_plus_deep",
            "deep_dispersion_core_tickets": 16,
            "deep_dispersion_tickets": 8,
            "deep_dispersion_min_rank": 501,
        },
    },
    {
        "name": "Y_ai75_geo25_core16_deep8",
        "description": "IA contextual 75% + Geo 25% y cartera 16+8",
        "overrides": {
            "resonance_blend_mode": "fixed",
            "hybrid_alpha": 0.75,
            "hybrid_beta": 0.25,
            "sniper_soft_reserve_fraction": 0.10,
            "fitness_selector_mode": "core_plus_deep",
            "deep_dispersion_core_tickets": 16,
            "deep_dispersion_tickets": 8,
            "deep_dispersion_min_rank": 501,
        },
    },
    {
        "name": "Z_ai50_geo50_core16_deep8",
        "description": "IA contextual 50% + Geo 50% y cartera 16+8",
        "overrides": {
            "resonance_blend_mode": "fixed",
            "hybrid_alpha": 0.50,
            "hybrid_beta": 0.50,
            "sniper_soft_reserve_fraction": 0.10,
            "fitness_selector_mode": "core_plus_deep",
            "deep_dispersion_core_tickets": 16,
            "deep_dispersion_tickets": 8,
            "deep_dispersion_min_rank": 501,
        },
    },
    {
        "name": "AA_ai25_geo75_core16_deep8",
        "description": "IA contextual 25% + Geo 75% y cartera 16+8",
        "overrides": {
            "resonance_blend_mode": "fixed",
            "hybrid_alpha": 0.25,
            "hybrid_beta": 0.75,
            "sniper_soft_reserve_fraction": 0.10,
            "fitness_selector_mode": "core_plus_deep",
            "deep_dispersion_core_tickets": 16,
            "deep_dispersion_tickets": 8,
            "deep_dispersion_min_rank": 501,
        },
    },
    {
        "name": "AB_geo_only_core16_deep8",
        "description": "Geo puro y cartera 16+8",
        "overrides": {
            "resonance_blend_mode": "fixed",
            "hybrid_alpha": 0.0,
            "hybrid_beta": 1.0,
            "sniper_soft_reserve_fraction": 0.10,
            "fitness_selector_mode": "core_plus_deep",
            "deep_dispersion_core_tickets": 16,
            "deep_dispersion_tickets": 8,
            "deep_dispersion_min_rank": 501,
        },
    },
    {
        "name": "AC_number15_adaptive_core16_deep8",
        "description": "IA por número 15% dentro de IA y mezcla adaptativa",
        "overrides": {
            "ai_context_weight": 0.85,
            "ai_number_weight": 0.15,
            "resonance_blend_mode": "adaptive",
            "sniper_soft_reserve_fraction": 0.10,
            "fitness_selector_mode": "core_plus_deep",
            "deep_dispersion_core_tickets": 16,
            "deep_dispersion_tickets": 8,
            "deep_dispersion_min_rank": 501,
        },
    },
    {
        "name": "AD_number50_adaptive_core16_deep8",
        "description": "IA contextual 50% + IA por número 50%, mezcla adaptativa",
        "overrides": {
            "ai_context_weight": 0.50,
            "ai_number_weight": 0.50,
            "resonance_blend_mode": "adaptive",
            "sniper_soft_reserve_fraction": 0.10,
            "fitness_selector_mode": "core_plus_deep",
            "deep_dispersion_core_tickets": 16,
            "deep_dispersion_tickets": 8,
            "deep_dispersion_min_rank": 501,
        },
    },
)

FIVE_HIT_VALIDATION_VARIANTS = (
    FIVE_HIT_SELECTOR_VARIANTS[0],
    FIVE_HIT_SELECTOR_VARIANTS[1],
    FIVE_HIT_SELECTOR_VARIANTS[5],
    FIVE_HIT_SCORER_VARIANTS[1],
    FIVE_HIT_SCORER_VARIANTS[3],
    FIVE_HIT_SCORER_VARIANTS[5],
)

FIVE_HIT_RESERVE_VARIANTS = (
    {
        "name": "AE_overlap3_adaptive_no_reserve",
        "description": "Cobertura de distancia adaptativa sin reemplazo Sniper posterior",
        "overrides": {
            "fitness_selector_mode": "five_hit_coverage",
            "resonance_blend_mode": "adaptive",
            "five_hit_elite_tickets": 1,
            "five_hit_candidate_max_rank": 500,
            "five_hit_max_overlap": 3,
            "sniper_soft_reserve_fraction": 0.0,
        },
    },
    {
        "name": "AF_ai_only_core16_deep8_no_reserve",
        "description": "IA pura 16+8 sin reemplazo Sniper posterior",
        "overrides": {
            "resonance_blend_mode": "fixed",
            "hybrid_alpha": 1.0,
            "hybrid_beta": 0.0,
            "fitness_selector_mode": "core_plus_deep",
            "deep_dispersion_core_tickets": 16,
            "deep_dispersion_tickets": 8,
            "deep_dispersion_min_rank": 501,
            "sniper_soft_reserve_fraction": 0.0,
        },
    },
)

FIVE_HIT_INTERACTION_VARIANTS = (
    {
        "name": "AG_ai_only_overlap3_no_reserve",
        "description": "IA pura con mejor rank sujeto a cobertura 5/6 disjunta",
        "overrides": {
            "resonance_blend_mode": "fixed",
            "hybrid_alpha": 1.0,
            "hybrid_beta": 0.0,
            "fitness_selector_mode": "five_hit_coverage",
            "five_hit_elite_tickets": 1,
            "five_hit_candidate_max_rank": 500,
            "five_hit_max_overlap": 3,
            "sniper_soft_reserve_fraction": 0.0,
        },
    },
    {
        "name": "AH_geo_only_overlap3_no_reserve",
        "description": "Geo puro con mejor rank sujeto a cobertura 5/6 disjunta",
        "overrides": {
            "resonance_blend_mode": "fixed",
            "hybrid_alpha": 0.0,
            "hybrid_beta": 1.0,
            "fitness_selector_mode": "five_hit_coverage",
            "five_hit_elite_tickets": 1,
            "five_hit_candidate_max_rank": 500,
            "five_hit_max_overlap": 3,
            "sniper_soft_reserve_fraction": 0.0,
        },
    },
)

FIVE_HIT_HOLDOUT_VARIANTS = (
    FIVE_HIT_SELECTOR_VARIANTS[0],
    FIVE_HIT_RESERVE_VARIANTS[0],
    FIVE_HIT_RESERVE_VARIANTS[1],
    {
        "name": "AI_geo_only_core16_deep8_no_reserve",
        "description": "Geo puro 16+8 sin reemplazo Sniper posterior",
        "overrides": {
            "resonance_blend_mode": "fixed",
            "hybrid_alpha": 0.0,
            "hybrid_beta": 1.0,
            "fitness_selector_mode": "core_plus_deep",
            "deep_dispersion_core_tickets": 16,
            "deep_dispersion_tickets": 8,
            "deep_dispersion_min_rank": 501,
            "sniper_soft_reserve_fraction": 0.0,
        },
    },
)

UNIVERSE_V17_VARIANTS = (
    {
        "name": "legacy_hard_geo_v16",
        "description": "Control V16 con filtros Geo y Sniper duros",
        "overrides": dict(LEGACY_HARD_FILTER_OVERRIDES),
    },
    {
        "name": "balanced_mixed50_v17",
        "description": "Producción V17: núcleo suave 50% + exploración 50%",
        "overrides": {},
    },
)

UNIVERSE_V17_2_BASE_OVERRIDES = {
    "sniper_mode": "soft",
    "sniper_soft_reserve_fraction": 0.0,
    "sum_filter_enabled": False,
    "structure_filter_enabled": False,
    "parity_filter_enabled": False,
    "prime_filter_enabled": False,
    "consecutive_filter_enabled": False,
    "max_delta_filter_enabled": False,
    "terminal_filter_enabled": False,
    "decade_profile_filter_enabled": False,
    "entropy_filter_enabled": False,
    "digital_root_filter_enabled": False,
    "ac_filter_enabled": False,
    "positional_filter_enabled": False,
    "spatial_filter_enabled": False,
    "std_filter_enabled": False,
    "candidate_selection_mode": "balanced_mixed",
    "universe_exploration_fraction": 0.50,
    "universe_ticket_limit": 45000,
    "radar_percentile": 0.0,
    "resonance_blend_mode": "fixed",
    "hybrid_alpha": 1.0,
    "hybrid_beta": 0.0,
    "ai_context_weight": 1.0,
    "ai_number_weight": 0.0,
    "fitness_selector_mode": "core_plus_deep",
    "deep_dispersion_core_tickets": 16,
    "deep_dispersion_tickets": 8,
    "deep_dispersion_min_rank": 501,
}


UNIVERSE_DYNAMIC_VARIANTS = (
    {
        "name": "UA_v17_2_balanced45_reference",
        "description": "V17.2: 45k, núcleo 50% y exploración 50%",
        "overrides": {
            **UNIVERSE_V17_2_BASE_OVERRIDES,
            "universe_ticket_limit": 45000,
            "candidate_selection_mode": "balanced_mixed",
            "universe_exploration_fraction": 0.50,
        },
    },
    {
        "name": "UB_balanced30",
        "description": "V17.2 balanceado con universo de 30k",
        "overrides": {
            **UNIVERSE_V17_2_BASE_OVERRIDES,
            "universe_ticket_limit": 30000,
        },
    },
    {
        "name": "UC_balanced60",
        "description": "V17.2 balanceado con universo de 60k",
        "overrides": {
            **UNIVERSE_V17_2_BASE_OVERRIDES,
            "universe_ticket_limit": 60000,
        },
    },
    {
        "name": "UD_balanced90",
        "description": "V17.2 balanceado con universo de 90k",
        "overrides": {
            **UNIVERSE_V17_2_BASE_OVERRIDES,
            "universe_ticket_limit": 90000,
        },
    },
    {
        "name": "UE_core100_45",
        "description": "45k sólo núcleo de score suave, sin exploración",
        "overrides": {
            **UNIVERSE_V17_2_BASE_OVERRIDES,
            "universe_exploration_fraction": 0.0,
        },
    },
    {
        "name": "UF_core75_explore25_45",
        "description": "45k con núcleo 75% y exploración 25%",
        "overrides": {
            **UNIVERSE_V17_2_BASE_OVERRIDES,
            "universe_exploration_fraction": 0.25,
        },
    },
    {
        "name": "UG_core25_explore75_45",
        "description": "45k con núcleo 25% y exploración 75%",
        "overrides": {
            **UNIVERSE_V17_2_BASE_OVERRIDES,
            "universe_exploration_fraction": 0.75,
        },
    },
    {
        "name": "UH_random100_45",
        "description": "45k de exploración uniforme, sin núcleo suave",
        "overrides": {
            **UNIVERSE_V17_2_BASE_OVERRIDES,
            "universe_exploration_fraction": 1.0,
        },
    },
    {
        "name": "UI_legacy_hard_complete",
        "description": "Pipeline V16 completo: hard filters, Sniper hard y radar 50",
        "overrides": {
            **UNIVERSE_V17_2_BASE_OVERRIDES,
            **LEGACY_HARD_FILTER_OVERRIDES,
        },
    },
    {
        "name": "UJ_topology_hard_current_scorer",
        "description": "Todos los filtros topológicos duros con scorer V17.2",
        "overrides": {
            **UNIVERSE_V17_2_BASE_OVERRIDES,
            **{
                key: value
                for key, value in LEGACY_HARD_FILTER_OVERRIDES.items()
                if key.endswith("_filter_enabled")
            },
        },
    },
    {
        "name": "UK_sum_hard_only",
        "description": "Sólo suma 95-115 como filtro duro",
        "overrides": {
            **UNIVERSE_V17_2_BASE_OVERRIDES,
            "sum_filter_enabled": True,
        },
    },
    {
        "name": "UL_structure_hard_only",
        "description": "Paridad, primos, consecutivos y delta como filtro duro",
        "overrides": {
            **UNIVERSE_V17_2_BASE_OVERRIDES,
            "structure_filter_enabled": True,
            "parity_filter_enabled": True,
            "prime_filter_enabled": True,
            "consecutive_filter_enabled": True,
            "max_delta_filter_enabled": True,
        },
    },
    {
        "name": "UM_entropy_hard_only",
        "description": "Sólo entropía como filtro duro",
        "overrides": {
            **UNIVERSE_V17_2_BASE_OVERRIDES,
            "entropy_filter_enabled": True,
        },
    },
    {
        "name": "UN_decade_oos_hard",
        "description": "Perfiles de décadas OOS como único filtro duro",
        "overrides": {
            **UNIVERSE_V17_2_BASE_OVERRIDES,
            "decade_profile_filter_enabled": True,
            "valid_decade_profiles": list(PROFILE_OOS_SET),
        },
    },
    {
        "name": "UO_topology_hard_radar50",
        "description": "Filtros topológicos duros con scorer V17.2 y radar 50",
        "overrides": {
            **UNIVERSE_V17_2_BASE_OVERRIDES,
            **{
                key: value
                for key, value in LEGACY_HARD_FILTER_OVERRIDES.items()
                if key.endswith("_filter_enabled")
            },
            "radar_percentile": 50.0,
        },
    },
    {
        "name": "UP_balanced45_radar50",
        "description": "Universo V17.2 de 45k sin filtros duros y radar 50",
        "overrides": {
            **UNIVERSE_V17_2_BASE_OVERRIDES,
            "radar_percentile": 50.0,
        },
    },
)

UNIVERSE_HYBRID_VARIANTS = (
    UNIVERSE_DYNAMIC_VARIANTS[0],
    UNIVERSE_DYNAMIC_VARIANTS[8],
    UNIVERSE_DYNAMIC_VARIANTS[9],
    {
        "name": "HQ_v17_2_18_topology_6",
        "description": "18 tickets V17.2 + 6 profundos del carril topológico",
        "overrides": {
            **UNIVERSE_V17_2_BASE_OVERRIDES,
            "universe_strategy_mode": "hybrid_soft_topology",
            "fitness_selector_mode": "hybrid_dual_lane",
            "hybrid_primary_tickets": 18,
            "hybrid_topology_tickets": 6,
            "hybrid_topology_min_rank": 501,
        },
    },
    {
        "name": "HR_v17_2_16_topology_8",
        "description": "16 tickets V17.2 + 8 profundos del carril topológico",
        "overrides": {
            **UNIVERSE_V17_2_BASE_OVERRIDES,
            "universe_strategy_mode": "hybrid_soft_topology",
            "fitness_selector_mode": "hybrid_dual_lane",
            "hybrid_primary_tickets": 16,
            "hybrid_topology_tickets": 8,
            "hybrid_topology_min_rank": 501,
        },
    },
    {
        "name": "HS_v17_2_12_topology_12",
        "description": "12 tickets V17.2 + 12 profundos del carril topológico",
        "overrides": {
            **UNIVERSE_V17_2_BASE_OVERRIDES,
            "universe_strategy_mode": "hybrid_soft_topology",
            "fitness_selector_mode": "hybrid_dual_lane",
            "hybrid_primary_tickets": 12,
            "hybrid_topology_tickets": 12,
            "hybrid_topology_min_rank": 501,
        },
    },
)

UNIVERSE_NESTED_CAP_VARIANTS = tuple(
    {
        "name": f"HN_nested_primary_{limit // 1000}k",
        "description": (
            "Carril suave anidado determinista con límite " f"{limit:,}"
        ),
        "overrides": {
            **UNIVERSE_V17_2_BASE_OVERRIDES,
            "universe_strategy_mode": "hybrid_soft_topology",
            "hybrid_primary_selection_mode": "nested_balanced",
            "hybrid_primary_universe_limit": limit,
            "hybrid_topology_universe_limit": 45000,
            "fitness_selector_mode": "hybrid_dual_lane",
            "hybrid_primary_tickets": 12,
            "hybrid_topology_tickets": 12,
            "hybrid_topology_min_rank": 501,
        },
    }
    for limit in (20000, 25000, 30000, 35000, 40000, 45000)
)

HYBRID_SELECTOR_BC_BASE_OVERRIDES = {
    **UNIVERSE_V17_2_BASE_OVERRIDES,
    "universe_strategy_mode": "hybrid_soft_topology",
    "hybrid_primary_selection_mode": "nested_balanced",
    "hybrid_primary_universe_limit": 30000,
    "hybrid_topology_universe_limit": 45000,
    "fitness_selector_mode": "hybrid_dual_lane",
    "hybrid_primary_tickets": 12,
    "hybrid_topology_tickets": 12,
    "hybrid_topology_min_rank": 501,
}

HYBRID_SELECTOR_BC_VARIANTS = (
    {
        "name": "HA_current_legacy_primary8_topology12deep",
        "description": "Control V17.3 anterior: primario legacy 8+4 y topología 0+12",
        "overrides": {
            **HYBRID_SELECTOR_BC_BASE_OVERRIDES,
            "hybrid_primary_selector_mode": "legacy_core_deep",
            "hybrid_topology_selector_mode": "deep_dispersion",
        },
    },
    {
        "name": "HB_stable_primary5_frontier3_deep4",
        "description": (
            "B: primario estable con 5 élites, 3 de frontera 6-500 y 4 profundos"
        ),
        "overrides": {
            **HYBRID_SELECTOR_BC_BASE_OVERRIDES,
            "hybrid_primary_selector_mode": "stable_frontier_deep",
            "hybrid_primary_elite_tickets": 5,
            "hybrid_primary_frontier_tickets": 3,
            "hybrid_primary_deep_tickets": 4,
            "hybrid_primary_frontier_max_rank": 500,
            "hybrid_primary_min_deep_rank": 501,
            "hybrid_topology_selector_mode": "deep_dispersion",
        },
    },
    {
        "name": "HC_stable_primary_and_topology_frontier",
        "description": (
            "C: B + topología estable con 1 élite, 3 de frontera y 8 profundos"
        ),
        "overrides": {
            **HYBRID_SELECTOR_BC_BASE_OVERRIDES,
            "hybrid_primary_selector_mode": "stable_frontier_deep",
            "hybrid_primary_elite_tickets": 5,
            "hybrid_primary_frontier_tickets": 3,
            "hybrid_primary_deep_tickets": 4,
            "hybrid_primary_frontier_max_rank": 500,
            "hybrid_primary_min_deep_rank": 501,
            "hybrid_topology_selector_mode": "stable_frontier_deep",
            "hybrid_topology_elite_tickets": 1,
            "hybrid_topology_frontier_tickets": 3,
            "hybrid_topology_deep_tickets": 8,
            "hybrid_topology_frontier_max_rank": 500,
            "hybrid_topology_min_deep_rank": 501,
        },
    },
)

HYBRID_SELECTOR_FOLLOWUP_VARIANTS = (
    HYBRID_SELECTOR_BC_VARIANTS[0],
    {
        "name": "HD_legacy_primary_topology1_frontier3_deep8",
        "description": (
            "Aísla C: primario legacy y topología con 1 élite, 3 frontera, 8 profundos"
        ),
        "overrides": {
            **HYBRID_SELECTOR_BC_BASE_OVERRIDES,
            "hybrid_primary_selector_mode": "legacy_core_deep",
            "hybrid_topology_selector_mode": "stable_frontier_deep",
            "hybrid_topology_elite_tickets": 1,
            "hybrid_topology_frontier_tickets": 3,
            "hybrid_topology_deep_tickets": 8,
            "hybrid_topology_frontier_max_rank": 500,
            "hybrid_topology_min_deep_rank": 501,
        },
    },
    {
        "name": "HE_stable_primary7_frontier1_deep4",
        "description": (
            "B conservador: primario 7 élites, 1 frontera 8-500 y 4 profundos"
        ),
        "overrides": {
            **HYBRID_SELECTOR_BC_BASE_OVERRIDES,
            "hybrid_primary_selector_mode": "stable_frontier_deep",
            "hybrid_primary_elite_tickets": 7,
            "hybrid_primary_frontier_tickets": 1,
            "hybrid_primary_deep_tickets": 4,
            "hybrid_primary_frontier_max_rank": 500,
            "hybrid_primary_min_deep_rank": 501,
            "hybrid_topology_selector_mode": "deep_dispersion",
        },
    },
    {
        "name": "HF_stable_primary5_frontier3_rank40_deep4",
        "description": (
            "Reparación local: primario estable 5 élites, 3 frontera 6-40 y 4 profundos"
        ),
        "overrides": {
            **HYBRID_SELECTOR_BC_BASE_OVERRIDES,
            "hybrid_primary_selector_mode": "stable_frontier_deep",
            "hybrid_primary_elite_tickets": 5,
            "hybrid_primary_frontier_tickets": 3,
            "hybrid_primary_deep_tickets": 4,
            "hybrid_primary_frontier_max_rank": 40,
            "hybrid_primary_min_deep_rank": 501,
            "hybrid_topology_selector_mode": "deep_dispersion",
        },
    },
    {
        "name": "HG_legacy_primary_topology1_frontier1_deep10",
        "description": (
            "C conservador: primario legacy y topología 1 élite, 1 frontera, 10 profundos"
        ),
        "overrides": {
            **HYBRID_SELECTOR_BC_BASE_OVERRIDES,
            "hybrid_primary_selector_mode": "legacy_core_deep",
            "hybrid_topology_selector_mode": "stable_frontier_deep",
            "hybrid_topology_elite_tickets": 1,
            "hybrid_topology_frontier_tickets": 1,
            "hybrid_topology_deep_tickets": 10,
            "hybrid_topology_frontier_max_rank": 500,
            "hybrid_topology_min_deep_rank": 501,
        },
    },
    {
        "name": "HH_legacy_primary_topology12deep_highorder",
        "description": (
            "Aísla objetivo: 12 topológicos profundos con novedad triple/cuádruple"
        ),
        "overrides": {
            **HYBRID_SELECTOR_BC_BASE_OVERRIDES,
            "hybrid_primary_selector_mode": "legacy_core_deep",
            "hybrid_topology_selector_mode": "stable_frontier_deep",
            "hybrid_topology_elite_tickets": 0,
            "hybrid_topology_frontier_tickets": 0,
            "hybrid_topology_deep_tickets": 12,
            "hybrid_topology_min_deep_rank": 501,
        },
    },
    {
        "name": "HI_legacy_primary_topology1_frontier1_deep10_legacyweights",
        "description": (
            "Aísla puertas: topología 1+1+10 con el objetivo profundo legacy"
        ),
        "overrides": {
            **HYBRID_SELECTOR_BC_BASE_OVERRIDES,
            "hybrid_primary_selector_mode": "legacy_core_deep",
            "hybrid_topology_selector_mode": "stable_frontier_deep",
            "hybrid_topology_elite_tickets": 1,
            "hybrid_topology_frontier_tickets": 1,
            "hybrid_topology_deep_tickets": 10,
            "hybrid_topology_frontier_max_rank": 500,
            "hybrid_topology_min_deep_rank": 501,
            "hybrid_topology_pair_novelty_weight": 0.40,
            "hybrid_topology_triple_novelty_weight": 0.0,
            "hybrid_topology_quad_novelty_weight": 0.0,
            "hybrid_topology_number_rarity_weight": 0.25,
            "hybrid_topology_dissimilarity_weight": 0.20,
            "hybrid_topology_local_quality_weight": 0.15,
        },
    },
    {
        "name": "HJ_legacy_primary_topology1_frontier1_deep10_quintetmass",
        "description": (
            "HI + masa local de quintetos ponderada por rank en profundidad"
        ),
        "overrides": {
            **HYBRID_SELECTOR_BC_BASE_OVERRIDES,
            "hybrid_primary_selector_mode": "legacy_core_deep",
            "hybrid_topology_selector_mode": "stable_frontier_deep",
            "hybrid_topology_elite_tickets": 1,
            "hybrid_topology_frontier_tickets": 1,
            "hybrid_topology_deep_tickets": 10,
            "hybrid_topology_frontier_max_rank": 500,
            "hybrid_topology_min_deep_rank": 501,
            "hybrid_topology_pair_novelty_weight": 0.10,
            "hybrid_topology_triple_novelty_weight": 0.10,
            "hybrid_topology_quad_novelty_weight": 0.10,
            "hybrid_topology_number_rarity_weight": 0.05,
            "hybrid_topology_dissimilarity_weight": 0.05,
            "hybrid_topology_local_quality_weight": 0.10,
            "hybrid_topology_quintet_mass_weight": 0.50,
            "hybrid_topology_quintet_rank_scale": 5000.0,
        },
    },
)

DEEP_SCORER_VARIANTS = (
    {
        "name": "SA_hi_rank_quality_control",
        "description": "Control productivo V17.4 HI con calidad por rank contextual",
        "overrides": {},
    },
    {
        "name": "SB_hi_geo_quality15",
        "description": "HI con Geo como calidad local conservadora al 15%",
        "overrides": {
            "hybrid_topology_deep_quality_mode": "geo",
        },
    },
    {
        "name": "SC_hi_geo_quality50",
        "description": "HI con Geo como señal local dominante al 50%",
        "overrides": {
            "hybrid_topology_deep_quality_mode": "geo",
            "hybrid_topology_pair_novelty_weight": 0.25,
            "hybrid_topology_number_rarity_weight": 0.15,
            "hybrid_topology_dissimilarity_weight": 0.10,
            "hybrid_topology_local_quality_weight": 0.50,
        },
    },
    {
        "name": "SD_hi_rank_tiebreak_geo",
        "description": "HI con Geo sólo como desempate exacto del rank profundo",
        "overrides": {
            "hybrid_topology_deep_quality_mode": "rank_tiebreak_geo",
        },
    },
    {
        "name": "SE_hi_rank_tiebreak_number",
        "description": (
            "HI con modelo por número sólo como desempate exacto del rank profundo"
        ),
        "overrides": {
            "hybrid_topology_deep_quality_mode": "rank_tiebreak_number",
        },
    },
    {
        "name": "SF_hi_rank_window25_geo",
        "description": "HI con Geo reordenando ventanas contextuales de 25 ranks",
        "overrides": {
            "hybrid_topology_deep_quality_mode": "rank_window_geo",
            "hybrid_topology_deep_quality_rank_window": 25,
        },
    },
    {
        "name": "SG_hi_rank_window100_geo",
        "description": "HI con Geo reordenando ventanas contextuales de 100 ranks",
        "overrides": {
            "hybrid_topology_deep_quality_mode": "rank_window_geo",
            "hybrid_topology_deep_quality_rank_window": 100,
        },
    },
    {
        "name": "SH_hi_rank_window500_geo",
        "description": "HI con Geo reordenando ventanas contextuales de 500 ranks",
        "overrides": {
            "hybrid_topology_deep_quality_mode": "rank_window_geo",
            "hybrid_topology_deep_quality_rank_window": 500,
        },
    },
    {
        "name": "SI_hi_marginal5_uniform15",
        "description": (
            "HI reemplazando calidad local por cobertura marginal 5/6 uniforme"
        ),
        "overrides": {
            "hybrid_topology_local_quality_weight": 0.0,
            "hybrid_topology_quintet_marginal_coverage_weight": 0.15,
            "hybrid_topology_quintet_coverage_rank_scale": 0.0,
        },
    },
    {
        "name": "SJ_hi_marginal5_uniform30",
        "description": (
            "HI con cobertura marginal 5/6 uniforme al 30% y dispersión reducida"
        ),
        "overrides": {
            "hybrid_topology_pair_novelty_weight": 0.30,
            "hybrid_topology_number_rarity_weight": 0.20,
            "hybrid_topology_dissimilarity_weight": 0.15,
            "hybrid_topology_local_quality_weight": 0.05,
            "hybrid_topology_quintet_marginal_coverage_weight": 0.30,
            "hybrid_topology_quintet_coverage_rank_scale": 0.0,
        },
    },
    {
        "name": "SK_hi_marginal5_rankweighted15",
        "description": (
            "HI con cobertura marginal 5/6 ponderada por rank al 15%"
        ),
        "overrides": {
            "hybrid_topology_local_quality_weight": 0.0,
            "hybrid_topology_quintet_marginal_coverage_weight": 0.15,
            "hybrid_topology_quintet_coverage_rank_scale": 5000.0,
        },
    },
    {
        "name": "SL_hi_marginal5_uniform_tiebreak",
        "description": (
            "HI con cobertura marginal 5/6 uniforme sólo como desempate"
        ),
        "overrides": {
            "hybrid_topology_quintet_marginal_coverage_weight": 0.000001,
            "hybrid_topology_quintet_coverage_rank_scale": 0.0,
        },
    },
    {
        "name": "SM_hi_marginal5_rankweighted05",
        "description": (
            "HI reemplazando 5% de calidad local por cobertura 5/6 ponderada"
        ),
        "overrides": {
            "hybrid_topology_local_quality_weight": 0.10,
            "hybrid_topology_quintet_marginal_coverage_weight": 0.05,
            "hybrid_topology_quintet_coverage_rank_scale": 5000.0,
        },
    },
)

VARIANT_SUITES = {
    "core": CORE_VARIANTS,
    "blend-sweep": BLEND_SWEEP_VARIANTS,
    "selector-shadows": SELECTOR_SHADOW_VARIANTS,
    "controlled-weights": CONTROLLED_WEIGHT_VARIANTS,
    "deep-dispersion": DEEP_DISPERSION_VARIANTS,
    "elite-coverage-deep": ELITE_COVERAGE_DEEP_VARIANTS,
    "five-hit-selector": FIVE_HIT_SELECTOR_VARIANTS,
    "five-hit-scorer": FIVE_HIT_SCORER_VARIANTS,
    "five-hit-validation": FIVE_HIT_VALIDATION_VARIANTS,
    "five-hit-reserve": FIVE_HIT_RESERVE_VARIANTS,
    "five-hit-interaction": FIVE_HIT_INTERACTION_VARIANTS,
    "five-hit-holdout": FIVE_HIT_HOLDOUT_VARIANTS,
    "universe-v17": UNIVERSE_V17_VARIANTS,
    "universe-dynamic": UNIVERSE_DYNAMIC_VARIANTS,
    "universe-hybrid": UNIVERSE_HYBRID_VARIANTS,
    "universe-nested-cap": UNIVERSE_NESTED_CAP_VARIANTS,
    "selector-hybrid-bc": HYBRID_SELECTOR_BC_VARIANTS,
    "selector-hybrid-followup": HYBRID_SELECTOR_FOLLOWUP_VARIANTS,
    "deep-scorer": DEEP_SCORER_VARIANTS,
}


def _plain_number(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _summarize(result, forensic_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = np.asarray([row.get("rank", 0) for row in forensic_rows], dtype=float)
    proximity = np.asarray(
        [row.get("proximity", 999) for row in forensic_rows], dtype=float
    )
    ai_scores = np.asarray(
        [row.get("ai_score", 0.0) for row in forensic_rows], dtype=float
    )
    geo_scores = np.asarray(
        [row.get("geo_score", 0.0) for row in forensic_rows], dtype=float
    )
    jackpots = [row for row in forensic_rows if int(row.get("hits", 0)) == 6]
    hit_distribution = {
        str(key): int(value) for key, value in result.hit_distribution.items()
    }
    selected_max_hits = []
    earnings_by_draw = []
    prefix_20_max_hits = []
    prefix_20_earnings = []
    selected_ticket_observations = 0
    selected_ranks_by_draw = []
    selected_stable_ranks_by_draw = []
    deep_ranks_by_draw = []
    portfolio_elite_ranks_by_draw = []
    portfolio_coverage_ranks_by_draw = []
    portfolio_deep_ranks_by_draw = []
    portfolio_unique_pairs = []
    portfolio_unique_triples = []
    portfolio_unique_quads = []
    selected_unique_pairs = []
    selected_unique_triples = []
    selected_unique_quads = []
    selected_unique_quintuples = []
    selected_radius_one_coverage = []
    selected_radius_one_max = []
    winner_selected_max_overlap = []
    universe_sizes = []
    winner_exact_in_universe = []
    hybrid_primary_counts = []
    hybrid_topology_counts = []
    hybrid_topology_radius_one_targets = []
    hybrid_primary_lane_ranks = []
    hybrid_topology_lane_ranks = []
    hybrid_primary_phase_counts: dict[str, int] = {}
    hybrid_topology_phase_counts: dict[str, int] = {}
    radar_jackpot_diagnostics = []
    deep_neighbor_signal_diagnostics = []
    deep_band_signal_diagnostics = []
    for row in forensic_rows:
        metrics = row.get("metrics_json", {})
        if not isinstance(metrics, dict) or "selected_max_hits" not in metrics:
            selected_max_hits = []
            break
        selected_max_hits.append(int(metrics["selected_max_hits"]))
        ticket_hits = [int(value) for value in metrics.get("selected_ticket_hits", [])]
        ticket_prizes = [
            float(value) for value in metrics.get("selected_ticket_prizes", [])
        ]
        selected_ticket_observations += len(ticket_hits)
        if ticket_prizes:
            earnings_by_draw.append(float(sum(ticket_prizes)))
        if len(ticket_hits) >= 20 and len(ticket_prizes) >= 20:
            prefix_20_max_hits.append(int(max(ticket_hits[:20])))
            prefix_20_earnings.append(float(sum(ticket_prizes[:20])))
        selected_ranks_by_draw.extend(
            int(rank) for rank in metrics.get("selected_ranks", [])
        )
        selected_stable_ranks_by_draw.extend(
            int(rank) for rank in metrics.get("selected_stable_ranks", [])
        )
        deep_ranks_by_draw.extend(
            int(rank) for rank in metrics.get("deep_dispersion_ranks", [])
        )
        portfolio_elite_ranks_by_draw.extend(
            int(rank) for rank in metrics.get("portfolio_elite_ranks", [])
        )
        portfolio_coverage_ranks_by_draw.extend(
            int(rank) for rank in metrics.get("portfolio_coverage_ranks", [])
        )
        portfolio_deep_ranks_by_draw.extend(
            int(rank) for rank in metrics.get("portfolio_deep_ranks", [])
        )
        for metric_name, destination in (
            ("portfolio_unique_pairs", portfolio_unique_pairs),
            ("portfolio_unique_triples", portfolio_unique_triples),
            ("portfolio_unique_quads", portfolio_unique_quads),
        ):
            if metrics.get(metric_name) is not None:
                destination.append(int(metrics[metric_name]))
        for metric_name, destination in (
            ("selected_unique_pairs", selected_unique_pairs),
            ("selected_unique_triples", selected_unique_triples),
            ("selected_unique_quads", selected_unique_quads),
            ("selected_unique_quintuples", selected_unique_quintuples),
            ("selected_radius_one_coverage", selected_radius_one_coverage),
            ("selected_radius_one_max", selected_radius_one_max),
        ):
            if metrics.get(metric_name) is not None:
                destination.append(int(metrics[metric_name]))
        if metrics.get("winner_selected_max_overlap") is not None:
            winner_selected_max_overlap.append(
                int(metrics["winner_selected_max_overlap"])
            )
        if row.get("univ_size") is not None:
            universe_sizes.append(int(row["univ_size"]))
        if metrics.get("winner_in_universe") is not None:
            winner_exact_in_universe.append(int(metrics["winner_in_universe"]))
        if metrics.get("hybrid_primary_selected") is not None:
            hybrid_primary_counts.append(int(metrics["hybrid_primary_selected"]))
        if metrics.get("hybrid_topology_selected") is not None:
            hybrid_topology_counts.append(int(metrics["hybrid_topology_selected"]))
        if metrics.get("hybrid_topology_radius_one_targets") is not None:
            hybrid_topology_radius_one_targets.append(
                int(metrics["hybrid_topology_radius_one_targets"])
            )
        hybrid_primary_lane_ranks.extend(
            int(rank) for rank in metrics.get("hybrid_primary_lane_ranks", [])
        )
        hybrid_topology_lane_ranks.extend(
            int(rank) for rank in metrics.get("hybrid_topology_lane_ranks", [])
        )
        for phase in metrics.get("hybrid_primary_lane_phases", []):
            key = str(phase)
            hybrid_primary_phase_counts[key] = (
                hybrid_primary_phase_counts.get(key, 0) + 1
            )
        for phase in metrics.get("hybrid_topology_lane_phases", []):
            key = str(phase)
            hybrid_topology_phase_counts[key] = (
                hybrid_topology_phase_counts.get(key, 0) + 1
            )
        if metrics.get("winner_topology_deep_oracle_max_overlap") is not None:
            deep_band_signal_diagnostics.append(
                {
                    "draw_id": int(row["draw_id"]),
                    "oracle": {
                        "max_overlap": int(
                            metrics["winner_topology_deep_oracle_max_overlap"]
                        ),
                        "overlap_sum": int(
                            metrics["winner_topology_deep_oracle_overlap_sum"]
                        ),
                        "count_ge_4": int(
                            metrics["winner_topology_deep_oracle_count_ge_4"]
                        ),
                        "count_ge_5": int(
                            metrics["winner_topology_deep_oracle_count_ge_5"]
                        ),
                    },
                    "selected": (
                        {
                            "max_overlap": int(
                                metrics[
                                    "winner_topology_deep_selected_max_overlap"
                                ]
                            ),
                            "overlap_sum": int(
                                metrics[
                                    "winner_topology_deep_selected_overlap_sum"
                                ]
                            ),
                            "count_ge_4": int(
                                metrics[
                                    "winner_topology_deep_selected_count_ge_4"
                                ]
                            ),
                            "count_ge_5": int(
                                metrics[
                                    "winner_topology_deep_selected_count_ge_5"
                                ]
                            ),
                            "count": int(
                                metrics["winner_topology_deep_selected_count"]
                            ),
                        }
                        if metrics.get(
                            "winner_topology_deep_selected_max_overlap"
                        )
                        is not None
                        else None
                    ),
                    "signals": {
                        name: {
                            "top1_max_overlap": metrics.get(
                                f"winner_topology_deep_{name}_top1_max_overlap"
                            ),
                            "top1_overlap_sum": metrics.get(
                                f"winner_topology_deep_{name}_top1_overlap_sum"
                            ),
                            "top1_count_ge_4": metrics.get(
                                f"winner_topology_deep_{name}_top1_count_ge_4"
                            ),
                            "top1_count_ge_5": metrics.get(
                                f"winner_topology_deep_{name}_top1_count_ge_5"
                            ),
                            "top10_max_overlap": metrics.get(
                                f"winner_topology_deep_{name}_top10_max_overlap"
                            ),
                            "top_tie_mean": metrics.get(
                                f"winner_topology_deep_{name}_top_tie_mean"
                            ),
                        }
                        for name in ("hybrid", "ai", "number", "geo")
                        if metrics.get(
                            f"winner_topology_deep_{name}_top1_max_overlap"
                        )
                        is not None
                    },
                }
            )
        if int(metrics.get("winner_topology_neighbor_band_count", 0)) > 0:
            deep_neighbor_signal_diagnostics.append(
                {
                    "draw_id": int(row["draw_id"]),
                    "band_count": int(
                        metrics["winner_topology_neighbor_band_count"]
                    ),
                    "best_ranks": {
                        name: metrics.get(
                            f"winner_topology_neighbor_{name}_best_rank"
                        )
                        for name in ("hybrid", "ai", "number", "geo")
                    },
                    "best_tie_sizes": {
                        name: metrics.get(
                            f"winner_topology_neighbor_{name}_best_tie_size"
                        )
                        for name in ("hybrid", "ai", "number", "geo")
                    },
                    "best_overlaps": {
                        name: metrics.get(
                            f"winner_topology_neighbor_{name}_best_overlap"
                        )
                        for name in ("hybrid", "ai", "number", "geo")
                    },
                }
            )
        if int(metrics.get("winner_in_universe", 0)) == 1:
            radar_jackpot_diagnostics.append(
                {
                    "draw_id": int(row["draw_id"]),
                    "winner_rank": int(row.get("rank", 0)),
                    "winner_stable_rank": (
                        int(metrics["winner_stable_rank"])
                        if metrics.get("winner_stable_rank") is not None
                        else None
                    ),
                    "winner_score_tie_size": int(
                        metrics.get("winner_score_tie_size", 0)
                    ),
                    "winner_stable_rank_proximity": int(
                        metrics.get("winner_stable_rank_proximity", 999)
                    ),
                    "selected_max_overlap": int(
                        metrics.get("winner_selected_max_overlap", 0)
                    ),
                    "selected_missing_numbers": int(
                        metrics.get("winner_selected_min_missing", 6)
                    ),
                    "selected_count_ge_4": int(
                        metrics.get("winner_selected_count_ge_4", 0)
                    ),
                    "selected_count_ge_5": int(
                        metrics.get("winner_selected_count_ge_5", 0)
                    ),
                    "selected_best_ranks": [
                        int(rank)
                        for rank in metrics.get("winner_selected_best_ranks", [])
                    ],
                    "selected_best_stable_ranks": [
                        int(rank)
                        for rank in metrics.get(
                            "winner_selected_best_stable_ranks", []
                        )
                    ],
                    "topology_deep_signal_ranks": {
                        name: (
                            int(metrics[f"winner_topology_deep_{name}_rank"])
                            if metrics.get(
                                f"winner_topology_deep_{name}_rank"
                            )
                            is not None
                            else None
                        )
                        for name in ("hybrid", "ai", "number", "geo")
                    },
                    "topology_deep_signal_tie_sizes": {
                        name: (
                            int(
                                metrics[
                                    f"winner_topology_deep_{name}_tie_size"
                                ]
                            )
                            if metrics.get(
                                f"winner_topology_deep_{name}_tie_size"
                            )
                            is not None
                            else None
                        )
                        for name in ("hybrid", "ai", "number", "geo")
                    },
                    "topology_deep_band_rank": metrics.get(
                        "winner_topology_deep_band_rank"
                    ),
                    "topology_deep_band_size": metrics.get(
                        "winner_topology_deep_band_size"
                    ),
                }
            )
    selected_ranks_array = np.asarray(selected_ranks_by_draw, dtype=float)
    selected_stable_ranks_array = np.asarray(
        selected_stable_ranks_by_draw, dtype=float
    )
    deep_ranks_array = np.asarray(deep_ranks_by_draw, dtype=float)
    portfolio_elite_ranks_array = np.asarray(
        portfolio_elite_ranks_by_draw, dtype=float
    )
    portfolio_coverage_ranks_array = np.asarray(
        portfolio_coverage_ranks_by_draw, dtype=float
    )
    portfolio_deep_ranks_array = np.asarray(
        portfolio_deep_ranks_by_draw, dtype=float
    )
    winner_overlap_array = np.asarray(winner_selected_max_overlap, dtype=int)
    universe_size_array = np.asarray(universe_sizes, dtype=int)
    prefix_20_array = np.asarray(prefix_20_max_hits, dtype=int)
    prefix_20_earnings_array = np.asarray(prefix_20_earnings, dtype=float)
    inferred_ticket_cost = (
        float(result.investment / selected_ticket_observations)
        if selected_ticket_observations
        else 0.0
    )
    prefix_20_investment = float(
        len(prefix_20_max_hits) * 20 * inferred_ticket_cost
    )
    deep_band_signal_summary = {}
    if deep_band_signal_diagnostics:
        oracle_rows = [row["oracle"] for row in deep_band_signal_diagnostics]
        deep_band_signal_summary["oracle"] = {
            "draws": len(oracle_rows),
            "mean_max_overlap": float(
                np.mean([row["max_overlap"] for row in oracle_rows])
            ),
            "draws_ge_4": int(
                sum(row["max_overlap"] >= 4 for row in oracle_rows)
            ),
            "draws_ge_5": int(
                sum(row["max_overlap"] >= 5 for row in oracle_rows)
            ),
            "mean_overlap_sum": float(
                np.mean([row["overlap_sum"] for row in oracle_rows])
            ),
        }
        selected_rows = [
            row["selected"]
            for row in deep_band_signal_diagnostics
            if row.get("selected") is not None
        ]
        if selected_rows:
            deep_band_signal_summary["selected"] = {
                "draws": len(selected_rows),
                "mean_max_overlap": float(
                    np.mean([row["max_overlap"] for row in selected_rows])
                ),
                "draws_ge_4": int(
                    sum(row["max_overlap"] >= 4 for row in selected_rows)
                ),
                "draws_ge_5": int(
                    sum(row["max_overlap"] >= 5 for row in selected_rows)
                ),
                "total_count_ge_4": int(
                    sum(row["count_ge_4"] for row in selected_rows)
                ),
                "total_count_ge_5": int(
                    sum(row["count_ge_5"] for row in selected_rows)
                ),
                "mean_overlap_sum": float(
                    np.mean([row["overlap_sum"] for row in selected_rows])
                ),
                "mean_ticket_count": float(
                    np.mean([row["count"] for row in selected_rows])
                ),
            }
        for signal_name in ("hybrid", "ai", "number", "geo"):
            signal_rows = [
                row["signals"][signal_name]
                for row in deep_band_signal_diagnostics
                if signal_name in row["signals"]
            ]
            if not signal_rows:
                continue
            deep_band_signal_summary[signal_name] = {
                "draws": len(signal_rows),
                "mean_top1_max_overlap": float(
                    np.mean([row["top1_max_overlap"] for row in signal_rows])
                ),
                "draws_top1_ge_4": int(
                    sum(row["top1_max_overlap"] >= 4 for row in signal_rows)
                ),
                "draws_top1_ge_5": int(
                    sum(row["top1_max_overlap"] >= 5 for row in signal_rows)
                ),
                "total_top1_count_ge_4": int(
                    sum(row["top1_count_ge_4"] for row in signal_rows)
                ),
                "total_top1_count_ge_5": int(
                    sum(row["top1_count_ge_5"] for row in signal_rows)
                ),
                "mean_top1_overlap_sum": float(
                    np.mean([row["top1_overlap_sum"] for row in signal_rows])
                ),
                "mean_top10_max_overlap": float(
                    np.mean([row["top10_max_overlap"] for row in signal_rows])
                ),
                "draws_top10_ge_5": int(
                    sum(row["top10_max_overlap"] >= 5 for row in signal_rows)
                ),
                "mean_top_tie_size": float(
                    np.mean([row["top_tie_mean"] for row in signal_rows])
                ),
            }
    return {
        "draws": int(result.total_draws_tested),
        "draw_ids": [int(row["draw_id"]) for row in forensic_rows],
        "event_id": forensic_rows[0].get("event_id", "") if forensic_rows else "",
        "investment": float(result.investment),
        "earnings": float(result.earnings),
        "net_balance": float(result.net_balance),
        "gross_return_ratio": (
            float(result.earnings / result.investment)
            if result.investment
            else 0.0
        ),
        "net_roi": (
            float(result.net_balance / result.investment)
            if result.investment
            else 0.0
        ),
        "prize_breakdown": getattr(result, "prize_breakdown", {}),
        "ticket_hit_distribution": hit_distribution,
        "selected_hits_4": hit_distribution.get("4", 0),
        "selected_hits_5": hit_distribution.get("5", 0),
        "selected_jackpots": hit_distribution.get("6", 0),
        "selected_max_hits_by_draw": selected_max_hits,
        "earnings_by_draw": earnings_by_draw,
        "prefix_20": {
            "draws": int(prefix_20_array.size),
            "tickets_per_draw": 20,
            "investment": prefix_20_investment,
            "earnings": float(prefix_20_earnings_array.sum()),
            "net_balance": float(
                prefix_20_earnings_array.sum() - prefix_20_investment
            ),
            "gross_return_ratio": (
                float(prefix_20_earnings_array.sum() / prefix_20_investment)
                if prefix_20_investment
                else 0.0
            ),
            "avg_max_hits": (
                float(prefix_20_array.mean()) if prefix_20_array.size else None
            ),
            "draws_ge_4": int(np.sum(prefix_20_array >= 4)),
            "draws_ge_5": int(np.sum(prefix_20_array >= 5)),
            "draws_eq_6": int(np.sum(prefix_20_array == 6)),
            "max_hits_distribution": {
                str(hit): int(np.sum(prefix_20_array == hit)) for hit in range(7)
            },
        },
        "selected_rank_min": (
            int(np.min(selected_ranks_array)) if selected_ranks_array.size else None
        ),
        "selected_rank_median": (
            float(np.median(selected_ranks_array))
            if selected_ranks_array.size
            else None
        ),
        "selected_rank_max": (
            int(np.max(selected_ranks_array)) if selected_ranks_array.size else None
        ),
        "selected_stable_rank_min": (
            int(np.min(selected_stable_ranks_array))
            if selected_stable_ranks_array.size
            else None
        ),
        "selected_stable_rank_median": (
            float(np.median(selected_stable_ranks_array))
            if selected_stable_ranks_array.size
            else None
        ),
        "selected_stable_rank_max": (
            int(np.max(selected_stable_ranks_array))
            if selected_stable_ranks_array.size
            else None
        ),
        "deep_rank_min": (
            int(np.min(deep_ranks_array)) if deep_ranks_array.size else None
        ),
        "deep_rank_median": (
            float(np.median(deep_ranks_array)) if deep_ranks_array.size else None
        ),
        "deep_rank_max": (
            int(np.max(deep_ranks_array)) if deep_ranks_array.size else None
        ),
        "portfolio_elite_rank_min": (
            int(np.min(portfolio_elite_ranks_array))
            if portfolio_elite_ranks_array.size
            else None
        ),
        "portfolio_elite_rank_max": (
            int(np.max(portfolio_elite_ranks_array))
            if portfolio_elite_ranks_array.size
            else None
        ),
        "portfolio_coverage_rank_median": (
            float(np.median(portfolio_coverage_ranks_array))
            if portfolio_coverage_ranks_array.size
            else None
        ),
        "portfolio_deep_rank_median": (
            float(np.median(portfolio_deep_ranks_array))
            if portfolio_deep_ranks_array.size
            else None
        ),
        "portfolio_unique_pairs_mean": (
            float(np.mean(portfolio_unique_pairs))
            if portfolio_unique_pairs
            else None
        ),
        "portfolio_unique_triples_mean": (
            float(np.mean(portfolio_unique_triples))
            if portfolio_unique_triples
            else None
        ),
        "portfolio_unique_quads_mean": (
            float(np.mean(portfolio_unique_quads))
            if portfolio_unique_quads
            else None
        ),
        "selected_unique_pairs_mean": (
            float(np.mean(selected_unique_pairs)) if selected_unique_pairs else None
        ),
        "selected_unique_triples_mean": (
            float(np.mean(selected_unique_triples))
            if selected_unique_triples
            else None
        ),
        "selected_unique_quads_mean": (
            float(np.mean(selected_unique_quads)) if selected_unique_quads else None
        ),
        "selected_unique_quintuples_mean": (
            float(np.mean(selected_unique_quintuples))
            if selected_unique_quintuples
            else None
        ),
        "selected_radius_one_coverage_mean": (
            float(np.mean(selected_radius_one_coverage))
            if selected_radius_one_coverage
            else None
        ),
        "selected_radius_one_efficiency_mean": (
            float(
                np.mean(
                    np.asarray(selected_radius_one_coverage, dtype=float)
                    / np.asarray(selected_radius_one_max, dtype=float)
                )
            )
            if selected_radius_one_coverage
            and selected_radius_one_max
            and all(selected_radius_one_max)
            else None
        ),
        "winner_selected_max_overlap_distribution": {
            str(hit): int(np.sum(winner_overlap_array == hit)) for hit in range(7)
        },
        "universe_size_min": (
            int(np.min(universe_size_array)) if universe_size_array.size else None
        ),
        "universe_size_mean": (
            float(np.mean(universe_size_array)) if universe_size_array.size else None
        ),
        "universe_size_max": (
            int(np.max(universe_size_array)) if universe_size_array.size else None
        ),
        "winner_exact_in_universe": int(sum(winner_exact_in_universe)),
        "winner_exact_in_universe_rate": (
            float(np.mean(winner_exact_in_universe))
            if winner_exact_in_universe
            else None
        ),
        "hybrid_primary_selected_mean": (
            float(np.mean(hybrid_primary_counts)) if hybrid_primary_counts else None
        ),
        "hybrid_topology_selected_mean": (
            float(np.mean(hybrid_topology_counts))
            if hybrid_topology_counts
            else None
        ),
        "hybrid_topology_radius_one_targets_mean": (
            float(np.mean(hybrid_topology_radius_one_targets))
            if hybrid_topology_radius_one_targets
            else None
        ),
        "hybrid_primary_lane_rank_min": (
            int(np.min(hybrid_primary_lane_ranks))
            if hybrid_primary_lane_ranks
            else None
        ),
        "hybrid_primary_lane_rank_median": (
            float(np.median(hybrid_primary_lane_ranks))
            if hybrid_primary_lane_ranks
            else None
        ),
        "hybrid_primary_lane_rank_max": (
            int(np.max(hybrid_primary_lane_ranks))
            if hybrid_primary_lane_ranks
            else None
        ),
        "hybrid_topology_lane_rank_min": (
            int(np.min(hybrid_topology_lane_ranks))
            if hybrid_topology_lane_ranks
            else None
        ),
        "hybrid_topology_lane_rank_median": (
            float(np.median(hybrid_topology_lane_ranks))
            if hybrid_topology_lane_ranks
            else None
        ),
        "hybrid_topology_lane_rank_max": (
            int(np.max(hybrid_topology_lane_ranks))
            if hybrid_topology_lane_ranks
            else None
        ),
        "hybrid_primary_phase_counts": hybrid_primary_phase_counts,
        "hybrid_topology_phase_counts": hybrid_topology_phase_counts,
        "radar_jackpot_diagnostics": radar_jackpot_diagnostics,
        "deep_neighbor_signal_diagnostics": deep_neighbor_signal_diagnostics,
        "deep_band_signal_diagnostics": deep_band_signal_diagnostics,
        "deep_band_signal_summary": deep_band_signal_summary,
        "jackpots_in_selector_radar": int(
            sum(
                row.get("winner_stable_rank") is not None
                for row in radar_jackpot_diagnostics
            )
        ),
        "jackpots_outside_selector_radar": int(
            sum(
                row.get("winner_stable_rank") is None
                for row in radar_jackpot_diagnostics
            )
        ),
        "draws_with_max_hits_at_least_4": sum(
            value >= 4 for value in selected_max_hits
        ),
        "draws_with_max_hits_at_least_5": sum(
            value >= 5 for value in selected_max_hits
        ),
        "universe_jackpots": len(jackpots),
        "universe_hits_5": sum(
            int(row.get("hits", 0)) == 5 for row in forensic_rows
        ),
        "universe_hits_4": sum(
            int(row.get("hits", 0)) == 4 for row in forensic_rows
        ),
        "rank_median": float(np.median(ranks)) if ranks.size else None,
        "rank_mean": float(np.mean(ranks)) if ranks.size else None,
        "proximity_median": float(np.median(proximity)) if proximity.size else None,
        "proximity_mean": float(np.mean(proximity)) if proximity.size else None,
        "proximity_zero": int(np.sum(proximity == 0)),
        "proximity_le_50": int(np.sum(proximity <= 50)),
        "ai_score_mean": float(np.mean(ai_scores)) if ai_scores.size else None,
        "geo_score_mean": float(np.mean(geo_scores)) if geo_scores.size else None,
        "jackpot_details": [
            {
                "draw_id": int(row["draw_id"]),
                "rank": int(row["rank"]),
                "proximity": int(row["proximity"]),
                "ai_score": float(row.get("ai_score", 0.0)),
                "geo_score": float(row.get("geo_score", 0.0)),
            }
            for row in jackpots
        ],
    }


def _exact_mcnemar(reference: np.ndarray, challenger: np.ndarray) -> dict[str, Any]:
    reference_only = int(np.sum(reference & ~challenger))
    challenger_only = int(np.sum(~reference & challenger))
    both = int(np.sum(reference & challenger))
    neither = int(np.sum(~reference & ~challenger))
    discordant = reference_only + challenger_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(reference_only, challenger_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "reference_only": reference_only,
        "challenger_only": challenger_only,
        "both": both,
        "neither": neither,
        "discordant": discordant,
        "exact_two_sided_p": float(p_value),
    }


def _paired_comparisons(
    results: list[dict[str, Any]], seed: int, resamples: int = 20_000
) -> list[dict[str, Any]]:
    if not results:
        return []
    reference = np.asarray(results[0].get("selected_max_hits_by_draw", []), dtype=int)
    if reference.size == 0:
        return []

    comparisons = []
    for offset, challenger_summary in enumerate(results[1:], start=1):
        challenger = np.asarray(
            challenger_summary.get("selected_max_hits_by_draw", []), dtype=int
        )
        if challenger.shape != reference.shape:
            raise RuntimeError("Resultados A/B sin pares completos por sorteo")
        differences = challenger.astype(float) - reference.astype(float)
        rng = np.random.default_rng(int(seed) + 10_000 + offset)
        bootstrap_idx = rng.integers(
            0, differences.size, size=(int(resamples), differences.size)
        )
        bootstrap_means = differences[bootstrap_idx].mean(axis=1)
        ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
        signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(int(resamples), differences.size),
        )
        observed = abs(float(differences.mean()))
        permuted = np.abs((signs * differences).mean(axis=1))
        permutation_p = float((np.sum(permuted >= observed) + 1) / (resamples + 1))
        reference_earnings = np.asarray(
            results[0].get("earnings_by_draw", []), dtype=float
        )
        challenger_earnings = np.asarray(
            challenger_summary.get("earnings_by_draw", []), dtype=float
        )
        earnings_comparison = None
        if (
            reference_earnings.shape == challenger_earnings.shape
            and reference_earnings.size == reference.size
        ):
            earnings_delta = challenger_earnings - reference_earnings
            earnings_bootstrap = earnings_delta[bootstrap_idx].mean(axis=1)
            earnings_observed = abs(float(earnings_delta.mean()))
            earnings_permuted = np.abs((signs * earnings_delta).mean(axis=1))
            earnings_comparison = {
                "wins": int(np.sum(earnings_delta > 0)),
                "losses": int(np.sum(earnings_delta < 0)),
                "ties": int(np.sum(earnings_delta == 0)),
                "total_reference": float(reference_earnings.sum()),
                "total_challenger": float(challenger_earnings.sum()),
                "total_delta": float(earnings_delta.sum()),
                "mean_delta_per_draw": float(earnings_delta.mean()),
                "bootstrap_95_ci_mean_delta": [
                    float(value)
                    for value in np.percentile(earnings_bootstrap, [2.5, 97.5])
                ],
                "paired_permutation_two_sided_p": float(
                    (np.sum(earnings_permuted >= earnings_observed) + 1)
                    / (resamples + 1)
                ),
            }
        comparisons.append(
            {
                "reference": results[0]["name"],
                "challenger": challenger_summary["name"],
                "paired_draws": int(reference.size),
                "max_hits_wins": int(np.sum(differences > 0)),
                "max_hits_losses": int(np.sum(differences < 0)),
                "max_hits_ties": int(np.sum(differences == 0)),
                "mean_max_hits_reference": float(reference.mean()),
                "mean_max_hits_challenger": float(challenger.mean()),
                "mean_paired_delta": float(differences.mean()),
                "bootstrap_95_ci_mean_delta": [float(ci_low), float(ci_high)],
                "paired_permutation_two_sided_p": permutation_p,
                "mcnemar_ge_4": _exact_mcnemar(reference >= 4, challenger >= 4),
                "mcnemar_ge_5": _exact_mcnemar(reference >= 5, challenger >= 5),
                "mcnemar_eq_6": _exact_mcnemar(reference == 6, challenger == 6),
                "paired_earnings": earnings_comparison,
            }
        )
    return comparisons


def run_experiments(
    backtest_size: int,
    tickets: int,
    seed: int,
    variants=CORE_VARIANTS,
    experiment_name: str = "melate_resonance_ab_v1",
    isolate_ledger: bool = True,
    end_contest: int | None = None,
) -> dict[str, Any]:
    profile = LOTTERY_PROFILES["melate_retro"]
    history = LotteryLoader(profile).load_data()
    if end_contest is not None:
        ordered = sort_history_chronologically(history)
        keep = [
            index
            for index, contest in enumerate(ordered.concursos)
            if int(contest) <= int(end_contest)
        ]
        history = DrawHistoryDTO(
            dates=[ordered.dates[index] for index in keep],
            winning_numbers=[ordered.winning_numbers[index] for index in keep],
            concursos=[ordered.concursos[index] for index in keep],
        )
    artifacts = prepare_fixed_origin_models(history, backtest_size)
    results: list[dict[str, Any]] = []
    reference_draw_ids: list[int] | None = None

    with TemporaryDirectory(prefix="mrpro_ab_ledger_") as ledger_directory:
        ledger_root = Path(ledger_directory)
        for variant in variants:
            overrides = dict(BEST_SETTINGS)
            overrides.update(variant["overrides"])
            overrides.update(
                {
                    "seed": int(seed),
                    "backtest_model_mode": "fixed_origin",
                    "fixed_origin_training_cutoff": artifacts.training_cutoff_contest,
                    "fixed_origin_test_start": artifacts.test_start_contest,
                    "fixed_origin_test_end": artifacts.test_end_contest,
                    "fixed_origin_dataset_hash": artifacts.dataset_hash,
                }
            )

            config = PredictionConfigDTO(
                TOTAL_BALLS,
                TICKET_SIZE,
                tickets,
                backtest_size=backtest_size,
                filter_overrides=overrides,
            )
            engine = BacktestEngine()
            if isolate_ledger:
                engine.tracker = PerformanceTracker(
                    log_path=ledger_root / "detailed_forensic_log.csv",
                    json_path=ledger_root / "backtest_results.json",
                    archive_directory=ledger_root / "forensic_log_archive",
                )
            result = engine.run(
                GeneticSelectorStrategy(
                    model_path=artifacts.context_model_path,
                    number_model_path=artifacts.number_model_path,
                ),
                history,
                config,
                verbose=False,
                pre_process_strategy=(
                    HybridUniverseReductionStrategy()
                    if overrides.get("universe_strategy_mode")
                    == "hybrid_soft_topology"
                    else UniverseReductionStrategy()
                ),
            )
            summary = _summarize(result, engine.forensic_data)
            summary["name"] = variant["name"]
            summary["description"] = variant["description"]
            summary["overrides"] = variant["overrides"]

            draw_ids = summary.pop("draw_ids")
            if reference_draw_ids is None:
                reference_draw_ids = draw_ids
            elif draw_ids != reference_draw_ids:
                raise RuntimeError(
                    f"Ventanas distintas: {variant['name']} no evaluó los mismos sorteos"
                )
            results.append(summary)

    return {
        "generated_at": datetime.now().isoformat(),
        "experiment": experiment_name,
        "requested_backtest_size": int(backtest_size),
        "tickets_per_draw": int(tickets),
        "seed": int(seed),
        "history_end_contest": int(end_contest) if end_contest is not None else None,
        "fixed_origin": artifacts.to_dict(),
        "evaluated_draw_ids": reference_draw_ids or [],
        "variants": results,
        "paired_comparisons": _paired_comparisons(results, seed),
        "ledger_isolated": bool(isolate_ledger),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=108)
    parser.add_argument("--tickets", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--end-contest", type=int, default=None)
    parser.add_argument(
        "--suite",
        choices=tuple(VARIANT_SUITES),
        default="core",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="Nombre exacto de variante; puede repetirse para ejecutar un subconjunto",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    default_names = {
        "core": "melate_ab_experiment.json",
        "blend-sweep": "melate_blend_sweep.json",
        "selector-shadows": "melate_selector_shadows.json",
        "controlled-weights": "melate_controlled_weights.json",
        "deep-dispersion": "melate_deep_dispersion_20_10.json",
        "elite-coverage-deep": "melate_elite_coverage_deep.json",
        "five-hit-selector": "melate_five_hit_selector.json",
        "five-hit-scorer": "melate_five_hit_scorer.json",
        "five-hit-validation": "melate_five_hit_validation.json",
        "five-hit-reserve": "melate_five_hit_reserve.json",
        "five-hit-interaction": "melate_five_hit_interaction.json",
        "five-hit-holdout": "melate_five_hit_holdout.json",
        "universe-v17": "melate_universe_v17.json",
        "universe-dynamic": "melate_universe_dynamic.json",
        "universe-hybrid": "melate_universe_hybrid.json",
        "universe-nested-cap": "melate_universe_nested_cap.json",
        "selector-hybrid-bc": "melate_selector_hybrid_bc.json",
        "selector-hybrid-followup": "melate_selector_hybrid_followup.json",
        "deep-scorer": "melate_deep_scorer.json",
    }
    output = args.output or Path(DATA_FOLDER_PATH / default_names[args.suite])
    selected_variants = VARIANT_SUITES[args.suite]
    if args.variant:
        requested = set(args.variant)
        selected_variants = tuple(
            variant for variant in selected_variants if variant["name"] in requested
        )
        missing = requested.difference(
            variant["name"] for variant in selected_variants
        )
        if missing:
            parser.error(f"Variantes desconocidas para {args.suite}: {sorted(missing)}")
    payload = run_experiments(
        args.draws,
        args.tickets,
        args.seed,
        variants=selected_variants,
        experiment_name=f"melate_resonance_{args.suite}_v1",
        end_contest=args.end_contest,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_plain_number),
        encoding="utf-8",
    )
    print(f"Reporte A/B guardado en {output}")


if __name__ == "__main__":
    main()
