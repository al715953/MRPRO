import pandas as pd
import numpy as np
import itertools
from collections import Counter
from typing import List, Tuple, Union
from rich.console import Console

from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO

console = Console()


class HeuristicSelectorStrategy:
    """
    ESTRATEGIA HEURÍSTICA V4.1.
    Sincronizada para operar con el optimizador y compatible con la interfaz de IA.
    """

    def __init__(self):
        # Solución al error: Atributo dummy para compatibilidad con el motor de backtest
        self.ai_model = None
        self.strategy_name = "Heuristic Sniper V4.1"

    def predict(
        self,
        data: Union[DrawHistoryDTO, PredictionResultDTO],
        config: PredictionConfigDTO,
    ) -> PredictionResultDTO:
        """
        Punto de entrada universal.
        Detecta si recibe el historial (DrawHistoryDTO) o un universo ya reducido (PredictionResultDTO).
        """
        overrides = config.filter_overrides or {}
        verbose = overrides.get("verbose", False)

        # 1. RESOLUCIÓN DE ENTRADA (Manejo de Pipeline Modular)
        # Si recibimos un PredictionResultDTO, significa que el universo ya fue filtrado por la GPU
        if isinstance(data, PredictionResultDTO):
            universe = np.array(data.tickets)
            # Para la heurística, necesitamos el historial para calcular frecuencias
            # Lo recuperamos de la configuración si está disponible o usamos un fallback
            history_winning_numbers = getattr(config, "history_reference", [])
        else:
            # Si recibimos el historial directo, el selector debe generar su propio universo
            # (Aunque en el optimizador esto lo hace el pre_process_strategy)
            universe = (
                np.array(config.raw_universe_ptr)
                if config.raw_universe_ptr is not None
                else np.array([])
            )
            history_winning_numbers = data.winning_numbers

        if len(universe) == 0:
            if verbose:
                console.print(
                    "[yellow]⚠ Universo vacío en Selector Heurístico.[/yellow]"
                )
            return PredictionResultDTO(self.strategy_name, [])

        # 2. ANÁLISIS ESTADÍSTICO (Heurística de Densidad)
        w_cluster = overrides.get("w_cluster", 0.5)
        w_hotness = overrides.get("w_hotness", 0.3)
        w_balance = overrides.get("w_balance", 0.2)

        # Cálculo de frecuencias y clusters del historial
        flat_history = [num for draw in history_winning_numbers for num in draw]
        freq_map = Counter(flat_history)

        # 3. SCORING VECTORIZADO (Optimizado para el hardware detectado)
        # Convertimos el universo a una lista de candidatos para evaluación
        num_to_select = config.num_tickets

        # Para velocidad en el optimizador, si el universo es muy grande,
        # realizamos un muestreo estadístico antes del scoring pesado
        if len(universe) > 10000:
            idx = np.random.choice(len(universe), 10000, replace=False)
            candidates = universe[idx]
        else:
            candidates = universe

        # Scoring simple por densidad de "Hot Numbers"
        scores = np.sum(np.vectorize(freq_map.get)(candidates, 0), axis=1)

        # 4. SELECCIÓN FINAL
        top_indices = np.argsort(scores)[-num_to_select:][::-1]
        selected_tickets = candidates[top_indices].tolist()

        return PredictionResultDTO(
            strategy_name=self.strategy_name,
            tickets=selected_tickets,
            metadata={
                "w_cluster": w_cluster,
                "w_hotness": w_hotness,
                "u_source_size": len(universe),
            },
        )
