# src/strategies/genetic/resonance.py

import numpy as np
import itertools
from src.core.ai_scorer import LotteryAIModel

class ResonanceEngine:
    def __init__(self):
        """
        Inicializa el motor de resonancia con el modelo de IA base.
        """
        self.ai_model = LotteryAIModel()
        self._matrix_cache = {"cluster_matrix": None}

    def calculate_resonance(self, u_xp, history, config, xp):
        """
        Calcula la resonancia híbrida (IA + Geometría).
        history: Puede ser un DrawHistoryDTO o una lista de sorteos (Backtest).
        config: Objeto de configuración que contiene total_balls.
        """
        if self._matrix_cache["cluster_matrix"] is None:
            raw_history = history.winning_numbers if hasattr(history, 'winning_numbers') else history
            self._compute_base_scores(u_xp, raw_history, config, xp)

        # 1. NORMALIZACIÓN DE ENTRADA (Sniper Fix para Backtester)
        # Extraemos la lista de números ganadores sin importar el tipo de objeto
        raw_history = history.winning_numbers if hasattr(history, 'winning_numbers') else history
        n_balls = config.total_balls

        # 2. ASEGURAR ENTRENAMIENTO
        if not self.ai_model.is_trained:
            self.ai_model.train(raw_history, n_balls)

        if u_xp is None or len(u_xp) == 0:
            return None

        # 3. CÁLCULO DE NÚMEROS TERMALES (Basado en n_balls)
        recent_draws = raw_history[-5:]
        flat_recent = set([num for draw in recent_draws for num in draw[:6]])
        thermal_numbers = sorted(
            list(set(range(1, n_balls + 1)) - flat_recent)
        )

        # 4. OBTENCIÓN DE SCORES BASE
        # Pasamos raw_history ya normalizado como lista
        ai_scores, geo_scores_raw = self._compute_base_scores(u_xp, raw_history, config, xp)

        if len(ai_scores) == 0:
            return None

        # 5. NORMALIZACIÓN SIGMOIDAL (Fusión Magneto)
        ai_norm = (ai_scores - ai_scores.min()) / (
            ai_scores.max() - ai_scores.min() + 1e-10
        )
        geo_norm = (geo_scores_raw - geo_scores_raw.min()) / (
            geo_scores_raw.max() - geo_scores_raw.min() + 1e-10
        )

        # Filtro de radar: Solo procesamos combinaciones con potencial real (>0.4)
        radar_indices = xp.where(ai_norm > 0.4)[0]
        if len(radar_indices) == 0:
            radar_indices = xp.argsort(ai_norm)[-5000:]

        u_reduced = u_xp[radar_indices]
        ai_subset = ai_norm[radar_indices]

        # Aplicación de Señal de Disrupción (Alpha)
        # Buscamos el 'Sweet Spot' cerca del percentil de éxito histórico
        sweet_spot = 1.0 - xp.abs(ai_subset - 0.85)
        
        x0, k = 0.5, 10.0
        disruption_signal = ai_subset * sweet_spot
        sigmoidal_boost = 1.0 / (1.0 + xp.exp(-k * (disruption_signal - x0)))

        # Score final: IA domina, Geometría ajusta el Rank
        final_scores_reduced = sigmoidal_boost * 10.0 + (geo_norm[radar_indices] * 0.05)

        return {
            "u_reduced": u_reduced,
            "final_scores_reduced": final_scores_reduced,
            "radar_indices": radar_indices,
            "ai_norm": ai_norm,
            "geo_scores": geo_norm,
            "geo_matrix_xp": xp.asarray(self._matrix_cache["cluster_matrix"]), # KEY REQUERIDA POR V7.17
            "thermal_numbers": thermal_numbers,
        }

    def _compute_base_scores(self, u_xp, raw_history, config, xp):
        """
        Motor Interno: Calcula la matriz de clústeres y puntúa el universo.
        """
        ai_scores = xp.asarray(self.ai_model.score_tickets(u_xp))
        n_balls = config.total_balls

        # Generación de Matriz Nexus (Geometría de pares)
        if self._matrix_cache["cluster_matrix"] is None:
            # Dimensionamos n_balls + 1 para manejar índices 1-39 sin desbordamiento
            m = np.zeros((n_balls + 1, n_balls + 1), dtype=np.uint16)
            for draw in raw_history:
                # Solo tomamos los primeros 6 (ganadores reales) para la matriz
                for a, b in itertools.combinations(sorted(draw[:6]), 2):
                    if a <= n_balls and b <= n_balls:
                        m[a, b] += 1
                        m[b, a] += 1
            self._matrix_cache["cluster_matrix"] = m
            
        m_xp = xp.asarray(self._matrix_cache["cluster_matrix"])
        
        # Cálculo de Proximidad Geométrica Vectorizada
        # (Aquí podrías añadir tu lógica específica de geo_scores)
        geo_scores = xp.zeros(len(ai_scores)) 
        
        return ai_scores, geo_scores