import numpy as np
import os
import itertools
from typing import List, Tuple, Dict, Any
from rich.console import Console

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO
from src.core.ai_scorer import LotteryAIModel


class GeneticSelectorStrategy(ILotteryStrategy):
    """
    SELECTOR V10.5: Quantum Alpha Core & Dynamic Mesh.
    - Alpha-Core: Reserva determinista de los Ranks #1-#3 para evitar saltos de malla.
    - GPA Dinámico: Ajusta la repulsión según la confianza de la IA (Top 10 Mean).
    - Zero-Leak VRAM: Gestión explícita de memoria para estabilidad en Windows.
    """

    def __init__(self):
        self.ai_model = LotteryAIModel()
        self.console = Console()
        self._last_trained_date = None
        self._matrix_cache = {"cluster_matrix": None}
        self._forensic_snapshot = {}

    def _update_heuristics(self, history: DrawHistoryDTO, total_balls: int):
        """Actualiza la matriz de co-ocurrencia para el cálculo de resonancia."""
        matrix = np.zeros((total_balls + 2, total_balls + 2), dtype=np.uint16)
        for draw in history.winning_numbers:
            for a, b in itertools.combinations(sorted(draw[:6]), 2):
                matrix[a, b] += 1
                matrix[b, a] += 1
        self._matrix_cache["cluster_matrix"] = matrix

    def _calculate_geo_scores(self, candidates_np: np.ndarray) -> np.ndarray:
        """Calcula el Geo Score basado en la fuerza de los pares históricos."""
        matrix = self._matrix_cache["cluster_matrix"]
        if matrix is None:
            return np.zeros(len(candidates_np), dtype=np.float32)

        scores = np.zeros(len(candidates_np), dtype=np.float32)
        # Vectorización del cálculo de pares para el universo actual
        for i in range(6):
            for j in range(i + 1, 6):
                scores += matrix[candidates_np[:, i], candidates_np[:, j]]

        max_s = np.max(scores)
        return scores / max_s if max_s > 0 else scores

    def _quantum_attraction_mesh(self, candidates_np, ai_scores, n_tickets=20):
        """
        Motor de Interferencia Gravitacional con Ajuste Dinámico de Repulsión.
        """
        xp = cp if HAS_CUPY else np
        X = xp.asarray(candidates_np, dtype=xp.float32)
        S = xp.asarray(ai_scores, dtype=xp.float32)

        # 1. ALPHA-CORE: Aseguramos la captura del Top 3 absoluto de la IA
        sorted_indices = xp.argsort(S)[::-1]
        alpha_core_idx = sorted_indices[:5].tolist()

        # 2. CALIBRACIÓN DE REPULSIÓN DINÁMICA
        # Evaluamos la densidad de probabilidad en el pico
        top_10_mean = xp.mean(S[sorted_indices[:10]])
        # Si la confianza es > 0.85, reducimos la repulsión para concentrar fuego
        dynamic_repulsion = (
            8.0 if top_10_mean < 0.70 else 5.0 if top_10_mean < 0.85 else 4.5
        )

        # 3. MALLA DE INTERFERENCIA (GPA)
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-6)

        elite_mask = sorted_indices[:500]
        Elite_X = X_norm[elite_mask]
        Elite_S = S[elite_mask]

        potential = xp.zeros(len(X_norm), dtype=xp.float32)
        chunk_size = 100
        for i in range(0, 500, chunk_size):
            e_x = Elite_X[i : i + chunk_size]
            e_s = Elite_S[i : i + chunk_size]
            # Distancia Euclidiana 6D vectorizada
            dists_sq = xp.sum((X_norm[:, xp.newaxis, :] - e_x) ** 2, axis=2)
            potential += xp.sum(e_s / (dists_sq + 0.05), axis=1)

        # 4. SELECCIÓN POR DESPLAZAMIENTO (Resto de cuota)
        mesh_indices = []
        current_potential = potential.copy()

        # Aplicamos repulsión inicial desde los puntos del Alpha-Core
        for idx in alpha_core_idx:
            dists_to_sel = xp.sum((X_norm - X_norm[idx]) ** 2, axis=1)
            current_potential *= 1.0 - xp.exp(-dists_to_sel * dynamic_repulsion)

        for _ in range(n_tickets - len(alpha_core_idx)):
            best_idx = int(xp.argmax(current_potential))
            mesh_indices.append(best_idx)
            # Actualizamos zona de repulsión gaussiana
            dists_to_new = xp.sum((X_norm - X_norm[best_idx]) ** 2, axis=1)
            current_potential *= 1.0 - xp.exp(-dists_to_new * dynamic_repulsion)

        return [int(i) for i in alpha_core_idx] + [int(i) for i in mesh_indices]

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        candidates_np = config.raw_universe_ptr
        if candidates_np is None or len(candidates_np) == 0:
            return PredictionResultDTO("Empty Universe", [])

        # Sincronización de entrenamiento y heurística
        if self._last_trained_date != history.dates[-1]:
            # El entrenamiento ya incluye el bucle forense inyectado por el Backtester
            self.ai_model.train(history.winning_numbers, config.total_balls)
            self._update_heuristics(history, config.total_balls)
            self._last_trained_date = history.dates[-1]

        # Scoring Híbrido
        raw_ai_scores = self.ai_model.score_tickets([tuple(x) for x in candidates_np])
        geo_scores = self._calculate_geo_scores(candidates_np)

        # Selección V10.5
        final_indices = self._quantum_attraction_mesh(
            candidates_np, raw_ai_scores, config.num_tickets
        )

        # Mapeo de Ranks para auditoría forense
        sorted_ai_idx = np.argsort(raw_ai_scores)[::-1]
        selected_ranks = [
            int(np.where(sorted_ai_idx == idx)[0][0] + 1) for idx in final_indices
        ]

        # Snapshot para Telemetría Sniper V6.3.3
        self._forensic_snapshot = {
            "universe": candidates_np,
            "ai_scores": raw_ai_scores,
            "geo_scores": geo_scores,
            "selected_ranks": sorted(selected_ranks),
            "univ_size": len(candidates_np),
        }

        if HAS_CUPY:
            cp.get_default_memory_pool().free_all_blocks()

        return PredictionResultDTO(
            "V10.5 Alpha Core", [list(candidates_np[idx]) for idx in final_indices]
        )

    def audit_winner(self, history, config, winning_ticket) -> dict:
        """Auditoría de alta resolución para detectar la Distancia Crítica."""
        snap = self._forensic_snapshot
        if "universe" not in snap:
            return {"found": False, "hits": 0}

        target = np.array(sorted(winning_ticket[:6]))
        hits_array = np.sum(np.isin(snap["universe"], target), axis=1)
        max_hits = int(np.max(hits_array)) if len(hits_array) > 0 else 0

        if max_hits == 0:
            return {"found": False, "hits": 0}

        best_indices = np.where(hits_array == max_hits)[0]
        # Seleccionamos el representante con mejor IA score dentro de la zona de éxito
        idx_audit = best_indices[np.argsort(snap["ai_scores"][best_indices])[-1]]
        w_rank = np.sum(snap["ai_scores"] > snap["ai_scores"][idx_audit]) + 1

        return {
            "found": max_hits >= 4,
            "hits": max_hits,
            "rank": int(w_rank),
            "proximity": int(min([abs(w_rank - r) for r in snap["selected_ranks"]])),
            "ai_score": float(snap["ai_scores"][idx_audit]),
            "geo_score": float(snap["geo_scores"][idx_audit]),
            "univ_size": snap["univ_size"],
            "selected_ranks": snap["selected_ranks"],
        }
