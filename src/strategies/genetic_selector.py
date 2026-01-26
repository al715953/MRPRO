import numpy as np
import itertools
from src.core.ai_scorer import LotteryAIModel
from src.domain.dtos import PredictionResultDTO

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class GeneticSelectorStrategy:
    """
    SELECTOR V34.2: Mutant Hunter Edition.
    Lógica de selección pura, desacoplada de auditoría.
    """

    def __init__(self):
        self.ai_model = LotteryAIModel()
        self._matrix_cache = {"cluster_matrix": None}

    def predict(self, history, config) -> PredictionResultDTO:
        univ = config.raw_universe_ptr
        if univ is None or len(univ) == 0:
            return PredictionResultDTO("Empty", [])

        if not self.ai_model.is_trained:
            self.ai_model.train(history.winning_numbers, config.total_balls)

        xp = cp if (HAS_CUPY and hasattr(univ, "get")) else np
        u_xp = xp.asarray(univ)

        # --- FASE DE SCORING ---
        ai_scores, geo_scores = self._compute_hybrid_scores(u_xp, history, config, xp)

        # Normalización
        ai_norm = (ai_scores - ai_scores.min()) / (
            ai_scores.max() - ai_scores.min() + 1e-10
        )
        geo_norm = (geo_scores - geo_scores.min()) / (
            geo_scores.max() - geo_scores.min() + 1e-10
        )

        # Fusión Mutant Hunter (0.5 Sigmoide + 0.5 Lineal)
        ai_w = 0.5 * (1 / (1 + xp.exp(-4 * (ai_norm - 0.5)))) + 0.5 * ai_norm
        hybrid_scores = (ai_w * ai_norm) + ((1.0 - ai_w) * geo_norm)

        # --- FASE DE MALLA (Zonificación Mutant) ---
        selected_indices = self._apply_mutant_mesh(u_xp, hybrid_scores, ai_norm, xp)

        tickets = [
            u_xp[i].get().tolist() if hasattr(u_xp[i], "get") else u_xp[i].tolist()
            for i in selected_indices
        ]

        # Devolvemos los tickets y el snapshot para que Forensics lo procese fuera
        snapshot = {
            "universe": univ,
            "ai_scores": ai_scores,
            "geo_scores": geo_scores,
            "hybrid_scores": hybrid_scores,
            "selected_ranks": list(range(1, len(selected_indices) + 1)),
        }

        return PredictionResultDTO("Mutant Hunter V34.2", tickets), snapshot

    def _compute_hybrid_scores(self, u_xp, history, config, xp):
        # Lógica de matriz de clústeres y scores (Se mantiene igual, pero aislada)
        ai_scores = xp.asarray(self.ai_model.score_tickets(u_xp))

        if self._matrix_cache["cluster_matrix"] is None:
            m = np.zeros(
                (config.total_balls + 1, config.total_balls + 1), dtype=np.uint16
            )
            for d in history.winning_numbers:
                for a, b in itertools.combinations(sorted(d[:6]), 2):
                    m[a, b] += 1
                    m[b, a] += 1
            self._matrix_cache["cluster_matrix"] = m

        m_xp = xp.asarray(self._matrix_cache["cluster_matrix"])
        geo_scores = xp.zeros(len(u_xp))
        for i, j in itertools.combinations(range(6), 2):
            geo_scores += m_xp[u_xp[:, i], u_xp[:, j]]

        return ai_scores, geo_scores

    def _apply_mutant_mesh(self, u_xp, hybrid_scores, ai_norm, xp):
        """
        MALLA DE DIVERSIDAD V35.0: Elite Handler Edition.
        Protege a los candidatos de alta resonancia (AI > 0.85) y aplica filtro N-4.
        """
        # 1. Ordenar por Score Híbrido (Mezcla IA + Histórica)
        sorted_indices = xp.argsort(hybrid_scores)[::-1]
        selected_indices = []

        # 2. Zonas de Quota Dinámicas (V35.0): Más profundidad en el Top 2000
        # (inicio, fin, cantidad_a_seleccionar)
        zones = [(0, 50, 10), (51, 500, 15), (501, 2000, 15)]

        for start, end, quota in zones:
            zone_count = 0
            for idx in sorted_indices[start:end]:
                idx_int = int(idx)

                # Conversión eficiente GPU -> CPU para comparación de sets
                ticket_val = (
                    u_xp[idx_int].get()
                    if hasattr(u_xp[idx_int], "get")
                    else u_xp[idx_int]
                )
                ticket_set = set(ticket_val.tolist())

                # --- 🟢 FASE A: RESCATE DE ÉLITE (Quantum Jump) ---
                # Si el score de IA es > 0.85, el ticket es una anomalía de alta confianza.
                # Lo inyectamos directamente ignorando la redundancia moderada.
                if float(ai_norm[idx_int]) > 0.85:
                    if idx_int not in selected_indices:
                        selected_indices.append(idx_int)
                        zone_count += 1
                    continue

                # --- 🔴 FASE B: FILTRO DE REDUNDANCIA N-4 ---
                is_redundant = False
                for s_idx in selected_indices:
                    sel_val = (
                        u_xp[s_idx].get()
                        if hasattr(u_xp[s_idx], "get")
                        else u_xp[s_idx]
                    )
                    sel_ticket = set(sel_val.tolist())

                    overlap = len(ticket_set & sel_ticket)

                    # Criterio N-4:
                    # - Si comparten 5 o 6 números: Redundancia Absoluta.
                    # - Si comparten 4 números: Solo se acepta si el AI Score es muy alto (>0.92).
                    if overlap >= 5 or (
                        overlap == 4 and float(ai_norm[idx_int]) < 0.92
                    ):
                        is_redundant = True
                        break

                if not is_redundant:
                    selected_indices.append(idx_int)
                    zone_count += 1

                # Salir de la zona si ya cumplimos la cuota
                if zone_count >= quota:
                    break

        return selected_indices
