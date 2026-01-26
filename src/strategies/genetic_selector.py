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
    SELECTOR V5.5: Geo-Deterministic Tuning (Weak-Link Shift).
    Blindaje de memoria GPU-CPU para evitar errores de conversión implícita.
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

        ai_scores, geo_scores = self._compute_hybrid_scores(u_xp, history, config, xp)

        # Normalización de Resonancia
        ai_min, ai_max = ai_scores.min(), ai_scores.max()
        ai_norm = (ai_scores - ai_min) / (ai_max - ai_min + 1e-10)
        geo_norm = (geo_scores - geo_scores.min()) / (
            geo_scores.max() - geo_scores.min() + 1e-10
        )

        # Fusión de Resonancia V5.5
        ai_boost = xp.where(ai_norm > 0.90, 0.95, 0.65)
        hybrid_scores = (ai_norm * ai_boost) + (geo_norm * (1.0 - ai_boost))

        final_tickets = self._apply_neighborhood_mesh(
            u_xp, hybrid_scores, xp, config.num_tickets
        )

        snapshot = {
            "universe": univ,
            "ai_scores": ai_scores,
            "geo_scores": geo_scores,
            "hybrid_scores": hybrid_scores,
            "selected_ranks": list(range(1, len(final_tickets) + 1)),
        }

        return (
            PredictionResultDTO("ENGINE V5.5: Geo-Deterministic Tuning", final_tickets),
            snapshot,
        )

    def _compute_hybrid_scores(self, u_xp, history, config, xp):
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

    def _to_flat_list(self, arr):
        if hasattr(arr, "get"):
            arr = arr.get()
        return [int(x) for x in np.asarray(arr).ravel()]

    def _generate_harmonic_neighbors(self, base_ticket, m_xp, xp):
        """
        V5.8: Omega Phase Shift.
        Genera variantes de fase simple y fase doble (2 números desplazados).
        """
        ticket_arr = xp.array(base_ticket)
        scores = xp.zeros(len(ticket_arr))
        for i in range(len(ticket_arr)):
            for j in range(len(ticket_arr)):
                if i != j:
                    scores[i] += m_xp[ticket_arr[i], ticket_arr[j]]

        # Identificamos los DOS eslabones más débiles (menor co-ocurrencia)
        weak_indices = xp.argsort(scores)[:2]
        w1, w2 = int(weak_indices[0]), int(weak_indices[1])

        variants = []
        base_tkt_cpu = self._to_flat_list(ticket_arr)

        # Variante A: Fase Simple (+/- 1 en el más débil)
        for shift in [-1, 1]:
            new_val = base_tkt_cpu[w1] + shift
            if 1 <= new_val <= 39 and new_val not in base_tkt_cpu:
                v = list(base_tkt_cpu)
                v[w1] = new_val
                variants.append(sorted(v))

        # Variante B: Fase Doble (Desplazamiento armónico en los dos más débiles)
        # Solo si la resonancia base es masiva (> 0.98)
        new_v1 = base_tkt_cpu[w1] + 1
        new_v2 = base_tkt_cpu[w2] - 1
        if (
            1 <= new_v1 <= 39
            and 1 <= new_v2 <= 39
            and new_v1 not in base_tkt_cpu
            and new_v2 not in base_tkt_cpu
        ):
            v_double = list(base_tkt_cpu)
            v_double[w1], v_double[w2] = new_v1, new_v2
            variants.append(sorted(v_double))

        return variants

    def _apply_neighborhood_mesh(self, u_xp, hybrid_scores, xp, n_target):
        """
        Malla V5.8: Quantum Dithering + Omega Cloud.
        El Top 1 genera una nube de 3 variantes; el resto compite.
        """
        sorted_indices = xp.argsort(hybrid_scores)[::-1]
        m_xp = xp.asarray(self._matrix_cache["cluster_matrix"])
        candidates_pool = []
        seen_sets = []

        # 1. POOL DE CANDIDATOS (Extendemos el radar a los mejores 30)
        for i in range(min(30, len(sorted_indices))):
            idx_int = int(sorted_indices[i])
            score = float(hybrid_scores[idx_int])
            ticket = self._to_flat_list(u_xp[idx_int])
            candidates_pool.append((score, ticket, "Laser"))

            # 2. GENERACIÓN OMEGA (Solo para el Top 1 Absoluto)
            # Creamos una 'nube' de vecinos con alta prioridad
            if i == 0:
                neighbors = self._generate_harmonic_neighbors(ticket, m_xp, xp)
                for n in neighbors:
                    # Los vecinos del Top 1 entran con un score casi idéntico (99%)
                    candidates_pool.append((score * 0.99, n, "Omega_Neighbor"))

        # 3. SELECCIÓN COMPETITIVA FINAL
        candidates_pool.sort(key=lambda x: x[0], reverse=True)

        final_tickets = []
        for score, ticket, source in candidates_pool:
            if len(final_tickets) >= n_target:
                break
            t_set = set(ticket)
            if not any(len(t_set & s) >= 5 for s in seen_sets):
                final_tickets.append(ticket)
                seen_sets.append(t_set)

        return final_tickets
