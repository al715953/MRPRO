import numpy as np
import itertools

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class CompetitiveMesh:
    """
    Motor V7.15: Quantum-Annealing-Mesh & Entropy-Closure.
    Utiliza análisis de entropía para identificar números de alta varianza
    y genera un cierre por "recocido" sobre los candidatos de élite.
    """

    def __init__(self, cloud_generator):
        self.cloud_gen = cloud_generator

    def apply_mesh(
        self, u_reduced, final_scores, n_target, m_xp, xp, thermal_numbers=None
    ):
        """
        Aplica el cierre por entropía y saturación de élite.
        """
        # 1. IDENTIFICACIÓN DE PILARES Y CÁLCULO DE ENTROPÍA
        sorted_idx = xp.argsort(final_scores)[::-1]
        p_alpha = u_reduced[sorted_idx[0]]
        alpha_score = final_scores[sorted_idx[0]]

        hybrid_list = []

        # --- FASE V7.15: ENTROPY-CLOSURE ---
        # Si el candidato es Alpha, analizamos los números con mayor varianza
        # (Entropía) para usarlos como cierres alternativos.
        alpha_base = p_alpha.get() if hasattr(p_alpha, "get") else p_alpha
        alpha_list = [int(x) for x in alpha_base]

        if alpha_score > 9.0:
            # Definimos "Números de Entropía" (Comodines de alta frecuencia reciente)
            # En una implementación real, esto vendría del decade_analyzer.
            # Aquí usamos un set de seguridad basado en el espejo de la V7.13
            entropy_boosters = [max(1, min(39, 40 - x)) for x in alpha_list[4:]]

            # Bloqueamos los primeros 4 o 5 números
            base_4 = alpha_list[:4]
            base_5 = alpha_list[:5]

            # Generamos el enjambre de cierre (Quantum-Annealing)
            for eb in entropy_boosters:
                # Variante 5/6 con booster de entropía
                h_ent = xp.asarray(sorted(list(set(base_5 + [eb]))))
                if len(h_ent) == 6:
                    hybrid_list.append(h_ent)

                # Variante de vecindad +/- 1 sobre el booster
                for offset in [-1, 1]:
                    h_ent_v = xp.asarray(
                        sorted(list(set(base_5 + [max(1, min(39, eb + offset))])))
                    )
                    if len(h_ent_v) == 6:
                        hybrid_list.append(h_ent_v)

        # --- FASE ALPHA-RESCUE (V7.11/14 Core) ---
        for i in range(6):
            variant = list(alpha_list)
            variant[i] = max(1, min(39, variant[i] + (1 if i % 2 == 0 else -1)))
            hybrid_list.append(xp.asarray(sorted(list(set(variant)))))

        # 2. CONSOLIDACIÓN Y BONO DE SUPERVIVENCIA
        hybrids = xp.vstack(hybrid_list)
        pool_tickets = xp.vstack([hybrids, u_reduced[sorted_idx[:1500]]])

        # Incrementamos el bono de supervivencia para forzar la dominancia de la entropía
        bonus_val = xp.max(final_scores) * 1.75
        pool_scores = xp.concatenate(
            [xp.full(len(hybrids), bonus_val), final_scores[sorted_idx[:1500]]]
        )

        # 3. SELECCIÓN COMPETITIVA (Quantum-Squeeze)
        one_hot = xp.zeros((len(pool_tickets), 41), dtype=xp.float32)
        rows = xp.arange(len(pool_tickets))[:, xp.newaxis]
        one_hot[rows, pool_tickets.astype(xp.int32)] = 1.0

        final_tickets = []
        current_scores = pool_scores.copy()
        immunity_threshold = 8.5

        while len(final_tickets) < n_target:
            best_idx = int(current_scores.argmax())
            if current_scores[best_idx] < -1e6:
                break

            winner = pool_tickets[best_idx]
            winner_score = pool_scores[best_idx]
            final_tickets.append(
                winner.get().tolist() if hasattr(winner, "get") else winner.tolist()
            )
            current_scores[best_idx] = -2e6

            # Redundancia Vectorizada
            matches = xp.dot(one_hot, one_hot[best_idx])

            # Protección de variantes (Máximo nivel histórico)
            protection_factor = 0.15 if winner_score > immunity_threshold else 1.0

            decay = xp.ones_like(matches)
            decay[matches >= 5] = 0.01
            decay[matches == 4] = 0.08 * protection_factor  # Ultra-agresivo
            decay[matches == 3] = 0.20 * protection_factor

            current_scores *= decay

        return final_tickets
