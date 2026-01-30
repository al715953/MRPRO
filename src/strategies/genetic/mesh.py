# src/strategies/genetic/mesh.py

import numpy as np


class CompetitiveMesh:
    """
    Motor V7.4: Slot-Mapping & Soft-Squeeze.
    Optimizado para velocidad (RTX 4070 Ti) y consistencia estructural.
    """

    def __init__(self, cloud_generator):
        self.cloud_gen = cloud_generator

    def apply_mesh(self, u_reduced, final_scores, n_target, m_xp, xp):
        # 1. PRE-PROCESAMIENTO VECTORIZADO (Recuperación de Velocidad)
        # Tomamos los 1000 mejores para tener un pool ancho tras el Slot-Mapping
        sorted_idx = xp.argsort(final_scores)[::-1][:1000]
        pool_tickets_xp = u_reduced[sorted_idx]

        # SUTURA TÉCNICA: Movemos a NumPy una sola vez para evitar errores de 'dtype'
        pool_scores = (
            final_scores[sorted_idx].get()
            if hasattr(final_scores, "get")
            else final_scores[sorted_idx]
        )

        # Convertimos todos los tickets a CPU de golpe (Adiós a los 700 micro-saltos)
        pool_tickets = [self.cloud_gen.to_flat_list(t) for t in pool_tickets_xp]

        final_tickets = []
        selected_sets = []
        current_scores = pool_scores.copy()

        # 2. SELECCIÓN POR PENALIZACIÓN DINÁMICA (Soft-Squeeze)
        while len(final_tickets) < n_target:
            # CORRECCIÓN DE CRASH: Usamos el método nativo de NumPy
            best_idx = current_scores.argmax()

            if current_scores[best_idx] < -1e6:
                break

            winner_ticket = pool_tickets[best_idx]
            winner_set = set(winner_ticket)

            final_tickets.append(winner_ticket)
            selected_sets.append(winner_set)

            # Marcamos como usado con un valor centinela
            current_scores[best_idx] = -2e6

            # --- SOFT-SQUEEZE: Redundancia Elástica ---
            for i in range(len(current_scores)):
                if current_scores[i] < 0:
                    continue

                # Intersección rápida de sets
                matches = len(winner_set & set(pool_tickets[i]))

                # Multiplicadores de decaimiento (Evita el descarte binario)
                if matches >= 5:
                    decay = 0.05  # Castigo extremo (Solo 5% de score)
                elif matches == 4:
                    decay = 0.40  # Castigo fuerte
                elif matches == 3:
                    decay = 0.80  # Castigo suave
                else:
                    decay = 1.0

                current_scores[i] *= decay

        return final_tickets
