# src/strategies/genetic/mesh.py


class CompetitiveMesh:
    """
    Motor de Selección (Árbitro).
    Administra la competencia entre tickets originales y sus nubes de difusión.
    Aplica filtros de redundancia adaptativos (N-5/N-4).
    """

    def __init__(self, cloud_generator):
        self.cloud_gen = cloud_generator

    def apply_mesh(self, u_reduced, final_scores, n_target, m_xp, xp):
        """Ejecuta la lógica Omni-Cloud Diffusion V5.9.8."""
        sorted_idx = xp.argsort(final_scores)[::-1]
        elite_pool = []

        # 1. GENERACIÓN DE CANDIDATOS (Analizamos el Top 40 del radar)
        for i in range(min(40, len(sorted_idx))):
            idx_int = int(sorted_idx[i])
            score = float(final_scores[idx_int])
            # Usamos el helper del generador de nubes para obtener la lista plana
            ticket_xp = u_reduced[idx_int]
            ticket_cpu = self.cloud_gen.to_flat_list(ticket_xp)

            # El ticket original entra al pool
            elite_pool.append((score, ticket_cpu, "Laser"))

            # Difusión: Generamos la nube usando el motor de difusión
            # Heredan el 99.8% del score
            cloud = self.cloud_gen.generate_omega_cloud(ticket_xp, m_xp, xp)
            for neighbor in cloud:
                elite_pool.append((score * 0.998, neighbor, "Cloud"))

        # 2. RE-RANKING COMPETITIVO
        elite_pool.sort(key=lambda x: x[0], reverse=True)

        # 3. FILTRADO Y SELECCIÓN FINAL
        final_tickets = []
        seen_sets = []
        for _, ticket, source in elite_pool:
            if len(final_tickets) >= n_target:
                break
            t_set = set(ticket)

            # Filtro Adaptativo: N-5 para láseres, N-4 para nubes
            threshold = 5 if source == "Laser" else 4

            if not any(len(t_set & s) >= threshold for s in seen_sets):
                final_tickets.append(ticket)
                seen_sets.append(t_set)

        return final_tickets
