# src/strategies/genetic/mesh.py


class CompetitiveMesh:
    """
    Motor de Selección (Árbitro) V6.2: Harmonic Contract.
    Administra la competencia entre tickets láser y nubes armónicas.
    Optimizado para universos reducidos por contracción de suma (120-128).
    """

    def __init__(self, cloud_generator):
        self.cloud_gen = cloud_generator

    def apply_mesh(self, u_reduced, final_scores, n_target, m_xp, xp):
        """
        Ejecuta la selección competitiva aumentando la profundidad a 80 candidatos.
        """
        sorted_idx = xp.argsort(final_scores)[::-1]
        elite_pool = []

        # 1. GENERACIÓN DE CANDIDATOS (Profundidad extendida a 80)
        # Al tener un universo de ~14k, analizar 80 es proporcionalmente más potente
        for i in range(min(80, len(sorted_idx))):
            idx_int = int(sorted_idx[i])
            score = float(final_scores[idx_int])

            # Usamos el helper para obtener la lista plana de Python
            ticket_xp = u_reduced[idx_int]
            ticket_cpu = self.cloud_gen.to_flat_list(ticket_xp)

            # Entrada del Ticket Original (Laser)
            elite_pool.append((score, ticket_cpu, "Laser"))

            # Entrada de la Nube Armónica (Salto de fase)
            # El Salto Armónico busca la pieza que falta según la matriz de co-ocurrencia
            cloud = self.cloud_gen.generate_harmonic_leap(ticket_xp, m_xp, xp)
            for j, neighbor in enumerate(cloud):
                # Aplicamos un decaimiento más suave (0.001) para que los saltos
                # armónicos compitan codo a codo con los originales
                decay = 0.999 - (j * 0.001)
                elite_pool.append((score * decay, neighbor, "Harmonic"))

        # 2. RE-RANKING COMPETITIVO
        # Ordenamos todo el pool (Láseres + Armónicos) por su nuevo score de resonancia
        elite_pool.sort(key=lambda x: x[0], reverse=True)

        # 3. FILTRADO Y SELECCIÓN FINAL (Poda Quirúrgica)
        final_tickets = []
        seen_sets = []
        for _, ticket, source in elite_pool:
            if len(final_tickets) >= n_target:
                break
            t_set = set(ticket)

            # Filtro de Redundancia Adaptativo:
            # - Laser (N-5): No permitimos más de 4 números iguales entre originales.
            # - Harmonic (N-4): Somos más estrictos con los saltos para forzar diversidad.
            threshold = 5 if source == "Laser" else 4

            if not any(len(t_set & s) >= threshold for s in seen_sets):
                final_tickets.append(ticket)
                seen_sets.append(t_set)

        return final_tickets
