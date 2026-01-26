# src/strategies/genetic/diffusion.py

import numpy as np


class CloudGenerator:
    """
    Motor de Difusión.
    Responsable de la lógica de generación de vecinos (Nubes) y helpers de tickets.
    """

    def generate_omega_cloud(self, base_ticket_xp, m_xp, xp):
        """Genera variantes armónicas basadas en el eslabón más débil."""
        ticket_arr = xp.array(base_ticket_xp)
        # Cálculo de resonancia interna por número
        scores = xp.zeros(len(ticket_arr))
        for i in range(len(ticket_arr)):
            for j in range(len(ticket_arr)):
                if i != j:
                    scores[i] += m_xp[ticket_arr[i], ticket_arr[j]]

        # Identificar eslabón débil
        weak_idx = int(xp.argmin(scores))

        variants = []
        # Convertir a CPU para manipulación de listas de Python
        base_cpu = self.to_flat_list(ticket_arr)

        # Desplazamiento de fase +/- 1
        for shift in [-1, 1]:
            nv = base_cpu[weak_idx] + shift
            if 1 <= nv <= 39 and nv not in base_cpu:
                v = list(base_cpu)
                v[weak_idx] = nv
                variants.append(sorted(v))
        return variants

    def to_flat_list(self, arr):
        """Helper estático para asegurar listas planas de Python."""
        if hasattr(arr, "get"):
            arr = arr.get()
        return [int(x) for x in np.asarray(arr).ravel()]
