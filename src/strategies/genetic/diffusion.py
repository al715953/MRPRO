# src/strategies/genetic/diffusion.py

import numpy as np


class CloudGenerator:
    """
    V6.1: Harmonic Leap Generator.
    Sustituye la difusión lineal por resonancia de afinidad geométrica.
    """

    def generate_harmonic_leap(self, base_ticket_xp, m_xp, xp, top_n=3):
        """Busca la 'pieza faltante' más armónica en el universo."""
        ticket_arr = xp.array(base_ticket_xp)

        # 1. Identificar eslabón débil por co-ocurrencia interna
        scores = xp.zeros(len(ticket_arr))
        for i in range(len(ticket_arr)):
            for j in range(len(ticket_arr)):
                if i != j:
                    scores[i] += m_xp[ticket_arr[i], ticket_arr[j]]

        weak_idx = int(xp.argmin(scores))
        base_cpu = self.to_flat_list(ticket_arr)

        # 2. Núcleo Estable (los otros 5 números)
        stable_core = [base_cpu[i] for i in range(len(base_cpu)) if i != weak_idx]

        # 3. Escaneo del Universo (1-39) por afinidad con el núcleo estable
        resonance_vals = xp.zeros(40)
        mask = xp.ones(40, dtype=bool)
        mask[0] = False
        for val in stable_core:
            resonance_vals += m_xp[val, :]
            mask[val] = False  # No repetir números del núcleo

        resonance_vals = resonance_vals * mask

        # 4. Seleccionar los top_n mejores saltos
        best_candidates = xp.argsort(resonance_vals)[::-1][:top_n]

        variants = []
        for candidate in best_candidates:
            if int(candidate) > 0:
                v = sorted(stable_core + [int(candidate)])
                variants.append(v)

        return variants

    def to_flat_list(self, arr):
        if hasattr(arr, "get"):
            arr = arr.get()
        return [int(x) for x in np.asarray(arr).ravel()]
