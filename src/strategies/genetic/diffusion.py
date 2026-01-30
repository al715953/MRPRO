# src/strategies/genetic/diffusion.py

import numpy as np


class CloudGenerator:
    """
    V6.4: Residue-Aware Harmonic Leap.
    Calcula el salto no solo por afinidad, sino por la 'vibración' de los fallos históricos.
    """

    def generate_harmonic_leap(self, base_ticket_xp, m_xp, xp, top_n=3):
        ticket_arr = xp.asarray(base_ticket_xp)

        # 1. Análisis de Resonancia Interna (Fuerza de los eslabones)
        # Identificamos el número con menor coherencia respecto al grupo
        scores = xp.zeros(len(ticket_arr))
        for i in range(len(ticket_arr)):
            for j in range(len(ticket_arr)):
                if i != j:
                    scores[i] += m_xp[ticket_arr[i], ticket_arr[j]]

        weak_idx = int(xp.argmin(scores))
        stable_core = [
            int(ticket_arr[i]) for i in range(len(ticket_arr)) if i != weak_idx
        ]

        # 2. CÁLCULO DE SALTO ARMÓNICO (Innovación V6.4)
        # En lugar de solo afinidad, aplicamos una 'Máscara de Interferencia'
        # que favorece números que históricamente completan los 5/6 en el rango 1-39.
        resonance_vals = xp.zeros(40)
        mask = xp.ones(40, dtype=bool)
        mask[0] = False
        for val in stable_core:
            resonance_vals += m_xp[val, :]
            mask[val] = False

        # Aplicamos un sesgo por 'Distancia de Residuo':
        # Los números altos (30-39) suelen requerir saltos de fase mayores (±2 o ±3)
        # mientras que los bajos son más estables (±1).
        pos_bias = xp.linspace(1.0, 1.15, 40)  # Mayor peso a la cola de la distribución
        resonance_vals = (resonance_vals * mask) * pos_bias

        best_candidates = xp.argsort(resonance_vals)[::-1][:top_n]

        variants = []
        for candidate in best_candidates:
            if int(candidate) > 0:
                # El salto armónico final
                v = sorted(stable_core + [int(candidate)])
                variants.append(v)

        return variants

    def to_flat_list(self, arr):
        if hasattr(arr, "get"):
            arr = arr.get()
        return [int(x) for x in np.asarray(arr).ravel()]
