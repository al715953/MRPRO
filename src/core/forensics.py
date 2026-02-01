# src/core/forensics.py
import numpy as np
from typing import Dict, Any


class LotteryForensics:
    @staticmethod
    def audit_winner(
        snapshot: Dict[str, Any], target_numbers: list, xp_audit
    ) -> Dict[str, Any]:
        if not snapshot or "universe" not in snapshot:
            return {"hits": 0, "rank": 0, "proximity": 999, "univ_size": 0}

        univ = snapshot["universe"]

        # --- FIX: DETECCIÓN DE BACKEND ---
        # Si el universo es NumPy (CPU), usamos NumPy. Si no, usamos el xp de la GPU.
        is_numpy = isinstance(univ, np.ndarray)
        xp = np if is_numpy else xp_audit

        target = xp.asarray(sorted(target_numbers[:6]), dtype=xp.uint8)

        # 1. Auditoría de aciertos
        try:
            hits_vec = xp.sum(xp.isin(univ, target), axis=1)
            max_h = int(xp.max(hits_vec))
        except Exception:
            # Fallback a CPU si hay colisión de memoria
            univ_cpu = univ.get() if hasattr(univ, "get") else univ
            target_cpu = np.array(sorted(target_numbers[:6]), dtype=np.uint8)
            hits_vec = np.sum(np.isin(univ_cpu, target_cpu), axis=1)
            max_h = int(np.max(hits_vec))
            xp = np

        if max_h == 0:
            return {"hits": 0, "rank": 0, "proximity": 999, "univ_size": len(univ)}

        # 2. Identificación del mejor candidato
        best_indices = xp.where(hits_vec == max_h)[0]
        scores_xp = xp.asarray(snapshot["hybrid_scores"])
        idx_best = int(best_indices[xp.argmax(scores_xp[best_indices])])

        # 3. Métricas finales (Siempre en CPU para el DTO)
        scores_cpu = snapshot["hybrid_scores"]
        rank = int(np.sum(scores_cpu > scores_cpu[idx_best]) + 1)

        # --- FIX DISTANCIA: Proximity Calculation ---
        selected_ranks = snapshot.get("selected_ranks", [])
        # Calculamos qué tan lejos está nuestro mejor Rank de los 20 seleccionados
        proximity = (
            int(min([abs(rank - r) for r in selected_ranks])) if selected_ranks else 999
        )

        return {
            "hits": max_h,
            "rank": rank,
            "proximity": proximity,
            "univ_size": len(univ),
            "hybrid_score": float(scores_cpu[idx_best]),
            "ai_score": (
                float(snapshot["ai_scores"][idx_best])
                if "ai_scores" in snapshot
                else 0.0
            ),
            "geo_score": (
                float(snapshot["geo_scores"][idx_best])
                if "geo_scores" in snapshot
                else 0.0
            ),
        }
