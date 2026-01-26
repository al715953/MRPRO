import numpy as np
from typing import Dict, Any


class LotteryForensics:
    """Analizador forense desacoplado para auditoría de Jackpots."""

    @staticmethod
    def audit_winner(
        snapshot: Dict[str, Any], target_numbers: list, xp
    ) -> Dict[str, Any]:
        """
        Calcula el rendimiento real comparando el universo con el ticket ganador.
        Procesa el snapshot (Dict) para extraer el ADN de la Trifecta.
        """
        # Validación de integridad del snapshot
        if not snapshot or "universe" not in snapshot:
            return {"hits": 0, "rank": 0, "proximity": 999, "univ_size": 0}

        univ = snapshot["universe"]
        target = xp.asarray(sorted(target_numbers[:6]), dtype=xp.uint8)

        # 1. Detección de aciertos vectorizada en el universo
        hits_vec = xp.sum(xp.isin(univ, target), axis=1)
        max_h = int(xp.max(hits_vec))

        if max_h == 0:
            return {"hits": 0, "rank": 0, "proximity": 999, "univ_size": len(univ)}

        # 2. Localización del mejor candidato por Score Híbrido
        best_indices = xp.where(hits_vec == max_h)[0]
        scores = snapshot["hybrid_scores"]

        # Sincronización segura CPU/GPU para cálculo de Rank y búsqueda
        scores_cpu = scores.get() if hasattr(scores, "get") else scores
        best_idx_cpu = (
            best_indices.get() if hasattr(best_indices, "get") else best_indices
        )

        # Identificamos el índice del ticket con el mejor score entre los aciertos máximos
        idx_best = int(best_idx_cpu[np.argsort(scores_cpu[best_idx_cpu])[-1]])

        # 3. Cálculo de métricas finales y ADN de la Trifecta
        rank = int(np.sum(scores_cpu > scores_cpu[idx_best]) + 1)
        selected_ranks = snapshot.get("selected_ranks", [])
        proximity = (
            int(min([abs(rank - r) for r in selected_ranks])) if selected_ranks else 999
        )

        # Construcción del reporte forense enriquecido para el JSON
        return {
            "hits": max_h,
            "rank": rank,
            "proximity": proximity,
            "univ_size": len(univ),
            "hybrid_score": float(scores_cpu[idx_best]),
            # Integración de scores individuales para diagnóstico de la Trifecta
            "ai_score": (
                float(snapshot["ai_scores"][idx_best])
                if "ai_scores" in snapshot
                else 0.0
            ),
            "geo_score": (
                float(snapshot["geo_scores"][idx_best])
                if "geo_scores" in snapshot
                else 0.0
            ),  # Corregido typo 'geo_score' a 'geo_scores'
            "score_alpha": (
                float(snapshot["ai_alpha"][idx_best]) if "ai_alpha" in snapshot else 0.0
            ),
            "score_beta": (
                float(snapshot["ai_beta"][idx_best]) if "ai_beta" in snapshot else 0.0
            ),
            "score_omega": (
                float(snapshot["ai_omega"][idx_best]) if "ai_omega" in snapshot else 0.0
            ),
        }
