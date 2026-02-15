# src/core/forensics.py
import numpy as np
from typing import Dict, Any


class LotteryForensics:
    """Módulo de Auditoría de Alta Fidelidad V15 (Omega Stride)."""

    @staticmethod
    def audit_winner(
        snapshot: Dict[str, Any], target_numbers: list, xp_audit
    ) -> Dict[str, Any]:
        """
        Analiza el desempeño de un sorteo específico comparando el universo contra el ganador.
        """
        if not snapshot or "universe" not in snapshot:
            return {"hits": 0, "rank": 0, "proximity": 999, "univ_size": 0}

        univ = snapshot["universe"]

        # --- DETECCIÓN DE BACKEND (Resonancia CuPy/NumPy) ---
        is_numpy = isinstance(univ, np.ndarray)
        xp = np if is_numpy else xp_audit

        # Preparar objetivo (Top 6 números del sorteo)
        target = xp.asarray(sorted(target_numbers[:6]), dtype=xp.uint8)

        # 1. Cálculo de Aciertos Vectorizado
        try:
            # Comparamos cada fila del universo contra el vector target
            hits_vec = xp.sum(xp.isin(univ, target), axis=1)
            max_h = int(xp.max(hits_vec))
        except Exception:
            # Fallback de Seguridad: Mover a CPU si la VRAM está saturada
            univ_cpu = univ.get() if hasattr(univ, "get") else univ
            target_cpu = np.array(sorted(target_numbers[:6]), dtype=np.uint8)
            hits_vec = np.sum(np.isin(univ_cpu, target_cpu), axis=1)
            max_h = int(np.max(hits_vec))
            xp = np

        if max_h == 0:
            return {"hits": 0, "rank": 0, "proximity": 999, "univ_size": len(univ)}

        # 2. Identificación de Coordenadas de Éxito
        best_indices = xp.where(hits_vec == max_h)[0]
        scores_xp = xp.asarray(snapshot["hybrid_scores"])
        idx_best = int(best_indices[xp.argmax(scores_xp[best_indices])])

        # 3. Extracción de Métricas (Cumplimiento de Protocolo de Memoria .get())
        scores_cpu = snapshot["hybrid_scores"]
        if hasattr(scores_cpu, "get"):
            scores_cpu = scores_cpu.get()

        rank = int(np.sum(scores_cpu > scores_cpu[idx_best]) + 1)

        # 4. Cálculo de Proximidad al Top Rank Seleccionado
        selected_ranks = snapshot.get("selected_ranks", [])
        proximity = (
            int(min([abs(rank - r) for r in selected_ranks])) if selected_ranks else 999
        )

        return {
            "hits": max_h,
            "rank": rank,
            "proximity": proximity,
            "univ_size": len(univ),
            "hybrid_score": float(scores_cpu[idx_best]),
            "ai_score": float(snapshot.get("ai_scores", [0])[idx_best]),
            "geo_score": float(snapshot.get("geo_scores", [0])[idx_best]),
            "sniper_log": snapshot.get("sniper_msg", "N/A"),
        }
