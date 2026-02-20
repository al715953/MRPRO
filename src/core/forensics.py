# src/core/forensics.py
import numpy as np
from typing import Dict, Any


class LotteryForensics:
    """Módulo de Auditoría de Alta Fidelidad V15 (Omega Stride)."""

    @staticmethod
    def _minimal_result(univ_size: int = 0) -> Dict[str, Any]:
        return {"hits": 0, "rank": 0, "proximity": 999, "univ_size": univ_size}

    @staticmethod
    def audit_winner(
        snapshot: Dict[str, Any], target_numbers: list, xp_audit
    ) -> Dict[str, Any]:
        """
        Analiza el desempeño de un sorteo específico comparando el universo contra el ganador.
        """
        if not snapshot:
            # Snapshot vacío o nulo: no hay material de auditoría.
            return LotteryForensics._minimal_result(0)

        if "universe" not in snapshot:
            # Ruta Tris/no-universe: auditoría sobre tickets predichos.
            pred_tickets = snapshot.get("pred_tickets") or snapshot.get("_pred_tickets")
            if not pred_tickets:
                return LotteryForensics._minimal_result(0)

            target_digits = [int(x) for x in target_numbers[:5]]
            hits_vec = []
            for ticket in pred_tickets:
                t = [int(x) for x in ticket[:5]]
                hits_pos = sum(1 for i in range(5) if t[i] == target_digits[i])
                hits_vec.append(hits_pos)

            max_h = max(hits_vec) if hits_vec else 0
            best_idx = hits_vec.index(max_h) if hits_vec else 0
            hamming_min = 5 - max_h
            rank = (best_idx + 1) if max_h == 5 else 0

            return {
                "hits": int(max_h),
                "rank": int(rank),
                "proximity": int(hamming_min),
                "univ_size": len(pred_tickets),
                "hybrid_score": 0.0,
                "ai_score": 0.0,
                "geo_score": 0.0,
                "sniper_log": snapshot.get("sniper_msg", "N/A"),
            }

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
            return LotteryForensics._minimal_result(len(univ))

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
