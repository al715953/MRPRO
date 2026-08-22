# src/core/forensics.py
import numpy as np
from typing import Dict, Any


class LotteryForensics:
    """Módulo de Auditoría de Alta Fidelidad V15 (Omega Stride)."""

    @staticmethod
    def _minimal_result(univ_size: int = 0) -> Dict[str, Any]:
        return {"hits": 0, "rank": 0, "proximity": 999, "univ_size": univ_size}

    @staticmethod
    def _selected_overlap_metrics(
        snapshot: Dict[str, Any], target_numbers: list
    ) -> Dict[str, Any]:
        """Post-draw diagnostic of how the selected portfolio covered the winner."""
        raw_tickets = snapshot.get("_pred_tickets") or snapshot.get("tickets") or []
        target = set(int(number) for number in target_numbers[:6])
        tickets = [
            [int(number) for number in ticket[:6]]
            for ticket in raw_tickets
            if len(ticket) >= 6
        ]
        overlaps = [len(target.intersection(ticket)) for ticket in tickets]
        selected_ranks = snapshot.get("selected_ranks", [])
        selected_stable_ranks = snapshot.get("selected_stable_ranks", [])
        if not overlaps:
            return {
                "winner_selected_max_overlap": 0,
                "winner_selected_min_missing": 6,
                "winner_selected_count_ge_4": 0,
                "winner_selected_count_ge_5": 0,
                "winner_selected_exact": 0,
                "winner_selected_overlap_counts": {str(hit): 0 for hit in range(7)},
                "winner_selected_best_ranks": [],
                "winner_selected_best_stable_ranks": [],
            }
        maximum = int(max(overlaps))
        best_positions = [
            index for index, overlap in enumerate(overlaps) if overlap == maximum
        ]
        return {
            "winner_selected_max_overlap": maximum,
            "winner_selected_min_missing": int(6 - maximum),
            "winner_selected_count_ge_4": int(sum(hit >= 4 for hit in overlaps)),
            "winner_selected_count_ge_5": int(sum(hit >= 5 for hit in overlaps)),
            "winner_selected_exact": int(maximum == 6),
            "winner_selected_overlap_counts": {
                str(hit): int(sum(value == hit for value in overlaps))
                for hit in range(7)
            },
            "winner_selected_best_ranks": [
                int(selected_ranks[index])
                for index in best_positions
                if index < len(selected_ranks)
            ],
            "winner_selected_best_stable_ranks": [
                int(selected_stable_ranks[index])
                for index in best_positions
                if index < len(selected_stable_ranks)
            ],
            "winner_selected_best_ticket": tickets[best_positions[0]],
        }

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
        selected_overlap = LotteryForensics._selected_overlap_metrics(
            snapshot, target_numbers
        )

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
            return {
                **LotteryForensics._minimal_result(len(univ)),
                "winner_in_universe": 0,
                "winner_universe_max_overlap": 0,
                **selected_overlap,
            }

        # 2. Identificación de Coordenadas de Éxito
        best_indices = xp.where(hits_vec == max_h)[0]
        scores_xp = xp.asarray(snapshot["hybrid_scores"])
        idx_best = int(best_indices[xp.argmax(scores_xp[best_indices])])

        # 3. Extracción de Métricas (Cumplimiento de Protocolo de Memoria .get())
        scores_cpu = snapshot["hybrid_scores"]
        if hasattr(scores_cpu, "get"):
            scores_cpu = scores_cpu.get()
        scores_cpu = np.asarray(scores_cpu)

        rank = int(np.sum(scores_cpu > scores_cpu[idx_best]) + 1)
        radar_indices = snapshot.get("radar_indices")
        if hasattr(radar_indices, "get"):
            radar_indices = radar_indices.get()
        if radar_indices is None:
            radar_indices = np.arange(len(scores_cpu), dtype=np.int64)
        radar_indices = np.asarray(radar_indices, dtype=np.int64)
        valid_radar = radar_indices[
            (radar_indices >= 0) & (radar_indices < len(scores_cpu))
        ]
        radar_scores = scores_cpu[valid_radar]
        stable_order = np.argsort(-radar_scores, kind="stable")
        stable_ranks = np.empty(len(valid_radar), dtype=np.int32)
        stable_ranks[stable_order] = np.arange(
            1, len(valid_radar) + 1, dtype=np.int32
        )
        winner_positions = np.flatnonzero(valid_radar == idx_best)
        winner_stable_rank = (
            int(stable_ranks[int(winner_positions[0])])
            if winner_positions.size
            else None
        )
        winner_score_tie_size = int(np.sum(radar_scores == scores_cpu[idx_best]))

        ai_scores_cpu = snapshot.get("ai_scores", np.zeros(len(univ)))
        if hasattr(ai_scores_cpu, "get"):
            ai_scores_cpu = ai_scores_cpu.get()
        ai_scores_cpu = np.asarray(ai_scores_cpu, dtype=np.float64)
        ai_score = float(ai_scores_cpu[idx_best])
        below = int(np.sum(ai_scores_cpu < ai_score))
        tied = int(np.sum(ai_scores_cpu == ai_score))
        ai_percentile_rank = (
            100.0 * (below + 0.5 * tied) / len(ai_scores_cpu)
            if len(ai_scores_cpu)
            else 0.0
        )

        geo_score = float(snapshot.get("geo_scores", [0])[idx_best])
        blend_mode = str(snapshot.get("resonance_blend_mode", "adaptive")).lower()
        if blend_mode == "fixed":
            ai_weight = max(0.0, float(snapshot.get("hybrid_alpha", 0.5)))
            geo_weight = max(0.0, float(snapshot.get("hybrid_beta", 0.5)))
            weight_total = ai_weight + geo_weight
            if weight_total <= 0.0:
                ai_weight, geo_weight, weight_total = 0.5, 0.5, 1.0
            ai_weight /= weight_total
            geo_weight /= weight_total
        elif ai_score < 0.15:
            ai_weight, geo_weight = 0.10, 0.90
        elif geo_score > 0.4:
            ai_weight, geo_weight = 0.40, 0.60
        else:
            ai_weight, geo_weight = 0.80, 0.20

        # 4. Cálculo de Proximidad al Top Rank Seleccionado
        selected_ranks = snapshot.get("selected_ranks", [])
        proximity = (
            int(min([abs(rank - r) for r in selected_ranks])) if selected_ranks else 999
        )
        selected_stable_ranks = [
            int(value)
            for value in snapshot.get("selected_stable_ranks", [])
            if int(value) > 0
        ]
        stable_proximity = (
            int(
                min(
                    abs(int(winner_stable_rank) - selected_rank)
                    for selected_rank in selected_stable_ranks
                )
            )
            if winner_stable_rank is not None and selected_stable_ranks
            else 999
        )

        return {
            "hits": max_h,
            "winner_in_universe": int(max_h == 6),
            "winner_universe_max_overlap": int(max_h),
            "rank": rank,
            "winner_stable_rank": winner_stable_rank,
            "winner_score_tie_size": winner_score_tie_size,
            "proximity": proximity,
            "winner_stable_rank_proximity": stable_proximity,
            "univ_size": len(univ),
            "hybrid_score": float(scores_cpu[idx_best]),
            "ai_score": ai_score,
            "ai_score_kind": "relative_minmax",
            "ai_percentile_rank": float(ai_percentile_rank),
            "ai_weight_effective": float(ai_weight),
            "geo_weight_effective": float(geo_weight),
            "geo_score": geo_score,
            "ai_signal_enabled": bool(snapshot.get("ai_signal_enabled", True)),
            "ai_signal_validated": bool(snapshot.get("ai_signal_validated", True)),
            "ai_validation_scope": str(
                snapshot.get("ai_validation_scope", "model")
            ),
            "temporal_holdout_auc": snapshot.get("temporal_holdout_auc"),
            "resonance_blend_mode": blend_mode,
            "sniper_log": snapshot.get("sniper_msg", "N/A"),
            **selected_overlap,
        }
