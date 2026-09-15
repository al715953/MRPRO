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
        primary_count = max(0, int(snapshot.get("hybrid_primary_selected", 0)))
        topology_phases = snapshot.get("hybrid_topology_lane_phases", [])
        topology_deep_overlaps = [
            int(overlaps[primary_count + offset])
            for offset, phase in enumerate(topology_phases)
            if str(phase) == "deep" and primary_count + offset < len(overlaps)
        ]
        best_positions = [
            index for index, overlap in enumerate(overlaps) if overlap == maximum
        ]
        result = {
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
        if topology_deep_overlaps:
            result.update(
                {
                    "winner_topology_deep_selected_max_overlap": int(
                        max(topology_deep_overlaps)
                    ),
                    "winner_topology_deep_selected_overlap_sum": int(
                        sum(topology_deep_overlaps)
                    ),
                    "winner_topology_deep_selected_count_ge_4": int(
                        sum(value >= 4 for value in topology_deep_overlaps)
                    ),
                    "winner_topology_deep_selected_count_ge_5": int(
                        sum(value >= 5 for value in topology_deep_overlaps)
                    ),
                    "winner_topology_deep_selected_count": int(
                        len(topology_deep_overlaps)
                    ),
                }
            )
        return result

    @staticmethod
    def _ticket_codes(rows: np.ndarray, base: int) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.uint64)
        if rows.ndim != 2 or rows.shape[1] < 6:
            return np.empty(0, dtype=np.uint64)
        codes = np.zeros(len(rows), dtype=np.uint64)
        for position in range(6):
            codes = codes * np.uint64(base) + rows[:, position]
        return codes

    @staticmethod
    def _topology_deep_signal_metrics(
        snapshot: Dict[str, Any],
        winner_idx: int,
        hits_vec,
        *,
        winner_is_exact: bool,
    ) -> Dict[str, Any]:
        """Rank the exact winner and 5/6 neighbors inside topology bands."""
        topology = snapshot.get("_hybrid_topology_universe")
        bands = snapshot.get("hybrid_topology_deep_bands")
        universe = snapshot.get("universe")
        if topology is None or universe is None or not isinstance(bands, list):
            return {}

        universe = universe.get() if hasattr(universe, "get") else universe
        topology = topology.get() if hasattr(topology, "get") else topology
        universe = np.asarray(universe, dtype=np.uint8)
        topology = np.asarray(topology, dtype=np.uint8)
        if universe.ndim != 2 or topology.ndim != 2 or not len(topology):
            return {}

        base = max(2, int(universe.max()) + 1)
        universe_codes = LotteryForensics._ticket_codes(universe, base)
        topology_codes = LotteryForensics._ticket_codes(topology, base)
        topology_mask = np.isin(universe_codes, topology_codes)

        primary_count = int(snapshot.get("hybrid_primary_selected", 0))
        selected = snapshot.get("_pred_tickets") or []
        if primary_count > 0 and selected:
            primary_codes = LotteryForensics._ticket_codes(
                np.asarray(selected[:primary_count], dtype=np.uint8),
                base,
            )
            topology_mask &= ~np.isin(universe_codes, primary_codes)

        topology_idx = np.flatnonzero(topology_mask)
        if not topology_idx.size:
            return {}

        hybrid = np.asarray(snapshot.get("hybrid_scores"), dtype=np.float64)
        if hybrid.size != len(universe):
            return {}
        order = topology_idx[np.argsort(-hybrid[topology_idx], kind="stable")]
        ranks = np.empty(len(universe), dtype=np.int32)
        ranks.fill(-1)
        ranks[order] = np.arange(1, len(order) + 1, dtype=np.int32)
        raw_signals = {
            "hybrid": hybrid,
            "ai": snapshot.get("ai_scores"),
            "number": snapshot.get("number_ai_scores"),
            "geo": snapshot.get("geo_scores"),
        }
        signals = {}
        for name, raw_values in raw_signals.items():
            if raw_values is None:
                continue
            values = raw_values.get() if hasattr(raw_values, "get") else raw_values
            values = np.asarray(values, dtype=np.float64)
            if values.size != len(universe):
                continue
            signals[name] = values

        band_indices = []
        for item in bands:
            rank_min = int(item.get("rank_min", 0))
            rank_max = int(item.get("rank_max", -1))
            band_idx = order[
                (ranks[order] >= rank_min) & (ranks[order] <= rank_max)
            ]
            if band_idx.size:
                band_indices.append((item, band_idx))

        result = {}
        if winner_is_exact and topology_mask[int(winner_idx)]:
            winner_lane_rank = int(ranks[int(winner_idx)])
            winner_band = next(
                (
                    band_idx
                    for item, band_idx in band_indices
                    if int(item.get("rank_min", 0))
                    <= winner_lane_rank
                    <= int(item.get("rank_max", -1))
                ),
                None,
            )
            if winner_band is not None:
                result["winner_topology_deep_band_rank"] = winner_lane_rank
                result["winner_topology_deep_band_size"] = int(winner_band.size)
                for name, values in signals.items():
                    winner_value = float(values[int(winner_idx)])
                    band_values = values[winner_band]
                    result[f"winner_topology_deep_{name}_rank"] = int(
                        np.sum(band_values > winner_value) + 1
                    )
                    result[f"winner_topology_deep_{name}_tie_size"] = int(
                        np.sum(band_values == winner_value)
                    )

        hits_cpu = hits_vec.get() if hasattr(hits_vec, "get") else hits_vec
        hits_cpu = np.asarray(hits_cpu, dtype=np.int8)
        if hits_cpu.size != len(universe):
            return result

        # The useful scorer question is conditional: once a ticket already
        # belongs to one of the deep bands, can the signal choose a better
        # candidate than its band peers?  Global AUC does not answer that.
        # Record one stable top pick and a small top-10 shortlist per band, as
        # well as the target-informed oracle ceiling (diagnostic only).
        oracle_band_overlaps = []
        signal_band_picks: dict[str, list[int]] = {
            name: [] for name in signals
        }
        signal_band_top10: dict[str, list[int]] = {
            name: [] for name in signals
        }
        signal_band_top_ties: dict[str, list[int]] = {
            name: [] for name in signals
        }
        for _item, band_idx in band_indices:
            band_hits = hits_cpu[band_idx]
            oracle_band_overlaps.append(int(np.max(band_hits)))
            natural = np.arange(band_idx.size, dtype=np.int64)
            for name, values in signals.items():
                band_values = values[band_idx]
                signal_order = np.lexsort((natural, -band_values))
                best_local = int(signal_order[0])
                signal_band_picks[name].append(int(band_hits[best_local]))
                top10_local = signal_order[: min(10, signal_order.size)]
                signal_band_top10[name].append(
                    int(np.max(band_hits[top10_local]))
                )
                signal_band_top_ties[name].append(
                    int(np.sum(band_values == band_values[best_local]))
                )

        if oracle_band_overlaps:
            result["winner_topology_deep_oracle_max_overlap"] = int(
                max(oracle_band_overlaps)
            )
            result["winner_topology_deep_oracle_overlap_sum"] = int(
                sum(oracle_band_overlaps)
            )
            result["winner_topology_deep_oracle_count_ge_4"] = int(
                sum(value >= 4 for value in oracle_band_overlaps)
            )
            result["winner_topology_deep_oracle_count_ge_5"] = int(
                sum(value >= 5 for value in oracle_band_overlaps)
            )
        for name, picked_overlaps in signal_band_picks.items():
            if not picked_overlaps:
                continue
            result[f"winner_topology_deep_{name}_top1_max_overlap"] = int(
                max(picked_overlaps)
            )
            result[f"winner_topology_deep_{name}_top1_overlap_sum"] = int(
                sum(picked_overlaps)
            )
            result[f"winner_topology_deep_{name}_top1_count_ge_4"] = int(
                sum(value >= 4 for value in picked_overlaps)
            )
            result[f"winner_topology_deep_{name}_top1_count_ge_5"] = int(
                sum(value >= 5 for value in picked_overlaps)
            )
            result[f"winner_topology_deep_{name}_top10_max_overlap"] = int(
                max(signal_band_top10[name])
            )
            result[f"winner_topology_deep_{name}_top_tie_mean"] = float(
                np.mean(signal_band_top_ties[name])
            )

        best_by_signal: dict[str, tuple[int, int, int]] = {}
        neighbor_band_count = 0
        for _item, band_idx in band_indices:
            neighbor_idx = band_idx[hits_cpu[band_idx] >= 5]
            if not neighbor_idx.size:
                continue
            neighbor_band_count += 1
            for name, values in signals.items():
                neighbor_values = values[neighbor_idx]
                best_value = float(np.max(neighbor_values))
                best_candidates = neighbor_idx[neighbor_values == best_value]
                best_idx = int(best_candidates[0])
                band_values = values[band_idx]
                candidate = (
                    int(np.sum(band_values > best_value) + 1),
                    int(np.sum(band_values == best_value)),
                    int(hits_cpu[best_idx]),
                )
                previous = best_by_signal.get(name)
                if previous is None or candidate[:2] < previous[:2]:
                    best_by_signal[name] = candidate

        if neighbor_band_count:
            result["winner_topology_neighbor_band_count"] = neighbor_band_count
        for name, (rank, tie_size, overlap) in best_by_signal.items():
            result[f"winner_topology_neighbor_{name}_best_rank"] = rank
            result[f"winner_topology_neighbor_{name}_best_tie_size"] = tie_size
            result[f"winner_topology_neighbor_{name}_best_overlap"] = overlap
        return result

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

        topology_deep_metrics = LotteryForensics._topology_deep_signal_metrics(
            snapshot,
            idx_best,
            hits_vec,
            winner_is_exact=max_h == 6,
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
            **topology_deep_metrics,
        }
