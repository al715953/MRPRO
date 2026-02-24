from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.data_access.config import BEST_SETTINGS_TRIS
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO
from src.strategies.tris.gating import choose_strategy
from src.strategies.tris.structural_filters import (
    StructuralFilterConfig,
    StructuralFilterEngine,
)
from src.strategies.tris.feature_lr_model import FeatureLRModel
from src.strategies.tris.ticket_ngram_model import TicketNgramModel
from src.strategies.tris.topk import beam_search, select_diverse
from src.strategies.tris.uniform_baseline import TrisUniformBaselineStrategy
from src.strategies.tris.universe_gate import should_use_topk
from src.strategies.tris.universe_5d import get_universe_and_static_mask
from src.strategies.tris.v1a_model import TrisV1AModel, _extract_tris_series


class TrisForecastV1A:
    def __init__(self):
        self.strategy_name = "Tris Forecast V1-A"
        self.model_version = "tris_v1a_bayes_markov_001"

    @staticmethod
    def _get_override(cfg: Dict, key: str, default):
        return cfg.get(key, default) if isinstance(cfg, dict) else default

    @staticmethod
    def _uniform_probs() -> np.ndarray:
        return np.full((5, 10), 0.1, dtype=np.float64)

    def _get_structural_override(self, overrides: Dict, key: str, default):
        fallback = BEST_SETTINGS_TRIS.get(key, default)
        return self._get_override(overrides, key, fallback)

    @staticmethod
    def _filter_ranked_candidates(
        ranked: List[Tuple[List[int], float]], accepted_digits: List[List[int]]
    ) -> List[Tuple[List[int], float]]:
        counts = {}
        for digits in accepted_digits:
            key = tuple(int(d) for d in digits)
            counts[key] = counts.get(key, 0) + 1

        filtered = []
        for digits, lp in ranked:
            key = tuple(int(d) for d in digits)
            remaining = counts.get(key, 0)
            if remaining > 0:
                filtered.append((digits, lp))
                counts[key] = remaining - 1
        return filtered

    @staticmethod
    def _seed_from_overrides(overrides: Dict, history_len: int):
        if not isinstance(overrides, dict):
            return None
        raw = overrides.get("seed")
        if raw in (None, ""):
            return None
        try:
            return int(raw) + int(history_len)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _rank_universe_digits(
        universe_digits: np.ndarray,
        *,
        pos_probs: np.ndarray | None,
        ticket_scores: np.ndarray | None,
        score_mode: str,
    ) -> List[Tuple[List[int], float]]:
        digits = np.asarray(universe_digits)
        if digits.ndim != 2 or digits.shape[1] != 5 or digits.shape[0] == 0:
            return []

        mode = str(score_mode or "positional_logp").lower()
        if mode not in {"positional_logp", "ticket_score"}:
            mode = "positional_logp"

        scores: np.ndarray | None = None
        if mode == "ticket_score" and ticket_scores is not None:
            scores_candidate = np.asarray(ticket_scores, dtype=np.float64).reshape(-1)
            if scores_candidate.shape[0] == digits.shape[0]:
                scores = scores_candidate

        if scores is None:
            probs = np.asarray(pos_probs, dtype=np.float64) if pos_probs is not None else None
            if probs is None or probs.shape != (5, 10):
                scores = np.zeros(digits.shape[0], dtype=np.float64)
            else:
                eps = 1e-12
                logits = np.log(np.clip(probs, eps, None))
                scores = (
                    logits[0, digits[:, 0]]
                    + logits[1, digits[:, 1]]
                    + logits[2, digits[:, 2]]
                    + logits[3, digits[:, 3]]
                    + logits[4, digits[:, 4]]
                )

        order_idx = np.argsort(scores)[::-1]
        return [
            ([int(d) for d in digits[idx].tolist()], float(scores[idx]))
            for idx in order_idx.tolist()
        ]

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        def _int_default(value, default: int) -> int:
            try:
                if value is None:
                    return int(default)
                return int(value)
            except (TypeError, ValueError):
                return int(default)

        overrides = config.filter_overrides or {}
        verbose_struct = bool(self._get_override(overrides, "verbose", False))
        gate_universe_mode = str(
            self._get_override(overrides, "universe_mode", "full_filtered_universe")
        ).lower()
        gate_score_model = str(
            self._get_override(overrides, "score_model", "positional_logp")
        ).lower()
        gate_choice, gate_report = choose_strategy(history, config)
        bypass_uniform_gate = (
            gate_choice == "uniform"
            and gate_universe_mode == "topk_scored_universe"
            and gate_score_model == "feature_lr"
        )
        if gate_choice == "uniform" and not bypass_uniform_gate:
            baseline_pred = TrisUniformBaselineStrategy().predict(history, config)
            metadata = dict(baseline_pred.metadata or {})
            metadata["gate"] = gate_report
            return PredictionResultDTO(
                strategy_name=self.strategy_name,
                tickets=baseline_pred.tickets[: config.num_tickets],
                metadata=metadata,
            )

        short_window = int(self._get_override(overrides, "short_window", 200))
        long_window = int(self._get_override(overrides, "long_window", 2000))
        alpha_bayes = float(self._get_override(overrides, "alpha_bayes", 0.5))
        mix_lambda = float(self._get_override(overrides, "mix_lambda", 0.7))
        markov_window = int(self._get_override(overrides, "markov_window", 2000))
        alpha_markov = float(self._get_override(overrides, "alpha_markov", 0.2))
        blend_markov = float(self._get_override(overrides, "blend_markov", 0.35))
        uniform_mix = float(self._get_override(overrides, "uniform_mix", 0.0))
        uniform_floor_mu = float(self._get_override(overrides, "uniform_floor_mu", 0.35))
        peak_max_prob = float(self._get_override(overrides, "peak_max_prob", 0.22))
        peak_mu_boost = float(self._get_override(overrides, "peak_mu_boost", 0.20))
        temperature = float(self._get_override(overrides, "temperature", 1.4))
        topk_k = int(self._get_override(overrides, "topk_k", 2000))
        per_pos_topm = int(self._get_override(overrides, "per_pos_topm", 6))
        beam_width = int(self._get_override(overrides, "beam_width", 2500))
        diversity_min_hamming = int(
            self._get_override(overrides, "diversity_min_hamming", 2)
        )
        topk_preview = int(self._get_override(overrides, "topk_preview", 50))

        digits_list, mult_list = _extract_tris_series(history)
        n_draws = len(digits_list)

        if n_draws > 0:
            context_last_digits = digits_list[-1]
        else:
            context_last_digits = [0, 0, 0, 0, 0]

        if bypass_uniform_gate:
            pos_probs = self._uniform_probs()
            p_multiplier = 0.5
            entropy_pos = -np.sum(pos_probs * np.log(pos_probs), axis=1)
            entropy_mean = float(np.mean(entropy_pos))
            prob_guardrails = {
                "mu_used": float(0.0),
                "max_probs": np.max(pos_probs, axis=1).tolist(),
            }
        elif n_draws < 50:
            pos_probs = self._uniform_probs()
            positives = sum(1 for v in mult_list if v)
            p_multiplier = float((positives + 1.0) / (len(mult_list) + 2.0)) if mult_list else 0.5
            entropy_pos = -np.sum(pos_probs * np.log(pos_probs), axis=1)
            entropy_mean = float(np.mean(entropy_pos))
            prob_guardrails = {
                "mu_used": float(0.0),
                "max_probs": np.max(pos_probs, axis=1).tolist(),
            }
        else:
            model = TrisV1AModel(
                blend_markov=blend_markov,
                uniform_mix=uniform_mix,
                uniform_floor_mu=uniform_floor_mu,
                peak_max_prob=peak_max_prob,
                peak_mu_boost=peak_mu_boost,
                temperature=temperature,
                bayes_params={
                    "alpha": alpha_bayes,
                    "short_window": short_window,
                    "long_window": long_window,
                    "mix_lambda": mix_lambda,
                },
                markov_params={
                    "alpha": alpha_markov,
                    "window": markov_window,
                },
            )
            model.fit(digits_list, mult_list)
            pos_probs, p_multiplier, entropy_pos, entropy_mean, prob_guardrails = model.predict(
                context_last_digits
            )

        candidates: List[Tuple[List[int], float]] = []
        prev_digits = (
            [int(d) for d in history.winning_numbers[-1][:5]]
            if history.winning_numbers
            else None
        )

        struct_allowed_even = self._get_structural_override(
            overrides, "structural_allowed_even_counts", [2, 3]
        )
        if struct_allowed_even is None:
            struct_allowed_even = [2, 3]
        elif not isinstance(struct_allowed_even, (list, tuple, set)):
            struct_allowed_even = [struct_allowed_even]
        structural_cfg = StructuralFilterConfig(
            enabled=bool(
                self._get_structural_override(overrides, "structural_enabled", True)
            ),
            sum_min=_int_default(
                self._get_structural_override(overrides, "structural_sum_min", 15), 15
            ),
            sum_max=_int_default(
                self._get_structural_override(overrides, "structural_sum_max", 30), 30
            ),
            allowed_even_counts=tuple(int(v) for v in struct_allowed_even),
            min_unique_digits=_int_default(
                self._get_structural_override(
                    overrides, "structural_min_unique_digits", 3
                ),
                3,
            ),
            max_consecutive_run=_int_default(
                self._get_structural_override(
                    overrides, "structural_max_consecutive_run", 3
                ),
                3,
            ),
            max_positional_repeats_vs_prev=_int_default(
                self._get_structural_override(
                    overrides, "structural_max_positional_repeats_vs_prev", 2
                ),
                2,
            ),
            hard_filter=bool(
                self._get_structural_override(overrides, "structural_hard_filter", True)
            ),
            soft_penalties=self._get_structural_override(
                overrides, "structural_soft_penalties", None
            ),
        )

        universe_mode = str(
            self._get_override(overrides, "universe_mode", "full_filtered_universe")
        ).lower()
        score_model_raw = str(
            self._get_override(overrides, "score_model", "positional_logp")
        ).lower()
        if score_model_raw in {"v1a_positional", "positional_logp"}:
            score_model = "positional_logp"
        elif score_model_raw == "feature_lr":
            score_model = "feature_lr"
        elif score_model_raw == "ticket_ngram":
            score_model = "ticket_ngram"
        else:
            score_model = "positional_logp"
        universe_topk_k = int(self._get_override(overrides, "universe_topk_k", topk_k))
        universe_topk_k = max(0, universe_topk_k)
        use_topk_gate = bool(self._get_override(overrides, "use_topk_gate", False))
        topk_gate_calib_size = int(
            self._get_override(overrides, "topk_gate_calib_size", 300)
        )
        topk_gate_alpha = float(self._get_override(overrides, "topk_gate_alpha", 1.0))
        topk_gate_threshold_z = float(
            self._get_override(overrides, "topk_gate_threshold_z", 1.0)
        )
        selection_mode = str(
            self._get_override(overrides, "selection_mode", "ranked")
        ).lower()
        if selection_mode not in {"random", "ranked"}:
            selection_mode = "ranked"
        rank_score_mode = str(
            self._get_override(overrides, "rank_score_mode", "positional_logp")
        ).lower()
        if rank_score_mode not in {"positional_logp", "ticket_score"}:
            rank_score_mode = "positional_logp"

        filtered_ranked: List[Tuple[List[int], float]] = []
        structural_diag = {
            "enabled": bool(structural_cfg.enabled),
            "fallback": False,
        }
        tickets: List[List[int]] = []
        final_universe_digits = None
        score_stats_min = None
        score_stats_mean = None
        score_stats_max = None
        ngram_model = None
        if score_model == "ticket_ngram":
            ngram_alpha = max(alpha_bayes, 1.0) if n_draws < 50 else alpha_bayes
            ngram_model = TicketNgramModel(
                alpha=ngram_alpha,
                window=long_window,
                short_window=short_window,
                long_window=long_window,
                mix_lambda=mix_lambda,
                uniform_mix=uniform_mix,
            )
            ngram_model.fit(digits_list)

        topk_gate_pass = None
        effective_universe_mode = universe_mode
        if universe_mode == "topk_scored_universe" and use_topk_gate:
            topk_gate_pass = should_use_topk(
                digits_list,
                gate_calib_size=topk_gate_calib_size,
                K=universe_topk_k,
                alpha=topk_gate_alpha,
                threshold_z=topk_gate_threshold_z,
            )
            if not topk_gate_pass:
                effective_universe_mode = "full_filtered_universe"

        used_full_filtered_universe = effective_universe_mode == "full_filtered_universe"
        used_topk_scored_universe = effective_universe_mode == "topk_scored_universe"
        if used_full_filtered_universe:
            all_tickets, _, static_mask = get_universe_and_static_mask(structural_cfg)
            if structural_cfg.enabled:
                final_mask = StructuralFilterEngine.mask_all(
                    all_tickets, prev_digits, static_mask, structural_cfg
                )
            else:
                final_mask = np.ones(all_tickets.shape[0], dtype=bool)
            universe_digits = all_tickets[final_mask]
            final_universe_digits = universe_digits
            universe_size = int(universe_digits.shape[0])

            static_accepted = int(np.sum(static_mask)) if structural_cfg.enabled else int(
                all_tickets.shape[0]
            )
            mirror_rejected = (
                max(0, static_accepted - universe_size) if structural_cfg.enabled else 0
            )
            structural_diag.update(
                {
                    "total_in": int(all_tickets.shape[0]),
                    "accepted": int(universe_size),
                    "total_out": int(all_tickets.shape[0] - universe_size),
                    "acceptance_rate": float(
                        universe_size / int(all_tickets.shape[0])
                        if int(all_tickets.shape[0]) > 0
                        else 0.0
                    ),
                    "static_accepted": int(static_accepted),
                    "mirror_rejected": int(mirror_rejected),
                }
            )
            structural_diag["reject_reasons"] = {
                "sum": 0,
                "parity": 0,
                "uniques": 0,
                "consecutive": 0,
                "mirror_prev": int(mirror_rejected),
            }

            if universe_size > 0:
                if selection_mode == "random":
                    rng = np.random.default_rng(
                        self._seed_from_overrides(overrides, n_draws)
                    )
                    order_idx = rng.permutation(universe_size)
                    sampled = universe_digits[order_idx, :]
                    target_n = int(max(0, config.num_tickets))
                    if target_n > 0:
                        tickets = [
                            [int(d) for d in row.tolist()]
                            for row in sampled[: min(target_n, universe_size), :]
                        ]
                    filtered_ranked = [
                        ([int(d) for d in row.tolist()], 0.0) for row in sampled
                    ]
                else:
                    aligned_ticket_scores = None
                    effective_rank_score_mode = rank_score_mode
                    if score_model == "ticket_ngram" and ngram_model is not None:
                        ticket_scores_all = ngram_model.score_all(all_tickets)
                        aligned_ticket_scores = ticket_scores_all[final_mask]
                        effective_rank_score_mode = "ticket_score"
                    elif rank_score_mode == "ticket_score":
                        eps = 1e-12
                        logits = np.log(np.clip(pos_probs, eps, None))
                        all_ticket_scores = (
                            logits[0, all_tickets[:, 0]]
                            + logits[1, all_tickets[:, 1]]
                            + logits[2, all_tickets[:, 2]]
                            + logits[3, all_tickets[:, 3]]
                            + logits[4, all_tickets[:, 4]]
                        )
                        aligned_ticket_scores = all_ticket_scores[final_mask]

                    filtered_ranked = self._rank_universe_digits(
                        universe_digits,
                        pos_probs=pos_probs,
                        ticket_scores=aligned_ticket_scores,
                        score_mode=effective_rank_score_mode,
                    )
            else:
                filtered_ranked = []

            candidates = filtered_ranked
            structural_diag["topk_k_initial"] = int(topk_k)
            structural_diag["topk_k_used"] = int(topk_k)
            structural_diag["topk_k_max"] = int(topk_k)
            structural_diag["beam_expansions"] = 0
            structural_diag["universe_mode"] = "full_filtered_universe"
            structural_diag["selection_mode"] = selection_mode
        elif used_topk_scored_universe:
            all_tickets, features_cache, static_mask = get_universe_and_static_mask(structural_cfg)
            if structural_cfg.enabled:
                base_mask = StructuralFilterEngine.mask_all(
                    all_tickets, prev_digits, static_mask, structural_cfg
                )
            else:
                base_mask = np.ones(all_tickets.shape[0], dtype=bool)
            base_count = int(np.sum(base_mask))
            K = int(self._get_override(overrides, "universe_topk_k", topk_k))
            K = max(0, min(K, base_count))
            universe_topk_k = int(K)

            if score_model == "feature_lr":
                lr_alpha = float(self._get_override(overrides, "feature_lr_alpha", 1.0))
                lr_short = int(
                    self._get_override(overrides, "feature_lr_short_window", 200)
                )
                lr_long = int(
                    self._get_override(overrides, "feature_lr_long_window", 2000)
                )
                lr_mix = float(
                    self._get_override(overrides, "feature_lr_mix_lambda", 0.7)
                )
                lr_use_mirror = bool(
                    self._get_override(overrides, "feature_lr_use_mirror", True)
                )
                lr_model = FeatureLRModel(
                    alpha=lr_alpha,
                    short_window=lr_short,
                    long_window=lr_long,
                    mix_lambda=lr_mix,
                    use_mirror=lr_use_mirror,
                ).fit(digits_list)
                scores_all = lr_model.score_all(
                    all_tickets, features_cache, prev_digits=prev_digits
                )
            elif score_model == "ticket_ngram" and ngram_model is not None:
                scores_all = ngram_model.score_all(all_tickets)
            else:
                eps = 1e-12
                logits = np.log(np.clip(pos_probs, eps, None))
                scores_all = (
                    logits[0, all_tickets[:, 0]]
                    + logits[1, all_tickets[:, 1]]
                    + logits[2, all_tickets[:, 2]]
                    + logits[3, all_tickets[:, 3]]
                    + logits[4, all_tickets[:, 4]]
                )

            scores_all = np.asarray(scores_all, dtype=np.float64).copy()
            scores_all[~base_mask] = -np.inf

            valid_idx = np.flatnonzero(np.isfinite(scores_all))
            k_eff = int(min(K, valid_idx.size))
            top_idx = np.empty(0, dtype=np.int64)
            if k_eff > 0:
                if k_eff == valid_idx.size:
                    top_idx = valid_idx.astype(np.int64, copy=False)
                else:
                    top_idx = np.argpartition(scores_all, -k_eff)[-k_eff:].astype(
                        np.int64, copy=False
                    )
                top_idx = top_idx[np.argsort(scores_all[top_idx])[::-1]]

            final_mask = np.zeros(all_tickets.shape[0], dtype=bool)
            if top_idx.size > 0:
                final_mask[top_idx] = True
            universe_digits = all_tickets[final_mask]
            final_universe_digits = universe_digits
            universe_size = int(universe_digits.shape[0])

            static_accepted = int(np.sum(static_mask)) if structural_cfg.enabled else int(
                all_tickets.shape[0]
            )
            mirror_rejected = (
                max(0, static_accepted - int(np.sum(base_mask)))
                if structural_cfg.enabled
                else 0
            )
            structural_diag.update(
                {
                    "total_in": int(all_tickets.shape[0]),
                    "accepted": int(universe_size),
                    "total_out": int(all_tickets.shape[0] - universe_size),
                    "acceptance_rate": float(
                        universe_size / int(all_tickets.shape[0])
                        if int(all_tickets.shape[0]) > 0
                        else 0.0
                    ),
                    "static_accepted": int(static_accepted),
                    "mirror_rejected": int(mirror_rejected),
                }
            )
            structural_diag["reject_reasons"] = {
                "sum": 0,
                "parity": 0,
                "uniques": 0,
                "consecutive": 0,
                "mirror_prev": int(mirror_rejected),
            }

            if universe_size > 0:
                universe_scores = scores_all[final_mask]
                score_stats_min = float(np.min(universe_scores))
                score_stats_mean = float(np.mean(universe_scores))
                score_stats_max = float(np.max(universe_scores))
            else:
                universe_scores = np.zeros(0, dtype=np.float64)

            if universe_size > 0:
                if selection_mode == "random":
                    rng = np.random.default_rng(
                        self._seed_from_overrides(overrides, n_draws)
                    )
                    order_idx = rng.permutation(universe_size)
                    sampled = universe_digits[order_idx, :]
                    target_n = int(max(0, config.num_tickets))
                    if target_n > 0:
                        tickets = [
                            [int(d) for d in row.tolist()]
                            for row in sampled[: min(target_n, universe_size), :]
                        ]
                    filtered_ranked = [
                        ([int(d) for d in row.tolist()], 0.0) for row in sampled
                    ]
                else:
                    filtered_ranked = self._rank_universe_digits(
                        universe_digits,
                        pos_probs=pos_probs,
                        ticket_scores=universe_scores,
                        score_mode="ticket_score",
                    )
            else:
                filtered_ranked = []

            candidates = filtered_ranked
            structural_diag["topk_k_initial"] = int(topk_k)
            structural_diag["topk_k_used"] = int(K)
            structural_diag["topk_k_max"] = int(K)
            structural_diag["beam_expansions"] = 0
            structural_diag["universe_mode"] = "topk_scored_universe"
            structural_diag["selection_mode"] = selection_mode
            structural_diag["score_model"] = str(score_model)
            structural_diag["universe_topk_k"] = int(K)
        elif structural_cfg.enabled:
            candidates = beam_search(
                pos_probs,
                k=topk_k,
                per_pos_topm=per_pos_topm,
                beam_width=beam_width,
            )
            engine = StructuralFilterEngine(structural_cfg)
            current_topk = int(max(1, topk_k))
            max_topk = int(
                self._get_structural_override(
                    overrides,
                    "structural_max_topk_k",
                    max(current_topk, beam_width, config.num_tickets * 20, current_topk * 8),
                )
            )
            expansions = 0

            while True:
                candidates_digits = [digits for digits, _ in candidates]
                accepted_digits, diag = engine.apply(candidates_digits, prev_digits)
                structural_diag.update(diag)
                structural_diag["topk_k_initial"] = int(topk_k)
                structural_diag["topk_k_used"] = int(current_topk)
                structural_diag["topk_k_max"] = int(max_topk)
                structural_diag["beam_expansions"] = int(expansions)
                filtered_ranked = self._filter_ranked_candidates(candidates, accepted_digits)
                structural_diag["universe_mode"] = "topk_beam"
                structural_diag["selection_mode"] = "ranked"

                if len(filtered_ranked) >= config.num_tickets:
                    break
                if current_topk >= max_topk:
                    break

                next_topk = min(
                    max_topk,
                    max(current_topk + 1, int(np.ceil(current_topk * 1.5))),
                )
                if next_topk <= current_topk:
                    break

                current_topk = int(next_topk)
                expansions += 1
                candidates = beam_search(
                    pos_probs,
                    k=current_topk,
                    per_pos_topm=per_pos_topm,
                    beam_width=max(beam_width, current_topk),
                )
        else:
            candidates = beam_search(
                pos_probs,
                k=topk_k,
                per_pos_topm=per_pos_topm,
                beam_width=beam_width,
            )
            filtered_ranked = candidates
            structural_diag["universe_mode"] = "topk_beam"
            structural_diag["selection_mode"] = "ranked"

        if not tickets:
            tickets = select_diverse(
                filtered_ranked,
                n=config.num_tickets,
                min_hamming=diversity_min_hamming,
            )

        if structural_cfg.enabled and len(tickets) < config.num_tickets:
            structural_diag["fallback"] = True
            tickets = select_diverse(
                candidates,
                n=config.num_tickets,
                min_hamming=diversity_min_hamming,
            )
        elif structural_cfg.enabled:
            structural_diag["fallback"] = False

        if structural_cfg.enabled and verbose_struct:
            reject_counts = structural_diag.get("reject_reasons", {})
            ranked_reasons = sorted(
                reject_counts.items(), key=lambda kv: int(kv[1]), reverse=True
            )
            top_reasons = [f"{k}:{v}" for k, v in ranked_reasons if int(v) > 0][:3]
            reasons_text = ", ".join(top_reasons) if top_reasons else "none"
            print(
                "[TRIS][Structural] "
                f"acceptance_rate={float(structural_diag.get('acceptance_rate', 0.0)):.4f} "
                f"accepted={int(structural_diag.get('accepted', 0))}/"
                f"{int(structural_diag.get('total_in', 0))} "
                f"top_rejects={reasons_text} "
                f"fallback={bool(structural_diag.get('fallback', False))}"
            )

        if len(tickets) < config.num_tickets:
            fallback = [int(np.argmax(pos_probs[pos])) for pos in range(5)]
            while len(tickets) < config.num_tickets:
                tickets.append(fallback[:])

        metadata = {
            "pos_probs": pos_probs.tolist(),
            "p_multiplier": float(p_multiplier),
            "entropy_pos": entropy_pos.tolist(),
            "entropy_mean": float(entropy_mean),
            "model_version": self.model_version,
            "mu_used": float(prob_guardrails.get("mu_used", 0.0)),
            "max_probs": prob_guardrails.get("max_probs", []),
            "gate": gate_report,
            "topk_preview": [
                {"digits": d, "logp": float(lp)}
                for d, lp in candidates[: max(0, topk_preview)]
            ],
            "score_model": score_model,
            "score_preview": [
                {"digits": d, "score": float(sc)}
                for d, sc in filtered_ranked[: max(0, topk_preview)]
            ],
            "universe_topk_k": int(universe_topk_k),
            "score_stats": {
                "min": score_stats_min,
                "mean": score_stats_mean,
                "max": score_stats_max,
            },
            "topk_gate": {
                "enabled": bool(use_topk_gate),
                "pass": topk_gate_pass,
                "calib_size": int(topk_gate_calib_size),
                "alpha": float(topk_gate_alpha),
                "threshold_z": float(topk_gate_threshold_z),
            },
        }
        if structural_cfg.enabled or used_topk_scored_universe:
            metadata["structural_filters"] = structural_diag
        metadata["universe_size"] = int(structural_diag.get("accepted", len(filtered_ranked)))
        metadata["universe_mode"] = str(
            structural_diag.get(
                "universe_mode",
                (
                    "full_filtered_universe"
                    if used_full_filtered_universe
                    else (
                        "topk_scored_universe"
                        if used_topk_scored_universe
                        else "topk_beam"
                    )
                ),
            )
        )
        metadata["selection_mode"] = str(
            structural_diag.get("selection_mode", selection_mode)
        )
        if used_full_filtered_universe or used_topk_scored_universe:
            if final_universe_digits is None:
                final_universe_digits = np.empty((0, 5), dtype=np.uint8)
            metadata["raw_ndarray"] = np.asarray(final_universe_digits, dtype=np.uint8)

        return PredictionResultDTO(
            strategy_name=self.strategy_name,
            tickets=tickets[: config.num_tickets],
            metadata=metadata,
        )
