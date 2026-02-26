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
from src.strategies.tris.positional_analyzers import PositionalAnalyzers
from src.strategies.tris.universe_5d import (
    get_universe_and_static_mask,
    get_universe_with_positional_mask,
)
from src.strategies.tris.v1a_model import TrisV1AModel, _extract_tris_series


class TrisForecastV1A:
    _WARNED_ONCE_KEYS: set[str] = set()

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

    def _warn_once(self, key: str, msg: str):
        token = str(key or msg)
        if token in self._WARNED_ONCE_KEYS:
            return
        self._WARNED_ONCE_KEYS.add(token)
        print(msg)

    @classmethod
    def _reset_warn_once(cls):
        cls._WARNED_ONCE_KEYS.clear()

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
    def _to_bool(value, default: bool = False) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)):
            return bool(int(value))
        if isinstance(value, str):
            raw = value.strip().lower()
            if raw in {"1", "true", "t", "yes", "y", "on"}:
                return True
            if raw in {"0", "false", "f", "no", "n", "off", ""}:
                return False
        return bool(value)

    @staticmethod
    def _normalize_disallow_positions(raw) -> tuple[bool, ...]:
        if isinstance(raw, (list, tuple, np.ndarray)):
            arr = np.asarray(raw, dtype=bool).reshape(-1)
            if arr.size < 5:
                arr = np.pad(
                    arr,
                    (0, 5 - arr.size),
                    mode="constant",
                    constant_values=False,
                )
            return tuple(bool(v) for v in arr[:5].tolist())
        return (False, False, False, False, False)

    @staticmethod
    def _pos_unique_digits(universe_nd) -> list[int]:
        arr = np.asarray(universe_nd)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 5:
            return [0, 0, 0, 0, 0]
        arr = arr[:, :5]
        return [int(np.unique(arr[:, pos]).size) for pos in range(5)]

    def _apply_structural_override_mapping(self, cfg: StructuralFilterConfig, overrides: Dict):
        ov = overrides if isinstance(overrides, dict) else {}
        cfg.enable_global_sum_filter = self._to_bool(
            ov.get("structural_enable_global_sum_filter", cfg.enable_global_sum_filter),
            default=bool(cfg.enable_global_sum_filter),
        )
        cfg.enable_global_parity_filter = self._to_bool(
            ov.get(
                "structural_enable_global_parity_filter",
                cfg.enable_global_parity_filter,
            ),
            default=bool(cfg.enable_global_parity_filter),
        )
        cfg.immediate_repeat_mode = str(
            ov.get("structural_immediate_repeat_mode", cfg.immediate_repeat_mode)
        ).lower()
        cfg.immediate_repeat_disallow_positions = self._normalize_disallow_positions(
            ov.get(
                "structural_immediate_repeat_disallow_positions",
                cfg.immediate_repeat_disallow_positions,
            )
        )
        cfg.positional_limits = ov.get("structural_positional_limits", cfg.positional_limits)
        cfg.camera_entropy_rules = ov.get(
            "structural_camera_entropy_rules", cfg.camera_entropy_rules
        )
        return cfg

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
            and gate_score_model in {"feature_lr", "random_topk", "camera_mech_v1"}
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
        camera_topm_per_position = int(
            self._get_override(overrides, "camera_topm_per_position", 10)
        )
        camera_alpha = float(self._get_override(overrides, "camera_alpha", 1.0))
        camera_short_window = int(
            self._get_override(overrides, "camera_short_window", 100)
        )
        camera_long_window = int(
            self._get_override(overrides, "camera_long_window", 1000)
        )
        camera_mix_lambda = float(self._get_override(overrides, "camera_mix_lambda", 0.3))
        camera_latency_boost = float(
            self._get_override(overrides, "camera_latency_boost", 0.0)
        )
        camera_immediate_repeat_penalty = float(
            self._get_override(overrides, "camera_immediate_repeat_penalty", 0.0)
        )
        camera_parity_bias_strength = float(
            self._get_override(overrides, "camera_parity_bias_strength", 0.0)
        )
        camera_mech_blend_with_v1a = float(
            self._get_override(overrides, "camera_mech_blend_with_v1a", 0.5)
        )
        camera_use_slot_context = bool(
            self._get_override(overrides, "camera_use_slot_context", False)
        )
        camera_masked_universe = self._to_bool(
            self._get_override(overrides, "camera_masked_universe", True),
            default=True,
        )

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
        structural_positional_limits = self._get_structural_override(
            overrides, "structural_positional_limits", None
        )
        structural_immediate_repeat_mode = str(
            self._get_structural_override(
                overrides, "structural_immediate_repeat_mode", "global_count"
            )
        ).lower()
        structural_disallow_positions = self._normalize_disallow_positions(
            self._get_structural_override(
                overrides,
                "structural_immediate_repeat_disallow_positions",
                (False, False, False, False, False),
            )
        )
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
            enable_global_sum_filter=bool(
                self._get_structural_override(
                    overrides, "structural_enable_global_sum_filter", True
                )
            ),
            enable_global_parity_filter=bool(
                self._get_structural_override(
                    overrides, "structural_enable_global_parity_filter", True
                )
            ),
            positional_limits=structural_positional_limits,
            immediate_repeat_mode=str(structural_immediate_repeat_mode),
            immediate_repeat_disallow_positions=structural_disallow_positions,
            camera_entropy_rules=self._get_structural_override(
                overrides, "structural_camera_entropy_rules", None
            ),
        )
        structural_cfg = self._apply_structural_override_mapping(structural_cfg, overrides)

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
        elif score_model_raw == "random_topk":
            score_model = "random_topk"
        elif score_model_raw in {"camera_mech_v1", "positional_mech"}:
            score_model = "camera_mech_v1"
        else:
            score_model = "positional_logp"
        camera_debug_strict = self._to_bool(
            self._get_override(overrides, "camera_debug_strict", False),
            default=False,
        )
        if score_model == "camera_mech_v1" and not (
            1 <= int(camera_topm_per_position) <= 10
        ):
            raise ValueError(
                "camera_topm_per_position must be in [1..10] when score_model='camera_mech_v1'."
            )
        valid_camera_mask_controls = {"camera_mech_v1", "random_topk"}
        if bool(camera_masked_universe) and score_model not in valid_camera_mask_controls:
            invalid_msg = (
                "camera_masked_universe=True but "
                f"score_model='{score_model}'. Positional mask wiring expects "
                "camera_mech_v1 (or random_topk control)."
            )
            if camera_debug_strict:
                raise RuntimeError(invalid_msg)
            self._warn_once(
                f"camera_masked_universe_invalid::{score_model}",
                f"[TRIS][CameraMask][WARN] {invalid_msg}",
            )
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

        pos_probs_v1a = np.asarray(pos_probs, dtype=np.float64).copy()
        camera_analyzer_out = None
        camera_pmf = None
        camera_positional_mask = None
        camera_diag_summary = {}
        camera_universe_diag = {}
        camera_debug = {
            "pre_mask_universe_size": None,
            "post_static_mask_size": None,
            "post_topk_size": None,
            "pos_unique_digits_pre_topk": [0, 0, 0, 0, 0],
            "pos_unique_digits_final": [0, 0, 0, 0, 0],
            "structural_flags_effective": {
                "enable_global_sum_filter": bool(
                    structural_cfg.enable_global_sum_filter
                ),
                "enable_global_parity_filter": bool(
                    structural_cfg.enable_global_parity_filter
                ),
                "immediate_repeat_mode": str(structural_cfg.immediate_repeat_mode),
            },
        }
        if score_model == "camera_mech_v1":
            topm = int(max(1, min(10, camera_topm_per_position)))
            blend = float(np.clip(camera_mech_blend_with_v1a, 0.0, 1.0))
            slot_context = None
            if camera_use_slot_context:
                slot_context = self._get_override(overrides, "camera_slot_context", None)

            camera_model = PositionalAnalyzers(
                alpha=float(camera_alpha),
                short_window=int(camera_short_window),
                long_window=int(camera_long_window),
                mix_lambda=float(camera_mix_lambda),
                latency_boost=float(camera_latency_boost),
                immediate_repeat_penalty=float(camera_immediate_repeat_penalty),
                parity_bias_strength=float(camera_parity_bias_strength),
                topm_per_position=int(topm),
                pmf_floor=1e-6,
            ).fit(digits_list)
            camera_analyzer_out = camera_model.predict(
                prev_digits=prev_digits,
                slot_context=slot_context,
            )

            pmf_cam_raw = camera_analyzer_out.get("pmf_pos", camera_analyzer_out.get("pmf"))
            pmf_cam = np.asarray(pmf_cam_raw, dtype=np.float64)
            if pmf_cam.shape != (5, 10):
                pmf_cam = self._uniform_probs()
            pmf_cam = pmf_cam / np.clip(np.sum(pmf_cam, axis=1, keepdims=True), 1e-12, None)
            if pmf_cam.shape != (5, 10):
                raise RuntimeError("camera_mech_v1 pmf_cam must have shape (5,10).")
            pmf_row_sums = np.sum(pmf_cam, axis=1, dtype=np.float64)
            if not np.all(np.isfinite(pmf_row_sums)) or not np.all(
                np.abs(pmf_row_sums - 1.0) <= 1e-6
            ):
                raise RuntimeError(
                    "camera_mech_v1 pmf_cam row sums must be approximately 1.0."
                )
            camera_pmf = pmf_cam

            mask_raw = camera_analyzer_out.get("positional_mask")
            if mask_raw is None:
                camera_positional_mask = np.ones((5, 10), dtype=bool)
            else:
                camera_positional_mask = np.asarray(mask_raw, dtype=bool)
                if camera_positional_mask.shape != (5, 10):
                    camera_positional_mask = np.ones((5, 10), dtype=bool)
            camera_positional_mask = np.asarray(camera_positional_mask, dtype=bool)
            if bool(camera_masked_universe):
                if camera_positional_mask.shape != (5, 10):
                    raise RuntimeError(
                        "camera_masked_universe=True requires positional_digit_mask shape (5,10)."
                    )
                if camera_positional_mask.dtype != np.bool_:
                    raise RuntimeError(
                        "camera_masked_universe=True requires positional_digit_mask dtype bool."
                    )

            combined_pos_probs = (1.0 - blend) * pos_probs_v1a + blend * pmf_cam
            combined_pos_probs = combined_pos_probs / np.clip(
                np.sum(combined_pos_probs, axis=1, keepdims=True),
                1e-12,
                None,
            )
            pos_probs = combined_pos_probs
            entropy_pos = -np.sum(pos_probs * np.log(np.clip(pos_probs, 1e-12, None)), axis=1)
            entropy_mean = float(np.mean(entropy_pos))
            prob_guardrails = {
                "mu_used": float(prob_guardrails.get("mu_used", 0.0)),
                "max_probs": np.max(pos_probs, axis=1).tolist(),
            }

            camera_forbidden = camera_analyzer_out.get("forbidden_digits_by_pos")
            camera_favored = camera_analyzer_out.get("favored_digits_by_pos")
            if isinstance(camera_forbidden, (list, tuple)):
                limits_existing = structural_cfg.positional_limits
                if not isinstance(limits_existing, list) or len(limits_existing) < 5:
                    limits_existing = [{} for _ in range(5)]
                limits_out: list[dict] = []
                for pos in range(5):
                    base_rule = (
                        dict(limits_existing[pos])
                        if pos < len(limits_existing) and isinstance(limits_existing[pos], dict)
                        else {}
                    )
                    forb_values = set(int(v) % 10 for v in base_rule.get("forbidden_digits", []))
                    if pos < len(camera_forbidden):
                        forb_values.update(int(v) % 10 for v in (camera_forbidden[pos] or []))
                    if forb_values:
                        base_rule["forbidden_digits"] = sorted(forb_values)
                    if isinstance(camera_favored, (list, tuple)) and pos < len(camera_favored):
                        base_rule["favored_digits"] = [int(v) % 10 for v in (camera_favored[pos] or [])]
                    limits_out.append(base_rule)
                structural_cfg.positional_limits = limits_out

            camera_diag = camera_analyzer_out.get("diagnostics", {})
            cam_latency = np.asarray(camera_diag.get("latency", np.zeros((5, 10), dtype=np.int32)))
            cam_parity = np.asarray(camera_diag.get("parity_local_prob", np.full((5, 2), 0.5)))
            cam_entropy = np.asarray(camera_diag.get("entropy_pos", entropy_pos), dtype=np.float64)
            camera_diag_summary = {
                "latency_mean_by_pos": np.mean(cam_latency, axis=1).astype(float).tolist()
                if cam_latency.shape == (5, 10)
                else [],
                "latency_max_by_pos": np.max(cam_latency, axis=1).astype(int).tolist()
                if cam_latency.shape == (5, 10)
                else [],
                "parity_local_prob": cam_parity.tolist()
                if cam_parity.shape == (5, 2)
                else [],
                "entropy_pos": cam_entropy.tolist()
                if cam_entropy.shape == (5,)
                else [],
            }
            if pmf_cam.shape == (5, 10):
                camera_diag_summary["pmf_row_sums"] = pmf_row_sums.astype(float).tolist()

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
        use_camera_masked_universe = bool(
            score_model == "camera_mech_v1"
            and camera_masked_universe
            and camera_positional_mask is not None
        )
        if score_model == "camera_mech_v1" and bool(camera_masked_universe):
            if camera_positional_mask is None:
                raise RuntimeError(
                    "camera_masked_universe=True requires camera_positional_mask to be present."
                )
            if np.asarray(camera_positional_mask).shape != (5, 10):
                raise RuntimeError(
                    "camera_masked_universe=True requires camera_positional_mask shape (5,10)."
                )
        if used_full_filtered_universe:
            if use_camera_masked_universe:
                all_tickets, _, static_mask, camera_universe_diag = get_universe_with_positional_mask(
                    structural_cfg,
                    camera_positional_mask,
                    return_diag=True,
                )
                camera_debug["pre_mask_universe_size"] = int(all_tickets.shape[0])
            else:
                all_tickets, _, static_mask = get_universe_and_static_mask(structural_cfg)
            if structural_cfg.enabled:
                final_mask = StructuralFilterEngine.mask_all(
                    all_tickets, prev_digits, static_mask, structural_cfg
                )
            else:
                final_mask = np.ones(all_tickets.shape[0], dtype=bool)
            universe_digits = all_tickets[final_mask]
            camera_debug["post_static_mask_size"] = int(np.sum(final_mask))
            camera_debug["pos_unique_digits_pre_topk"] = self._pos_unique_digits(universe_digits)
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
            camera_debug["post_topk_size"] = int(universe_size)
        elif used_topk_scored_universe:
            if use_camera_masked_universe:
                all_tickets, features_cache, static_mask, camera_universe_diag = (
                    get_universe_with_positional_mask(
                        structural_cfg,
                        camera_positional_mask,
                        return_diag=True,
                    )
                )
                camera_debug["pre_mask_universe_size"] = int(all_tickets.shape[0])
            else:
                all_tickets, features_cache, static_mask = get_universe_and_static_mask(
                    structural_cfg
                )
            if structural_cfg.enabled:
                base_mask = StructuralFilterEngine.mask_all(
                    all_tickets, prev_digits, static_mask, structural_cfg
                )
            else:
                base_mask = np.ones(all_tickets.shape[0], dtype=bool)
            base_count = int(np.sum(base_mask))
            base_universe_digits = all_tickets[base_mask]
            camera_debug["post_static_mask_size"] = int(base_count)
            camera_debug["pos_unique_digits_pre_topk"] = self._pos_unique_digits(
                base_universe_digits
            )
            K = int(self._get_override(overrides, "universe_topk_k", topk_k))
            K = max(0, min(K, base_count))
            universe_topk_k = int(K)
            top_idx = np.empty(0, dtype=np.int64)

            if score_model == "random_topk":
                seed_raw = self._get_override(overrides, "random_topk_seed", 12345)
                try:
                    random_topk_seed = int(seed_raw)
                except (TypeError, ValueError):
                    random_topk_seed = 12345
                rng_topk = np.random.default_rng(int(random_topk_seed) + int(n_draws))
                idx_pool = np.flatnonzero(base_mask).astype(np.int64, copy=False)
                if K > 0 and idx_pool.size > 0:
                    if idx_pool.size >= K:
                        top_idx = rng_topk.choice(idx_pool, size=K, replace=False).astype(
                            np.int64, copy=False
                        )
                    else:
                        top_idx = idx_pool
                    top_idx = np.sort(top_idx)
                scores_all = np.full(all_tickets.shape[0], -np.inf, dtype=np.float64)
                if top_idx.size > 0:
                    scores_all[top_idx] = rng_topk.random(top_idx.size)
            elif score_model == "feature_lr":
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
                lr_shrink_c = float(
                    self._get_override(overrides, "feature_lr_shrink_c", 3000.0)
                )
                feature_lr_params = {
                    "alpha": float(lr_alpha),
                    "short_window": int(lr_short),
                    "long_window": int(lr_long),
                    "mix_lambda": float(lr_mix),
                    "use_mirror": bool(lr_use_mirror),
                    "shrink_c": float(lr_shrink_c),
                }
                structural_diag["feature_lr"] = feature_lr_params
                lr_model = FeatureLRModel(
                    alpha=lr_alpha,
                    short_window=lr_short,
                    long_window=lr_long,
                    mix_lambda=lr_mix,
                    use_mirror=lr_use_mirror,
                    shrink_c=lr_shrink_c,
                ).fit(digits_list)
                scores_all = lr_model.score_all(
                    all_tickets, features_cache, prev_digits=prev_digits
                )
            elif score_model == "ticket_ngram" and ngram_model is not None:
                scores_all = ngram_model.score_all(all_tickets)
            elif score_model == "camera_mech_v1":
                eps = 1e-12
                logits = np.log(np.clip(pos_probs, eps, None))
                scores_all = (
                    logits[0, all_tickets[:, 0]]
                    + logits[1, all_tickets[:, 1]]
                    + logits[2, all_tickets[:, 2]]
                    + logits[3, all_tickets[:, 3]]
                    + logits[4, all_tickets[:, 4]]
                )
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

            if score_model != "random_topk":
                scores_all = np.asarray(scores_all, dtype=np.float64).copy()
                scores_all[~base_mask] = -np.inf

                valid_idx = np.flatnonzero(np.isfinite(scores_all))
                k_eff = int(min(K, valid_idx.size))
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
            if score_model == "random_topk":
                final_universe_digits = all_tickets[top_idx]
            else:
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
            camera_debug["post_topk_size"] = int(universe_size)
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

        structural_diag["flags_effective"] = dict(
            camera_debug.get("structural_flags_effective", {})
        )
        if (
            use_camera_masked_universe
            and int(camera_topm_per_position) < 10
            and score_model == "camera_mech_v1"
        ):
            pre_unique = camera_debug.get("pos_unique_digits_pre_topk", [0, 0, 0, 0, 0])
            if (
                isinstance(pre_unique, list)
                and len(pre_unique) == 5
                and all(int(v) == 10 for v in pre_unique)
            ):
                msg = (
                    "camera mask not reducing positional support before topK "
                    f"(pre_topk_unique={pre_unique})"
                )
                print(f"[TRIS][CameraMask][WARN] {msg}")
                if camera_debug_strict:
                    raise RuntimeError(msg)

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

        if final_universe_digits is None and (used_full_filtered_universe or used_topk_scored_universe):
            final_universe_digits = np.empty((0, 5), dtype=np.uint8)
        if final_universe_digits is not None:
            camera_debug["post_topk_size"] = int(np.asarray(final_universe_digits).shape[0])
            camera_debug["pos_unique_digits_final"] = self._pos_unique_digits(
                final_universe_digits
            )

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
            "score_model_requested": score_model_raw,
            "score_model_effective": score_model,
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
        if score_model == "feature_lr":
            metadata["feature_lr"] = structural_diag.get("feature_lr", {})
        if score_model == "camera_mech_v1":
            if camera_pmf is None:
                camera_pmf = np.asarray(pos_probs, dtype=np.float64)
            if camera_positional_mask is None:
                camera_positional_mask = np.ones((5, 10), dtype=bool)
            metadata["score_model"] = "camera_mech_v1"
            metadata["camera_masked_universe"] = bool(camera_masked_universe)
            metadata["camera_topm_per_position"] = int(camera_topm_per_position)
            metadata["camera_pmf"] = np.asarray(camera_pmf, dtype=np.float64).tolist()
            metadata["camera_entropy_pos"] = (
                camera_diag_summary.get("entropy_pos")
                or (-np.sum(np.asarray(camera_pmf) * np.log(np.clip(np.asarray(camera_pmf), 1e-12, None)), axis=1)).tolist()
            )
            metadata["camera_positional_mask"] = (
                np.asarray(camera_positional_mask, dtype=bool).astype(np.uint8).tolist()
            )
            metadata["camera_analyzer_diag"] = dict(camera_diag_summary)
            if camera_universe_diag:
                metadata["camera_analyzer_diag"]["universe_mask_diag"] = dict(
                    camera_universe_diag
                )
            metadata["camera_mech_blend_with_v1a"] = float(
                np.clip(camera_mech_blend_with_v1a, 0.0, 1.0)
            )
            metadata["camera_debug"] = dict(camera_debug)
            metadata["camera_debug"]["strict"] = bool(camera_debug_strict)
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
            metadata["raw_ndarray"] = np.asarray(final_universe_digits, dtype=np.uint8)

        return PredictionResultDTO(
            strategy_name=self.strategy_name,
            tickets=tickets[: config.num_tickets],
            metadata=metadata,
        )
