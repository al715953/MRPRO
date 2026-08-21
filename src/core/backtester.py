# src/core/backtester.py

import csv
import time
import os
from uuid import uuid4
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from src.domain.dtos import (
    BacktestResultDTO,
    DrawHistoryDTO,
    sort_history_chronologically,
)
from src.core.rules import MelateRetroRules
from src.core.analytics import PerformanceTracker
from src.core.forensics import LotteryForensics
from src.core.prob_metrics import (
    logloss_positional,
    brier_positional,
    ece_positional,
)
from src.data_access.config import VERSION_TAG, DATA_FOLDER, get_lottery_profile
from src.data_access.dataset_version import compute_dataset_version
from src.strategies.tris.structural_filters import (
    StructuralFilterConfig,
    StructuralFilterEngine,
)
from src.strategies.tris.universe_5d import get_universe_and_static_mask
from src.strategies.tris.random_within_filters import (
    RandomWithinStructuralFiltersStrategy,
)
from src.strategies.tris.uniform_baseline import TrisUniformBaselineStrategy


class BacktestEngine:
    """Motor Sniper V14.10: Full Data Capture (Visual + CSV)."""

    def __init__(self, rules=None):
        self.rules = rules or MelateRetroRules()
        self.console = Console()
        self.tracker, self.forensic_data = PerformanceTracker(), []

    def _infer_profile_code(self, config, history: DrawHistoryDTO) -> str:
        overrides = (
            config.filter_overrides
            if hasattr(config, "filter_overrides")
            and isinstance(config.filter_overrides, dict)
            else {}
        )
        profile_code = overrides.get("profile_code")
        if profile_code:
            return str(profile_code)

        if config.ticket_size == 5 and getattr(config, "total_balls", None) == 10:
            return "tris_multiplicador"
        if config.ticket_size == 6 and getattr(config, "total_balls", None) == 39:
            return "melate_retro"

        if history.winning_numbers:
            first = history.winning_numbers[0]
            if len(first) <= 6 and config.ticket_size == 5:
                return "tris_multiplicador"

        return "melate_retro"

    def _build_tracking_context(self, config, history: DrawHistoryDTO, test_size: int):
        overrides = (
            config.filter_overrides
            if hasattr(config, "filter_overrides")
            and isinstance(config.filter_overrides, dict)
            else {}
        )
        profile_code = self._infer_profile_code(config, history)
        csv_path = ""
        try:
            profile = get_lottery_profile(profile_code)
            csv_path = os.path.join(DATA_FOLDER, profile.csv_filename)
        except Exception:
            csv_path = ""

        if csv_path:
            try:
                dataset_info = compute_dataset_version(csv_path)
            except Exception:
                dataset_info = {
                    "dataset_hash": "",
                    "row_count": 0,
                    "max_concurso": None,
                }
        else:
            dataset_info = {
                "dataset_hash": "",
                "row_count": 0,
                "max_concurso": None,
            }
        return {
            "event_id": str(uuid4()),
            "profile_code": profile_code,
            "dataset_hash": dataset_info.get("dataset_hash", ""),
            "seed": overrides.get("seed", ""),
            "split_id": f"bt_last_{test_size}",
        }

    @staticmethod
    def _coerce_bool(value, default: bool = False) -> bool:
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
    def _build_tris_structural_config(overrides):
        ov = overrides if isinstance(overrides, dict) else {}
        allowed_even = ov.get("structural_allowed_even_counts", [2, 3])
        if allowed_even is None:
            allowed_even = [2, 3]
        if not isinstance(allowed_even, (list, tuple, set)):
            allowed_even = [allowed_even]
        cfg = StructuralFilterConfig(
            enabled=BacktestEngine._coerce_bool(ov.get("structural_enabled", True), True),
            sum_min=int(ov.get("structural_sum_min", 15)),
            sum_max=int(ov.get("structural_sum_max", 30)),
            allowed_even_counts=tuple(int(v) for v in allowed_even),
            min_unique_digits=int(ov.get("structural_min_unique_digits", 3)),
            max_consecutive_run=int(ov.get("structural_max_consecutive_run", 3)),
            max_positional_repeats_vs_prev=int(
                ov.get("structural_max_positional_repeats_vs_prev", 2)
            ),
            hard_filter=BacktestEngine._coerce_bool(
                ov.get("structural_hard_filter", True), True
            ),
            soft_penalties=ov.get("structural_soft_penalties", None),
        )
        cfg.enable_global_sum_filter = BacktestEngine._coerce_bool(
            ov.get("structural_enable_global_sum_filter", cfg.enable_global_sum_filter),
            default=bool(cfg.enable_global_sum_filter),
        )
        cfg.enable_global_parity_filter = BacktestEngine._coerce_bool(
            ov.get(
                "structural_enable_global_parity_filter",
                cfg.enable_global_parity_filter,
            ),
            default=bool(cfg.enable_global_parity_filter),
        )
        cfg.immediate_repeat_mode = str(
            ov.get("structural_immediate_repeat_mode", cfg.immediate_repeat_mode)
        ).lower()
        disallow_raw = ov.get(
            "structural_immediate_repeat_disallow_positions",
            cfg.immediate_repeat_disallow_positions,
        )
        if isinstance(disallow_raw, (list, tuple, np.ndarray)):
            disallow_arr = np.asarray(disallow_raw, dtype=bool).reshape(-1)
            if disallow_arr.size < 5:
                disallow_arr = np.pad(
                    disallow_arr,
                    (0, 5 - disallow_arr.size),
                    mode="constant",
                    constant_values=False,
                )
            cfg.immediate_repeat_disallow_positions = tuple(
                bool(v) for v in disallow_arr[:5].tolist()
            )
        cfg.positional_limits = ov.get("structural_positional_limits", cfg.positional_limits)
        cfg.camera_entropy_rules = ov.get(
            "structural_camera_entropy_rules", cfg.camera_entropy_rules
        )
        return cfg

    def run(
        self,
        strategy,
        history: DrawHistoryDTO,
        config,
        verbose=True,
        pre_process_strategy=None,
    ):
        self.forensic_data = []
        total_inv, total_earn, jackpot_coverage = 0.0, 0.0, 0
        max_hits = getattr(self.rules, "max_hits", config.ticket_size)
        hits_dist = {i: 0 for i in range(max_hits + 1)}
        reduced_sizes = []
        high_hit_levels = list(range(max(0, max_hits - 2), max_hits + 1))
        max_hits_by_draw = {h: 0 for h in high_hit_levels}
        expects_universe_coverage = pre_process_strategy is not None
        has_universe_data = False

        # --- LOG: print once per run ---
        sniper_header_printed = False
        sniper_header_msg = ""
        # -----------------------------

        is_reduction_only = (
            pre_process_strategy is None
            and strategy.__class__.__name__ == "UniverseReductionStrategy"
        )

        chronological_history = sort_history_chronologically(history)
        full_h = list(
            zip(
                chronological_history.dates,
                chronological_history.winning_numbers,
                chronological_history.concursos,
            )
        )
        test_size = min(config.backtest_size, len(full_h))
        start_idx = len(full_h) - test_size
        model_overrides = (
            config.filter_overrides
            if hasattr(config, "filter_overrides")
            and isinstance(config.filter_overrides, dict)
            else {}
        )
        training_cutoff = getattr(strategy, "training_cutoff_contest", None)
        temporal_auc = getattr(strategy, "temporal_holdout_auc", None)
        model_ai_enabled = getattr(strategy, "ai_signal_enabled", True)
        model_ai_validated = getattr(strategy, "ai_signal_validated", True)
        number_temporal_auc = getattr(strategy, "number_temporal_holdout_auc", None)
        if verbose and temporal_auc is not None:
            ai_status = "ON" if model_ai_enabled else "OFF"
            validation_status = "VALIDADA" if model_ai_validated else "NO VALIDADA"
            status_color = "green" if model_ai_validated else "yellow"
            self.console.print(
                f"[{status_color}]AI temporal: {ai_status} ({validation_status})[/] | "
                f"AUC fuera de muestra: {float(temporal_auc):.4f} | "
                "umbral: 0.5100"
            )
        if verbose and number_temporal_auc is not None:
            model_overrides = getattr(config, "filter_overrides", None) or {}
            context_weight = float(model_overrides.get("ai_context_weight", 1.0))
            number_weight = float(model_overrides.get("ai_number_weight", 0.0))
            self.console.print(
                "[cyan]AI ensemble:[/] "
                f"contexto={context_weight:.2f}, números={number_weight:.2f} | "
                f"AUC números={float(number_temporal_auc):.4f}"
            )
        if training_cutoff is not None:
            expected_cutoff = model_overrides.get("fixed_origin_training_cutoff")
            expected_start = model_overrides.get("fixed_origin_test_start")
            requested_start_contest = int(full_h[start_idx][2])
            if expected_cutoff is not None and int(training_cutoff) != int(
                expected_cutoff
            ):
                raise ValueError(
                    "El modelo fixed-origin no coincide con el corte solicitado: "
                    f"modelo #{int(training_cutoff)}, esperado #{int(expected_cutoff)}."
                )
            if expected_start is not None and requested_start_contest != int(
                expected_start
            ):
                raise ValueError(
                    "La ventana fixed-origin no coincide con el backtest solicitado: "
                    f"inicio #{requested_start_contest}, esperado #{int(expected_start)}."
                )
            unseen_start_idx = next(
                (
                    idx
                    for idx, (_, _, contest) in enumerate(full_h)
                    if int(contest) > int(training_cutoff)
                ),
                len(full_h),
            )
            start_idx = max(start_idx, unseen_start_idx)
            test_size = len(full_h) - start_idx
            if test_size <= 0:
                raise ValueError(
                    "El modelo ya fue entrenado con todos los sorteos solicitados. "
                    "Reentrena para generar el modelo temporal de backtest."
                )
            if verbose and unseen_start_idx > len(full_h) - config.backtest_size:
                self.console.print(
                    "[yellow]⚠ Backtest temporal:[/] se evaluarán solo "
                    f"{test_size} sorteos posteriores al concurso "
                    f"#{int(training_cutoff)}."
                )
            if verbose and expected_cutoff is not None:
                self.console.print(
                    "[bold cyan]🧠 Fixed-origin:[/] "
                    f"entrenamiento hasta #{int(training_cutoff)} | "
                    f"evaluación #{int(full_h[start_idx][2])}-#{int(full_h[-1][2])} "
                    f"({test_size} sorteos)."
                )
        tracking_ctx = self._build_tracking_context(config, history, test_size)
        strategy_model_version = getattr(strategy, "model_version", "")
        is_tris_profile = tracking_ctx["profile_code"] == "tris_multiplicador" or (
            config.ticket_size == 5 and getattr(config, "total_balls", None) == 10
        )
        overrides = (
            config.filter_overrides
            if hasattr(config, "filter_overrides")
            and isinstance(config.filter_overrides, dict)
            else {}
        )
        run_baseline = self._coerce_bool(overrides.get("run_baseline", True), True)
        tris_backtest_mode_raw = str(
            overrides.get("tris_backtest_mode", "selector")
        ).lower()
        tris_backtest_mode = (
            "universe_strategy"
            if tris_backtest_mode_raw == "compare_models"
            else tris_backtest_mode_raw
        )
        is_tris_universe_mode = is_tris_profile and tris_backtest_mode in (
            "universe",
            "universe_strategy",
        )
        tris_compare_models = bool(
            is_tris_profile
            and tris_backtest_mode == "universe_strategy"
            and (
                tris_backtest_mode_raw == "compare_models"
                or self._coerce_bool(overrides.get("compare_models", False), False)
                or self._coerce_bool(
                    overrides.get("tris_compare_models", False), False
                )
            )
        )
        tris_universe_mode_raw = str(
            overrides.get("universe_mode", "full_filtered_universe")
        ).lower()
        tris_score_model_raw = str(overrides.get("score_model", "positional_logp")).lower()
        try:
            tris_universe_topk_k_raw = int(
                overrides.get("universe_topk_k", overrides.get("topk_k", 0))
            )
        except (TypeError, ValueError):
            tris_universe_topk_k_raw = 0
        try:
            tris_camera_topm_raw = int(overrides.get("camera_topm_per_position", 10))
        except (TypeError, ValueError):
            tris_camera_topm_raw = 10
        camera_debug_strict = self._coerce_bool(
            overrides.get("camera_debug_strict", False), False
        )
        compare_model_a_score_model = str(
            overrides.get(
                "compare_model_a_score_model",
                overrides.get("score_model", "feature_lr"),
            )
        ).lower()
        compare_model_b_score_model = str(
            overrides.get("compare_model_b_score_model", "random_topk")
        ).lower()
        compare_model_a_name = str(
            overrides.get("compare_model_a_name", compare_model_a_score_model)
        ) or str(compare_model_a_score_model or "feature_lr")
        compare_model_b_name = str(
            overrides.get("compare_model_b_name", compare_model_b_score_model)
        ) or str(compare_model_b_score_model or "random_topk")
        baseline_strategy = (
            TrisUniformBaselineStrategy()
            if (is_tris_profile and run_baseline and not is_tris_universe_mode)
            else None
        )
        compare_baselines = self._coerce_bool(
            overrides.get("compare_baselines", False), False
        )
        random_filter_baseline = (
            RandomWithinStructuralFiltersStrategy()
            if (is_tris_profile and compare_baselines and not is_tris_universe_mode)
            else None
        )
        baseline_compare_stats = (
            {
                "draws_evaluated": 0,
                "model_exact_hits": 0,
                "model_draw_hits": 0,
                "model_tickets_total": 0,
                "random_exact_hits": 0,
                "random_draw_hits": 0,
                "random_tickets_total": 0,
                "random_errors": 0,
            }
            if random_filter_baseline is not None
            else None
        )
        ll_base_const = 2.302585092994046
        br_base_const = 0.9
        prob_metric_sums = {"logloss": 0.0, "brier": 0.0, "ece": 0.0}
        prob_metric_count = 0
        baseline_metric_sums = {"logloss": 0.0, "brier": 0.0, "ece": 0.0}
        baseline_metric_count = 0
        delta_ll_values = []
        delta_ll_draw_ids = []
        delta_br_values = []
        delta_ll_details = []
        tris_universe_sizes = []
        tris_fs_pass_count = 0
        tris_compare_in_lr = []
        tris_compare_in_rand = []
        tris_compare_u_lr = []
        tris_compare_u_rand = []
        tris_winner_fail_reasons = {
            "sum": 0,
            "parity": 0,
            "uniques": 0,
            "consecutive": 0,
            "mirror_prev": 0,
            "score_topk": 0,
        }
        tris_pos_top1_hits = np.zeros(5, dtype=int)
        tris_pos_top1_total = 0
        tris_pos_mask_hits = np.zeros(5, dtype=int)
        tris_pos_mask_total = 0
        tris_pos_universe_hits = np.zeros(5, dtype=int)
        tris_pos_universe_total = 0
        tris_pos_weight_sum = np.zeros(5, dtype=np.float64)
        tris_pos_weight_count = np.zeros(5, dtype=np.int32)
        tris_pos_target_cov_sum = np.zeros(5, dtype=np.float64)
        tris_pos_target_cov_count = np.zeros(5, dtype=np.int32)
        tris_pos_volatility_sum = np.zeros(5, dtype=np.float64)
        tris_pos_volatility_count = np.zeros(5, dtype=np.int32)
        camera_mask_missing_warned = False
        camera_full_support_draws = 0
        camera_support_checks = 0
        camera_mask_present_draws = 0
        tris_layered_mesh_draws = 0
        tris_mesh_pre_sizes = []
        tris_mesh_post_guardrails_sizes = []
        tris_mesh_post_topk_sizes = []
        tris_mesh_pre_hits = 0
        tris_mesh_pre_hit_count = 0
        tris_mesh_post_guardrails_hits = 0
        tris_mesh_post_guardrails_hit_count = 0
        tris_mesh_topk_hits = 0
        tris_mesh_topk_hit_count = 0
        tris_mesh_conditional_topk_num = 0
        tris_mesh_conditional_topk_den = 0
        tris_mesh_attrition_pre_to_guardrails = []
        tris_mesh_attrition_guardrails_to_topk = []
        tris_mesh_attrition_total = []
        tris_struct_cfg = None
        tris_struct_cfg_effective = (
            self._build_tris_structural_config(overrides) if is_tris_profile else None
        )
        fast_flags_enabled = bool(is_tris_profile)
        fast_mode_enabled = bool(
            fast_flags_enabled
            and self._coerce_bool(overrides.get("backtest_fast_mode", False), False)
        )
        skip_forensics = bool(
            fast_flags_enabled
            and (
                fast_mode_enabled
                or self._coerce_bool(
                    overrides.get("backtest_skip_forensics", False), False
                )
            )
        )
        skip_prob_metrics = bool(
            fast_flags_enabled
            and (
                fast_mode_enabled
                or self._coerce_bool(
                    overrides.get("backtest_skip_prob_metrics", False), False
                )
            )
        )
        skip_baseline_probs = bool(
            fast_flags_enabled
            and (
                fast_mode_enabled
                or self._coerce_bool(
                    overrides.get("backtest_skip_baseline_probs", False), False
                )
            )
        )
        skip_outlier_csv = bool(
            fast_flags_enabled
            and (
                fast_mode_enabled
                or self._coerce_bool(
                    overrides.get("backtest_skip_outlier_csv", False), False
                )
            )
        )
        skip_bootstrap_ci = bool(
            fast_flags_enabled
            and (
                fast_mode_enabled
                or self._coerce_bool(
                    overrides.get("backtest_skip_bootstrap_ci", False), False
                )
            )
        )
        if fast_flags_enabled:
            try:
                render_every_n = max(1, int(overrides.get("backtest_render_every_n", 1)))
            except (TypeError, ValueError):
                render_every_n = 1
            try:
                gpu_pool_cleanup_every_n = max(
                    1, int(overrides.get("gpu_pool_cleanup_every_n", 1))
                )
            except (TypeError, ValueError):
                gpu_pool_cleanup_every_n = 1
        else:
            render_every_n = 1
            gpu_pool_cleanup_every_n = 1
        fast_mode_summary = None
        fast_mode_omissions = []
        if skip_forensics:
            fast_mode_omissions.append("forensics")
        if skip_prob_metrics:
            fast_mode_omissions.append("prob_metrics")
        if skip_baseline_probs:
            fast_mode_omissions.append("baseline_probs")
        if skip_outlier_csv:
            fast_mode_omissions.append("outlier_csv")
        if skip_bootstrap_ci:
            fast_mode_omissions.append("bootstrap_ci")
        if fast_mode_enabled or fast_mode_omissions or render_every_n > 1:
            fast_mode_summary = {
                "enabled": bool(fast_mode_enabled),
                "omitted_modules": list(fast_mode_omissions),
                "render_every_n": int(render_every_n),
                "gpu_pool_cleanup_every_n": int(gpu_pool_cleanup_every_n),
            }
        if isinstance(overrides, dict) and "backtest_incremental_history" in overrides:
            use_incremental_history = self._coerce_bool(
                overrides.get("backtest_incremental_history", False), False
            )
        else:
            use_incremental_history = bool(is_tris_profile)
        tris_run_context = {}
        if is_tris_profile and isinstance(tris_struct_cfg_effective, StructuralFilterConfig):
            tris_run_context = {
                "tris_backtest_mode": str(tris_backtest_mode),
                "universe_mode": str(tris_universe_mode_raw),
                "score_model": str(tris_score_model_raw),
                "universe_topk_k": int(max(0, tris_universe_topk_k_raw)),
                "camera_masked_universe": bool(
                    self._coerce_bool(
                        overrides.get("camera_masked_universe", False), False
                    )
                ),
                "camera_topm_per_position": int(max(1, min(10, tris_camera_topm_raw))),
                "structural_enable_global_sum_filter": bool(
                    tris_struct_cfg_effective.enable_global_sum_filter
                ),
                "structural_enable_global_parity_filter": bool(
                    tris_struct_cfg_effective.enable_global_parity_filter
                ),
                "structural_immediate_repeat_mode": str(
                    tris_struct_cfg_effective.immediate_repeat_mode
                ),
                "camera_debug_strict": bool(camera_debug_strict),
            }
            if verbose:
                self.console.print(
                    "[cyan]TRIS RUN CONTEXT:[/] "
                    + ", ".join(f"{k}={v}" for k, v in tris_run_context.items())
                )
        tris_struct_engine = None
        tris_all_tickets = None
        tris_static_mask = None
        if is_tris_universe_mode:
            tris_struct_cfg = tris_struct_cfg_effective
            tris_struct_engine = StructuralFilterEngine(tris_struct_cfg)
            tris_all_tickets, _, tris_static_mask = get_universe_and_static_mask(
                tris_struct_cfg
            )

        def _coerce_universe_ptr(universe_nd):
            if universe_nd is None:
                return np.empty((0, 5), dtype=np.int16)
            universe_ptr = np.asarray(universe_nd)
            if universe_ptr.ndim == 1:
                universe_ptr = (
                    universe_ptr.reshape(1, -1)
                    if universe_ptr.shape[0] >= 5
                    else np.empty((0, 5), dtype=np.int16)
                )
            if universe_ptr.ndim != 2 or universe_ptr.shape[1] < 5:
                return np.empty((0, 5), dtype=np.int16)
            return universe_ptr[:, :5].astype(np.int16, copy=False)

        def _coerce_optional_nonneg_int(value):
            if value is None:
                return None
            try:
                iv = int(value)
            except (TypeError, ValueError):
                return None
            return iv if iv >= 0 else None

        def _coerce_optional_binary(value):
            if value is None:
                return None
            return int(1 if self._coerce_bool(value, False) else 0)

        def _winner_in_universe_payload(universe_payload, winner_digits):
            if universe_payload is None:
                return None
            try:
                universe_ptr = _coerce_universe_ptr(universe_payload)
                if universe_ptr.shape[0] == 0:
                    return 0
                return int(
                    np.any(
                        np.all(
                            universe_ptr.astype(np.int16, copy=False)
                            == winner_digits[None, :],
                            axis=1,
                        )
                    )
                )
            except Exception:
                return None

        def _accumulate_optional_pos_metric(values, sums, counts):
            if values is None:
                return
            try:
                arr = np.asarray(values, dtype=np.float64).reshape(-1)
            except Exception:
                return
            if arr.size == 0:
                return
            if arr.size < 5:
                arr = np.pad(arr, (0, 5 - arr.size), mode="edge")
            arr = arr[:5]
            finite = np.isfinite(arr)
            if not np.any(finite):
                return
            sums[finite] += arr[finite]
            counts[finite] += 1

        def _predict_snapshot(curr_history, local_overrides):
            base_overrides = (
                dict(config.filter_overrides)
                if hasattr(config, "filter_overrides")
                and isinstance(config.filter_overrides, dict)
                else {}
            )
            merged_overrides = dict(base_overrides)
            merged_overrides.update(local_overrides)
            config.filter_overrides = merged_overrides
            try:
                try:
                    pred = strategy.predict(curr_history, config, verbose=False)
                except TypeError:
                    pred = strategy.predict(curr_history, config)
            finally:
                config.filter_overrides = base_overrides
            snap = pred.metadata if pred else {}
            universe_nd_local = snap.get("raw_ndarray") if isinstance(snap, dict) else None
            return (
                pred,
                snap,
                _coerce_universe_ptr(universe_nd_local),
                universe_nd_local is not None,
            )

        self.console.print(
            f"\n[bold magenta]🚀 INICIANDO MISIÓN ALPHA GLOBAL ({VERSION_TAG})[/bold magenta]"
        )

        hist_dates = []
        hist_nums = []
        hist_ids = []
        if use_incremental_history and start_idx > 0:
            for past_date, past_numbers, past_id in full_h[:start_idx]:
                hist_dates.append(past_date)
                hist_nums.append(past_numbers)
                hist_ids.append(past_id)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]📡 Sniper Lab:[/][white] Analizando Malla...[/]"),
            BarColumn(bar_width=20),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
            disable=(not verbose) or is_reduction_only,
        ) as progress:
            task = progress.add_task("Misión", total=test_size)

            for i in range(start_idx, len(full_h)):
                draw_idx = (i - start_idx) + 1
                t_start = time.time()
                _, target, t_id = full_h[i]
                used_cupy_this_draw = False

                if use_incremental_history:
                    curr_h = DrawHistoryDTO(hist_dates, hist_nums, hist_ids)
                else:
                    past = full_h[:i]
                    d_past, n_past, ids_past = zip(*past)
                    curr_h = DrawHistoryDTO(list(d_past), list(n_past), list(ids_past))

                # --- FASE 1: REDUCCIÓN ---
                sniper_msg = ""
                sniper_msg_for_line = ""  # <- no repetimos el log por sorteo
                if pre_process_strategy:
                    res_univ = pre_process_strategy.predict(
                        curr_h, config, verbose=False
                    )
                    config.raw_universe_ptr = res_univ.metadata.get("raw_ndarray")
                    sniper_msg = res_univ.metadata.get("sniper_log", "")

                    # --- LOG: imprimir solo 1 vez en toda la corrida ---
                    if verbose and (not sniper_header_printed) and sniper_msg:
                        sniper_header_msg = sniper_msg
                        self.console.print(
                            f"[cyan]🧷 SNIPER (run):[/] {sniper_header_msg}"
                        )
                        sniper_header_printed = True
                    # ---------------------------------------------------

                    if config.raw_universe_ptr is not None:
                        has_universe_data = True
                        xp = (
                            cp
                            if (HAS_CUPY and hasattr(config.raw_universe_ptr, "get"))
                            else np
                        )
                        if HAS_CUPY and xp is cp:
                            used_cupy_this_draw = True
                        universe_ptr = config.raw_universe_ptr[:, : config.ticket_size]
                        target_slice = target[: config.ticket_size]

                        if isinstance(self.rules, MelateRetroRules):
                            t_xp = xp.asarray(target_slice, dtype=xp.uint8)
                            matches = xp.sum(xp.isin(universe_ptr, t_xp), axis=1)
                            if int(xp.max(matches)) == max_hits:
                                jackpot_coverage += 1
                        else:
                            # Modo agnóstico: deferimos la semántica de acierto a las reglas del juego.
                            univ_cpu = (
                                universe_ptr.get()
                                if hasattr(universe_ptr, "get")
                                else np.asarray(universe_ptr)
                            )
                            best_hits = 0
                            for candidate in univ_cpu:
                                h_n, _ = self.rules.validate_ticket(
                                    candidate.tolist(), target
                                )
                                if h_n > best_hits:
                                    best_hits = h_n
                                if best_hits == max_hits:
                                    break
                            if best_hits == max_hits:
                                jackpot_coverage += 1

                # --- FASE 2: ESTRATEGIA ---
                univ_size_curr = 0
                fs_pass = False
                winner_fail_reasons_curr = {}
                compare_metrics_curr = {}
                camera_mask_present_curr = False
                camera_mask_hits_curr = None
                camera_pos_unique_digits_final_curr = None
                mesh_pre_universe_size_curr = None
                mesh_post_guardrails_size_curr = None
                mesh_post_topk_size_curr = None
                winner_in_mesh_pre_curr = None
                winner_in_mesh_post_guardrails_curr = None
                if is_tris_universe_mode:
                    prediction = None
                    snapshot = {}
                    prev_digits = (
                        [int(d) for d in curr_h.winning_numbers[-1][:5]]
                        if curr_h.winning_numbers
                        else None
                    )
                    winner_digits = [int(d) for d in target[:5]]
                    winner_arr = np.asarray(winner_digits, dtype=np.int16)
                    universe_ptr = np.empty((0, 5), dtype=np.int16)
                    has_raw_universe = False
                    if tris_backtest_mode == "universe_strategy":
                        if tris_compare_models:
                            compare_base = dict(overrides)
                            try:
                                compare_k = int(
                                    compare_base.get(
                                        "universe_topk_k",
                                        compare_base.get("topk_k", 10000),
                                    )
                                )
                            except (TypeError, ValueError):
                                compare_k = 10000
                            compare_k = max(0, compare_k)
                            compare_common = {
                                "tris_backtest_mode": "universe_strategy",
                                "universe_mode": "topk_scored_universe",
                                "universe_topk_k": compare_k,
                            }
                            shared_compare_used = False
                            t_prepare_ctx_ms = None
                            t_model_main_ms = None
                            t_model_rand_ms = None
                            compare_a_overrides = {
                                **compare_base,
                                **compare_common,
                                "score_model": compare_model_a_score_model,
                            }
                            compare_b_overrides = {
                                **compare_base,
                                **compare_common,
                                "score_model": compare_model_b_score_model,
                            }

                            if (
                                hasattr(strategy, "_prepare_tris_context")
                                and hasattr(strategy, "_run_score_model_on_context")
                                and str(compare_model_b_score_model).lower()
                                == "random_topk"
                            ):
                                try:
                                    t_ctx_start = time.perf_counter()
                                    shared_ctx = strategy._prepare_tris_context(
                                        curr_h,
                                        config,
                                        compare_a_overrides,
                                    )
                                    if isinstance(shared_ctx, dict):
                                        out_a = strategy._run_score_model_on_context(
                                            shared_ctx,
                                            compare_model_a_score_model,
                                            compare_a_overrides,
                                        )
                                        t_rand_start = time.perf_counter()
                                        out_b = strategy._run_score_model_on_context(
                                            shared_ctx,
                                            compare_model_b_score_model,
                                            compare_b_overrides,
                                        )
                                        t_model_rand_ms = float(
                                            (time.perf_counter() - t_rand_start) * 1000.0
                                        )
                                        if isinstance(out_a, dict) and isinstance(out_b, dict):
                                            snap_a = dict(out_a.get("metadata", {}) or {})
                                            snap_b = dict(out_b.get("metadata", {}) or {})
                                            universe_a = _coerce_universe_ptr(
                                                out_a.get("raw_ndarray")
                                            )
                                            universe_b = _coerce_universe_ptr(
                                                out_b.get("raw_ndarray")
                                            )
                                            has_a_raw = bool(out_a.get("has_raw_universe", False))
                                            has_b_raw = bool(out_b.get("has_raw_universe", False))
                                            timing_payload = (
                                                snap_a.get("timings", {})
                                                if isinstance(snap_a.get("timings"), dict)
                                                else {}
                                            )
                                            t_prepare_ctx_ms = timing_payload.get(
                                                "t_prepare_ctx_ms"
                                            )
                                            t_model_main_ms = timing_payload.get(
                                                "t_model_main_ms"
                                            )
                                            if t_prepare_ctx_ms is None:
                                                t_prepare_ctx_ms = float(
                                                    (time.perf_counter() - t_ctx_start) * 1000.0
                                                )
                                            shared_compare_used = True
                                except Exception:
                                    shared_compare_used = False

                            if not shared_compare_used:
                                _, snap_a, universe_a, has_a_raw = _predict_snapshot(
                                    curr_h,
                                    compare_a_overrides,
                                )
                                _, snap_b, universe_b, has_b_raw = _predict_snapshot(
                                    curr_h,
                                    compare_b_overrides,
                                )
                            snapshot = (
                                dict(snap_a)
                                if isinstance(snap_a, dict)
                                else {}
                            )
                            if not compare_model_a_name or compare_model_a_name == "none":
                                compare_model_a_name = str(
                                    snapshot.get("score_model", compare_model_a_score_model)
                                )
                            if (
                                (not compare_model_b_name or compare_model_b_name == "none")
                                and isinstance(snap_b, dict)
                            ):
                                compare_model_b_name = str(
                                    snap_b.get("score_model", compare_model_b_score_model)
                                )
                            has_raw_universe = bool(has_a_raw and has_b_raw)
                            if has_raw_universe:
                                u_a = int(universe_a.shape[0])
                                u_b = int(universe_b.shape[0])
                                in_a = bool(
                                    u_a > 0
                                    and np.any(
                                        np.all(
                                            universe_a.astype(np.int16, copy=False)
                                            == winner_arr[None, :],
                                            axis=1,
                                        )
                                    )
                                )
                                in_b = bool(
                                    u_b > 0
                                    and np.any(
                                        np.all(
                                            universe_b.astype(np.int16, copy=False)
                                            == winner_arr[None, :],
                                            axis=1,
                                        )
                                    )
                                )
                                compare_metrics_curr = {
                                    "in_lr": int(in_a),
                                    "in_rand": int(in_b),
                                    "u_lr": int(u_a),
                                    "u_rand": int(u_b),
                                    "in_model_a": int(in_a),
                                    "in_model_b": int(in_b),
                                    "u_model_a": int(u_a),
                                    "u_model_b": int(u_b),
                                    "model_a_name": str(compare_model_a_name),
                                    "model_b_name": str(compare_model_b_name),
                                }
                                if t_prepare_ctx_ms is not None:
                                    compare_metrics_curr["t_prepare_ctx_ms"] = float(
                                        t_prepare_ctx_ms
                                    )
                                if t_model_main_ms is not None:
                                    compare_metrics_curr["t_model_main_ms"] = float(
                                        t_model_main_ms
                                    )
                                if t_model_rand_ms is not None:
                                    compare_metrics_curr["t_model_rand_ms"] = float(
                                        t_model_rand_ms
                                    )
                                if compare_metrics_curr:
                                    snapshot["compare_model_timings"] = {
                                        "t_prepare_ctx_ms": compare_metrics_curr.get(
                                            "t_prepare_ctx_ms"
                                        ),
                                        "t_model_main_ms": compare_metrics_curr.get(
                                            "t_model_main_ms"
                                        ),
                                        "t_model_rand_ms": compare_metrics_curr.get(
                                            "t_model_rand_ms"
                                        ),
                                        "shared_ctx_reused": bool(shared_compare_used),
                                    }
                                tris_compare_in_lr.append(int(in_a))
                                tris_compare_in_rand.append(int(in_b))
                                tris_compare_u_lr.append(int(u_a))
                                tris_compare_u_rand.append(int(u_b))

                                universe_ptr = universe_a
                                univ_size_curr = int(u_a)
                                tris_universe_sizes.append(univ_size_curr)
                                fs_pass = bool(in_a)
                        else:
                            prediction, snapshot, universe_ptr, has_raw_universe = (
                                _predict_snapshot(curr_h, {})
                            )

                    if tris_backtest_mode != "universe_strategy" or not has_raw_universe:
                        if tris_struct_cfg is not None and bool(tris_struct_cfg.enabled):
                            final_mask = StructuralFilterEngine.mask_all(
                                tris_all_tickets,
                                prev_digits,
                                tris_static_mask,
                                tris_struct_cfg,
                            )
                        else:
                            final_mask = np.ones(tris_all_tickets.shape[0], dtype=bool)
                        univ_size_curr = int(np.sum(final_mask))
                        tris_universe_sizes.append(univ_size_curr)

                        if tris_struct_cfg is not None and bool(tris_struct_cfg.enabled):
                            accepted_winner, winner_diag = tris_struct_engine.apply(
                                [winner_digits], prev_digits
                            )
                        else:
                            accepted_winner, winner_diag = [winner_digits], {
                                "reject_reasons": {}
                            }
                        fs_pass = bool(len(accepted_winner) > 0)
                        winner_rr = (
                            winner_diag.get("reject_reasons", {})
                            if isinstance(winner_diag, dict)
                            else {}
                        )
                        winner_fail_reasons_curr = {
                            k: int(v) for k, v in winner_rr.items() if int(v) > 0
                        }
                    else:
                        if not tris_compare_models:
                            univ_size_curr = int(universe_ptr.shape[0])
                            tris_universe_sizes.append(univ_size_curr)
                            fs_pass = bool(
                                univ_size_curr > 0
                                and np.any(
                                    np.all(
                                        universe_ptr.astype(np.int16, copy=False)
                                        == winner_arr[None, :],
                                        axis=1,
                                    )
                                )
                            )

                        if tris_struct_cfg is not None and bool(tris_struct_cfg.enabled):
                            accepted_winner, winner_diag = tris_struct_engine.apply(
                                [winner_digits], prev_digits
                            )
                        else:
                            accepted_winner, winner_diag = [winner_digits], {
                                "reject_reasons": {}
                            }
                        winner_rr = (
                            winner_diag.get("reject_reasons", {})
                            if isinstance(winner_diag, dict)
                            else {}
                        )
                        if len(accepted_winner) == 0:
                            winner_fail_reasons_curr = {
                                k: int(v) for k, v in winner_rr.items() if int(v) > 0
                            }
                        elif not fs_pass:
                            winner_fail_reasons_curr = {"score_topk": 1}
                        else:
                            winner_fail_reasons_curr = {}

                    if fs_pass:
                        tris_fs_pass_count += 1
                        jackpot_coverage += 1
                    else:
                        for k, v in winner_fail_reasons_curr.items():
                            if k in tris_winner_fail_reasons:
                                tris_winner_fail_reasons[k] += int(v)
                else:
                    if is_reduction_only:
                        prediction = strategy.predict(curr_h, config, verbose=False)
                    else:
                        prediction = strategy.predict(curr_h, config)
                    snapshot = prediction.metadata
                    if is_reduction_only:
                        reduced_sizes.append(
                            int(snapshot.get("final_size", len(prediction.tickets)))
                        )

                winner_in_topk_curr = int(fs_pass) if is_tris_universe_mode else None
                if is_tris_profile and isinstance(snapshot, dict):
                    winner_digits_pos = np.asarray(
                        [int(d) % 10 for d in target[:5]], dtype=np.int16
                    )

                    probs_payload = snapshot.get("pos_probs")
                    if probs_payload is None:
                        probs_payload = snapshot.get("camera_pmf")
                    if probs_payload is not None:
                        try:
                            probs_arr = np.asarray(probs_payload, dtype=np.float64)
                            if probs_arr.shape == (5, 10):
                                probs_arr = np.clip(probs_arr, 1e-12, None)
                                probs_arr = probs_arr / np.clip(
                                    probs_arr.sum(axis=1, keepdims=True), 1e-12, None
                                )
                                pred_top1 = np.argmax(probs_arr, axis=1).astype(np.int16)
                                tris_pos_top1_hits += (
                                    pred_top1 == winner_digits_pos
                                ).astype(int)
                                tris_pos_top1_total += 1
                        except Exception:
                            pass

                    layered_meta = snapshot.get("layered_mesh")
                    if not isinstance(layered_meta, dict):
                        layered_meta = {}
                    _accumulate_optional_pos_metric(
                        snapshot.get(
                            "camera_weights_effective",
                            layered_meta.get("camera_weights_effective"),
                        ),
                        tris_pos_weight_sum,
                        tris_pos_weight_count,
                    )
                    _accumulate_optional_pos_metric(
                        snapshot.get(
                            "target_coverage_per_pos_effective",
                            layered_meta.get("target_coverage_per_pos_effective"),
                        ),
                        tris_pos_target_cov_sum,
                        tris_pos_target_cov_count,
                    )
                    _accumulate_optional_pos_metric(
                        snapshot.get(
                            "camera_volatility_pos",
                            layered_meta.get("camera_volatility_pos"),
                        ),
                        tris_pos_volatility_sum,
                        tris_pos_volatility_count,
                    )

                    score_model_snapshot = str(
                        snapshot.get("score_model", tris_score_model_raw)
                    ).lower()
                    camera_masked_snapshot = bool(
                        self._coerce_bool(
                            snapshot.get(
                                "camera_masked_universe",
                                overrides.get("camera_masked_universe", False),
                            ),
                            False,
                        )
                    )
                    try:
                        camera_topm_snapshot = int(
                            snapshot.get(
                                "camera_topm_per_position",
                                overrides.get("camera_topm_per_position", 10),
                            )
                        )
                    except (TypeError, ValueError):
                        camera_topm_snapshot = 10

                    mask_payload = snapshot.get("camera_positional_mask")
                    if (
                        mask_payload is None
                        and score_model_snapshot == "camera_mech_v1"
                        and not camera_mask_missing_warned
                    ):
                        self.console.print(
                            "[bold yellow]WARNING:[/] camera_positional_mask missing in metadata"
                        )
                        camera_mask_missing_warned = True
                    if mask_payload is not None:
                        try:
                            mask_arr = np.asarray(mask_payload, dtype=bool)
                            if mask_arr.shape == (5, 10):
                                idx = winner_digits_pos.astype(np.int64, copy=False)
                                mask_hits_curr = mask_arr[
                                    np.arange(5, dtype=np.int64), idx
                                ].astype(int)
                                tris_pos_mask_hits += mask_hits_curr
                                tris_pos_mask_total += 1
                                camera_mask_present_curr = True
                                camera_mask_hits_curr = [
                                    int(v) for v in mask_hits_curr.tolist()
                                ]
                                camera_mask_present_draws += 1
                        except Exception:
                            pass

                    if "raw_ndarray" in snapshot:
                        try:
                            universe_pos = _coerce_universe_ptr(snapshot.get("raw_ndarray"))
                            for pos in range(5):
                                if universe_pos.shape[0] > 0 and np.any(
                                    universe_pos[:, pos] == winner_digits_pos[pos]
                                ):
                                    tris_pos_universe_hits[pos] += 1
                            tris_pos_universe_total += 1

                            if (
                                score_model_snapshot == "camera_mech_v1"
                                and camera_masked_snapshot
                                and int(camera_topm_snapshot) < 10
                            ):
                                pos_unique_digits_final = [
                                    int(np.unique(universe_pos[:, pos]).size)
                                    if universe_pos.shape[0] > 0
                                    else 0
                                    for pos in range(5)
                                ]
                                camera_pos_unique_digits_final_curr = (
                                    pos_unique_digits_final
                                )
                                camera_support_checks += 1
                                if all(v == 10 for v in pos_unique_digits_final):
                                    camera_full_support_draws += 1
                        except Exception:
                            pass

                    layered_mesh_payload = snapshot.get("layered_mesh")
                    if isinstance(layered_mesh_payload, dict):
                        tris_layered_mesh_draws += 1
                        winner_ticket = winner_digits_pos.astype(np.int16, copy=False)
                        mesh_pre_universe_size_curr = _coerce_optional_nonneg_int(
                            layered_mesh_payload.get("pre_mask_universe_size")
                        )
                        mesh_post_guardrails_size_curr = _coerce_optional_nonneg_int(
                            layered_mesh_payload.get("post_guardrails_size")
                        )
                        mesh_post_topk_size_curr = _coerce_optional_nonneg_int(
                            layered_mesh_payload.get("post_topk_size")
                        )

                        winner_in_mesh_pre_curr = _coerce_optional_binary(
                            layered_mesh_payload.get("winner_in_mesh_pre")
                        )
                        winner_in_mesh_post_guardrails_curr = _coerce_optional_binary(
                            layered_mesh_payload.get("winner_in_mesh_post_guardrails")
                        )
                        if winner_in_topk_curr is None:
                            winner_in_topk_curr = _coerce_optional_binary(
                                layered_mesh_payload.get("winner_in_topk")
                            )

                        if winner_in_mesh_pre_curr is None:
                            winner_in_mesh_pre_curr = _winner_in_universe_payload(
                                layered_mesh_payload.get("pre_mask_universe_raw_ndarray"),
                                winner_ticket,
                            )
                        if winner_in_mesh_pre_curr is None:
                            winner_in_mesh_pre_curr = _winner_in_universe_payload(
                                layered_mesh_payload.get("pre_mask_universe"),
                                winner_ticket,
                            )

                        if winner_in_mesh_post_guardrails_curr is None:
                            winner_in_mesh_post_guardrails_curr = _winner_in_universe_payload(
                                layered_mesh_payload.get("post_guardrails_raw_ndarray"),
                                winner_ticket,
                            )
                        if winner_in_mesh_post_guardrails_curr is None:
                            winner_in_mesh_post_guardrails_curr = _winner_in_universe_payload(
                                layered_mesh_payload.get("post_guardrails_universe"),
                                winner_ticket,
                            )

                        if winner_in_topk_curr == 1:
                            if winner_in_mesh_pre_curr is None:
                                winner_in_mesh_pre_curr = 1
                            if winner_in_mesh_post_guardrails_curr is None:
                                winner_in_mesh_post_guardrails_curr = 1

                        if mesh_pre_universe_size_curr is not None:
                            tris_mesh_pre_sizes.append(int(mesh_pre_universe_size_curr))
                        if mesh_post_guardrails_size_curr is not None:
                            tris_mesh_post_guardrails_sizes.append(
                                int(mesh_post_guardrails_size_curr)
                            )
                        if mesh_post_topk_size_curr is not None:
                            tris_mesh_post_topk_sizes.append(int(mesh_post_topk_size_curr))

                        if winner_in_mesh_pre_curr is not None:
                            tris_mesh_pre_hits += int(winner_in_mesh_pre_curr)
                            tris_mesh_pre_hit_count += 1
                        if winner_in_mesh_post_guardrails_curr is not None:
                            tris_mesh_post_guardrails_hits += int(
                                winner_in_mesh_post_guardrails_curr
                            )
                            tris_mesh_post_guardrails_hit_count += 1
                        if winner_in_topk_curr is not None:
                            tris_mesh_topk_hits += int(winner_in_topk_curr)
                            tris_mesh_topk_hit_count += 1

                        if (
                            winner_in_mesh_post_guardrails_curr is not None
                            and winner_in_topk_curr is not None
                            and int(winner_in_mesh_post_guardrails_curr) == 1
                        ):
                            tris_mesh_conditional_topk_den += 1
                            tris_mesh_conditional_topk_num += int(winner_in_topk_curr)

                        if (
                            mesh_pre_universe_size_curr is not None
                            and mesh_post_guardrails_size_curr is not None
                            and int(mesh_pre_universe_size_curr) > 0
                        ):
                            pre_sz = float(mesh_pre_universe_size_curr)
                            guard_sz = float(mesh_post_guardrails_size_curr)
                            tris_mesh_attrition_pre_to_guardrails.append(
                                1.0 - (guard_sz / pre_sz)
                            )
                        if (
                            mesh_post_guardrails_size_curr is not None
                            and mesh_post_topk_size_curr is not None
                            and int(mesh_post_guardrails_size_curr) > 0
                        ):
                            guard_sz = float(mesh_post_guardrails_size_curr)
                            topk_sz = float(mesh_post_topk_size_curr)
                            tris_mesh_attrition_guardrails_to_topk.append(
                                1.0 - (topk_sz / guard_sz)
                            )
                        if (
                            mesh_pre_universe_size_curr is not None
                            and mesh_post_topk_size_curr is not None
                            and int(mesh_pre_universe_size_curr) > 0
                        ):
                            pre_sz = float(mesh_pre_universe_size_curr)
                            topk_sz = float(mesh_post_topk_size_curr)
                            tris_mesh_attrition_total.append(1.0 - (topk_sz / pre_sz))

                prob_metrics = {}
                baseline_metrics = {}
                if is_tris_profile and not is_tris_universe_mode and not skip_prob_metrics:
                    y_digits = [int(d) for d in target[:5]]
                    if isinstance(snapshot, dict) and "pos_probs" in snapshot:
                        try:
                            ll = float(logloss_positional(snapshot["pos_probs"], y_digits))
                            br = float(brier_positional(snapshot["pos_probs"], y_digits))
                            ece = float(
                                ece_positional(snapshot["pos_probs"], y_digits, n_bins=10)
                            )
                            if np.isfinite(ll) and np.isfinite(br) and np.isfinite(ece):
                                probs_arr = np.asarray(snapshot["pos_probs"], dtype=np.float64)
                                probs_arr = np.clip(probs_arr, 1e-12, None)
                                probs_arr = probs_arr / np.clip(
                                    probs_arr.sum(axis=1, keepdims=True), 1e-12, None
                                )
                                prev_digits = (
                                    [int(d) for d in curr_h.winning_numbers[-1][:5]]
                                    if curr_h.winning_numbers
                                    else []
                                )
                                p_true_per_pos = [
                                    float(probs_arr[pos, y_digits[pos]]) for pos in range(5)
                                ]
                                max_prob_per_pos = [
                                    float(np.max(probs_arr[pos])) for pos in range(5)
                                ]
                                entropy_per_pos = [
                                    float(-np.sum(probs_arr[pos] * np.log(probs_arr[pos])))
                                    for pos in range(5)
                                ]
                                prob_metrics = {
                                    "logloss": ll,
                                    "brier": br,
                                    "ece": ece,
                                }
                                prob_metric_sums["logloss"] += ll
                                prob_metric_sums["brier"] += br
                                prob_metric_sums["ece"] += ece
                                prob_metric_count += 1
                                delta_ll = ll - ll_base_const
                                delta_ll_values.append(delta_ll)
                                delta_ll_draw_ids.append(int(t_id))
                                delta_br_values.append(br - br_base_const)
                                delta_ll_details.append(
                                    {
                                        "draw_id": int(t_id),
                                        "delta_ll": float(delta_ll),
                                        "y_digits": [int(d) for d in y_digits],
                                        "prev_digits": prev_digits,
                                        "p_true_per_pos": p_true_per_pos,
                                        "max_prob_per_pos": max_prob_per_pos,
                                        "entropy_per_pos": entropy_per_pos,
                                    }
                                )
                        except Exception:
                            prob_metrics = {}

                    if baseline_strategy is not None and not skip_baseline_probs:
                        try:
                            base_pred = baseline_strategy.predict(curr_h, config)
                            base_probs = (
                                base_pred.metadata.get("pos_probs")
                                if isinstance(base_pred.metadata, dict)
                                else None
                            )
                            if base_probs is not None:
                                b_ll = float(logloss_positional(base_probs, y_digits))
                                b_br = float(brier_positional(base_probs, y_digits))
                                b_ece = float(ece_positional(base_probs, y_digits, n_bins=10))
                                if (
                                    np.isfinite(b_ll)
                                    and np.isfinite(b_br)
                                    and np.isfinite(b_ece)
                                ):
                                    baseline_metrics = {
                                        "logloss": b_ll,
                                        "brier": b_br,
                                        "ece": b_ece,
                                    }
                                    baseline_metric_sums["logloss"] += b_ll
                                    baseline_metric_sums["brier"] += b_br
                                    baseline_metric_sums["ece"] += b_ece
                                    baseline_metric_count += 1
                        except Exception:
                            baseline_metrics = {}

                # Auditoría Forense
                if is_tris_universe_mode:
                    audit = {
                        "hits": int(max_hits if fs_pass else 0),
                        "rank": 1 if fs_pass else 0,
                        "proximity": 0 if fs_pass else 999,
                        "ai_score": 0.0,
                        "geo_score": 0.0,
                        "jackpot_coverage_universe": int(fs_pass),
                        "winner_fail_reasons": winner_fail_reasons_curr,
                    }
                    if compare_metrics_curr:
                        audit.update(compare_metrics_curr)
                elif is_reduction_only:
                    audit = None
                else:
                    if skip_forensics:
                        audit = None
                    else:
                        audit_snapshot = dict(snapshot) if snapshot else {}
                        audit_snapshot["_pred_tickets"] = [
                            [int(x) for x in t]
                            for t in prediction.tickets
                        ]
                        xp_audit = (
                            cp
                            if (HAS_CUPY and hasattr(config.raw_universe_ptr, "get"))
                            else np
                        )
                        if HAS_CUPY and xp_audit is cp:
                            used_cupy_this_draw = True
                        audit = LotteryForensics.audit_winner(
                            audit_snapshot, target, xp_audit
                        )

                if audit and not skip_forensics:
                    audit["draw_id"] = int(t_id)

                    # 1. Guardamos el Tamaño del Universo
                    if is_tris_universe_mode:
                        audit["univ_size"] = int(univ_size_curr)
                    else:
                        tris_struct_u = None
                        if is_tris_profile and isinstance(snapshot, dict):
                            sf_diag = snapshot.get("structural_filters")
                            if isinstance(sf_diag, dict):
                                try:
                                    tris_struct_u = int(sf_diag.get("accepted"))
                                except Exception:
                                    tris_struct_u = None
                        if tris_struct_u is not None and tris_struct_u >= 0:
                            audit["univ_size"] = tris_struct_u
                        else:
                            audit["univ_size"] = (
                                len(config.raw_universe_ptr)
                                if config.raw_universe_ptr is not None
                                else len(prediction.tickets)
                            )

                    # 2. Guardamos el Log del Sniper (sin cambios; se guarda en CSV igual)
                    audit["sniper_log"] = sniper_msg
                    audit["event_id"] = tracking_ctx["event_id"]
                    audit["profile_code"] = tracking_ctx["profile_code"]
                    audit["dataset_hash"] = tracking_ctx["dataset_hash"]
                    audit["model_version"] = (
                        snapshot.get("model_version", strategy_model_version)
                        if isinstance(snapshot, dict)
                        else strategy_model_version
                    )
                    audit["seed"] = tracking_ctx["seed"]
                    audit["split_id"] = tracking_ctx["split_id"]
                    if prob_metrics:
                        audit["logloss"] = prob_metrics["logloss"]
                        audit["brier"] = prob_metrics["brier"]
                        audit["ece"] = prob_metrics["ece"]
                    else:
                        audit["logloss"] = ""
                        audit["brier"] = ""
                        audit["ece"] = ""

                    metrics_payload = {
                        "hits_pos": int(audit.get("hits", 0)),
                        "winner_in_topk": (
                            int(winner_in_topk_curr)
                            if winner_in_topk_curr is not None
                            else ""
                        ),
                    }
                    if is_tris_profile:
                        metrics_payload["camera_mask_present"] = bool(
                            camera_mask_present_curr
                        )
                        if camera_mask_hits_curr is not None:
                            metrics_payload["camera_mask_hit_by_pos"] = [
                                int(v) for v in camera_mask_hits_curr
                            ]
                        if camera_pos_unique_digits_final_curr is not None:
                            metrics_payload["camera_pos_unique_digits_final"] = [
                                int(v) for v in camera_pos_unique_digits_final_curr
                            ]
                        if isinstance(snapshot, dict):
                            cam_debug_payload = snapshot.get("camera_debug")
                            if isinstance(cam_debug_payload, dict):
                                metrics_payload["camera_debug"] = cam_debug_payload
                        if mesh_pre_universe_size_curr is not None:
                            metrics_payload["mesh_pre_universe_size"] = int(
                                mesh_pre_universe_size_curr
                            )
                        if mesh_post_guardrails_size_curr is not None:
                            metrics_payload["mesh_post_guardrails_size"] = int(
                                mesh_post_guardrails_size_curr
                            )
                        if mesh_post_topk_size_curr is not None:
                            metrics_payload["mesh_post_topk_size"] = int(
                                mesh_post_topk_size_curr
                            )
                        if winner_in_mesh_pre_curr is not None:
                            metrics_payload["winner_in_mesh_pre"] = int(
                                winner_in_mesh_pre_curr
                            )
                        if winner_in_mesh_post_guardrails_curr is not None:
                            metrics_payload["winner_in_mesh_post_guardrails"] = int(
                                winner_in_mesh_post_guardrails_curr
                            )
                    if prob_metrics:
                        metrics_payload.update(prob_metrics)
                    if baseline_metrics:
                        metrics_payload["baseline_logloss"] = baseline_metrics["logloss"]
                        metrics_payload["baseline_brier"] = baseline_metrics["brier"]
                        metrics_payload["baseline_ece"] = baseline_metrics["ece"]
                    if is_tris_universe_mode:
                        metrics_payload["jackpot_coverage_universe"] = int(fs_pass)
                        if winner_fail_reasons_curr:
                            metrics_payload["winner_fail_reasons"] = (
                                winner_fail_reasons_curr
                            )
                        if compare_metrics_curr:
                            metrics_payload.update(compare_metrics_curr)
                    audit["metrics_json"] = metrics_payload

                    self.forensic_data.append(audit)

                # --- FASE 3: VALIDACIÓN FINANCIERA ---
                max_hit_this_draw = 0
                high_hits_this_draw = {h: 0 for h in high_hit_levels}
                if is_tris_universe_mode:
                    h_n = int(max_hits if fs_pass else 0)
                    hits_dist[h_n] += 1
                    max_hit_this_draw = h_n
                    if h_n in high_hits_this_draw:
                        high_hits_this_draw[h_n] += 1
                else:
                    for tkt in prediction.tickets:
                        total_inv += self.rules.ticket_cost
                        h_n, h_a = self.rules.validate_ticket(tkt, target)
                        total_earn += self.rules.calculate_prize(h_n, h_a)
                        hits_dist[h_n] += 1
                        if h_n > max_hit_this_draw:
                            max_hit_this_draw = h_n
                        if h_n in high_hits_this_draw:
                            high_hits_this_draw[h_n] += 1

                if baseline_compare_stats is not None and not is_tris_universe_mode:
                    model_exact = int(high_hits_this_draw.get(max_hits, 0))
                    baseline_compare_stats["draws_evaluated"] += 1
                    baseline_compare_stats["model_exact_hits"] += model_exact
                    baseline_compare_stats["model_tickets_total"] += int(
                        len(prediction.tickets)
                    )
                    if model_exact > 0:
                        baseline_compare_stats["model_draw_hits"] += 1

                    random_exact = 0
                    random_ticket_count = 0
                    try:
                        random_pred = random_filter_baseline.predict(curr_h, config)
                        random_ticket_count = int(len(random_pred.tickets))
                        for tkt in random_pred.tickets:
                            r_hits, _ = self.rules.validate_ticket(tkt, target)
                            if r_hits == max_hits:
                                random_exact += 1
                    except Exception:
                        baseline_compare_stats["random_errors"] += 1
                        random_exact = 0
                        random_ticket_count = 0

                    baseline_compare_stats["random_exact_hits"] += int(random_exact)
                    baseline_compare_stats["random_tickets_total"] += int(
                        random_ticket_count
                    )
                    if random_exact > 0:
                        baseline_compare_stats["random_draw_hits"] += 1

                if is_reduction_only and max_hit_this_draw in max_hits_by_draw:
                    max_hits_by_draw[max_hit_this_draw] += 1

                audit_for_render = audit
                if (
                    skip_forensics
                    and not is_reduction_only
                    and audit_for_render is None
                ):
                    audit_for_render = {
                        "hits": int(max_hit_this_draw),
                        "rank": int(1 if max_hit_this_draw == max_hits else 0),
                        "proximity": int(0 if max_hit_this_draw == max_hits else 999),
                        "ai_score": 0.0,
                        "geo_score": 0.0,
                        "univ_size": int(
                            univ_size_curr
                            if is_tris_universe_mode
                            else (
                                len(config.raw_universe_ptr)
                                if config.raw_universe_ptr is not None
                                else len(prediction.tickets)
                            )
                        ),
                    }

                if (
                    verbose
                    and is_reduction_only
                    and (draw_idx % render_every_n == 0 or draw_idx == test_size)
                ):
                    self._render_reduction_telemetry(
                        t_id=t_id,
                        univ_size=reduced_sizes[-1] if reduced_sizes else 0,
                        max_hit=max_hit_this_draw,
                        high_hits=high_hits_this_draw,
                        high_hit_levels=high_hit_levels,
                        max_hits=max_hits,
                        elapsed=time.time() - t_start,
                    )

                if (
                    verbose
                    and audit_for_render
                    and not is_reduction_only
                    and (draw_idx % render_every_n == 0 or draw_idx == test_size)
                ):
                    # <- log por sorteo apagado (solo se imprimió 1 vez arriba)
                    self._render_telemetry(
                        audit_for_render, t_id, t_start, max_hits, sniper_msg_for_line
                    )

                if (
                    HAS_CUPY
                    and used_cupy_this_draw
                    and (draw_idx % gpu_pool_cleanup_every_n == 0)
                ):
                    cp.get_default_memory_pool().free_all_blocks()

                if use_incremental_history:
                    hist_dates.append(full_h[i][0])
                    hist_nums.append(full_h[i][1])
                    hist_ids.append(full_h[i][2])

                progress.advance(task)

        if camera_support_checks > 0:
            full_support_ratio = float(camera_full_support_draws / camera_support_checks)
            if full_support_ratio > 0.80:
                msg = (
                    "Positional mask appears ineffective: final universe retains full support "
                    "0-9 in all positions"
                )
                detail = (
                    f"{msg} (draws={camera_full_support_draws}/{camera_support_checks}, "
                    f"ratio={full_support_ratio:.2%})"
                )
                if camera_debug_strict:
                    raise RuntimeError(detail)
                self.console.print(f"[bold yellow]WARNING:[/] {detail}")

        # Reporte Final
        res = BacktestResultDTO(
            f"Sniper Global (Dynamic Hybrid)",
            test_size,
            total_inv,
            total_earn,
            total_earn - total_inv,
            hits_dist,
        )
        tris_prob_summary = None
        baseline_prob_summary = None
        tris_delta_summary = None
        tris_outlier_summary = None
        tris_universe_summary = None
        tris_positional_summary = None
        tris_layered_mesh_summary = None
        baseline_compare_summary = None
        if is_tris_profile:
            tris_positional_summary = {
                "top1_hits": [int(v) for v in tris_pos_top1_hits.tolist()],
                "top1_total": int(tris_pos_top1_total),
                "top1_hit_rate_by_pos": [
                    (
                        float(tris_pos_top1_hits[pos] / tris_pos_top1_total)
                        if tris_pos_top1_total > 0
                        else None
                    )
                    for pos in range(5)
                ],
                "mask_hits": [int(v) for v in tris_pos_mask_hits.tolist()],
                "mask_total": int(tris_pos_mask_total),
                "mask_coverage_rate_by_pos": [
                    (
                        float(tris_pos_mask_hits[pos] / tris_pos_mask_total)
                        if tris_pos_mask_total > 0
                        else None
                    )
                    for pos in range(5)
                ],
                "universe_hits": [int(v) for v in tris_pos_universe_hits.tolist()],
                "universe_total": int(tris_pos_universe_total),
                "universe_position_coverage_rate_by_pos": [
                    (
                        float(tris_pos_universe_hits[pos] / tris_pos_universe_total)
                        if tris_pos_universe_total > 0
                        else None
                    )
                    for pos in range(5)
                ],
                "camera_weight_avg_by_pos": [
                    (
                        float(tris_pos_weight_sum[pos] / tris_pos_weight_count[pos])
                        if int(tris_pos_weight_count[pos]) > 0
                        else None
                    )
                    for pos in range(5)
                ],
                "camera_target_coverage_avg_by_pos": [
                    (
                        float(
                            tris_pos_target_cov_sum[pos] / tris_pos_target_cov_count[pos]
                        )
                        if int(tris_pos_target_cov_count[pos]) > 0
                        else None
                    )
                    for pos in range(5)
                ],
                "camera_volatility_avg_by_pos": [
                    (
                        float(
                            tris_pos_volatility_sum[pos] / tris_pos_volatility_count[pos]
                        )
                        if int(tris_pos_volatility_count[pos]) > 0
                        else None
                    )
                    for pos in range(5)
                ],
                "camera_mask_present_draws": int(camera_mask_present_draws),
                "run_context": dict(tris_run_context),
            }
            if is_tris_universe_mode:
                if tris_universe_sizes:
                    u_arr = np.asarray(tris_universe_sizes, dtype=np.float64)
                    draws_eval = int(u_arr.size)
                    tris_universe_summary = {
                        "avg_u": float(np.mean(u_arr)),
                        "min_u": int(np.min(u_arr)),
                        "max_u": int(np.max(u_arr)),
                        "fs_pass_rate": (
                            float(tris_fs_pass_count / draws_eval) if draws_eval else 0.0
                        ),
                        "draws": draws_eval,
                        "winner_fail_reasons": {
                            k: int(v) for k, v in tris_winner_fail_reasons.items()
                        },
                        "run_context": dict(tris_run_context),
                        "structural_flags_effective": {
                            "enable_global_sum_filter": bool(
                                tris_run_context.get(
                                    "structural_enable_global_sum_filter", True
                                )
                            ),
                            "enable_global_parity_filter": bool(
                                tris_run_context.get(
                                    "structural_enable_global_parity_filter", True
                                )
                            ),
                            "immediate_repeat_mode": str(
                                tris_run_context.get(
                                    "structural_immediate_repeat_mode", "global_count"
                                )
                            ),
                        },
                    }
                    if tris_compare_models and tris_compare_in_lr and tris_compare_in_rand:
                        in_lr_arr = np.asarray(tris_compare_in_lr, dtype=np.float64)
                        in_rand_arr = np.asarray(tris_compare_in_rand, dtype=np.float64)
                        u_lr_arr = np.asarray(tris_compare_u_lr, dtype=np.float64)
                        u_rand_arr = np.asarray(tris_compare_u_rand, dtype=np.float64)
                        delta_arr = in_lr_arr - in_rand_arr
                        fs_lr = float(np.mean(in_lr_arr))
                        fs_rand = float(np.mean(in_rand_arr))
                        b = int(
                            np.sum(
                                (in_lr_arr.astype(np.int8) == 1)
                                & (in_rand_arr.astype(np.int8) == 0)
                            )
                        )
                        c = int(
                            np.sum(
                                (in_lr_arr.astype(np.int8) == 0)
                                & (in_rand_arr.astype(np.int8) == 1)
                            )
                        )
                        bc = b + c
                        mcnemar_chi2 = (
                            float(((abs(b - c) - 1.0) ** 2) / bc) if bc > 0 else None
                        )
                        delta_ci = None
                        if not skip_bootstrap_ci:
                            block_size = max(2, int(np.sqrt(max(1, int(delta_arr.size)))))
                            delta_ci = self._bootstrap_block_mean_ci(
                                delta_arr,
                                block_size=block_size,
                                n_resamples=2000,
                            )
                        tris_universe_summary["compare_models"] = {
                            "draws": int(delta_arr.size),
                            "fs_lr": fs_lr,
                            "fs_rand": fs_rand,
                            "delta": float(fs_lr - fs_rand),
                            "b": int(b),
                            "c": int(c),
                            "mcnemar_chi2": mcnemar_chi2,
                            "delta_ci": delta_ci,
                            "avg_u_lr": float(np.mean(u_lr_arr)),
                            "avg_u_rand": float(np.mean(u_rand_arr)),
                            "model_a_name": str(compare_model_a_name or tris_score_model_raw),
                            "model_b_name": str(compare_model_b_name or "random_topk"),
                            "model_a_score_model": str(
                                compare_model_a_score_model or tris_score_model_raw
                            ),
                            "model_b_score_model": str(
                                compare_model_b_score_model or "random_topk"
                            ),
                        }
                if tris_layered_mesh_draws > 0:
                    tris_layered_mesh_summary = {
                        "draws_with_layered_mesh": int(tris_layered_mesh_draws),
                        "avg_pre_mesh_u": (
                            float(np.mean(np.asarray(tris_mesh_pre_sizes, dtype=np.float64)))
                            if tris_mesh_pre_sizes
                            else None
                        ),
                        "avg_post_guardrails_u": (
                            float(
                                np.mean(
                                    np.asarray(
                                        tris_mesh_post_guardrails_sizes, dtype=np.float64
                                    )
                                )
                            )
                            if tris_mesh_post_guardrails_sizes
                            else None
                        ),
                        "avg_post_topk_u": (
                            float(
                                np.mean(
                                    np.asarray(tris_mesh_post_topk_sizes, dtype=np.float64)
                                )
                            )
                            if tris_mesh_post_topk_sizes
                            else None
                        ),
                        "recall_layer_1": (
                            float(tris_mesh_pre_hits / tris_mesh_pre_hit_count)
                            if tris_mesh_pre_hit_count > 0
                            else None
                        ),
                        "recall_layer_2": (
                            float(
                                tris_mesh_post_guardrails_hits
                                / tris_mesh_post_guardrails_hit_count
                            )
                            if tris_mesh_post_guardrails_hit_count > 0
                            else None
                        ),
                        "recall_final": (
                            float(tris_mesh_topk_hits / tris_mesh_topk_hit_count)
                            if tris_mesh_topk_hit_count > 0
                            else None
                        ),
                        "precision_conditional_selector": (
                            float(
                                tris_mesh_conditional_topk_num
                                / tris_mesh_conditional_topk_den
                            )
                            if tris_mesh_conditional_topk_den > 0
                            else None
                        ),
                        "attrition_pre_to_guardrails": (
                            float(
                                np.mean(
                                    np.asarray(
                                        tris_mesh_attrition_pre_to_guardrails,
                                        dtype=np.float64,
                                    )
                                )
                            )
                            if tris_mesh_attrition_pre_to_guardrails
                            else None
                        ),
                        "attrition_guardrails_to_topk": (
                            float(
                                np.mean(
                                    np.asarray(
                                        tris_mesh_attrition_guardrails_to_topk,
                                        dtype=np.float64,
                                    )
                                )
                            )
                            if tris_mesh_attrition_guardrails_to_topk
                            else None
                        ),
                        "attrition_total": (
                            float(
                                np.mean(
                                    np.asarray(
                                        tris_mesh_attrition_total,
                                        dtype=np.float64,
                                    )
                                )
                            )
                            if tris_mesh_attrition_total
                            else None
                        ),
                    }
            else:
                tris_prob_summary = {
                    "logloss": (
                        prob_metric_sums["logloss"] / prob_metric_count
                        if prob_metric_count
                        else None
                    ),
                    "brier": (
                        prob_metric_sums["brier"] / prob_metric_count
                        if prob_metric_count
                        else None
                    ),
                    "ece": (
                        prob_metric_sums["ece"] / prob_metric_count
                        if prob_metric_count
                        else None
                    ),
                    "count": prob_metric_count,
                }
                top_positive_outliers = sorted(
                    (
                        row
                        for row in delta_ll_details
                        if float(row.get("delta_ll", 0.0)) > 0.0
                    ),
                    key=lambda x: float(x["delta_ll"]),
                    reverse=True,
                )[:10]
                if top_positive_outliers:
                    csv_path = ""
                    if not skip_outlier_csv:
                        csv_path = self._dump_tris_outliers_csv(
                            tracking_ctx.get("event_id", "unknown"), top_positive_outliers
                        )
                    tris_outlier_summary = {
                        "rows": top_positive_outliers,
                        "csv_path": csv_path,
                    }
                if not skip_bootstrap_ci:
                    ll_delta_stats = self._bootstrap_mean_ci(
                        delta_ll_values, n_resamples=2000
                    )
                    br_delta_stats = self._bootstrap_mean_ci(
                        delta_br_values, n_resamples=2000
                    )
                    ll_delta_debug = self._delta_distribution_debug(
                        delta_ll_values, draw_ids=delta_ll_draw_ids, top_k=10
                    )
                    if ll_delta_stats is not None and ll_delta_debug is not None:
                        ll_delta_stats.update(ll_delta_debug)
                    tris_delta_summary = {
                        "logloss": ll_delta_stats,
                        "brier": br_delta_stats,
                    }
                if baseline_metric_count:
                    baseline_prob_summary = {
                        "logloss": baseline_metric_sums["logloss"] / baseline_metric_count,
                        "brier": baseline_metric_sums["brier"] / baseline_metric_count,
                        "ece": baseline_metric_sums["ece"] / baseline_metric_count,
                        "count": baseline_metric_count,
                    }
                if baseline_compare_stats is not None:
                    draws_eval = int(baseline_compare_stats["draws_evaluated"])
                    model_tickets_total = int(
                        baseline_compare_stats["model_tickets_total"]
                    )
                    random_tickets_total = int(
                        baseline_compare_stats["random_tickets_total"]
                    )
                    model_exact_hits = int(baseline_compare_stats["model_exact_hits"])
                    random_exact_hits = int(baseline_compare_stats["random_exact_hits"])
                    model_draw_hits = int(baseline_compare_stats["model_draw_hits"])
                    random_draw_hits = int(baseline_compare_stats["random_draw_hits"])
                    baseline_compare_summary = {
                        "hit_label": f"{max_hits}/{max_hits}",
                        "draws_evaluated": draws_eval,
                        "model_exact_hits": model_exact_hits,
                        "model_hit_rate": (
                            float(model_exact_hits / model_tickets_total)
                            if model_tickets_total > 0
                            else None
                        ),
                        "model_draw_hit_rate": (
                            float(model_draw_hits / draws_eval)
                            if draws_eval > 0
                            else None
                        ),
                        "random_exact_hits": random_exact_hits,
                        "random_hit_rate": (
                            float(random_exact_hits / random_tickets_total)
                            if random_tickets_total > 0
                            else None
                        ),
                        "random_draw_hit_rate": (
                            float(random_draw_hits / draws_eval)
                            if draws_eval > 0
                            else None
                        ),
                        "random_errors": int(baseline_compare_stats["random_errors"]),
                    }
        if is_reduction_only:
            self._print_reduction_summary(
                res, reduced_sizes, max_hits_by_draw, max_hits
            )
        else:
            self._print_final_report(
                res,
                jackpot_coverage,
                max_hits,
                expects_universe_coverage,
                has_universe_data,
                tris_prob_summary=tris_prob_summary,
                tris_delta_summary=tris_delta_summary,
                tris_outlier_summary=tris_outlier_summary,
                tris_universe_summary=tris_universe_summary,
                tris_positional_summary=tris_positional_summary,
                tris_layered_mesh_summary=tris_layered_mesh_summary,
                baseline_prob_summary=baseline_prob_summary,
                baseline_compare_summary=baseline_compare_summary,
                fast_mode_summary=fast_mode_summary,
            )
        self.tracker.log_run(res, VERSION_TAG, self.forensic_data)
        return res

    def _render_telemetry(self, audit, t_id, t_s, max_hits, sniper_msg=""):
        d, r, h = (
            audit.get("proximity", 999),
            audit.get("rank", 0),
            audit.get("hits", 0),
        )
        ai_s = audit.get("ai_score", 0.0)
        ai_enabled = bool(audit.get("ai_signal_enabled", True))
        ai_validated = bool(audit.get("ai_signal_validated", True))
        ai_percentile = audit.get("ai_percentile_rank")
        ai_weight = float(audit.get("ai_weight_effective", 0.0))
        geo_weight = float(audit.get("geo_weight_effective", 0.0))
        geo_s = audit.get("geo_score", 0.0)
        u_s = audit.get("univ_size", 0)

        st_c = "bold green" if d == 0 else "bold red"
        h_c = (
            "bold green"
            if h == max_hits
            else "bold yellow"
            if h == max(0, max_hits - 1)
            else "cyan"
            if h == max(0, max_hits - 2)
            else "white"
        )
        d_c = "bold green" if d == 0 else "bold yellow" if d < 50 else "white"
        status = "🎯 HIT" if d == 0 else "❌"

        if ai_enabled:
            percentile_cell = (
                f" p{float(ai_percentile):02.0f}"
                if ai_percentile is not None
                else ""
            )
            validation_cell = " NV" if not ai_validated else ""
            ai_cell = f"{ai_s:.3f}{percentile_cell}{validation_cell}"
        else:
            ai_cell = "OFF"
        mix_cell = (
            f"{int(round(ai_weight * 100)):02d}/{int(round(geo_weight * 100)):02d}"
            if ai_enabled
            else "00/100"
        )
        line = (
            f"[bold blue]#{t_id}[/] | "
            f"U: {u_s:>6,d} | "
            f"[{h_c}]{h}/{max_hits}[/] | "
            f"AIr: [bold yellow]{ai_cell:>11}[/] | "
            f"Geo: [bold cyan]{geo_s:.4f}[/] | "
            f"Mix: {mix_cell} | "
            f"Rank: #{r:<5} | "
            f"Dist: [{d_c}]{d:<4}[/] | "
            f"[{st_c}]{status}[/] | [dim]{time.time()-t_s:.2f}s[/dim]"
        )

        if sniper_msg:
            line += f" | [cyan]{sniper_msg}[/]"

        self.console.print(line)

    @staticmethod
    def _fmt_metric(value):
        if value is None:
            return "[bold yellow]N/A[/]"
        try:
            val = float(value)
        except (TypeError, ValueError):
            return "[bold yellow]N/A[/]"
        if not np.isfinite(val):
            return "[bold yellow]N/A[/]"
        return f"{val:.6f}"

    @staticmethod
    def _bootstrap_mean_ci(values, n_resamples: int = 2000):
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None

        mean = float(np.mean(arr))
        std = float(np.std(arr))
        n_resamples = max(1, int(n_resamples))
        rng = np.random.default_rng(20260221)
        idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
        boot_means = np.mean(arr[idx], axis=1)
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        p_lt_zero = float(np.sum(arr < 0.0) / arr.size)

        return {
            "mean": mean,
            "std": std,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "p_lt_zero": p_lt_zero,
            "count": int(arr.size),
        }

    @staticmethod
    def _bootstrap_block_mean_ci(
        values, block_size: int | None = None, n_resamples: int = 2000
    ):
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        n = int(arr.size)
        if n == 0:
            return None

        if block_size is None:
            block_size = max(2, int(np.sqrt(n)))
        block_size = int(max(1, min(int(block_size), n)))
        n_resamples = max(1, int(n_resamples))
        mean = float(np.mean(arr))
        std = float(np.std(arr))

        rng = np.random.default_rng(20260222)
        n_blocks = int(np.ceil(n / block_size))
        steps = np.arange(block_size, dtype=np.int64)
        boot_means = np.empty(n_resamples, dtype=np.float64)
        for j in range(n_resamples):
            starts = rng.integers(0, n, size=n_blocks, endpoint=False)
            idx = (starts[:, None] + steps[None, :]) % n
            sample = arr[idx.reshape(-1)[:n]]
            boot_means[j] = float(np.mean(sample))
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

        return {
            "mean": mean,
            "std": std,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "count": n,
            "block_size": int(block_size),
        }

    @staticmethod
    def _delta_distribution_debug(values, draw_ids=None, top_k: int = 10):
        arr_raw = np.asarray(values, dtype=np.float64)
        finite_mask = np.isfinite(arr_raw)
        arr = arr_raw[finite_mask]
        if arr.size == 0:
            return None

        ids = None
        if draw_ids is not None:
            ids_raw = np.asarray(draw_ids)
            if ids_raw.shape[0] == arr_raw.shape[0]:
                ids = ids_raw[finite_mask]

        n_neg = int(np.sum(arr < 0.0))
        n_pos = int(np.sum(arr > 0.0))
        n_zero = int(arr.size - n_neg - n_pos)

        q_min, q05, q50, q95, q_max = np.percentile(arr, [0, 5, 50, 95, 100])
        neg_vals = arr[arr < 0.0]
        pos_vals = arr[arr > 0.0]

        top_positive = []
        if ids is not None:
            order = np.argsort(arr)[::-1]
            limit = max(1, int(top_k))
            for idx in order:
                delta_val = float(arr[idx])
                if delta_val <= 0.0:
                    break
                top_positive.append(
                    {"draw_id": int(ids[idx]), "delta": delta_val}
                )
                if len(top_positive) >= limit:
                    break

        return {
            "delta_definition": "model-baseline",
            "n_neg": n_neg,
            "n_pos": n_pos,
            "n_zero": n_zero,
            "q_min": float(q_min),
            "q05": float(q05),
            "q50": float(q50),
            "q95": float(q95),
            "q_max": float(q_max),
            "mean_neg": float(np.mean(neg_vals)) if neg_vals.size else None,
            "mean_pos": float(np.mean(pos_vals)) if pos_vals.size else None,
            "top_positive": top_positive,
        }

    @staticmethod
    def _format_float_list(values):
        if not isinstance(values, (list, tuple)):
            return ""
        return "[" + ", ".join(f"{float(v):.6f}" for v in values) + "]"

    def _dump_tris_outliers_csv(self, event_id: str, rows):
        safe_event_id = str(event_id or "unknown")
        out_dir = os.path.join("artifacts", "tris")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"outliers_{safe_event_id}.csv")

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "draw_id",
                    "delta_ll",
                    "y_digits",
                    "prev_digits",
                    "p_true_per_pos",
                    "max_prob_per_pos",
                    "entropy_per_pos",
                ]
            )
            for row in rows or []:
                writer.writerow(
                    [
                        int(row.get("draw_id", 0)),
                        f"{float(row.get('delta_ll', 0.0)):.12f}",
                        " ".join(str(int(x)) for x in row.get("y_digits", [])),
                        " ".join(str(int(x)) for x in row.get("prev_digits", [])),
                        " ".join(
                            f"{float(x):.12f}" for x in row.get("p_true_per_pos", [])
                        ),
                        " ".join(
                            f"{float(x):.12f}"
                            for x in row.get("max_prob_per_pos", [])
                        ),
                        " ".join(
                            f"{float(x):.12f}" for x in row.get("entropy_per_pos", [])
                        ),
                    ]
                )
        return out_path

    def _print_final_report(
        self,
        res,
        jackpot_coverage,
        max_hits,
        expects_universe_coverage,
        has_universe_data,
        tris_prob_summary=None,
        tris_delta_summary=None,
        tris_outlier_summary=None,
        tris_universe_summary=None,
        tris_positional_summary=None,
        tris_layered_mesh_summary=None,
        baseline_prob_summary=None,
        baseline_compare_summary=None,
        fast_mode_summary=None,
    ):
        self.console.print("\n[bold green]📊 REPORTE FINAL DE MISIÓN[/bold green]")
        summary = Table(show_header=True, header_style="bold magenta")
        summary.add_column("Métrica Sniper", style="dim", width=20)
        summary.add_column("Valor", justify="right", width=15)
        summary.add_row("Sorteos Analizados", str(res.total_draws_tested))
        summary.add_row(
            "Balance Neto",
            f"[{'green' if res.net_balance >= 0 else 'red'}]${res.net_balance:,.2f}[/]",
        )
        if expects_universe_coverage and not has_universe_data:
            jackpot_value = "[bold yellow]N/A[/]"
        elif expects_universe_coverage:
            jackpot_value = f"[bold yellow]{jackpot_coverage}[/]"
        else:
            jackpot_value = "[bold yellow]0[/]"
        summary.add_row("Jackpots en Universo", jackpot_value)
        self.console.print(summary)
        if isinstance(fast_mode_summary, dict):
            omitted = fast_mode_summary.get("omitted_modules", [])
            if omitted or bool(fast_mode_summary.get("enabled", False)):
                omitted_text = ", ".join(str(v) for v in omitted) if omitted else "none"
                self.console.print(
                    "[yellow]Fast Mode:[/] "
                    f"enabled={bool(fast_mode_summary.get('enabled', False))} | "
                    f"omitted={omitted_text} | "
                    f"render_every_n={int(fast_mode_summary.get('render_every_n', 1))} | "
                    "gpu_cleanup_every_n="
                    f"{int(fast_mode_summary.get('gpu_pool_cleanup_every_n', 1))}"
                )

        dist_table = Table(
            title="Distribución de Aciertos",
            show_header=True,
            header_style="bold cyan",
        )
        dist_table.add_column("Rango", justify="center")
        dist_table.add_column("Tickets", justify="right")
        for h in range(max_hits + 1):
            count = res.hit_distribution.get(h, 0)
            style = "bold yellow" if h >= max(0, max_hits - 2) else "white"
            dist_table.add_row(f"{h}/{max_hits} aciertos", f"[{style}]{count}[/]")
        self.console.print(dist_table)

        if isinstance(tris_universe_summary, dict):
            u_table = Table(
                title="Tris Universe-Only (Solo Filtros)",
                show_header=True,
                header_style="bold magenta",
            )
            u_table.add_column("Métrica", justify="left")
            u_table.add_column("Valor", justify="right")
            u_table.add_row(
                "Avg U (universo filtrado)",
                f"{float(tris_universe_summary.get('avg_u', 0.0)):.2f}",
            )
            u_table.add_row(
                "Min U",
                str(int(tris_universe_summary.get("min_u", 0))),
            )
            u_table.add_row(
                "Max U",
                str(int(tris_universe_summary.get("max_u", 0))),
            )
            u_table.add_row(
                "FS-pass rate",
                f"{100.0 * float(tris_universe_summary.get('fs_pass_rate', 0.0)):.2f}%",
            )
            structural_flags = tris_universe_summary.get("structural_flags_effective")
            if isinstance(structural_flags, dict):
                u_table.add_row(
                    "Structural flags effective",
                    ", ".join(f"{k}={v}" for k, v in structural_flags.items()),
                )
            self.console.print(u_table)

            fail_counts = tris_universe_summary.get("winner_fail_reasons", {})
            fail_table = Table(
                title="Fail Reasons (Winner fuera de filtros)",
                show_header=True,
                header_style="bold yellow",
            )
            fail_table.add_column("Reason", justify="left")
            fail_table.add_column("Count", justify="right")
            for key in (
                "sum",
                "parity",
                "uniques",
                "consecutive",
                "mirror_prev",
                "score_topk",
            ):
                fail_table.add_row(key, str(int(fail_counts.get(key, 0))))
            self.console.print(fail_table)

            compare_summary = tris_universe_summary.get("compare_models")
            if isinstance(compare_summary, dict):
                model_a_name = str(
                    compare_summary.get("model_a_name", "feature_lr")
                    or "feature_lr"
                )
                model_b_name = str(
                    compare_summary.get("model_b_name", "random_topk")
                    or "random_topk"
                )
                cmp_table = Table(
                    title=f"Compare Models ({model_a_name} vs {model_b_name})",
                    show_header=True,
                    header_style="bold cyan",
                )
                cmp_table.add_column("Métrica", justify="left")
                cmp_table.add_column("Valor", justify="right")
                cmp_table.add_row(
                    "FS_lr",
                    f"{100.0 * float(compare_summary.get('fs_lr', 0.0)):.2f}%",
                )
                cmp_table.add_row(
                    "FS_rand",
                    f"{100.0 * float(compare_summary.get('fs_rand', 0.0)):.2f}%",
                )
                cmp_table.add_row(
                    "Delta (FS_lr - FS_rand)",
                    f"{float(compare_summary.get('delta', 0.0)):.6f}",
                )
                cmp_table.add_row("b (1,0)", str(int(compare_summary.get("b", 0))))
                cmp_table.add_row("c (0,1)", str(int(compare_summary.get("c", 0))))
                mcnemar_chi2 = compare_summary.get("mcnemar_chi2")
                cmp_table.add_row(
                    "McNemar chi2",
                    self._fmt_metric(mcnemar_chi2),
                )
                delta_ci = compare_summary.get("delta_ci")
                if isinstance(delta_ci, dict):
                    cmp_table.add_row(
                        "Delta CI 95% (block)",
                        f"[{float(delta_ci.get('ci_low', 0.0)):.6f}, {float(delta_ci.get('ci_high', 0.0)):.6f}]",
                    )
                    cmp_table.add_row(
                        "Block size",
                        str(int(delta_ci.get("block_size", 0))),
                    )
                cmp_table.add_row(
                    "Avg U LR",
                    f"{float(compare_summary.get('avg_u_lr', 0.0)):.2f}",
                )
                cmp_table.add_row(
                    "Avg U Rand",
                    f"{float(compare_summary.get('avg_u_rand', 0.0)):.2f}",
                )
                self.console.print(cmp_table)

        if isinstance(tris_positional_summary, dict):
            pos_table = Table(
                title="Hit Rate por Posicion (Camaras)",
                show_header=True,
                header_style="bold cyan",
            )
            pos_table.add_column("Camara", justify="left")
            pos_table.add_column("Top1 hit-rate", justify="right")
            pos_table.add_column("Mask coverage", justify="right")
            pos_table.add_column("Universe pos coverage", justify="right")

            top1_rates = tris_positional_summary.get("top1_hit_rate_by_pos", [])
            mask_rates = tris_positional_summary.get("mask_coverage_rate_by_pos", [])
            univ_rates = tris_positional_summary.get(
                "universe_position_coverage_rate_by_pos", []
            )
            weight_rates = tris_positional_summary.get("camera_weight_avg_by_pos", [])
            target_cov_rates = tris_positional_summary.get(
                "camera_target_coverage_avg_by_pos", []
            )
            volatility_rates = tris_positional_summary.get(
                "camera_volatility_avg_by_pos", []
            )
            has_weight = any(v is not None for v in weight_rates)
            has_target_cov = any(v is not None for v in target_cov_rates)
            has_volatility = any(v is not None for v in volatility_rates)
            if has_weight:
                pos_table.add_column("Weight", justify="right")
            if has_target_cov:
                pos_table.add_column("Target coverage", justify="right")
            if has_volatility:
                pos_table.add_column("Volatility", justify="right")

            def _fmt_rate(v):
                if v is None:
                    return "[bold yellow]N/A[/]"
                try:
                    val = float(v)
                except (TypeError, ValueError):
                    return "[bold yellow]N/A[/]"
                if not np.isfinite(val):
                    return "[bold yellow]N/A[/]"
                return f"{100.0 * val:.2f}%"

            for pos in range(5):
                t1 = top1_rates[pos] if pos < len(top1_rates) else None
                mk = mask_rates[pos] if pos < len(mask_rates) else None
                uv = univ_rates[pos] if pos < len(univ_rates) else None
                row = [
                    f"Camara {pos + 1}",
                    _fmt_rate(t1),
                    _fmt_rate(mk),
                    _fmt_rate(uv),
                ]
                if has_weight:
                    wt = weight_rates[pos] if pos < len(weight_rates) else None
                    row.append(self._fmt_metric(wt))
                if has_target_cov:
                    tc = target_cov_rates[pos] if pos < len(target_cov_rates) else None
                    row.append(_fmt_rate(tc))
                if has_volatility:
                    vol = volatility_rates[pos] if pos < len(volatility_rates) else None
                    row.append(self._fmt_metric(vol))
                pos_table.add_row(*row)
            self.console.print(pos_table)

        if isinstance(tris_layered_mesh_summary, dict):
            mesh_table = Table(
                title="Layered Mesh Telemetry",
                show_header=True,
                header_style="bold cyan",
            )
            mesh_table.add_column("Métrica", justify="left")
            mesh_table.add_column("Valor", justify="right")

            def _fmt_mesh_size(v):
                if v is None:
                    return "[bold yellow]N/A[/]"
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    return "[bold yellow]N/A[/]"
                if not np.isfinite(fv):
                    return "[bold yellow]N/A[/]"
                return f"{fv:.2f}"

            def _fmt_mesh_rate(v):
                if v is None:
                    return "[bold yellow]N/A[/]"
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    return "[bold yellow]N/A[/]"
                if not np.isfinite(fv):
                    return "[bold yellow]N/A[/]"
                return f"{100.0 * fv:.2f}%"

            mesh_table.add_row(
                "Avg pre-mesh U",
                _fmt_mesh_size(tris_layered_mesh_summary.get("avg_pre_mesh_u")),
            )
            mesh_table.add_row(
                "Avg post-guardrails U",
                _fmt_mesh_size(tris_layered_mesh_summary.get("avg_post_guardrails_u")),
            )
            mesh_table.add_row(
                "Avg post-topK U",
                _fmt_mesh_size(tris_layered_mesh_summary.get("avg_post_topk_u")),
            )
            mesh_table.add_row(
                "Recall capa 1 (winner_in_mesh_pre)",
                _fmt_mesh_rate(tris_layered_mesh_summary.get("recall_layer_1")),
            )
            mesh_table.add_row(
                "Recall capa 2 (winner_in_mesh_post_guardrails)",
                _fmt_mesh_rate(tris_layered_mesh_summary.get("recall_layer_2")),
            )
            mesh_table.add_row(
                "Recall final (FS-pass)",
                _fmt_mesh_rate(tris_layered_mesh_summary.get("recall_final")),
            )
            mesh_table.add_row(
                "Precision condicional selector",
                _fmt_mesh_rate(
                    tris_layered_mesh_summary.get("precision_conditional_selector")
                ),
            )
            attrition_line = (
                f"L1->L2: {_fmt_mesh_rate(tris_layered_mesh_summary.get('attrition_pre_to_guardrails'))}, "
                f"L2->TopK: {_fmt_mesh_rate(tris_layered_mesh_summary.get('attrition_guardrails_to_topk'))}, "
                f"Total: {_fmt_mesh_rate(tris_layered_mesh_summary.get('attrition_total'))}"
            )
            mesh_table.add_row(
                "Attrition por capa (% reducción de universo)",
                attrition_line,
            )
            self.console.print(mesh_table)

        if isinstance(tris_prob_summary, dict):
            prob_table = Table(
                title="Métricas Probabilísticas Tris",
                show_header=True,
                header_style="bold green",
            )
            prob_table.add_column("Métrica", justify="left")
            prob_table.add_column("Modelo", justify="right")
            prob_table.add_row(
                "Avg LogLoss (positional)",
                self._fmt_metric(tris_prob_summary.get("logloss")),
            )
            prob_table.add_row(
                "Avg Brier (positional)",
                self._fmt_metric(tris_prob_summary.get("brier")),
            )
            prob_table.add_row(
                "Avg ECE (positional)",
                self._fmt_metric(tris_prob_summary.get("ece")),
            )
            self.console.print(prob_table)

            if isinstance(tris_delta_summary, dict):
                ll_stats = (
                    tris_delta_summary.get("logloss")
                    if isinstance(tris_delta_summary.get("logloss"), dict)
                    else None
                )
                br_stats = (
                    tris_delta_summary.get("brier")
                    if isinstance(tris_delta_summary.get("brier"), dict)
                    else None
                )
                delta_table = Table(
                    title="Delta Tris vs Baseline Uniforme (Constante)",
                    show_header=True,
                    header_style="bold blue",
                )
                delta_table.add_column("Métrica", justify="left")
                delta_table.add_column("Valor", justify="right")

                if ll_stats is not None:
                    delta_table.add_row(
                        "Delta LogLoss mean ± std",
                        f"{ll_stats['mean']:.6f} ± {ll_stats['std']:.6f}",
                    )
                    delta_table.add_row(
                        "Delta LogLoss 95% CI",
                        f"[{ll_stats['ci_low']:.6f}, {ll_stats['ci_high']:.6f}]",
                    )
                    delta_table.add_row(
                        "P(delta LogLoss < 0)",
                        f"{ll_stats['p_lt_zero']:.4f}",
                    )
                    delta_table.add_row(
                        "Delta LogLoss definition",
                        str(ll_stats.get("delta_definition", "model-baseline")),
                    )
                    delta_table.add_row(
                        "Delta LogLoss counts (neg/pos/zero)",
                        f"{int(ll_stats.get('n_neg', 0))}/"
                        f"{int(ll_stats.get('n_pos', 0))}/"
                        f"{int(ll_stats.get('n_zero', 0))}",
                    )
                    delta_table.add_row(
                        "Delta LogLoss quantiles min/p05/p50/p95/max",
                        f"{float(ll_stats.get('q_min', np.nan)):.6f} / "
                        f"{float(ll_stats.get('q05', np.nan)):.6f} / "
                        f"{float(ll_stats.get('q50', np.nan)):.6f} / "
                        f"{float(ll_stats.get('q95', np.nan)):.6f} / "
                        f"{float(ll_stats.get('q_max', np.nan)):.6f}",
                    )
                    delta_table.add_row(
                        "Delta LogLoss mean_neg",
                        self._fmt_metric(ll_stats.get("mean_neg")),
                    )
                    delta_table.add_row(
                        "Delta LogLoss mean_pos",
                        self._fmt_metric(ll_stats.get("mean_pos")),
                    )
                else:
                    delta_table.add_row("Delta LogLoss mean ± std", "[bold yellow]N/A[/]")
                    delta_table.add_row("Delta LogLoss 95% CI", "[bold yellow]N/A[/]")
                    delta_table.add_row("P(delta LogLoss < 0)", "[bold yellow]N/A[/]")

                if br_stats is not None:
                    delta_table.add_row(
                        "Delta Brier mean ± std",
                        f"{br_stats['mean']:.6f} ± {br_stats['std']:.6f}",
                    )
                    delta_table.add_row(
                        "Delta Brier 95% CI",
                        f"[{br_stats['ci_low']:.6f}, {br_stats['ci_high']:.6f}]",
                    )
                else:
                    delta_table.add_row("Delta Brier mean ± std", "[bold yellow]N/A[/]")
                    delta_table.add_row("Delta Brier 95% CI", "[bold yellow]N/A[/]")

                self.console.print(delta_table)

                if ll_stats is not None:
                    top_positive = ll_stats.get("top_positive", [])
                    top_table = Table(
                        title="Top 10 Delta LogLoss Positivos (draw_id)",
                        show_header=True,
                        header_style="bold cyan",
                    )
                    top_table.add_column("Draw ID", justify="right")
                    top_table.add_column("Delta LogLoss", justify="right")
                    if isinstance(top_positive, list) and top_positive:
                        for item in top_positive[:10]:
                            draw_id = int(item.get("draw_id", 0))
                            delta_val = float(item.get("delta", 0.0))
                            top_table.add_row(str(draw_id), f"{delta_val:.6f}")
                    else:
                        top_table.add_row("-", "N/A")
                    self.console.print(top_table)

                if isinstance(tris_outlier_summary, dict):
                    outlier_rows = tris_outlier_summary.get("rows", [])
                    csv_path = tris_outlier_summary.get("csv_path", "")
                    outlier_table = Table(
                        title="Top 10 delta_ll positivos con detalle",
                        show_header=True,
                        header_style="bold green",
                    )
                    outlier_table.add_column("draw_id", justify="right")
                    outlier_table.add_column("delta_ll", justify="right")
                    outlier_table.add_column("y_digits", justify="left")
                    outlier_table.add_column("prev_digits", justify="left")
                    outlier_table.add_column("p_true_per_pos", justify="left")
                    outlier_table.add_column("max_prob_per_pos", justify="left")
                    outlier_table.add_column("entropy_per_pos", justify="left")
                    if outlier_rows:
                        for row in outlier_rows:
                            outlier_table.add_row(
                                str(int(row.get("draw_id", 0))),
                                f"{float(row.get('delta_ll', 0.0)):.6f}",
                                str(row.get("y_digits", [])),
                                str(row.get("prev_digits", [])),
                                self._format_float_list(row.get("p_true_per_pos", [])),
                                self._format_float_list(
                                    row.get("max_prob_per_pos", [])
                                ),
                                self._format_float_list(row.get("entropy_per_pos", [])),
                            )
                    else:
                        outlier_table.add_row("-", "N/A", "[]", "[]", "[]", "[]", "[]")
                    self.console.print(outlier_table)
                    if csv_path:
                        self.console.print(
                            f"[cyan]Outliers CSV:[/] [white]{csv_path}[/]"
                        )

            if isinstance(baseline_prob_summary, dict):
                cmp_table = Table(
                    title="Comparativa vs Baseline Uniforme",
                    show_header=True,
                    header_style="bold yellow",
                )
                cmp_table.add_column("Métrica", justify="left")
                cmp_table.add_column("Baseline", justify="right")
                cmp_table.add_column("Delta (modelo-baseline)", justify="right")
                for key, label in (
                    ("logloss", "LogLoss"),
                    ("brier", "Brier"),
                    ("ece", "ECE"),
                ):
                    model_val = tris_prob_summary.get(key)
                    base_val = baseline_prob_summary.get(key)
                    delta = (
                        model_val - base_val
                        if (model_val is not None and base_val is not None)
                        else None
                    )
                    cmp_table.add_row(
                        label,
                        self._fmt_metric(base_val),
                        self._fmt_metric(delta),
                    )
                self.console.print(cmp_table)

            if isinstance(baseline_compare_summary, dict):
                cmp_hits_table = Table(
                    title="Comparativa Hits Exactos vs Baseline Aleatorio+Filtros",
                    show_header=True,
                    header_style="bold magenta",
                )
                cmp_hits_table.add_column("Métrica", justify="left")
                cmp_hits_table.add_column("Valor", justify="right")
                hit_label = str(
                    baseline_compare_summary.get("hit_label", f"{max_hits}/{max_hits}")
                )
                cmp_hits_table.add_row(
                    f"Modelo exactos {hit_label}",
                    str(int(baseline_compare_summary.get("model_exact_hits", 0))),
                )
                cmp_hits_table.add_row(
                    f"Modelo hit-rate {hit_label}",
                    self._fmt_metric(baseline_compare_summary.get("model_hit_rate")),
                )
                cmp_hits_table.add_row(
                    f"Modelo draw-hit-rate {hit_label}",
                    self._fmt_metric(
                        baseline_compare_summary.get("model_draw_hit_rate")
                    ),
                )
                cmp_hits_table.add_row(
                    f"Random+Filtros exactos {hit_label}",
                    str(int(baseline_compare_summary.get("random_exact_hits", 0))),
                )
                cmp_hits_table.add_row(
                    f"Random+Filtros hit-rate {hit_label}",
                    self._fmt_metric(baseline_compare_summary.get("random_hit_rate")),
                )
                cmp_hits_table.add_row(
                    f"Random+Filtros draw-hit-rate {hit_label}",
                    self._fmt_metric(
                        baseline_compare_summary.get("random_draw_hit_rate")
                    ),
                )
                cmp_hits_table.add_row(
                    "Random+Filtros errors",
                    str(int(baseline_compare_summary.get("random_errors", 0))),
                )
                self.console.print(cmp_hits_table)

    def _print_reduction_summary(self, res, reduced_sizes, max_hits_by_draw, max_hits):
        final_universe = int(reduced_sizes[-1]) if reduced_sizes else 0
        sorted_levels = sorted(max_hits_by_draw.keys())

        self.console.print(
            "\n[bold green]📊 RESUMEN REDUCCIÓN DE UNIVERSO[/bold green]"
        )
        summary = Table(show_header=True, header_style="bold magenta")
        summary.add_column("Métrica", style="dim", width=30)
        summary.add_column("Valor", justify="right", width=15)
        summary.add_row("Universo final reducido", f"[bold cyan]{final_universe:,}[/]")
        for h in sorted_levels:
            summary.add_row(
                f"Hits {h}/{max_hits}",
                f"[bold yellow]{int(max_hits_by_draw.get(h, 0))}[/]",
            )
        self.console.print(summary)

    def _render_reduction_telemetry(
        self, t_id, univ_size, max_hit, high_hits, high_hit_levels, max_hits, elapsed
    ):
        hits_line = " ".join(
            [
                f"{h}/{max_hits}: [yellow]{high_hits.get(h, 0)}[/]"
                for h in high_hit_levels
            ]
        )
        self.console.print(
            f"[bold blue]#{t_id}[/] | "
            f"U_final: [bold cyan]{int(univ_size):,}[/] | "
            f"Hits -> {hits_line} | "
            f"Max: [bold]{int(max_hit)}/{max_hits}[/] | "
            f"[dim]{elapsed:.2f}s[/dim]"
        )
