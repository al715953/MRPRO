# src/strategies/universe_reduction.py

import time
import numpy as np
from src.domain.dtos import PredictionResultDTO
from .universe.backend import UniverseBackend
from .universe.filters import VectorizedFilters
from src.data_access.config import BEST_SETTINGS


class UniverseReductionStrategy:
    """
    Reductor V17 - núcleo suave con exploración uniforme reproducible.
    """

    def __init__(self):
        self.xp, self.backend_name = UniverseBackend.get_xp()
        self.filters = VectorizedFilters(self.xp)

    # ------------------------------
    # LOG HELPERS (solo log)
    # ------------------------------
    @staticmethod
    def _one_line(s: str) -> str:
        """Garantiza que el log jamás meta saltos de línea (para consola y CSV)."""
        if s is None:
            return ""
        s = str(s)
        s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        # colapsa espacios dobles
        while "  " in s:
            s = s.replace("  ", " ")
        return s.strip()

    @staticmethod
    def _fmt(v, nd: int = 2) -> str:
        """Formato compacto: floats con nd decimales, ints como int, None vacío."""
        if v is None:
            return ""
        try:
            if isinstance(v, bool):
                return "1" if v else "0"
            if isinstance(v, (int, np.integer)):
                return str(int(v))
            if isinstance(v, (float, np.floating)):
                # nd decimales, pero recorta ceros finales
                out = f"{float(v):.{nd}f}"
                out = out.rstrip("0").rstrip(".")
                return out or "0"
        except Exception:
            pass
        return str(v)

    @classmethod
    def _build_sniper_log_compact(cls, sniper_base: str, cfg: dict) -> str:
        """
        Log compacto pensado para NO wrappear en consola.
        Mantiene: exclusión, threshold, sum/std, delta, last-digit, vdp.
        """
        sniper_base = cls._one_line(sniper_base or "Sniper:None")

        thr = cls._fmt(cfg.get("sniper_threshold", 0.85), 2)
        sum_min = cls._fmt(cfg.get("sum_min", ""), 0)
        sum_max = cls._fmt(cfg.get("sum_max", ""), 0)
        std_min = cls._fmt(cfg.get("std_min", ""), 1)
        std_max = cls._fmt(cfg.get("std_max", ""), 1)
        max_delta = cls._fmt(cfg.get("max_delta", ""), 0)
        max_same_last = cls._fmt(cfg.get("max_same_last_digit", ""), 0)

        vdp = cfg.get("valid_decade_profiles", [])
        vdp_n = len(vdp) if isinstance(vdp, (list, tuple)) else 0

        # Ultra-compacto para que la línea completa (telemetría + log) quepa:
        # Ej: S:-16(0.95)|t.90|sum95-115|std7.5-13.2|d15|ld3|v6
        sum_marker = "" if bool(cfg.get("sum_filter_enabled", True)) else "~"
        std_hard = bool(cfg.get("std_filter_enabled", False)) or bool(
            cfg.get("auto_std_compensation", False)
        )
        std_marker = "" if std_hard else "~"
        selection_mode = str(cfg.get("candidate_selection_mode", "density"))
        exploration = cls._fmt(cfg.get("universe_exploration_fraction", 0.0), 2)
        radar = cls._fmt(cfg.get("radar_percentile", 50.0), 0)

        compact = (
            sniper_base.replace("Sniper:", "S:")
            + f"|t{thr}"
            + (
                f"|sum{sum_marker}{sum_min}-{sum_max}"
                if (sum_min or sum_max)
                else ""
            )
            + (
                f"|std{std_marker}{std_min}-{std_max}"
                if (std_min or std_max)
                else ""
            )
            + (f"|d{max_delta}" if max_delta else "")
            + (f"|ld{max_same_last}" if max_same_last else "")
            + f"|v{vdp_n}"
            + f"|sel:{selection_mode}"
            + (f"@{exploration}" if selection_mode == "balanced_mixed" else "")
            + f"|rad{radar}"
        )

        return cls._one_line(compact)

    def predict(self, history, config, verbose=True):

        universe, sniper_log, stage_stats = self.reduce(
            history, config, verbose=verbose
        )

        if hasattr(universe, "get"):
            universe_cpu = universe.get()
        else:
            universe_cpu = universe

        res = PredictionResultDTO(
            strategy_name=f"Universe V17 Balanced ({self.backend_name})",
            tickets=universe_cpu.tolist() if len(universe_cpu) > 0 else [],
        )

        res.metadata = {
            "raw_ndarray": universe,
            "final_size": len(universe),
            "sniper_log": sniper_log,
            "reduction_stage_stats": stage_stats,
            "sniper_mode": stage_stats.get("sniper_mode", "hard"),
            "sniper_candidates": stage_stats.get("sniper_candidates", []),
            "hard_excluded_numbers": stage_stats.get(
                "hard_excluded_numbers", []
            ),
            "universe_ticket_limit": stage_stats.get("universe_ticket_limit"),
            "candidate_selection_mode": stage_stats.get(
                "candidate_selection_mode", "density"
            ),
            "candidate_selection_seed": stage_stats.get(
                "candidate_selection_seed"
            ),
            "selection_core_count": stage_stats.get("selection_core_count", 0),
            "selection_exploration_count": stage_stats.get(
                "selection_exploration_count", 0
            ),
        }

        return res

    def reduce(self, history, config, verbose=True):

        start_time = time.time()
        runtime_overrides = getattr(config, "filter_overrides", None)
        cfg = runtime_overrides or BEST_SETTINGS
        std_filter_hard_enabled = bool(cfg.get("std_filter_enabled", False)) or bool(
            cfg.get("auto_std_compensation", False)
        )

        if verbose:
            print(f"🚀 Universe V17 Balanced [Backend: {self.backend_name}]")

        sniper_mode = str(cfg.get("sniper_mode", "hard")).strip().lower()
        if sniper_mode not in {"hard", "soft", "off"}:
            sniper_mode = "hard"
        sniper_threshold = float(cfg.get("sniper_threshold", 0.85))
        if bool(cfg.get("sniper_conservative", False)):
            sniper_threshold += max(
                0.0, float(cfg.get("sniper_threshold_boost", 0.0))
            )

        # SNIPER E1. En modo soft solamente produce una señal; no borra números.
        sniper_candidates, sniper_msg = self.filters.get_sniper_exclusion(
            history,
            threshold=sniper_threshold,
            weights=(
                cfg.get("w_gap", 0.25),
                cfg.get("w_term", 0.10),
                cfg.get("w_freq", 0.60),
            ),
            n_exclude=(
                0
                if sniper_mode == "off"
                else int(cfg.get("dynamic_exclude_count", 1))
            ),
        )
        excluded_pool = sniper_candidates if sniper_mode == "hard" else []
        # Contexto transitorio para que el selector pueda aplicar el veto suave en
        # producción y backtest sin acoplarse al reductor.
        if isinstance(runtime_overrides, dict) and runtime_overrides:
            runtime_overrides["sniper_soft_numbers"] = (
                [int(number) for number in sniper_candidates]
                if sniper_mode == "soft"
                else []
            )

        # ---- LOG (único cambio funcional): compact + 1 línea ----
        # Mantén el "base" de exclusión (Sniper:-N(score)) y agrega solo lo necesario para no wrappear.
        sniper_msg = self._build_sniper_log_compact(sniper_msg, cfg)
        # --------------------------------------------------------

        stage_sizes = []

        def apply_stage(name, operation, current):
            before = int(len(current))
            after_value = operation(current)
            after = int(len(after_value))
            stage_sizes.append(
                {
                    "stage": str(name),
                    "before": before,
                    "after": after,
                    "removed": before - after,
                }
            )
            return after_value

        universe = self.filters.generate_universe(excluded_pool=excluded_pool)
        generated_size = int(len(universe))

        if len(universe) > 0:
            universe = apply_stage(
                "positional",
                lambda value: self.filters.apply_positional_limits(value, cfg),
                universe,
            )
            universe = apply_stage(
                "sum",
                lambda value: self.filters.apply_aggregation(value, cfg),
                universe,
            )
            universe = apply_stage(
                "structure",
                lambda value: self.filters.apply_structure(value, cfg),
                universe,
            )
            before = int(len(universe))
            universe, _ = self.filters.apply_terminal_poda(universe, cfg)
            stage_sizes.append(
                {
                    "stage": "terminal",
                    "before": before,
                    "after": int(len(universe)),
                    "removed": before - int(len(universe)),
                }
            )
            before = int(len(universe))
            universe, d_vecs = self.filters.apply_spatial(universe, cfg)
            stage_sizes.append(
                {
                    "stage": "spatial",
                    "before": before,
                    "after": int(len(universe)),
                    "removed": before - int(len(universe)),
                }
            )
            universe = apply_stage(
                "decade_profile",
                lambda value: self.filters.apply_profile_poda(value, d_vecs, cfg),
                universe,
            )
            universe = apply_stage(
                "entropy",
                lambda value: self.filters.apply_entropy_shannon(value, cfg),
                universe,
            )
            universe = apply_stage(
                "digital_root",
                lambda value: self.filters.apply_digital_root_sum(value, cfg),
                universe,
            )
            universe = apply_stage(
                "ac_complexity",
                lambda value: self.filters.apply_ac_complexity(value, cfg),
                universe,
            )
            if bool(cfg.get("std_filter_enabled", False)) or bool(
                cfg.get("auto_std_compensation", False)
            ):
                universe = apply_stage(
                    "standard_deviation",
                    lambda value: self.filters.apply_standard_deviation(value, cfg),
                    universe,
                )

        legacy_target = cfg.get("target_universe_size", 0)
        configured_limit = cfg.get("universe_ticket_limit", 45000)
        try:
            legacy_target = int(legacy_target or 0)
            configured_limit = int(configured_limit or 0)
        except (TypeError, ValueError):
            legacy_target = 0
            configured_limit = 0
        if legacy_target > 0:
            configured_limit = legacy_target
        target_k = configured_limit if configured_limit > 0 else 45000

        topk_applied = False
        selection_mode = str(
            cfg.get("candidate_selection_mode", "density")
        ).strip().lower()
        selection_seed = self._candidate_selection_seed(history, cfg)
        selection_details = {
            "mode": selection_mode,
            "seed": selection_seed,
            "core_count": int(len(universe)),
            "exploration_count": 0,
        }
        if len(universe) > target_k:
            before = int(len(universe))
            if selection_mode == "balanced_mixed":
                universe, selection_details = self._balanced_mixed_selection(
                    universe,
                    cfg,
                    target_k,
                    selection_seed,
                )
            else:
                selection_mode = "density"
                universe = self._density_penalized_selection(universe, cfg, target_k)
                selection_details = {
                    "mode": selection_mode,
                    "seed": selection_seed,
                    "core_count": int(len(universe)),
                    "exploration_count": 0,
                }
            topk_applied = True
            stage_sizes.append(
                {
                    "stage": (
                        "balanced_topk"
                        if selection_mode == "balanced_mixed"
                        else "density_topk"
                    ),
                    "before": before,
                    "after": int(len(universe)),
                    "removed": before - int(len(universe)),
                }
            )

        elapsed = time.time() - start_time

        if verbose:
            print(f"✅ UNIVERSO V17: {len(universe):,} ({elapsed:.2f}s)")

        return (
            universe,
            sniper_msg,
            {
                "final_size": len(universe),
                "execution_time": elapsed,
                "backend": self.backend_name,
                "generated_size": generated_size,
                "stages": stage_sizes,
                "sniper_mode": sniper_mode,
                "sniper_threshold_effective": sniper_threshold,
                "sniper_candidates": [int(number) for number in sniper_candidates],
                "hard_excluded_numbers": [int(number) for number in excluded_pool],
                "universe_ticket_limit": int(target_k),
                "topk_applied": bool(topk_applied),
                "candidate_selection_mode": selection_mode,
                "candidate_selection_seed": int(selection_seed),
                "selection_core_count": int(selection_details["core_count"]),
                "selection_exploration_count": int(
                    selection_details["exploration_count"]
                ),
                "hard_filters_enabled": [
                    name
                    for name, enabled in (
                        ("positional", cfg.get("positional_filter_enabled", True)),
                        ("sum", cfg.get("sum_filter_enabled", True)),
                        ("structure", cfg.get("structure_filter_enabled", True)),
                        ("terminal", cfg.get("terminal_filter_enabled", True)),
                        ("spatial", cfg.get("spatial_filter_enabled", True)),
                        (
                            "decade_profile",
                            cfg.get("decade_profile_filter_enabled", True),
                        ),
                        ("entropy", cfg.get("entropy_filter_enabled", True)),
                        (
                            "digital_root",
                            cfg.get("digital_root_filter_enabled", True),
                        ),
                        ("ac_complexity", cfg.get("ac_filter_enabled", True)),
                        ("standard_deviation", std_filter_hard_enabled),
                    )
                    if bool(enabled)
                ],
            },
        )

    @staticmethod
    def _candidate_selection_seed(history, cfg):
        base = int(cfg.get("candidate_sampling_seed", 20260215))
        contests = getattr(history, "concursos", None) or []
        try:
            next_contest = int(contests[-1]) + 1 if contests else 1
        except (TypeError, ValueError):
            next_contest = len(contests) + 1
        return int((base + next_contest * 1_000_003) % (2**32))

    def _balanced_mixed_selection(self, universe, cfg, target_k, seed):
        """Blend a softly scored core with reproducible uniform exploration."""

        xp = self.xp
        n_candidates = int(len(universe))
        exploration_fraction = min(
            1.0,
            max(0.0, float(cfg.get("universe_exploration_fraction", 0.50))),
        )
        exploration_count = min(
            target_k, int(round(target_k * exploration_fraction))
        )
        core_count = target_k - exploration_count

        if core_count > 0:
            core_scores = self.filters.compute_survival_scores(universe, cfg)
            density_penalty = self.filters.compute_density_penalty(universe)
            penalty = max(
                0.0, float(cfg.get("density_penalty_strength", 0.15))
            )
            final_scores = core_scores - penalty * density_penalty
            core_indices = xp.argpartition(-final_scores, core_count - 1)[:core_count]
            core_cpu = (
                core_indices.get()
                if hasattr(core_indices, "get")
                else np.asarray(core_indices)
            )
            core_cpu = np.asarray(core_cpu, dtype=np.int64)
        else:
            core_cpu = np.empty(0, dtype=np.int64)

        available = np.ones(n_candidates, dtype=bool)
        available[core_cpu] = False
        remaining = np.flatnonzero(available)
        rng = np.random.default_rng(int(seed))
        if exploration_count > 0:
            exploration_cpu = rng.choice(
                remaining, size=exploration_count, replace=False
            ).astype(np.int64, copy=False)
        else:
            exploration_cpu = np.empty(0, dtype=np.int64)

        selected_cpu = np.sort(np.concatenate((core_cpu, exploration_cpu)))
        selected = xp.asarray(selected_cpu)
        return universe[selected], {
            "mode": "balanced_mixed",
            "seed": int(seed),
            "core_count": int(core_count),
            "exploration_count": int(exploration_count),
        }

    # ======================================================
    # DENSITY PENALIZED TOP-K
    # ======================================================
    def _density_penalized_selection(self, universe, cfg, target_k):

        xp = self.xp

        core_scores = self.filters.compute_survival_scores(universe, cfg)
        density_penalty = self.filters.compute_density_penalty(universe)

        lambda_penalty = max(
            0.0, float(cfg.get("density_penalty_strength", 0.15))
        )

        final_scores = core_scores - lambda_penalty * density_penalty

        idx = xp.argpartition(-final_scores, target_k - 1)[:target_k]

        return universe[idx]
