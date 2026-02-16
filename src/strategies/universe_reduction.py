# src/strategies/universe_reduction.py

import time
import numpy as np
from src.domain.dtos import PredictionResultDTO
from .universe.backend import UniverseBackend
from .universe.filters import VectorizedFilters
from src.data_access.config import BEST_SETTINGS


class UniverseReductionStrategy:
    """
    Sniper V15.2 - Density Penalized Survival
    45K fijo + Repulsión Estructural GPU
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
                return out
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
        compact = (
            sniper_base.replace("Sniper:", "S:")
            + f"|t{thr}"
            + (f"|sum{sum_min}-{sum_max}" if (sum_min or sum_max) else "")
            + (f"|std{std_min}-{std_max}" if (std_min or std_max) else "")
            + (f"|d{max_delta}" if max_delta else "")
            + (f"|ld{max_same_last}" if max_same_last else "")
            + f"|v{vdp_n}"
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
            strategy_name=f"Sniper V15.2 Density ({self.backend_name})",
            tickets=universe_cpu.tolist() if len(universe_cpu) > 0 else [],
        )

        res.metadata = {
            "raw_ndarray": universe,
            "final_size": len(universe),
            "sniper_log": sniper_log,
            "reduction_stage_stats": stage_stats,
        }

        return res

    def reduce(self, history, config, verbose=True):

        start_time = time.time()
        cfg = getattr(config, "filter_overrides", None) or BEST_SETTINGS

        if verbose:
            print(f"🚀 Sniper V15.2 [Backend: {self.backend_name}]")

        # SNIPER E1
        excluded_pool, sniper_msg = self.filters.get_sniper_exclusion(
            history,
            threshold=cfg.get("sniper_threshold", 0.85),
            weights=(
                cfg.get("w_gap", 0.25),
                cfg.get("w_term", 0.10),
                cfg.get("w_freq", 0.60),
            ),
            n_exclude=int(cfg.get("dynamic_exclude_count", 1)),
        )

        # ---- LOG (único cambio funcional): compact + 1 línea ----
        # Mantén el "base" de exclusión (Sniper:-N(score)) y agrega solo lo necesario para no wrappear.
        sniper_msg = self._build_sniper_log_compact(sniper_msg, cfg)
        # --------------------------------------------------------

        universe = self.filters.generate_universe(excluded_pool=excluded_pool)

        if len(universe) > 0:
            universe = self.filters.apply_positional_limits(universe, cfg)
            universe = self.filters.apply_aggregation(universe, cfg)
            universe = self.filters.apply_structure(universe, cfg)
            universe, _ = self.filters.apply_terminal_poda(universe, cfg)
            universe, d_vecs = self.filters.apply_spatial(universe, cfg)
            universe = self.filters.apply_profile_poda(universe, d_vecs, cfg)
            universe = self.filters.apply_entropy_shannon(universe, cfg)
            universe = self.filters.apply_digital_root_sum(universe, cfg)
            universe = self.filters.apply_ac_complexity(universe, cfg)

        target_k = 45000

        if len(universe) > target_k:
            universe = self._density_penalized_selection(universe, cfg, target_k)

        elapsed = time.time() - start_time

        if verbose:
            print(f"✅ UNIVERSO V15.2: {len(universe):,} ({elapsed:.2f}s)")

        return (
            universe,
            sniper_msg,
            {
                "final_size": len(universe),
                "execution_time": elapsed,
                "backend": self.backend_name,
            },
        )

    # ======================================================
    # DENSITY PENALIZED TOP-K
    # ======================================================
    def _density_penalized_selection(self, universe, cfg, target_k):

        xp = self.xp

        core_scores = self.filters.compute_survival_scores(universe, cfg)
        density_penalty = self.filters.compute_density_penalty(universe)

        lambda_penalty = 0.15

        final_scores = core_scores - lambda_penalty * density_penalty

        idx = xp.argpartition(-final_scores, target_k - 1)[:target_k]

        return universe[idx]
