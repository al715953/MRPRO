# src/strategies/universe_reduction.py
import time
import numpy as np
from src.domain.dtos import PredictionResultDTO
from .universe.backend import UniverseBackend
from .universe.filters import VectorizedFilters
from src.data_access.config import BEST_SETTINGS


class UniverseReductionStrategy:
    """
    Coordinador Sniper V14.2 (Protocolo Magneto S2 Activo + Log Unificado).
    Integra la exclusión inteligente y preserva la cadena de filtros exitosa.
    """

    def __init__(self):
        self.xp, self.backend_name = UniverseBackend.get_xp()
        self.filters = VectorizedFilters(self.xp)

    def predict(self, history, config, verbose=True) -> PredictionResultDTO:
        # Capturamos el retorno doble: el universo filtrado y el mensaje del Sniper
        universe, sniper_log, stage_stats = self.reduce(history, config, verbose=verbose)

        # FORZADO DE TRANSFERENCIA GPU -> CPU (CRÍTICO PARA WINDOWS)
        if hasattr(universe, "get"):  # Si es CuPy
            universe_cpu = universe.get()
        else:
            universe_cpu = universe

        # Empaquetamos resultados
        res = PredictionResultDTO(
            strategy_name=f"Sniper V14.2 ({self.backend_name})",
            tickets=universe_cpu.tolist() if len(universe_cpu) > 0 else [],
        )

        # INYECCIÓN DE METADATA PARA EL LOG UNIFICADO
        # Ahora 'sniper_log' viaja con el objeto para ser impreso en la misma línea del reporte
        res.metadata = {
            "raw_ndarray": universe,
            "final_size": len(universe),
            "sniper_log": sniper_log,
            "reduction_stage_stats": stage_stats,
        }
        return res

    def reduce(self, history, config, verbose=True):
        """Lógica V7.17: Integra la exclusión dinámica inteligente."""
        start_time = time.time()

        cfg = getattr(config, "filter_overrides", None) or BEST_SETTINGS

        if verbose:
            print(f"🚀 Sniper V14.2 [Mac Mode: {self.backend_name}]")

        # --- ETAPA 1: EXCLUSIÓN QUIRÚRGICA (SNIPER E1) ---
        # Cambio Crítico: Usamos get_sniper_exclusion que retorna (pool, mensaje).
        # Esto nos permite saber qué pasó sin ensuciar la consola con prints extra.
        threshold = cfg.get("sniper_threshold", 0.85)
        conservative_sniper = bool(cfg.get("sniper_conservative", False))
        if conservative_sniper:
            threshold = max(0.0, min(1.0, threshold + cfg.get("sniper_threshold_boost", 0.08)))
        weights = (
            cfg.get("w_gap", BEST_SETTINGS.get("w_gap", 0.25)),
            cfg.get("w_term", BEST_SETTINGS.get("w_term", 0.10)),
            cfg.get("w_freq", BEST_SETTINGS.get("w_freq", 0.60)),
        )
        n_exclude = int(cfg.get("dynamic_exclude_count", 1))
        if conservative_sniper:
            n_exclude = min(n_exclude, 1)
        excluded_pool, sniper_msg = self.filters.get_sniper_exclusion(
            history, threshold=threshold, weights=weights, n_exclude=n_exclude
        )

        if verbose:
            # Solo imprimimos aquí si estamos en modo debug manual.
            # En producción, el mensaje se verá en la línea del reporte.
            if excluded_pool:
                print(f"   ✂️  Exclusión Activa: {sniper_msg}")
            else:
                print(
                    f"   🛡️  Sniper E1: Silencio de Radio (Sin exclusiones de alta certeza)."
                )

        # 2. Generación del Universo Base (con la poda aplicada)
        universe = self.filters.generate_universe(excluded_pool=excluded_pool)
        stage_sizes = {"base": int(len(universe))}

        if verbose:
            print(f"   ├─ Universo Base: {len(universe):,}")

        # --- ETAPA 2: FILTROS ESTRUCTURALES (Lo que ya funciona) ---
        # Mantenemos el orden exacto que recuperó el 6/6

        # Filtros de Posición y Suma
        universe = self.filters.apply_positional_limits(universe, cfg)
        stage_sizes["positional"] = int(len(universe))
        universe = self.filters.apply_aggregation(universe, cfg)
        stage_sizes["aggregation"] = int(len(universe))
        universe = self.filters.apply_structure(universe, cfg)
        stage_sizes["structure"] = int(len(universe))

        # Poda de Terminales y Espacial
        universe, mask_p = self.filters.apply_terminal_poda(universe, cfg)
        stage_sizes["terminal"] = int(len(universe))
        universe, d_vecs = self.filters.apply_spatial(universe, cfg)
        stage_sizes["spatial"] = int(len(universe))
        std_bounds = {"std_min_effective": 0.0, "std_max_effective": 0.0}

        # --- ETAPA 3: FILTROS DE ALTA FIDELIDAD ---
        if len(universe) > 0:
            # Perfiles de Década
            universe = self.filters.apply_profile_poda(universe, d_vecs, cfg)
            stage_sizes["profile"] = int(len(universe))

            # Entropía y Raíz Digital
            universe = self.filters.apply_entropy_shannon(universe, cfg)
            stage_sizes["entropy"] = int(len(universe))
            universe = self.filters.apply_digital_root_sum(universe, cfg)
            stage_sizes["sdr"] = int(len(universe))

            # Complejidad Aritmética
            universe = self.filters.apply_ac_complexity(universe, cfg)
            stage_sizes["ac"] = int(len(universe))

            # Pulido Final (Desviación Estándar)
            # Este es el toque final que limpia el ruido estadístico
            stds = self.xp.std(universe.astype(self.xp.float32), axis=1)

            auto_std = bool(cfg.get("auto_std_compensation", False))
            target_size = int(cfg.get("target_universe_size", 0) or 0)
            std_center = (cfg.get("std_min", 8.2) + cfg.get("std_max", 12.4)) / 2.0
            std_bounds = {
                "std_min_effective": float(cfg.get("std_min", 8.2)),
                "std_max_effective": float(cfg.get("std_max", 12.4)),
            }

            if auto_std and target_size > 0 and len(universe) > target_size:
                # Selección de tamaño objetivo: conservamos los tickets con std más cercana
                # al centro configurado, manteniendo un tamaño final controlado.
                stds_cpu = stds.get() if hasattr(stds, "get") else stds
                distances = np.abs(stds_cpu - std_center)
                keep_idx_cpu = np.argpartition(distances, target_size - 1)[:target_size]
                keep_idx = self.xp.asarray(keep_idx_cpu)
                universe = universe[keep_idx]

                selected_stds = stds_cpu[keep_idx_cpu]
                std_bounds = {
                    "std_min_effective": float(np.min(selected_stds)),
                    "std_max_effective": float(np.max(selected_stds)),
                }
            else:
                std_mask = (stds >= cfg.get("std_min", 8.2)) & (
                    stds <= cfg.get("std_max", 12.4)
                )
                universe = universe[std_mask]

            stage_sizes["std"] = int(len(universe))
        else:
            stage_sizes["profile"] = 0
            stage_sizes["entropy"] = 0
            stage_sizes["sdr"] = 0
            stage_sizes["ac"] = 0
            stage_sizes["std"] = 0

        stage_names = list(stage_sizes.keys())
        stage_ratios = {}
        base_size = stage_sizes.get("base", 0)
        for idx, name in enumerate(stage_names):
            size = stage_sizes[name]
            prev_size = stage_sizes[stage_names[idx - 1]] if idx > 0 else size
            stage_ratios[f"{name}_vs_prev"] = round(
                (size / prev_size) if prev_size else 0.0, 4
            )
            stage_ratios[f"{name}_vs_base"] = round(
                (size / base_size) if base_size else 0.0, 4
            )

        elapsed = time.time() - start_time
        if verbose:
            print(f"✅ UNIVERSO V7.17: {len(universe):,} tickets ({elapsed:.2f}s)")

        # RETORNO: Universo + Mensaje de Log + Telemetría de reducción
        return universe, sniper_msg, {
            "stage_sizes": stage_sizes,
            "stage_ratios": stage_ratios,
            "std_bounds": std_bounds,
            "execution_time": elapsed,
            "backend": self.backend_name,
        }
