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
        universe, sniper_log = self.reduce(history, config, verbose=verbose)

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
        excluded_pool, sniper_msg = self.filters.get_sniper_exclusion(
            history, threshold=threshold
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

        if verbose:
            print(f"   ├─ Universo Base: {len(universe):,}")

        # --- ETAPA 2: FILTROS ESTRUCTURALES (Lo que ya funciona) ---
        # Mantenemos el orden exacto que recuperó el 6/6

        # Filtros de Posición y Suma
        universe = self.filters.apply_positional_limits(universe, cfg)
        universe = self.filters.apply_aggregation(universe, cfg)
        universe = self.filters.apply_structure(universe, cfg)

        # Poda de Terminales y Espacial
        universe, mask_p = self.filters.apply_terminal_poda(universe, cfg)
        universe, d_vecs = self.filters.apply_spatial(universe, cfg)

        # --- ETAPA 3: FILTROS DE ALTA FIDELIDAD ---
        if len(universe) > 0:
            # Perfiles de Década
            universe = self.filters.apply_profile_poda(universe, d_vecs, cfg)

            # Entropía y Raíz Digital
            universe = self.filters.apply_entropy_shannon(universe, cfg)
            universe = self.filters.apply_digital_root_sum(universe, cfg)

            # Complejidad Aritmética
            universe = self.filters.apply_ac_complexity(universe, cfg)

            # Pulido Final (Desviación Estándar)
            # Este es el toque final que limpia el ruido estadístico
            stds = self.xp.std(universe.astype(self.xp.float32), axis=1)
            universe = universe[
                (stds >= cfg.get("std_min", 8.2)) & (stds <= cfg.get("std_max", 12.4))
            ]

        elapsed = time.time() - start_time
        if verbose:
            print(f"✅ UNIVERSO V7.17: {len(universe):,} tickets ({elapsed:.2f}s)")

        # RETORNO DOBLE: Universo + Mensaje de Log
        return universe, sniper_msg
