import itertools
import sys
import numpy as np
from typing import Dict, Any
from colorama import Fore, Style

from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO
from src.data_access.config import TOTAL_BALLS, TICKET_SIZE, BEST_SETTINGS
from src.strategies.universe_reduction import UniverseReductionStrategy

# Definición de Colores
CYAN = Fore.CYAN
GREEN = Fore.GREEN
RED = Fore.RED
YELLOW = Fore.YELLOW
WHITE = Fore.WHITE
RESET = Style.RESET_ALL


class StrategyOptimizer:
    """
    Optimizador V8.8: Resolución Ultra-Fina.
    Explora el espacio de parámetros con pasos de 0.05 para máxima asfixia.
    """

    def __init__(self):
        self.reducer = UniverseReductionStrategy()
        self.xp = self.reducer.xp

    def _print_progress(
        self, current, total, hits_5_6, hits_4_6, label="Iter", u_size=0
    ):
        percent = int((current + 1) / (total if total > 0 else 1) * 100)

        color_5 = GREEN if hits_5_6 >= 25 else (YELLOW if hits_5_6 >= 15 else RED)
        color_4 = CYAN if hits_4_6 >= 15 else WHITE

        # El "Punto Dulce" ideal para 40 sorteos suele estar entre 35k y 45k
        u_color = (
            GREEN if 35000 <= u_size <= 48000 else (YELLOW if u_size <= 85000 else RED)
        )
        u_info = f" | Univ: {u_color}{u_size:,}{RESET}" if u_size > 0 else ""

        bar = "█" * (20 * (current + 1) // (total if total > 0 else 1))
        sys.stdout.write(
            f"\r   {CYAN}[{bar:<20}] {percent}%{RESET} | {label} {current+1}/{total} | 5/6: {color_5}{hits_5_6}{RESET} 4/6: {color_4}{hits_4_6}{RESET}{u_info}"
        )
        sys.stdout.flush()

    def optimize_filters(
        self, history: DrawHistoryDTO, draws_to_test: int = 40
    ) -> Dict[str, Any]:
        """Búsqueda exhaustiva del Golden Ratio con granularidad aumentada."""
        print(
            f"\n{CYAN}🔬 FASE 1: Búsqueda Exhaustiva de Alta Resolución (Hardware: {self.reducer.backend_name}){RESET}"
        )

        # Definimos ejes independientes para máxima precisión
        # Entropía: Pasos de 0.05
        e_mins = [2.00, 2.05, 2.10, 2.15]
        e_maxs = [2.45, 2.50, 2.55, 2.60]

        # SDR (Raíz Digital): Pasos de 1
        s_mins = [20, 22, 24]
        s_maxs = [42, 44, 46]

        # Complejidad AC
        ac_limits = [7, 8]

        # Dispersión (STD): Pasos de 0.2
        std_mins = [7.8, 8.0, 8.2]
        std_maxs = [12.4, 12.6, 12.8]

        # Generamos la red de búsqueda completa
        combinations = list(
            itertools.product(
                e_mins, e_maxs, s_mins, s_maxs, ac_limits, std_mins, std_maxs
            )
        )

        best_score = -float("inf")
        best_params = BEST_SETTINGS.copy()
        winners_to_check = np.array(
            history.winning_numbers[-draws_to_test:], dtype=np.uint8
        )

        for i, (emin, emax, smin, smax, ac, stdmin, stdmax) in enumerate(combinations):
            # Seguridad: Evitar rangos invertidos o absurdos
            if emin >= emax or smin >= smax or stdmin >= stdmax:
                continue

            params = BEST_SETTINGS.copy()
            params.update(
                {
                    "entropy_min": emin,
                    "entropy_max": emax,
                    "sdr_min": smin,
                    "sdr_max": smax,
                    "ac_min": ac,
                    "std_min": stdmin,
                    "std_max": stdmax,
                }
            )

            # Reducción en VRAM
            universe = self.reducer.reduce(
                history,
                PredictionConfigDTO(
                    TOTAL_BALLS, TICKET_SIZE, 20, filter_overrides=params
                ),
                verbose=False,
            )
            u_size = len(universe)

            # Filtro de Viabilidad Sniping
            if u_size > 110000 or u_size < 15000:
                self._print_progress(i, len(combinations), 0, 0, "Skip", u_size=u_size)
                continue

            # Auditoría Forense Vectorial
            hits_5_6 = 0
            hits_4_6 = 0
            u_data = self.xp.asarray(universe, dtype=self.xp.uint8)

            for winner in winners_to_check:
                matches = self.xp.zeros(u_size, dtype=self.xp.int8)
                for val in winner:
                    matches += self.xp.any(u_data == val, axis=1)

                max_h = int(self.xp.max(matches))
                if max_h >= 5:
                    hits_5_6 += 1
                elif max_h == 4:
                    hits_4_6 += 1

            # Métrica Sniper: Eficiencia de Densidad
            # Premiamos la cobertura absoluta pero con un mazo más pesado para el volumen
            # (hits * 1000) / sqrt(u_size) para no castigar tan fuerte el crecimiento necesario
            raw_quality = (hits_5_6 * 1000) + (hits_4_6 * 100)
            density_score = raw_quality / np.sqrt(u_size)

            self._print_progress(
                i, len(combinations), hits_5_6, hits_4_6, "Search", u_size=u_size
            )

            if density_score > best_score:
                best_score = density_score
                best_params = params.copy()
                best_params["u_size_avg"] = u_size
                best_params["hits_5_6_found"] = hits_5_6

        print(f"\n\n{GREEN}✅ CALIBRACIÓN ULTRA-FINA COMPLETADA{RESET}")
        return best_params
