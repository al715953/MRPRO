import itertools
import sys
import numpy as np
import time
from typing import Dict, Any, List
from colorama import Fore, Style

# Importación de infraestructura central
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO
from src.data_access.config import TOTAL_BALLS, TICKET_SIZE, BEST_SETTINGS, SEARCH_GRID
from src.strategies.universe_reduction import UniverseReductionStrategy

# Definición de Colores Sniper
CYAN, GREEN, RED, YELLOW, WHITE, RESET = (
    Fore.CYAN,
    Fore.GREEN,
    Fore.RED,
    Fore.YELLOW,
    Fore.WHITE,
    Style.RESET_ALL,
)


class StrategyOptimizer:
    """
    Optimizador V8.13: Deep Audit Edition.
    Incluye los números reales del sorteo en el reporte forense para validación manual.
    """

    def __init__(self):
        self.reducer = UniverseReductionStrategy()
        self.xp = self.reducer.xp

    def _print_progress(
        self, current, total, hits_5_6, hits_4_6, start_time, label="Iter", u_size=0
    ):
        percent = int((current + 1) / (total if total > 0 else 1) * 100)
        elapsed = time.time() - start_time
        color_5 = GREEN if hits_5_6 > 0 else RED
        u_info = f" | Univ: {u_size:,}" if u_size > 0 else ""
        bar = "█" * (20 * (current + 1) // (total if total > 0 else 1))

        sys.stdout.write(
            f"\r   {CYAN}[{bar:<20}] {percent}%{RESET} | "
            f"{label} {current+1}/{total} | "
            f"5/6: {color_5}{hits_5_6}{RESET} 4/6: {hits_4_6}{u_info} | "
            f"{YELLOW}⏱️ {elapsed:.1f}s{RESET}"
        )
        sys.stdout.flush()

    def optimize_filters(
        self,
        history: DrawHistoryDTO,
        draws_to_test: int = 50,
        custom_grid: Dict[str, List] = None,
    ) -> Dict[str, Any]:
        print(
            f"\n{CYAN}🔬 FASE 1: Calibración Forense con Verificación de Números (Hardware: {self.reducer.backend_name}){RESET}"
        )
        global_start = time.time()

        grid = custom_grid or SEARCH_GRID
        keys = list(grid.keys())
        combinations = list(itertools.product(*(grid[k] for k in keys)))
        total_comb = len(combinations)

        best_score = -float("inf")
        best_params = BEST_SETTINGS.copy()

        # Datos para auditoría
        winners_to_check = np.array(
            history.winning_numbers[-draws_to_test:], dtype=np.uint8
        )
        concursos = history.concursos[-draws_to_test:]
        fechas = history.dates[-draws_to_test:]
        best_audit_log = []

        for i, values in enumerate(combinations):
            c = dict(zip(keys, values))
            if (
                c.get("e_min", 0) >= c.get("e_max", 1)
                or c.get("s_min", 0) >= c.get("s_max", 1)
                or c.get("std_min", 0) >= c.get("std_max", 1)
            ):
                continue

            params = BEST_SETTINGS.copy()
            params.update(
                {
                    "entropy_min": c["e_min"],
                    "entropy_max": c["e_max"],
                    "sdr_min": c["s_min"],
                    "sdr_max": c["s_max"],
                    "ac_min": c["ac"],
                    "std_min": c["std_min"],
                    "std_max": c["std_max"],
                }
            )

            config_dto = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                num_tickets=20,
                filter_overrides=params,
            )

            universe = self.reducer.reduce(history, config_dto, verbose=False)
            u_size = len(universe)

            if u_size > 120000 or u_size < 15000:
                self._print_progress(
                    i, total_comb, 0, 0, global_start, "Skip", u_size=u_size
                )
                continue

            u_data = self.xp.asarray(universe, dtype=self.xp.uint8)
            current_hits_5, current_hits_4 = 0, 0
            temp_log = []

            for idx, winner in enumerate(winners_to_check):
                matches = self.xp.zeros(u_size, dtype=self.xp.int8)
                for val in winner:
                    matches += self.xp.any(u_data == val, axis=1)

                max_h = int(self.xp.max(matches))
                winner_str = str(list(winner))  # Convertimos a string para el log

                if max_h >= 5:
                    current_hits_5 += 1
                    temp_log.append(
                        f"Concurso {concursos[idx]} ({fechas[idx]}): {GREEN}Hit 5/6{RESET} -> Real: {WHITE}{winner_str}{RESET}"
                    )
                elif max_h == 4:
                    current_hits_4 += 1
                    temp_log.append(
                        f"Concurso {concursos[idx]} ({fechas[idx]}): {CYAN}Hit 4/6{RESET} -> Real: {WHITE}{winner_str}{RESET}"
                    )

            density_score = (
                (current_hits_5 * 1000) + (current_hits_4 * 100)
            ) / np.sqrt(u_size)
            self._print_progress(
                i,
                total_comb,
                current_hits_5,
                current_hits_4,
                global_start,
                "Search",
                u_size=u_size,
            )

            if density_score > best_score:
                best_score = density_score
                best_params = params.copy()
                best_params["u_size_avg"] = u_size
                best_params["hits_5_6_found"] = current_hits_5
                best_audit_log = temp_log

        # --- REPORTE DE EVIDENCIA FINAL ---
        print(f"\n\n{GREEN}✅ CALIBRACIÓN FINALIZADA - REPORTE FORENSE{RESET}")
        print("=" * 80)
        print(f"{'CONCURSO':<15} {'FECHA':<12} {'RESULTADO':<20} {'COMBINACIÓN REAL'}")
        print("-" * 80)
        for log in best_audit_log:
            # El log ya viene con colores, lo imprimimos directamente
            print(f" 🎯 {log}")
        print("=" * 80)
        print(
            f"📊 Resumen Sniper: {best_params['hits_5_6_found']}/{draws_to_test} aciertos 5/6 en {best_params['u_size_avg']:,} tickets."
        )

        return best_params
