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

    @staticmethod
    def _extract_universe(reduction_result):
        """Normaliza la salida de reduce() para soportar ndarray o tupla (universe, meta)."""
        if isinstance(reduction_result, tuple):
            return reduction_result[0]
        return reduction_result

    @staticmethod
    def _as_chronological(history: DrawHistoryDTO) -> DrawHistoryDTO:
        """
        Retorna una vista cronológica (concurso ascendente) para evitar
        validaciones con fuga temporal en optimización.
        """
        ordered = sorted(
            zip(history.concursos, history.dates, history.winning_numbers),
            key=lambda x: int(x[0]),
        )
        concursos = [int(c) for c, _, _ in ordered]
        dates = [d for _, d, _ in ordered]
        winners = [w for _, _, w in ordered]
        return DrawHistoryDTO(dates=dates, winning_numbers=winners, concursos=concursos)

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

    # Ubicación: src/core/optimizer.py

    def optimize_voter_weights(self, history: DrawHistoryDTO, n_draws: int = 200):
        print(f"\n{CYAN}⚖️  CALIBRANDO PESOS DE VOTANTES (Protocolo Sniper E1){RESET}")
        global_start = time.time()
        h = self._as_chronological(history)

        # 1. Generar Rejilla (G + T + F = 1.0)
        resolution = 0.05
        weights_grid = []
        for g in np.arange(0.1, 0.8, resolution):
            for t in np.arange(0.05, 0.5, resolution):
                f = 1.0 - g - t
                if f > 0.1:
                    weights_grid.append((round(g, 2), round(t, 2), round(f, 2)))

        total_comb = len(weights_grid)
        best_score = -float("inf")
        best_weights = (0.25, 0.10, 0.60)

        # 2. Preparar sub-historiales para backtesting rápido
        # Probamos los últimos 'n_draws' sorteos
        total_available = len(h.winning_numbers)
        start_idx = max(50, total_available - n_draws)

        for i, w_tuple in enumerate(weights_grid):
            errors = 0
            success_exclusions = 0

            for idx in range(start_idx, total_available):
                # Creamos un "falso presente" para el Sniper
                past_history = DrawHistoryDTO(
                    dates=h.dates[:idx],
                    winning_numbers=h.winning_numbers[:idx],
                    concursos=h.concursos[:idx],
                )
                real_winner = set(h.winning_numbers[idx][:6])

                # Ejecutamos Sniper con los pesos de la iteración actual
                excluded, _ = self.reducer.filters.get_sniper_exclusion(
                    past_history, weights=w_tuple
                )

                if excluded:
                    if excluded[0] in real_winner:
                        errors += 1  # ¡Fatal! Excluimos un número que iba a ganar
                    else:
                        success_exclusions += 1

            # Puntuación: Queremos muchas exclusiones pero PENALIZAMOS fuerte los errores
            # Un error (matar el Jackpot) resta mucho más que un acierto
            current_score = success_exclusions - (errors * 50)

            if current_score > best_score:
                best_score = current_score
                best_weights = w_tuple

            if i % 10 == 0 or i == total_comb - 1:
                self._print_progress(
                    i, total_comb, 0, errors, global_start, label="Weights"
                )

        print(f"\n\n{GREEN}✅ OPTIMIZACIÓN DE PESOS FINALIZADA{RESET}")
        print(
            f"{WHITE}Copia estos valores en 'BEST_SETTINGS' dentro de config.py:{RESET}"
        )

        return {
            "w_gap": best_weights[0],
            "w_term": best_weights[1],
            "w_freq": best_weights[2],
            "score": best_score,
        }

    def optimize_filters(
        self,
        history: DrawHistoryDTO,
        draws_to_test: int = 50,
        custom_grid: Dict[str, List] = None,
        target_universe_size: int = None,
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
        h = self._as_chronological(history)
        total_available = len(h.winning_numbers)
        eval_start_idx = max(50, total_available - draws_to_test)

        train_history = DrawHistoryDTO(
            dates=h.dates[:eval_start_idx],
            winning_numbers=h.winning_numbers[:eval_start_idx],
            concursos=h.concursos[:eval_start_idx],
        )
        winners_to_check = np.array(
            [w[:6] for w in h.winning_numbers[eval_start_idx:]], dtype=np.uint8
        )
        concursos = h.concursos[eval_start_idx:]
        fechas = h.dates[eval_start_idx:]
        best_audit_log = []

        # Universo objetivo para no crecer tamaño: si no se provee, usamos baseline actual.
        if target_universe_size is None:
            baseline_cfg = BEST_SETTINGS.copy()
            baseline_dto = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                num_tickets=20,
                filter_overrides=baseline_cfg,
            )
            baseline_universe = self._extract_universe(
                self.reducer.reduce(train_history, baseline_dto, verbose=False)
            )
            target_u_size = int(len(baseline_universe))
        else:
            target_u_size = int(max(1, target_universe_size))

        print(
            f"{YELLOW}🎯 Objetivo de universo fijo: {target_u_size:,} tickets (sin crecimiento).{RESET}"
        )

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
                    # Sniper conservador + compensación de std para sostener tamaño final.
                    "sniper_conservative": True,
                    "sniper_threshold_boost": 0.08,
                    "dynamic_exclude_count": min(
                        int(BEST_SETTINGS.get("dynamic_exclude_count", 1)), 1
                    ),
                    "auto_std_compensation": True,
                    "target_universe_size": target_u_size,
                }
            )

            config_dto = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                num_tickets=20,
                filter_overrides=params,
            )

            universe = self._extract_universe(
                self.reducer.reduce(train_history, config_dto, verbose=False)
            )
            u_size = len(universe)

            if u_size == 0 or u_size > 200000:
                self._print_progress(
                    i, total_comb, 0, 0, global_start, "Skip", u_size=u_size
                )
                continue

            u_data = self.xp.asarray(universe[:, :6], dtype=self.xp.uint8)
            current_hits_6, current_hits_5, current_hits_4 = 0, 0, 0
            temp_log = []

            for idx, winner in enumerate(winners_to_check):
                matches = self.xp.zeros(u_size, dtype=self.xp.int8)
                for val in winner:
                    matches += self.xp.any(u_data == val, axis=1)

                max_h = int(self.xp.max(matches))
                winner_str = str(list(winner))  # Convertimos a string para el log

                if max_h == 6:
                    current_hits_6 += 1
                    temp_log.append(
                        f"Concurso {concursos[idx]} ({fechas[idx]}): {GREEN}Hit 6/6{RESET} -> Real: {WHITE}{winner_str}{RESET}"
                    )
                elif max_h == 5:
                    current_hits_5 += 1
                    temp_log.append(
                        f"Concurso {concursos[idx]} ({fechas[idx]}): {GREEN}Hit 5/6{RESET} -> Real: {WHITE}{winner_str}{RESET}"
                    )
                elif max_h == 4:
                    current_hits_4 += 1
                    temp_log.append(
                        f"Concurso {concursos[idx]} ({fechas[idx]}): {CYAN}Hit 4/6{RESET} -> Real: {WHITE}{winner_str}{RESET}"
                    )

            size_delta = u_size - target_u_size
            oversize_penalty = (
                (max(0, size_delta) / max(1, target_u_size)) * 2500.0
            )
            undersize_penalty = (
                (max(0, -size_delta) / max(1, target_u_size)) * 300.0
            )
            density_score = (
                (current_hits_6 * 6000)
                + (current_hits_5 * 1000)
                + (current_hits_4 * 120)
                - oversize_penalty
                - undersize_penalty
            )
            self._print_progress(
                i,
                total_comb,
                current_hits_5 + current_hits_6,
                current_hits_4,
                global_start,
                "Search",
                u_size=u_size,
            )

            if density_score > best_score:
                best_score = density_score
                best_params = params.copy()
                best_params["u_size_avg"] = u_size
                best_params["hits_6_6_found"] = current_hits_6
                best_params["hits_5_6_found"] = current_hits_5
                best_params["hits_4_6_found"] = current_hits_4
                best_params["target_universe_size"] = target_u_size
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
        hits_6_6_found = best_params.get("hits_6_6_found", 0)
        hits_5_6_found = best_params.get("hits_5_6_found", 0)
        hits_4_6_found = best_params.get("hits_4_6_found", 0)
        u_size_avg = best_params.get("u_size_avg", 0)
        print(
            f"📊 Resumen Sniper: 6/6={hits_6_6_found} | 5/6={hits_5_6_found} | 4/6={hits_4_6_found} "
            f"en {u_size_avg:,} tickets (objetivo {target_u_size:,})."
        )

        return best_params
