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

    @staticmethod
    def _temporal_split(
        history: DrawHistoryDTO,
        requested_draws: int,
        validation_fraction: float = 0.70,
    ) -> Dict[str, Any]:
        """Create a chronological train/validation/test split with a held-out test."""
        total = len(history.winning_numbers)
        if total < 2:
            raise ValueError("Se requieren al menos dos sorteos para optimizar")
        window = min(total, max(2, int(requested_draws)))
        start = total - window
        if total > 52:
            start = max(50, start)
        available = total - start
        if available < 2:
            start = max(0, total - 2)
            available = total - start
        validation_count = int(round(available * float(validation_fraction)))
        validation_count = min(available - 1, max(1, validation_count))
        test_start = start + validation_count
        return {
            "train": (0, start),
            "validation": (start, test_start),
            "test": (test_start, total),
            "train_end_contest": (
                int(history.concursos[start - 1]) if start > 0 else None
            ),
            "validation_range": [
                int(history.concursos[start]),
                int(history.concursos[test_start - 1]),
            ],
            "test_range": [
                int(history.concursos[test_start]),
                int(history.concursos[-1]),
            ],
        }

    @staticmethod
    def _score_universe(
        universe,
        history: DrawHistoryDTO,
        start: int,
        end: int,
        target_size: int,
    ) -> Dict[str, Any]:
        universe_cpu = universe.get() if hasattr(universe, "get") else universe
        universe_cpu = np.asarray(universe_cpu, dtype=np.uint8)
        distribution = {hit: 0 for hit in range(7)}
        details = []
        for idx in range(int(start), int(end)):
            winner = np.asarray(history.winning_numbers[idx][:6], dtype=np.uint8)
            matches = np.sum(np.isin(universe_cpu, winner), axis=1)
            max_hits = int(np.max(matches)) if len(matches) else 0
            distribution[max_hits] += 1
            details.append(
                {
                    "contest": int(history.concursos[idx]),
                    "date": history.dates[idx],
                    "winner": winner.tolist(),
                    "max_hits": max_hits,
                }
            )
        size = int(len(universe_cpu))
        size_delta = size - int(target_size)
        oversize_penalty = max(0, size_delta) / max(1, target_size) * 2500.0
        undersize_penalty = max(0, -size_delta) / max(1, target_size) * 300.0
        score = (
            distribution[6] * 6000
            + distribution[5] * 1000
            + distribution[4] * 120
            - oversize_penalty
            - undersize_penalty
        )
        return {
            "draws": int(end - start),
            "universe_size": size,
            "hit_distribution": distribution,
            "hits_6_6": distribution[6],
            "hits_5_6": distribution[5],
            "hits_4_6": distribution[4],
            "score": float(score),
            "details": details,
        }

    def _evaluate_voter_weights(
        self,
        history: DrawHistoryDTO,
        weights,
        start: int,
        end: int,
    ) -> Dict[str, Any]:
        errors = 0
        successful_exclusions = 0
        active = 0
        for idx in range(int(start), int(end)):
            past_history = DrawHistoryDTO(
                dates=history.dates[:idx],
                winning_numbers=history.winning_numbers[:idx],
                concursos=history.concursos[:idx],
            )
            excluded, _ = self.reducer.filters.get_sniper_exclusion(
                past_history,
                threshold=float(BEST_SETTINGS.get("sniper_threshold", 0.90)),
                weights=weights,
                n_exclude=int(BEST_SETTINGS.get("dynamic_exclude_count", 1)),
            )
            if not excluded:
                continue
            active += 1
            if int(excluded[0]) in set(history.winning_numbers[idx][:6]):
                errors += 1
            else:
                successful_exclusions += 1
        return {
            "draws": int(end - start),
            "active_exclusions": active,
            "successful_exclusions": successful_exclusions,
            "errors": errors,
            "error_rate": float(errors / active) if active else 0.0,
            "score": float(successful_exclusions - errors * 50),
        }

    # Ubicación: src/core/optimizer.py

    def optimize_voter_weights(self, history: DrawHistoryDTO, n_draws: int = 200):
        print(f"\n{CYAN}⚖️  CALIBRANDO PESOS DE VOTANTES (Protocolo Sniper E1){RESET}")
        global_start = time.time()
        h = self._as_chronological(history)
        split = self._temporal_split(h, n_draws)
        validation_start, validation_end = split["validation"]
        test_start, test_end = split["test"]
        print(
            f"{YELLOW}Split temporal: validación "
            f"#{split['validation_range'][0]}-#{split['validation_range'][1]} | "
            f"test reservado #{split['test_range'][0]}-#{split['test_range'][1]}.{RESET}"
        )

        # 1. Generar Rejilla (G + T + F = 1.0)
        resolution = 0.05
        weights_grid = []
        for g in np.arange(0.1, 0.8, resolution):
            for t in np.arange(0.05, 0.5, resolution):
                f = 1.0 - g - t
                if f > 0.1:
                    weights_grid.append((round(g, 2), round(t, 2), round(f, 2)))

        total_comb = len(weights_grid)
        current_weights = (
            float(BEST_SETTINGS.get("w_gap", 0.25)),
            float(BEST_SETTINGS.get("w_term", 0.10)),
            float(BEST_SETTINGS.get("w_freq", 0.60)),
        )
        best_weights = current_weights
        best_validation = self._evaluate_voter_weights(
            h, current_weights, validation_start, validation_end
        )
        best_score = best_validation["score"]
        best_distance = 0.0

        for i, w_tuple in enumerate(weights_grid):
            validation = self._evaluate_voter_weights(
                h, w_tuple, validation_start, validation_end
            )
            current_score = validation["score"]

            distance = sum(
                abs(candidate - current)
                for candidate, current in zip(w_tuple, current_weights)
            )
            candidate_key = (
                current_score,
                -validation["errors"],
                validation["successful_exclusions"],
            )
            best_key = (
                best_score,
                -best_validation["errors"],
                best_validation["successful_exclusions"],
            )
            if candidate_key > best_key or (
                candidate_key == best_key and distance < best_distance
            ):
                best_score = current_score
                best_weights = w_tuple
                best_validation = validation
                best_distance = distance

            if i % 10 == 0 or i == total_comb - 1:
                self._print_progress(
                    i,
                    total_comb,
                    0,
                    validation["errors"],
                    global_start,
                    label="Weights-Validation",
                )

        validation_metrics = best_validation
        test_metrics = self._evaluate_voter_weights(
            h, best_weights, test_start, test_end
        )
        selection_inconclusive = validation_metrics["active_exclusions"] == 0

        print(f"\n\n{GREEN}✅ OPTIMIZACIÓN DE PESOS FINALIZADA{RESET}")
        print(
            f"{WHITE}Copia estos valores en 'BEST_SETTINGS' dentro de config.py:{RESET}"
        )
        if selection_inconclusive:
            print(
                f"{YELLOW}Sin exclusiones activas en validación: se conservan "
                f"los pesos vigentes; no hay evidencia para promover otros.{RESET}"
            )

        return {
            "w_gap": best_weights[0],
            "w_term": best_weights[1],
            "w_freq": best_weights[2],
            "score": validation_metrics["score"],
            "selection_inconclusive": selection_inconclusive,
            "optimizer_split": {
                key: value
                for key, value in split.items()
                if key not in {"train", "validation", "test"}
            },
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        }

    def optimize_filters(
        self,
        history: DrawHistoryDTO,
        draws_to_test: int = 50,
        custom_grid: Dict[str, List] = None,
        target_universe_size: int = None,
    ) -> Dict[str, Any]:
        print(
            f"\n{CYAN}🔬 CALIBRACIÓN TEMPORAL DE FILTROS "
            f"(Hardware: {self.reducer.backend_name}){RESET}"
        )
        global_start = time.time()
        grid = custom_grid or SEARCH_GRID
        keys = list(grid.keys())
        combinations = list(itertools.product(*(grid[k] for k in keys)))
        total_comb = len(combinations)
        best_score = -float("inf")
        best_params = BEST_SETTINGS.copy()
        h = self._as_chronological(history)
        split = self._temporal_split(h, draws_to_test)
        train_start, train_end = split["train"]
        validation_start, validation_end = split["validation"]
        test_start, test_end = split["test"]
        train_history = DrawHistoryDTO(
            dates=h.dates[train_start:train_end],
            winning_numbers=h.winning_numbers[train_start:train_end],
            concursos=h.concursos[train_start:train_end],
        )
        print(
            f"{YELLOW}Split temporal: entrenamiento hasta "
            f"#{split['train_end_contest']} | validación "
            f"#{split['validation_range'][0]}-#{split['validation_range'][1]} | "
            f"test reservado #{split['test_range'][0]}-#{split['test_range'][1]}.{RESET}"
        )

        if target_universe_size is None:
            baseline_cfg = BEST_SETTINGS.copy()
            # Los filtros estructurales se calibran aislados del veto temporal.
            baseline_cfg.update(
                {
                    "sniper_mode": "off",
                    "dynamic_exclude_count": 0,
                    "sniper_soft_numbers": [],
                }
            )
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
            f"{YELLOW}🎯 Objetivo fijo: {target_u_size:,} tickets. "
            f"El test no participa en la selección.{RESET}"
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
                    "std_filter_enabled": True,
                    "auto_std_compensation": True,
                    "target_universe_size": target_u_size,
                    "universe_ticket_limit": target_u_size,
                    "sniper_mode": "off",
                    "dynamic_exclude_count": 0,
                    "sniper_soft_numbers": [],
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
            validation = self._score_universe(
                universe,
                h,
                validation_start,
                validation_end,
                target_u_size,
            )
            self._print_progress(
                i,
                total_comb,
                validation["hits_5_6"] + validation["hits_6_6"],
                validation["hits_4_6"],
                global_start,
                "Validation",
                u_size=u_size,
            )
            if validation["score"] > best_score:
                best_score = validation["score"]
                best_params = params.copy()
                best_params["optimizer_validation_metrics"] = validation

        if best_score == -float("inf"):
            raise ValueError("Ninguna configuración produjo un universo válido")

        best_config = PredictionConfigDTO(
            total_balls=TOTAL_BALLS,
            ticket_size=TICKET_SIZE,
            num_tickets=20,
            filter_overrides=best_params,
        )
        best_universe = self._extract_universe(
            self.reducer.reduce(train_history, best_config, verbose=False)
        )
        validation_metrics = self._score_universe(
            best_universe,
            h,
            validation_start,
            validation_end,
            target_u_size,
        )
        test_metrics = self._score_universe(
            best_universe,
            h,
            test_start,
            test_end,
            target_u_size,
        )
        best_params["u_size_avg"] = int(len(best_universe))
        best_params["hits_6_6_found"] = test_metrics["hits_6_6"]
        best_params["hits_5_6_found"] = test_metrics["hits_5_6"]
        best_params["hits_4_6_found"] = test_metrics["hits_4_6"]
        best_params["target_universe_size"] = target_u_size
        best_params["optimizer_split"] = {
            key: value
            for key, value in split.items()
            if key not in {"train", "validation", "test"}
        }
        best_params["optimizer_validation_metrics"] = validation_metrics
        best_params["optimizer_test_metrics"] = test_metrics

        print(f"\n\n{GREEN}✅ CALIBRACIÓN TEMPORAL FINALIZADA{RESET}")
        print(
            "📊 Validación: "
            f"6/6={validation_metrics['hits_6_6']} | "
            f"5/6={validation_metrics['hits_5_6']} | "
            f"4/6={validation_metrics['hits_4_6']}"
        )
        print(
            "🧪 Test reservado: "
            f"6/6={test_metrics['hits_6_6']} | "
            f"5/6={test_metrics['hits_5_6']} | "
            f"4/6={test_metrics['hits_4_6']} | "
            f"universo={len(best_universe):,}."
        )
        return best_params
