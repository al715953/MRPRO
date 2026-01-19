import json
import os
import time
import numpy as np
from rich.console import Console
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

from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, BacktestResultDTO
from src.core.rules import MelateRetroRules
from src.core.analytics import PerformanceTracker  # Módulo de bitácora
from src.data_access.config import VERSION_TAG  # Tag de versión


class BacktestEngine:
    """
    Motor Sniper V6.3.4: Telemetría de Alta Resolución y Bitácora Automática.
    - Preserva el formato de log original.
    - Integra PerformanceTracker para persistencia histórica.
    - Optimizado para evitar saturación de memoria en Windows.
    """

    def __init__(self):
        self.rules = MelateRetroRules()
        self.console = Console()
        self.forensic_data = []  # Almacena errores de distancia para el Scorer
        self.tracker = PerformanceTracker()  # Inicialización de bitácora

    def run(
        self,
        strategy,
        history: DrawHistoryDTO,
        config: PredictionConfigDTO,
        verbose: bool = True,
        pre_process_strategy=None,
    ):
        total_investment = 0.0
        total_earnings = 0.0
        hits_distribution = {i: 0 for i in range(7)}
        coverage_6_count = 0

        full_history = list(
            zip(history.dates, history.winning_numbers, history.concursos)
        )
        full_history.sort(key=lambda x: x[2])

        test_size = min(config.backtest_size, len(full_history))
        start_index = len(full_history) - test_size

        self.console.print(
            f"\n[bold magenta]🚀 INICIANDO MISIÓN ALPHA GLOBAL ({VERSION_TAG})[/bold magenta]"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn(
                "[bold blue]📡 Sniper Lab:[/][white] Analizando picos de probabilidad...[/]"
            ),
            BarColumn(bar_width=20),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
            disable=not verbose,
        ) as progress:
            task_id = progress.add_task("Misión", total=test_size)

            for i in range(start_index, len(full_history)):
                iter_start = time.time()
                _, target_draw, target_id = full_history[i]

                # 1. Preparar Historial "Ciego"
                past_data = full_history[:i]
                p_dates, p_nums, p_ids = zip(*past_data)
                current_history = DrawHistoryDTO(
                    list(p_dates), list(p_nums), list(p_ids)
                )

                # 2. Fase 1: Reducción de Universo
                if pre_process_strategy:
                    config.filter_overrides["verbose"] = False
                    univ_result = pre_process_strategy.predict(current_history, config)
                    config.raw_universe_ptr = univ_result.metadata.get("raw_ndarray")

                    # Radar de Cobertura en Universo Reducido
                    if config.raw_universe_ptr is not None:
                        hits_in_univ = np.max(
                            np.sum(
                                np.isin(config.raw_universe_ptr, target_draw[:6]),
                                axis=1,
                            )
                        )
                        if hits_in_univ == 6:
                            coverage_6_count += 1

                # 3. Fase 2: Entrenamiento y Selección
                # Inyectamos el feedback forense acumulado
                if hasattr(strategy.ai_model, "train"):
                    strategy.ai_model.train(
                        current_history.winning_numbers,
                        config.total_balls,
                        feedback_loop=self.forensic_data,
                    )

                prediction = strategy.predict(current_history, config)

                # 4. Auditoría Forense (AI, Geo y Hits)
                audit = {}
                if hasattr(strategy, "audit_winner"):
                    audit = strategy.audit_winner(current_history, config, target_draw)
                    audit["draw_id"] = int(target_id)
                    audit["internal_idx"] = i
                    self.forensic_data.append(audit)

                # 5. Validación Financiera
                for ticket in prediction.tickets:
                    total_investment += self.rules.ticket_cost
                    h_nat, h_add = self.rules.validate_ticket(ticket, target_draw)
                    total_earnings += self.rules.calculate_prize(h_nat, h_add)
                    hits_distribution[h_nat] += 1

                # 6. TELEMETRÍA (FORMATO ORIGINAL PRESERVADO)
                if verbose and audit:
                    self._render_telemetry(audit, target_id, iter_start)

                # 7. Limpieza explícita de VRAM para Windows
                if HAS_CUPY:
                    cp.get_default_memory_pool().free_all_blocks()

                progress.advance(task_id)

        # Construir DTO final
        result = self._build_result(
            test_size,
            total_investment,
            total_earnings,
            hits_distribution,
            coverage_6_count,
        )

        # REGISTRO EN BITÁCORA CSV
        self.tracker.log_run(result, tag=VERSION_TAG, audit_history=self.forensic_data)

        return result

    def _render_telemetry(self, audit, target_id, start_time):
        """Mantiene el formato de log exacto solicitado."""
        dist = audit.get("proximity", 999)
        rank = audit.get("rank", 0)
        hits = audit.get("hits", 0)
        ai_score = audit.get("ai_score", 0.0)
        geo_score = audit.get("geo_score", 0.0)

        # Colores dinámicos para visibilidad técnica
        st_col = "bold green" if dist == 0 else "bold red"
        h_col = "bold yellow" if hits >= 5 else "cyan" if hits == 4 else "white"
        d_col = "bold green" if dist == 0 else "bold yellow" if dist < 20 else "white"

        hit_label = "🎯 HIT" if dist == 0 else "❌ MISSED"

        # Log de salida: #ID | U: size | X/6 | AI: score | Geo: score | Rank: #pos | Dist: dist | STATUS | TIME
        self.console.print(
            f"[bold blue]#{target_id}[/] | "
            f"U: [white]{audit.get('univ_size', 0):>6,d}[/] | "
            f"[{h_col}]{hits}/6[/] | "
            f"AI: [bold yellow]{ai_score:.4f}[/] | "
            f"Geo: [bold cyan]{geo_score:.4f}[/] | "
            f"Rank: [white]#{rank:<5,d}[/] | "
            f"Dist: [{d_col}]{dist:<4,d}[/] | "
            f"[{st_col}]{hit_label}[/] | [dim]⏱ {time.time()-start_time:.2f}s[/dim]"
        )

    def _build_result(self, size, investment, earnings, hits, coverage):
        """Alineado con BacktestResultDTO de dtos.py."""
        return BacktestResultDTO(
            strategy_name="Quantum Alpha V10.5",
            total_draws_tested=size,
            investment=investment,
            earnings=earnings,
            net_balance=earnings - investment,
            hit_distribution=hits,
        )
