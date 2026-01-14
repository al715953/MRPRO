import itertools
import sys
import time
from typing import Dict, Any, List, Tuple
from colorama import Fore, Style

# --- IMPORTS DE DOMINIO ---
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO
from src.data_access.config import TOTAL_BALLS, TICKET_SIZE

# --- IMPORTS DE ESTRATEGIAS ---
from src.strategies.universe_reduction import UniverseReductionStrategy
from src.strategies.genetic_selector import GeneticSelectorStrategy
from src.strategies.heuristic_selector import HeuristicSelectorStrategy

# --- IMPORTS DE MOTOR ---
from src.core.backtester import BacktestEngine

# --- COLORES ---
CYAN = Fore.CYAN
GREEN = Fore.GREEN
RED = Fore.RED
YELLOW = Fore.YELLOW
RESET = Style.RESET_ALL


class StrategyOptimizer:
    """
    Optimizador Híbrido V6 (Full Stack + Forensic Calibration).

    ARQUITECTURA DE 3 FASES:
    1. TOPOLOGÍA (Fuerza Bruta): Filtros de reducción (Suma, Pares, etc.).
    2. SINTONIZACIÓN (Fuerza Bruta): Pesos IA vs Heurística.
    3. ESTRUCTURA (Analítica Forense): Ajuste de cuotas (Elite/Mid/Low) basado en dónde caen los ganadores reales.
    """

    def __init__(self):
        self.backtester = BacktestEngine()

    def _print_progress(
        self, current, total, value, label="Iter", extra_info="", is_money=True
    ):
        """
        Barra de progreso visual mejorada.
        """
        if total == 0:
            total = 1  # Evitar división por cero
        percent = int((current + 1) / total * 100)

        # Formato del valor (Dinero o Score)
        if is_money:
            val_str = f"${value:,.0f}"
            status_color = GREEN if value > 0 else RED
        else:
            val_str = f"{value:.4f}"
            status_color = YELLOW

        bar_len = 20
        filled_len = int(bar_len * (current + 1) // total)
        bar = "█" * filled_len + "-" * (bar_len - filled_len)

        # Limpiar línea y escribir
        sys.stdout.write(
            f"\r   {CYAN}[{bar}] {percent}%{RESET} | {label} {current+1}/{total} | {extra_info} | Val: {status_color}{val_str}{RESET}"
        )
        sys.stdout.flush()

    # =========================================================================
    # FASE 1: OPTIMIZACIÓN DE FILTROS (TOPOLOGÍA)
    # =========================================================================
    def optimize_filters(
        self, history: DrawHistoryDTO, draws_to_test: int = 20
    ) -> Dict[str, Any]:
        """
        Prueba combinaciones de filtros geométricos para maximizar el ROI.
        """
        print(f"\n{CYAN}🔬 FASE 1: Calibración de Filtros (Topología)...{RESET}")

        # Grid de Búsqueda (Rangos probables)
        sums = [(110, 180), (120, 170), (115, 175)]
        evens = [(2, 4)]
        acs = [4, 5]

        combinations = list(itertools.product(sums, evens, acs))

        best_roi = -float("inf")
        best_params = {}

        # Usamos HeuristicSelectorStrategy como proxy rápido
        strategy = HeuristicSelectorStrategy()

        # Config base silenciosa
        config = PredictionConfigDTO(
            TOTAL_BALLS, TICKET_SIZE, 15, draws_to_test, {"verbose": False}
        )

        for i, (sum_r, even_r, ac_val) in enumerate(combinations):
            params = {
                "sum_min": sum_r[0],
                "sum_max": sum_r[1],
                "even_min": even_r[0],
                "even_max": even_r[1],
                "ac_min": ac_val,
                "verbose": False,
            }
            config.filter_overrides = params

            try:
                res = self.backtester.run(
                    strategy,
                    history,
                    config,
                    pre_process_strategy=UniverseReductionStrategy(),
                    verbose=False,
                )

                self._print_progress(i, len(combinations), res.net_balance, "Filtros")

                if res.net_balance > best_roi:
                    best_roi = res.net_balance
                    best_params = params.copy()
            except Exception:
                pass

        print(f"\n   ✅ Mejor ROI Fase 1: ${best_roi:,.0f}")
        return best_params

    # =========================================================================
    # FASE 2: OPTIMIZACIÓN DE PESOS (SENSIBILIDAD)
    # =========================================================================
    def optimize_heuristics(
        self, history: DrawHistoryDTO, base_params: Dict, draws_to_test: int = 20
    ) -> Dict[str, Any]:
        """
        Prueba combinaciones de pesos (Cluster, Hotness, IA) usando la estrategia Genética.
        """
        print(f"\n{CYAN}⚖️  FASE 2: Sintonización de Pesos (IA vs Heurística)...{RESET}")

        # Grid: (w_cluster, w_hotness, w_ai)
        weights_grid = [
            (0.6, 0.4, 0.3),  # Standard V9
            (0.5, 0.3, 0.5),  # IA Heavy
            (0.7, 0.3, 0.1),  # Classic Heavy
            (0.4, 0.4, 0.2),  # Balanceado
            (0.8, 0.2, 0.0),  # Estructural Puro
        ]

        best_roi = -float("inf")
        best_params = base_params.copy()

        # Aquí usamos GeneticSelectorStrategy porque es la que usa la IA
        strategy = GeneticSelectorStrategy()

        config = PredictionConfigDTO(
            TOTAL_BALLS, TICKET_SIZE, 15, draws_to_test, {"verbose": False}
        )

        for i, (wc, wh, wa) in enumerate(weights_grid):
            params = base_params.copy()
            params.update(
                {"w_cluster": wc, "w_hotness": wh, "w_ai": wa, "verbose": False}
            )
            config.filter_overrides = params

            try:
                # GeneticSelector incluye entrenamiento, así que esto prueba la IA real
                res = self.backtester.run(
                    strategy,
                    history,
                    config,
                    pre_process_strategy=UniverseReductionStrategy(),
                    verbose=False,
                )

                self._print_progress(i, len(weights_grid), res.net_balance, "Pesos")

                if res.net_balance > best_roi:
                    best_roi = res.net_balance
                    best_params = params.copy()
            except Exception:
                pass

        print(f"\n   ✅ Mejor ROI Fase 2: ${best_roi:,.0f}")
        return best_params

    # =========================================================================
    # FASE 3: OPTIMIZACIÓN ESTRUCTURAL (Cuotas + Umbrales)
    # =========================================================================
    def optimize_quotas(
        self, history: DrawHistoryDTO, base_params: Dict, draws_to_test: int = 20
    ) -> Dict[str, Any]:
        """
        Calibración Forense V2.
        Encuentra las mejores Cuotas Y los mejores Umbrales (Thresholds)
        basado en la distribución real de los ganadores.
        """
        print(
            f"\n{CYAN}🧱 FASE 3: Calibración Estructural (Cuotas & Umbrales)...{RESET}"
        )
        analyze_depth = max(draws_to_test, 50)
        print(
            f"   📊 Analizando últimos {analyze_depth} sorteos para definir umbrales..."
        )

        strategy = GeneticSelectorStrategy()
        strategy._train_model(history, TOTAL_BALLS)
        strategy._update_heuristic_metrics(history)

        scores_log = []

        # Pesos actuales
        scoring_weights = {
            "w_cluster": base_params.get("w_cluster", 0.6),
            "w_hotness": base_params.get("w_hotness", 0.4),
            "w_ai": base_params.get("w_ai", 0.3),
        }

        # 1. Escaneo de Scores de Ganadores
        start_idx = max(0, len(history.winning_numbers) - analyze_depth)
        winners = history.winning_numbers[start_idx:]

        for i, w_draw in enumerate(winners):
            target = tuple(sorted(w_draw[:6]))
            ai_score = strategy.ai_model.score_tickets([target])[0]

            # Usar API V11
            if hasattr(strategy, "_compute_v7_score"):
                try:
                    s, _, _ = strategy._compute_v7_score(
                        target, ai_score, scoring_weights
                    )
                except:
                    s, _, _ = strategy._compute_v7_score(target, ai_score)
            else:
                s = 0.5

            scores_log.append(s)
            self._print_progress(
                i,
                len(winners),
                s,
                label="Scan",
                extra_info=f"S:{s:.2f}",
                is_money=False,
            )

        # 2. Determinación de Umbrales Óptimos (Percentiles)
        # Elite = Top 20% de ganadores, Mid = Siguiente 60%, Low = Resto
        scores_log.sort(reverse=True)
        if not scores_log:
            return base_params

        # Percentil 20 superior para definir Elite
        idx_elite = int(len(scores_log) * 0.20)
        suggested_elite_th = (
            scores_log[idx_elite] if idx_elite < len(scores_log) else 0.70
        )

        # Percentil 80 (para cubrir la mayoría en Mid)
        idx_mid = int(len(scores_log) * 0.80)
        suggested_mid_th = scores_log[idx_mid] if idx_mid < len(scores_log) else 0.55

        # Redondeo estético
        suggested_elite_th = round(suggested_elite_th, 2)
        suggested_mid_th = round(suggested_mid_th, 2)

        # Safety
        if suggested_elite_th > 0.85:
            suggested_elite_th = 0.75
        if suggested_mid_th < 0.40:
            suggested_mid_th = 0.50
        if suggested_elite_th <= suggested_mid_th:
            suggested_elite_th = suggested_mid_th + 0.10

        print(
            f"\n   📏 Umbrales Sugeridos: Elite >= {suggested_elite_th}, Mid >= {suggested_mid_th}"
        )

        # 3. Recálculo de Cuotas con los Nuevos Umbrales
        zone_hits = {"Elite": 0, "Mid": 0, "Low": 0}

        for s in scores_log:
            if s >= suggested_elite_th:
                zone_hits["Elite"] += 1
            elif s >= suggested_mid_th:
                zone_hits["Mid"] += 1
            elif s >= (suggested_mid_th - 0.10):
                zone_hits["Low"] += 1

        total = sum(zone_hits.values()) or 1
        target_tickets = 15

        q_e = round((zone_hits["Elite"] / total) * target_tickets)
        q_m = round((zone_hits["Mid"] / total) * target_tickets)
        q_l = round((zone_hits["Low"] / total) * target_tickets)

        # Ajuste suma 15
        diff = target_tickets - (q_e + q_m + q_l)
        if diff != 0:
            q_m += diff  # Simplificación: echar al Mid

        # Safety
        q_e, q_m, q_l = max(1, int(q_e)), max(1, int(q_m)), max(1, int(q_l))
        while (q_e + q_m + q_l) > 15:
            q_l -= 1
        while (q_e + q_m + q_l) < 15:
            q_m += 1

        best_quotas = {
            "quota_elite": q_e,
            "quota_mid": q_m,
            "quota_low": q_l,
            "threshold_elite": suggested_elite_th,
            "threshold_mid": suggested_mid_th,
        }

        print(f"   ✅ Estructura Optimizada: {best_quotas}")

        final_params = base_params.copy()
        final_params.update(best_quotas)
        return final_params

    # =========================================================================
    # EJECUCIÓN MAESTRA
    # =========================================================================
    def optimize_full_stack(self, history: DrawHistoryDTO, draws_to_test: int = 20):
        print(f"\n{YELLOW}🚀 INICIANDO OPTIMIZACIÓN FULL STACK (V6){RESET}")
        print(f"{YELLOW}========================================={RESET}")

        start_time = time.time()

        # 1. Filtros (Fuerza Bruta es mejor aquí para asegurar universo válido)
        cfg_1 = self.optimize_filters(history, draws_to_test)

        # 2. Pesos (Fuerza Bruta para maximizar ROI)
        cfg_2 = self.optimize_heuristics(history, cfg_1, draws_to_test)

        # 3. Cuotas (CALIBRACIÓN FORENSE)
        # Llamamos a optimize_quotas que ahora contiene la lógica analítica
        final_cfg = self.optimize_quotas(history, cfg_2, draws_to_test)

        elapsed = time.time() - start_time

        print(f"\n\n{GREEN}📊 CONFIGURACIÓN MAESTRA RECOMENDADA:{RESET}")
        print(f"⏱️  Tiempo Total: {elapsed/60:.1f} min")
        print(f"--------------------------------------------------")
        print(
            f"   • [Filtros] Suma: {final_cfg['sum_min']}-{final_cfg['sum_max']} | AC: {final_cfg['ac_min']}"
        )
        print(
            f"   • [Pesos]   C:{final_cfg['w_cluster']} H:{final_cfg['w_hotness']} IA:{final_cfg['w_ai']}"
        )
        print(
            f"   • [Táctica] E:{final_cfg['quota_elite']} M:{final_cfg['quota_mid']} L:{final_cfg['quota_low']}"
        )
        print(f"--------------------------------------------------")
        return final_cfg
