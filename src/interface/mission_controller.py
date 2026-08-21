# src/interface/mission_controller.py

from pathlib import Path
import src.data_access.report as report
import src.data_access.scraper as scraper
import src.data_access.shadow_ledger as shadow_ledger
import subprocess
import sys
from colorama import Fore, Style
from rich.panel import Panel
from rich.table import Table
from src.domain.dtos import PredictionConfigDTO, sort_history_chronologically
from src.data_access.config import (
    BEST_SETTINGS,
    BEST_SETTINGS_TRIS,
    BEST_SETTINGS_TRIS_CAMERA_LAB,
    TOTAL_BALLS,
    TICKET_SIZE,
    VERSION_TAG,
)
from src.strategies.universe_reduction import UniverseReductionStrategy
from src.strategies.universe.shadow import build_promoted_universe_shadows
from src.strategies.genetic_selector import GeneticSelectorStrategy
from src.strategies.combinatorial.shadow import build_promoted_covering_shadows
from src.strategies.tris.tris_forecast import TrisForecastV1A
from src.core.backtester import BacktestEngine
from src.core.fixed_origin_training import prepare_fixed_origin_models
from src.core.optimizer import StrategyOptimizer
from src.core.coverage_tester import CoverageTester
from src.core.rules import TrisMultiplicadorRules


class MissionController:
    def __init__(self, ui, history, profile=None):
        self.ui = ui
        self.history = history
        self.profile = profile

    def run_mission(self, option):
        if self._is_tris():
            self._run_mission_tris(option)
            return
        self._run_mission_melate(option)

    def _run_mission_melate(self, option):
        option = option.upper()
        if option == "1":
            self._view_history()
        elif option == "2":
            self._analyze_frequency()

        elif option == "3":
            self._run_optimizer()
        elif option == "4":
            self._retrain_model()
        elif option == "5":
            self._update_history()
        elif option == "6":
            self._run_backtest_lab()
        elif option == "7":
            self._run_production()
        elif option == "8":
            self._validate_bets()
        elif option == "P":
            self._run_forensic_plot()
        elif option == "C":
            self._run_covering_lab()
        else:
            self.ui.console.print("[red]Opcion no valida.[/]")
            self._pause()

    def _run_mission_tris(self, option):
        option = option.upper()
        if option == "1":
            self._view_history()
        elif option == "2":
            self._analyze_frequency()
        elif option == "3":
            self._notify_beta_feature(
                "Optimizador Tris",
                "La logica de optimizacion para secuencias (00000-99999) aun no esta integrada.",
            )
        elif option == "4":
            self._notify_beta_feature(
                "Reentrenamiento Tris",
                "El pipeline de entrenamiento para Tris Multiplicador sigue en desarrollo.",
            )
        elif option == "5":
            self._update_history()
        elif option == "6":
            self._run_tris_backtest()
        elif option == "7":
            self._run_tris_production()
        elif option == "8":
            self._notify_beta_feature(
                "Liquidacion Tris",
                "La liquidacion de cartera para Tris requiere ledger separado y tabla oficial de premios.",
            )
        elif option == "P":
            self._notify_beta_feature(
                "Plot Forense Tris",
                "El reporte forense para Tris aun no tiene esquema final.",
            )
        else:
            self.ui.console.print("[red]Opcion no valida.[/]")
            self._pause()

    def _view_history(self):
        self.ui.show_history(self.history, self.profile)
        self._pause()

    def _analyze_frequency(self):
        self.ui.show_frequency_analysis(self.history, self.profile)
        self._pause()

    def _run_optimizer(self):
        self.ui.clear_screen()
        print(f"\n{Fore.MAGENTA}🔬 MÓDULO DE OPTIMIZACIÓN MRPRO V15{Style.RESET_ALL}")
        print("1. Calibración Forense (Filtros de Reducción)")
        print("2. Optimizar Pesos de Votantes (Sniper E1)")

        op = input(f"\n{Fore.CYAN}Selecciona opción: {Style.RESET_ALL}")

        try:
            n_draws = int(input(f"   ¿Cuántos sorteos analizar? (200): ") or 200)
        except:
            n_draws = 200

        opt = StrategyOptimizer()

        if op == "1":
            best_cfg = opt.optimize_filters(self.history, n_draws)
        elif op == "2":
            best_cfg = opt.optimize_voter_weights(self.history, n_draws)

            print(f"\n{Fore.GREEN}🏆 PESOS SUGERIDOS:{Style.RESET_ALL}")
            for k, v in best_cfg.items():
                print(f"   • {k:<10}: {Fore.CYAN}{v}{Style.RESET_ALL}")
            print(
                f"\n{Fore.YELLOW}ℹ️  Actualiza estos valores en src/data_access/config.py{Style.RESET_ALL}"
            )

        input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

    def _generate_universe(self):
        tester = CoverageTester()
        config = PredictionConfigDTO(TOTAL_BALLS, TICKET_SIZE, num_tickets=20)
        tester.run(UniverseReductionStrategy(), self.history, config)
        input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

    def _run_backtest_lab(self):
        """Restaurada la funcionalidad de personalización de Backtest."""
        self.ui.clear_screen()
        print(
            f"\n{Fore.MAGENTA}🧪 LABORATORIO DE PRUEBAS (V15 OMEGA STRIDE){Style.RESET_ALL}"
        )
        print("1. Sniper Mode (Solo Reducción)")
        print("2. Full Omega Stride (Fixed-origin automático)")

        sub_op = input(f"\n{Fore.CYAN}Selecciona modo: {Style.RESET_ALL}")

        # RESTAURACIÓN DE INPUTS FUNCIONALES
        try:
            b_size = int(
                input(f"   ¿Cuántos sorteos hacia atrás probar? (108): ") or 108
            )
            n_tkt = int(input(f"   ¿Cuántos tickets por sorteo? (24): ") or 24)
        except:
            b_size, n_tkt = 108, 24

        engine = BacktestEngine()
        config = PredictionConfigDTO(
            TOTAL_BALLS, TICKET_SIZE, n_tkt, backtest_size=b_size
        )

        if sub_op == "1":
            engine.run(UniverseReductionStrategy(), self.history, config, verbose=True)
        elif sub_op == "2":
            try:
                self.ui.console.print(
                    "[cyan]🧠 Preparando cerebro fixed-origin sin tocar producción...[/]"
                )
                artifacts = prepare_fixed_origin_models(self.history, b_size)
            except Exception as exc:
                self.ui.console.print(
                    f"[bold red]❌ No se pudo preparar el modelo de backtest:[/] {exc}"
                )
                input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")
                return
            cache_status = "reutilizado" if artifacts.reused_cache else "entrenado"
            self.ui.console.print(
                "[bold green]✅ Fixed-origin "
                f"{cache_status}:[/] train hasta "
                f"#{artifacts.training_cutoff_contest} | test "
                f"#{artifacts.test_start_contest}-#{artifacts.test_end_contest} | "
                f"AUC interna={artifacts.internal_context_auc:.4f}"
            )
            fixed_settings = dict(BEST_SETTINGS)
            fixed_settings.update(
                {
                    "backtest_model_mode": "fixed_origin",
                    "fixed_origin_training_cutoff": artifacts.training_cutoff_contest,
                    "fixed_origin_test_start": artifacts.test_start_contest,
                    "fixed_origin_test_end": artifacts.test_end_contest,
                    "fixed_origin_dataset_hash": artifacts.dataset_hash,
                }
            )
            config.filter_overrides = fixed_settings
            engine.run(
                GeneticSelectorStrategy(
                    model_path=artifacts.context_model_path,
                    number_model_path=artifacts.number_model_path,
                ),
                self.history,
                config,
                True,
                UniverseReductionStrategy(),
            )

        input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

    def _run_production(self):
        """Producción V15: Con inputs funcionales y Ledger Lock."""
        ultimo_id = max(self.history.concursos)
        proximo_id = ultimo_id + 1

        if report.tiene_apuestas_pendientes(proximo_id):
            self.ui.console.print(
                Panel(
                    f"[bold red]🚫 BLOQUEO DE SEGURIDAD[/]\n\nYa existen apuestas para el sorteo [bold cyan]#{proximo_id}[/].",
                    border_style="red",
                )
            )
            input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")
            return

        # RESTAURACIÓN DE INPUT FUNCIONAL
        try:
            n_prod = int(
                input(
                    f"\n   ¿Cuántos tickets generar para el sorteo #{proximo_id}? (24): "
                )
                or 24
            )
        except:
            n_prod = 24

        principal_settings = dict(BEST_SETTINGS)
        principal_settings["resonance_blend_mode"] = "adaptive"
        config = PredictionConfigDTO(
            TOTAL_BALLS,
            TICKET_SIZE,
            n_prod,
            filter_overrides=principal_settings,
        )
        production_history = sort_history_chronologically(self.history)

        print(f"   {Fore.YELLOW}⏳ Paso 1: Filtrado Titanium...{Style.RESET_ALL}")
        univ_res = UniverseReductionStrategy().predict(production_history, config)
        config.raw_universe_ptr = univ_res.metadata.get("raw_ndarray")

        print(f"   {Fore.CYAN}🧬 Paso 2: Ejecutando Omega Stride...{Style.RESET_ALL}")
        selector = GeneticSelectorStrategy()
        pred = selector.predict(production_history, config)

        if pred.metadata.get("ai_signal_enabled") is False:
            self.ui.console.print(
                "[bold red]❌ Producción cancelada:[/] el modelo principal de IA "
                "no está disponible. No se guardaron apuestas ni carteras sombra."
            )
            input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")
            return

        if pred.metadata.get("ai_signal_validated") is False:
            auc = pred.metadata.get("temporal_holdout_auc")
            auc_text = f" (AUC temporal: {float(auc):.4f})" if auc is not None else ""
            self.ui.console.print(
                "[yellow]⚠ Señal de IA activa, pero todavía no validada fuera de muestra"
                f"{auc_text}. Su score sí participa en la selección.[/]"
            )

        if pred.tickets:
            shadow_specs = (
                {
                    "key": "principal_ai_adaptive",
                    "label": "Principal IA adaptativa",
                    "official": True,
                    "settings": {
                        "resonance_blend_mode": "adaptive",
                        "ai_context_weight": 1.0,
                        "ai_number_weight": 0.0,
                    },
                    "prediction": pred,
                },
                {
                    "key": "benchmark_mrpro_native_m300",
                    "label": "Benchmark MRPRO nativo / 300",
                    "official": False,
                    "ticket_count": 300,
                    "settings": {
                        "shadow_family": "same_budget_benchmark",
                        "resonance_blend_mode": "adaptive",
                        "ai_context_weight": 1.0,
                        "ai_number_weight": 0.0,
                    },
                },
                {
                    "key": "challenger_ai10_geo90",
                    "label": "Sombra IA 10% / Geo 90%",
                    "official": False,
                    "settings": {
                        "resonance_blend_mode": "fixed",
                        "hybrid_alpha": 0.10,
                        "hybrid_beta": 0.90,
                        "ai_context_weight": 1.0,
                        "ai_number_weight": 0.0,
                    },
                },
                {
                    "key": "control_geo_only",
                    "label": "Control Geo puro",
                    "official": False,
                    "settings": {
                        "resonance_blend_mode": "fixed",
                        "hybrid_alpha": 0.0,
                        "hybrid_beta": 1.0,
                        "ai_context_weight": 1.0,
                        "ai_number_weight": 0.0,
                    },
                },
                {
                    "key": "challenger_context50_number50",
                    "label": "Sombra IA contexto 50% / números 50%",
                    "official": False,
                    "settings": {
                        "shadow_family": "ai_signal_mix",
                        "promotion_reference_key": "principal_ai_adaptive",
                        "resonance_blend_mode": "adaptive",
                        "ai_context_weight": 0.50,
                        "ai_number_weight": 0.50,
                    },
                },
                {
                    "key": "challenger_deep_rank_5000",
                    "label": "Sombra selector estratificado hasta rank 5000",
                    "official": False,
                    "settings": {
                        "shadow_family": "selector_depth",
                        "promotion_reference_key": "principal_ai_adaptive",
                        "resonance_blend_mode": "adaptive",
                        "ai_context_weight": 1.0,
                        "ai_number_weight": 0.0,
                        "fitness_focus_max_rank": 5000,
                        "fitness_candidate_max_rank": 5000,
                        "fitness_rank_edges": [
                            5,
                            20,
                            100,
                            300,
                            750,
                            1500,
                            3000,
                            5000,
                        ],
                        "fitness_bucket_plan": [
                            [6, 20, 2],
                            [21, 100, 3],
                            [101, 300, 3],
                            [301, 750, 3],
                            [751, 1500, 3],
                            [1501, 3000, 3],
                            [3001, 5000, 2],
                        ],
                    },
                },
            )

            shadow_variants = []
            for spec in shadow_specs:
                variant_pred = spec.get("prediction")
                if variant_pred is None:
                    variant_settings = dict(BEST_SETTINGS)
                    variant_settings.update(spec["settings"])
                    variant_config = PredictionConfigDTO(
                        TOTAL_BALLS,
                        TICKET_SIZE,
                        int(spec.get("ticket_count", n_prod)),
                        filter_overrides=variant_settings,
                    )
                    variant_config.raw_universe_ptr = config.raw_universe_ptr
                    variant_pred = selector.predict(production_history, variant_config)
                shadow_variants.append(
                    {
                        "key": spec["key"],
                        "label": spec["label"],
                        "official": spec["official"],
                        "settings": spec["settings"],
                        "tickets": variant_pred.tickets,
                        "metadata": variant_pred.metadata,
                    }
                )

            covering_shadow_count = 0
            try:
                covering_variants = build_promoted_covering_shadows(
                    pred.metadata,
                    total_balls=TOTAL_BALLS,
                    ticket_size=TICKET_SIZE,
                )
                shadow_variants.extend(covering_variants)
                covering_shadow_count = len(covering_variants)
            except Exception as exc:
                self.ui.console.print(
                    "[bold yellow]⚠ Las sombras covering no pudieron generarse; "
                    f"producción y sombras existentes continúan:[/] {exc}"
                )

            universe_shadow_count = 0
            try:
                universe_variants = build_promoted_universe_shadows(
                    production_history,
                    selector,
                    total_balls=TOTAL_BALLS,
                    ticket_size=TICKET_SIZE,
                    ticket_count=n_prod,
                )
                shadow_variants.extend(universe_variants)
                universe_shadow_count = len(universe_variants)
            except Exception as exc:
                self.ui.console.print(
                    "[bold yellow]⚠ Las sombras de universo no pudieron generarse; "
                    f"producción y demás sombras continúan:[/] {exc}"
                )

            report.guardar_prediccion(pred.tickets, proximo_id)
            report.generar_ticket_limpio(pred.tickets, proximo_id)
            try:
                saved = shadow_ledger.guardar_carteras_sombra(
                    concurso_id=proximo_id,
                    variants=shadow_variants,
                    dataset_through_concurso=ultimo_id,
                )
                if saved:
                    self.ui.console.print(
                        "[bold cyan]🌓 MODO SOMBRA ACTIVO:[/] principal + 10/90 + "
                        "Geo + IA números 50/50 + rank profundo + benchmark 300"
                        + (
                            f" + {covering_shadow_count} covering"
                            if covering_shadow_count
                            else ""
                        )
                        + (
                            f" + {universe_shadow_count} de universo"
                            if universe_shadow_count
                            else ""
                        )
                        + " guardados sin inversión adicional."
                    )
                else:
                    self.ui.console.print(
                        f"[yellow]La comparación sombra del sorteo #{proximo_id} "
                        "ya estaba registrada.[/]"
                    )
            except Exception as exc:
                self.ui.console.print(
                    f"[bold yellow]⚠ Apuestas guardadas, pero falló el ledger sombra:[/] {exc}"
                )
            self.ui.show_prediction_results(pred)
            print(
                f"\n{Fore.GREEN}🍀 Tickets bloqueados y listos en archivo .txt{Style.RESET_ALL}"
            )

        input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

    def _update_history(self):
        print(f"\n{Fore.YELLOW}🌐 Sincronizando datos...{Style.RESET_ALL}")
        game_code = self.profile.code if self.profile else "melate_retro"
        if scraper.actualizar_csv(game_code):
            print(f"{Fore.GREEN}✅ Historial actualizado.{Style.RESET_ALL}")
        input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

    def _validate_bets(self):
        self.ui.clear_screen()
        print(f"\n{Fore.CYAN}💰 LIQUIDACIÓN DE CARTERA (ROI REAL){Style.RESET_ALL}")
        totales = report.liquidar_cartera(self.history)
        if totales:
            report.mostrar_resumen_roi(totales)
        try:
            shadow_summary = shadow_ledger.liquidar_carteras_sombra(self.history)
            shadow_ledger.mostrar_resumen_sombra(shadow_summary)
        except Exception as exc:
            self.ui.console.print(
                f"[bold yellow]⚠ No se pudo liquidar el ledger sombra:[/] {exc}"
            )
        input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

    def _run_forensic_plot(self):
        print(f"\n{Fore.CYAN}📊 Generando visualización forense...{Style.RESET_ALL}")
        try:
            from src.data_access.visualizer import run_forensic_visualization

            run_forensic_visualization()
        except Exception as e:
            print(
                f"{Fore.RED}❌ No se pudo cargar el módulo de visualización: {e}{Style.RESET_ALL}"
            )
        input(f"\n{Fore.YELLOW}>> Reporte generado. Presiona ENTER...{Style.RESET_ALL}")

    def _run_covering_lab(self):
        """Laboratorio aislado de covering designs para Melate Retro."""
        self.ui.clear_screen()
        self.ui.console.print(
            "\n[bold magenta]🧩 LABORATORIO DE CONDENSACIÓN COMBINATORIA[/]"
        )
        self.ui.console.print(
            "[yellow]Este experimento mide cobertura matemática; no predice la física "
            "del sorteo. El modo oracle es exclusivamente un control no predictivo.[/]"
        )
        try:
            v = int(input("   Tamaño del conjunto candidato v (15): ") or 15)
            t_raw = input(f"   Tamaño target t ({TICKET_SIZE - 1}): ") or str(
                TICKET_SIZE - 1
            )
            budget = int(input("   Presupuesto de boletos m (300): ") or 300)
            draws = int(input("   Sorteos walk-forward (108): ") or 108)
            random_trials = int(input("   Repeticiones random (100): ") or 100)
        except (TypeError, ValueError):
            self.ui.console.print("[red]Parámetros inválidos.[/]")
            self._pause()
            return

        self.ui.console.print(
            "\n[cyan]Conjunto candidato:[/] 1) Oracle control  2) Random  3) MRPRO"
        )
        candidate_choice = input("   Selección (1): ").strip() or "1"
        candidate_method = {
            "1": "oracle_candidate_set",
            "2": "random_candidate_set",
            "3": "mrpro_candidate_set",
        }.get(candidate_choice, "oracle_candidate_set")

        project_root = Path(__file__).resolve().parents[2]
        command = [
            sys.executable,
            str(project_root / "run_covering_experiment.py"),
            "--v",
            str(v),
            "--t",
            str(t_raw),
            "--budget",
            str(budget),
            "--draws",
            str(draws),
            "--random-trials",
            str(random_trials),
            "--candidate-method",
            candidate_method,
            "--mode",
            "both",
        ]
        try:
            subprocess.run(command, check=True, cwd=project_root)
            self.ui.console.print(
                "[bold green]✅ Experimento terminado.[/] Revisa JSON, CSV y gráficos en data/."
            )
        except subprocess.CalledProcessError as exc:
            self.ui.console.print(f"[bold red]❌ Falló el laboratorio:[/] {exc}")
        self._pause()

    def _run_tris_backtest(self):
        self.ui.clear_screen()
        self.ui.console.print(
            "\n[bold magenta]🧪 BACKTEST TRIS V1-A (BAYES + MARKOV)[/]"
        )

        settings = BEST_SETTINGS_TRIS.copy()
        self.ui.console.print("\n[bold cyan]Modo de backtest Tris:[/]")
        self.ui.console.print("1) Universe-only (solo filtros)")
        self.ui.console.print("2) Full selector (ranking + selección final)")
        self.ui.console.print("3) Camera Lab (universe_strategy / camera_mech_v1)")
        mode_in = self.ui.console.input("\n[yellow]Selecciona modo [2]: [/]").strip()
        if mode_in == "1":
            tris_backtest_mode = "universe"
            n_tickets = int(settings.get("num_tickets", 200))
        elif mode_in == "3":
            settings = BEST_SETTINGS_TRIS_CAMERA_LAB.copy()
            tris_backtest_mode = "universe_strategy"
            n_tickets = int(settings.get("num_tickets", 200))
            try:
                universe_topk_k = int(
                    self.ui.console.input(
                        f"[yellow]K del universo topK ({int(settings.get('universe_topk_k', 10000))}): [/] "
                    ).strip()
                    or int(settings.get("universe_topk_k", 10000))
                )
            except Exception:
                universe_topk_k = int(settings.get("universe_topk_k", 10000))
            settings["tris_backtest_mode"] = "universe_strategy"
            settings["universe_mode"] = "topk_scored_universe"
            settings["score_model"] = "camera_mech_v1"
            settings["camera_masked_universe"] = True
            settings["universe_topk_k"] = int(max(0, universe_topk_k))
            guardrails_in = (
                self.ui.console.input(
                    "[yellow]¿Usar guardrails estructurales? [Y/n]: [/] "
                )
                .strip()
                .lower()
            )
            if guardrails_in in {"n", "no"}:
                settings["structural_enabled"] = False
        else:
            tris_backtest_mode = "selector"
            try:
                n_tickets = int(
                    self.ui.console.input(
                        f"[yellow]¿Cuántos tickets por sorteo? ({int(settings.get('num_tickets', 200))}): [/] "
                    ).strip()
                    or int(settings.get("num_tickets", 200))
                )
            except Exception:
                n_tickets = int(settings.get("num_tickets", 200))
        settings["tris_backtest_mode"] = tris_backtest_mode

        config = PredictionConfigDTO(
            total_balls=self.profile.total_balls,
            ticket_size=self.profile.ticket_size,
            num_tickets=int(max(1, n_tickets)),
            backtest_size=int(settings.get("backtest_size", 500)),
            filter_overrides=settings.copy(),
        )

        engine = BacktestEngine(rules=TrisMultiplicadorRules())
        engine.run(
            strategy=TrisForecastV1A(),
            history=self.history,
            config=config,
            verbose=True,
            pre_process_strategy=None,
        )
        self._pause()

    def _run_tris_production(self):
        self.ui.clear_screen()
        self.ui.console.print("\n[bold green]🎯 PRODUCCION TRIS V1-A (ONE-SHOT)[/]")

        settings = BEST_SETTINGS_TRIS.copy()
        config = PredictionConfigDTO(
            total_balls=self.profile.total_balls,
            ticket_size=self.profile.ticket_size,
            num_tickets=int(settings.get("num_tickets", 200)),
            backtest_size=int(settings.get("backtest_size", 500)),
            filter_overrides=settings.copy(),
        )

        predictor = TrisForecastV1A()
        production_history = sort_history_chronologically(self.history)
        pred = predictor.predict(production_history, config)

        preview_n = min(10, len(pred.tickets))
        self.ui.console.print(
            f"[cyan]Estrategia:[/] {pred.strategy_name} | [cyan]Tickets generados:[/] {len(pred.tickets)}"
        )
        self.ui.console.print(f"[cyan]Top {preview_n} tickets:[/]")
        for idx, t in enumerate(pred.tickets[:preview_n], start=1):
            t_str = "".join(str(int(d)) for d in t[:5])
            self.ui.console.print(f"  [bold]{idx:02d}.[/] {t_str}")

        pos_probs = pred.metadata.get("pos_probs", [])
        if pos_probs and len(pos_probs) == 5:
            table = Table(title="Top-3 digitos por posicion", show_header=True)
            table.add_column("Pos", justify="center")
            table.add_column("Top-1", justify="center")
            table.add_column("Top-2", justify="center")
            table.add_column("Top-3", justify="center")

            for pos in range(5):
                row = pos_probs[pos]
                ranked = sorted(
                    enumerate(row),
                    key=lambda x: x[1],
                    reverse=True,
                )[:3]
                top_cells = [f"{d} ({p:.3f})" for d, p in ranked]
                table.add_row(str(pos + 1), top_cells[0], top_cells[1], top_cells[2])

            self.ui.console.print(table)

        p_multiplier = float(pred.metadata.get("p_multiplier", 0.0))
        entropy_mean = float(pred.metadata.get("entropy_mean", 0.0))
        self.ui.console.print(
            f"[yellow]p_multiplier:[/] {p_multiplier:.4f} | [yellow]entropy_mean:[/] {entropy_mean:.4f}"
        )
        self._pause()

    def _retrain_model_bk(self):
        """Módulo de Calibración de Neuronas V15."""
        self.ui.clear_screen()
        # Corregido: Fore para color, Style para efectos (DIM)
        print(
            f"\n{Fore.MAGENTA}☢️  PROTOCOLO DE RECALIBRACIÓN CEREBRAL V8{Style.RESET_ALL}"
        )
        print(
            f"{Style.DIM}Iniciando XGBoost Engine sobre hardware detectado...{Style.RESET_ALL}\n"
        )

        try:
            # Ejecutamos el entrenamiento como modulo para preservar imports del paquete src.
            project_root = Path(__file__).resolve().parents[2]
            subprocess.run(
                [sys.executable, "-m", "src.core.train_static_model"],
                check=True,
                cwd=project_root,
            )
            print(
                f"\n{Fore.GREEN}✅ MODELO ACTUALIZADO: Los pesos han sido sincronizados.{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"\n{Fore.RED}❌ ERROR CRÍTICO EN ENTRENAMIENTO: {e}{Style.RESET_ALL}"
            )

        input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

    def _retrain_model(self):
        """Módulo de Calibración de Neuronas V15."""
        self.ui.clear_screen()
        # Estética de consola conservada
        print(
            f"\n{Fore.MAGENTA}☢️  PROTOCOLO DE RECALIBRACIÓN CEREBRAL V8{Style.RESET_ALL}"
        )
        print(
            f"{Style.DIM}Iniciando XGBoost Engine sobre hardware detectado...{Style.RESET_ALL}\n"
        )

        try:
            # Importación dinámica para evitar dependencias circulares al inicio
            from src.core.train_static_model import train_master_brain

            # Ejecución directa de la función del 'Cerebro'
            train_master_brain()

            print(
                f"\n{Fore.GREEN}✅ MODELO ACTUALIZADO: Los pesos han sido sincronizados.{Style.RESET_ALL}"
            )
        except Exception as e:
            # Captura cualquier error de rutas o librerías durante el entrenamiento
            print(
                f"\n{Fore.RED}❌ ERROR CRÍTICO EN ENTRENAMIENTO: {e}{Style.RESET_ALL}"
            )

        input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

    def _notify_beta_feature(self, title: str, details: str):
        self.ui.console.print(
            Panel(
                f"[bold yellow]MODULO EN BETA: {title}[/]\n\n{details}",
                border_style="yellow",
            )
        )
        self._pause()

    def _pause(self):
        self.ui.console.input(f"\n[yellow]>> Presiona ENTER para volver...[/]")

    def _is_tris(self) -> bool:
        return bool(self.profile and self.profile.code == "tris_multiplicador")
