import time
from colorama import Fore, Style
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO

class CoverageTester:
    """
    Motor Sniper V33.6: Telemetría Completa + Promedio de Densidad.
    """
    def run(self, strategy, history: DrawHistoryDTO, config: PredictionConfigDTO, verbose: bool = True):
        print(f"\n{Fore.CYAN}📡 INICIANDO TEST DE COBERTURA (HIT RATIO V33.6){Style.RESET_ALL}")
        
        full_history = list(zip(history.dates, history.winning_numbers, history.concursos))
        full_history.sort(key=lambda x: x[2])
        test_size = min(config.backtest_size, len(full_history))
        start_index = len(full_history) - test_size

        hits_stats = {i: 0 for i in range(7)}
        total_tickets_accumulated = 0
        total_universes_count = 0

        for i in range(start_index, len(full_history)):
            target_date, target_draw_full, target_id = full_history[i]
            real_winning_set = set(target_draw_full[:6])

            # Simulación de "Viaje en el Tiempo"
            past_data = full_history[:i]
            p_dates, p_nums, p_ids = zip(*past_data) if past_data else ([], [], [])
            current_history = DrawHistoryDTO(list(p_dates), list(p_nums), list(p_ids))

            # Ejecución de Estrategia
            if verbose:
                print(f"\n📅 Sorteo #{target_id} ({target_date})... Generando Universo...", end="", flush=True)

            prediction = strategy.predict(current_history, config)
            
            u_size = len(prediction.tickets)
            total_tickets_accumulated += u_size
            total_universes_count += 1

            if verbose:
                print(f" Hecho. ({u_size:,} tickets)")

            # Validación de Cobertura
            best_hit = 0
            for ticket in prediction.tickets:
                hits = len(real_winning_set & set(ticket))
                if hits > best_hit: best_hit = hits
                if hits == 6: break

            hits_stats[best_hit] += 1

            # --- RESTAURACIÓN DE DETALLE VISUAL POR SORTEO ---
            color = Fore.RED; icon = "❌"
            if best_hit >= 4: color = Fore.YELLOW; icon = "⚠️"
            if best_hit >= 5: color = Fore.GREEN; icon = "✅"
            if best_hit == 6: color = Fore.MAGENTA; icon = "💎"

            if verbose:
                print(f"   🎱 Real: {sorted(list(real_winning_set))}")
                print(f"   {color}{icon} Mejor Cobertura: {best_hit}/6 Aciertos{Style.RESET_ALL}")

        # --- REPORTE FINAL ---
        avg_tickets = total_tickets_accumulated / total_universes_count if total_universes_count > 0 else 0
        
        print("\n" + "=" * 60)
        print(f"📊 REPORTE DE COBERTURA FINAL")
        print("=" * 60)
        print(f"Sorteos Analizados : {test_size}")
        print(f"Promedio de Tickets: {avg_tickets:,.0f} tkts/sorteo")
        print("-" * 30)
        for h in range(6, 2, -1):
            pct = (hits_stats[h] / test_size) * 100
            print(f"Aciertos {h}/6: {hits_stats[h]:3d} veces ({pct:5.1f}%)")
        print("-" * 30)
        print(f"{Fore.MAGENTA}💎 JACKPOTS CAPTURADOS: {hits_stats[6]}/{test_size}{Style.RESET_ALL}")