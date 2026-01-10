import time
from typing import List, Set, Tuple
from colorama import Fore, Style
from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO


class CoverageTester:
    """
    Motor especializado en validar 'Universos Reducidos'.
    No calcula dinero, sino la presencia (Cobertura) de premios dentro del pool.
    """

    def run(
        self,
        strategy: ILotteryStrategy,
        history: DrawHistoryDTO,
        config: PredictionConfigDTO,
        verbose: bool = True,
    ):
        print(
            f"\n{Fore.CYAN}📡 INICIANDO TEST DE COBERTURA (HIT RATIO){Style.RESET_ALL}"
        )
        print(f"🎯 Buscando premios dentro de universos generados dinámicamente...")

        # 1. Preparar historia
        full_history = list(
            zip(history.dates, history.winning_numbers, history.concursos)
        )
        full_history.sort(key=lambda x: x[2])

        # 2. Rango de prueba
        total_draws = len(full_history)
        test_size = config.backtest_size
        if total_draws < test_size:
            test_size = total_draws

        start_index = total_draws - test_size

        # Estadísticas
        hits_stats = {6: 0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0}
        jackpots_found = 0
        total_universes_generated = 0

        start_time_global = time.time()

        # 3. Bucle Temporal
        for i in range(start_index, total_draws):
            target_date, target_draw_full, target_id = full_history[i]

            # El sorteo real (Solo los 6 naturales para comparar rápido)
            # Asumimos que loader ya ordenó los primeros 6
            real_winning_set = set(target_draw_full[:6])

            # Recorte de historia (Viaje en el tiempo)
            past_data = full_history[:i]
            if not past_data:
                continue

            p_dates, p_nums, p_ids = zip(*past_data)
            current_history = DrawHistoryDTO(list(p_dates), list(p_nums), list(p_ids))

            # --- GENERAR UNIVERSO ---
            if verbose:
                print(
                    f"\n📅 Sorteo #{target_id} ({target_date})... Generando Universo...",
                    end="",
                    flush=True,
                )

            # Ejecutamos la estrategia (UniverseReduction)
            # Nota: Esto puede tardar unos segundos por iteración
            prediction = strategy.predict(current_history, config)

            universe_tickets = prediction.tickets  # Aquí viene la lista de ~250k listas
            universe_size = len(universe_tickets)
            total_universes_generated += 1

            if verbose:
                print(f" Hecho. ({universe_size:,} tickets)")

            # --- VALIDACIÓN DE COBERTURA ---
            # Buscamos el MEJOR ticket dentro del universo generado
            best_hit = 0
            found_jackpot = False

            # Convertimos a sets para comparación rápida
            # (Aunque iterar 250k veces es rápido en Python puro si son sets)
            for ticket_list in universe_tickets:
                ticket_set = set(ticket_list)
                hits = len(real_winning_set & ticket_set)

                if hits > best_hit:
                    best_hit = hits

                if hits == 6:
                    found_jackpot = True
                    # No hacemos break para contar si hubo múltiples (opcional),
                    # pero por velocidad podríamos romper aquí.
                    break

            # Registrar estadísticas
            hits_stats[best_hit] += 1
            if found_jackpot:
                jackpots_found += 1

            # Reporte Visual del Sorteo
            color = Fore.RED
            icon = "❌"
            if best_hit >= 4:
                color = Fore.YELLOW
                icon = "⚠️"
            if best_hit >= 5:
                color = Fore.GREEN
                icon = "✅"
            if best_hit == 6:
                color = Fore.MAGENTA
                icon = "💎"

            if verbose:
                print(f"   🎱 Real: {sorted(list(real_winning_set))}")
                print(
                    f"   {color}{icon} Mejor Cobertura: {best_hit}/6 Aciertos{Style.RESET_ALL}"
                )

        # --- REPORTE FINAL ---
        print("\n" + "=" * 60)
        print(f"📊 REPORTE DE COBERTURA (HIT RATIO)")
        print("=" * 60)
        print(f"Sorteos Analizados: {test_size}")
        print(f"Duración: {time.time() - start_time_global:.1f} seg")
        print("-" * 30)

        # Cálculo de porcentajes
        for h in range(6, 2, -1):  # 6, 5, 4, 3
            count = hits_stats[h]
            pct = (count / test_size) * 100
            print(f"Aciertos {h}/6: {count:3d} veces ({pct:5.1f}%)")

        print("-" * 30)
        print(
            f"{Fore.MAGENTA}💎 JACKPOTS CAPTURADOS: {jackpots_found}/{test_size}{Style.RESET_ALL}"
        )
        print("=" * 60)
