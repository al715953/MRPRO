import os
from collections import Counter
from colorama import Fore, Style
from src.data_access.config import CYAN, RESET, VERDE, BLANCO_B
from src.domain.dtos import PredictionResultDTO, DrawHistoryDTO


class ConsoleUI:
    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def show_welcome(self):
        self.clear_screen()
        print("══════════════════════════════════════════════════")
        print(f"            🎱  MRPRO {VERDE}- {BLANCO_B}SYSTEM V3 (AI)  🎱       ")
        print("══════════════════════════════════════════════════")
        print(
            f"{Fore.CYAN}    Arquitectura: Clean Code | Motor: Híbrido (Numpy + AI){Style.RESET_ALL}"
        )
        print("══════════════════════════════════════════════════")

    def show_main_menu(self) -> str:
        """
        Muestra el menú principal alineado con las opciones de main.py
        """
        print("\nSelecciona una opción:")
        print(f"1. 📜 {Fore.WHITE}Ver Historial de Sorteos{Style.RESET_ALL}")
        print(f"2. 📊 {Fore.WHITE}Análisis de Frecuencia (Hot/Cold){Style.RESET_ALL}")
        print(f"3. 🎲 {Fore.WHITE}Simulación Monte Carlo (Baseline){Style.RESET_ALL}")
        print(
            f"4. 🧠 {Fore.MAGENTA}Optimizador de Parámetros (Grid Search){Style.RESET_ALL}"
        )
        print(f"5. 🌌 {Fore.CYAN}Generar Universo Reducido (Fase 1){Style.RESET_ALL}")
        print(
            f"6. 📡 {Fore.YELLOW}Laboratorio de Pruebas (Backtest & QA){Style.RESET_ALL}"
        )
        print(
            f"7. 🎯 {Fore.GREEN}SELECTOR GENÉTICO FINAL (Sniper + AI){Style.RESET_ALL}"
        )
        print("-" * 40)
        print("0. 🚪 Salir")
        return input(f"\n{Fore.GREEN}>> Tu orden, Arquitecto: {Style.RESET_ALL}")

    def show_history(self, history: DrawHistoryDTO):
        print(f"\n{Fore.CYAN}📜 ÚLTIMOS 10 SORTEOS:{Style.RESET_ALL}")
        # Mostramos los últimos 10 para no saturar
        total = len(history.winning_numbers)
        start = max(0, total - 10)

        for i in range(start, total):
            date = history.dates[i] if i < len(history.dates) else "??"
            concurso = history.concursos[i] if i < len(history.concursos) else "??"
            nums = history.winning_numbers[i]
            # Formateamos bonito
            nums_str = ", ".join([f"{n:02d}" for n in nums])
            print(f"📅 {date} | Sorteo #{concurso} | 🎱 [{nums_str}]")

    def analyze_frequency(self, history: DrawHistoryDTO, total_balls: int):
        print(f"\n{Fore.CYAN}📊 ANÁLISIS DE FRECUENCIA (Top & Flop){Style.RESET_ALL}")

        # Aplanar todas las listas de números en una sola
        all_nums = [n for draw in history.winning_numbers for n in draw[:6]]
        counts = Counter(all_nums)

        # Top 5 Calientes
        print(f"\n🔥 {Fore.RED}NÚMEROS MÁS CALIENTES:{Style.RESET_ALL}")
        for num, freq in counts.most_common(5):
            print(f"   🎱 {num:02d}: {freq} veces")

        # Top 5 Fríos (Calculando los que menos salen)
        # Inicializamos todos en 0 para contar incluso los que no han salido
        full_counts = {n: 0 for n in range(1, total_balls + 1)}
        full_counts.update(counts)

        sorted_cold = sorted(full_counts.items(), key=lambda x: x[1])

        print(f"\n❄️ {Fore.BLUE}NÚMEROS MÁS FRÍOS:{Style.RESET_ALL}")
        for num, freq in sorted_cold[:5]:
            print(f"   🧊 {num:02d}: {freq} veces")

    def show_prediction_results(self, result: PredictionResultDTO):
        print(
            f"\n{Fore.GREEN}🎫 RESULTADOS GENERADOS POR: {result.strategy_name}{Style.RESET_ALL}"
        )
        print("=" * 40)

        if not result.tickets:
            print(f"{Fore.RED}⚠ No se generaron tickets.{Style.RESET_ALL}")
            return

        for i, ticket in enumerate(result.tickets, 1):
            ticket_str = ", ".join([f"{num:02d}" for num in sorted(ticket)])
            print(f"Ticket #{i:02d}: [ {ticket_str} ]")

        print("=" * 40)
        print("✅ Proceso finalizado.")
