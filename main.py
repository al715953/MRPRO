# main.py

import sys
import os
from datetime import datetime
from colorama import Fore, Style

import src.data_access.report as report
import src.data_access.scraper as scraper
from src.data_access.loader import MelateLoader
from src.data_access.config import CSV_FILE_PATH
from src.interface.cli import ConsoleUI
from src.interface.mission_controller import MissionController


# Asegurar que el sistema reconozca las rutas del proyecto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def initialize_data_layer(ui):
    """
    Carga de datos con lógica de sincronización inteligente.
    Detecta si los datos son obsoletos basándose en la fecha máxima real.
    """
    loader = MelateLoader(CSV_FILE_PATH)

    # Intento de carga inicial
    try:
        history = loader.load_data()
    except Exception:
        ui.console.print(
            "[yellow]⚠️ Base de datos no encontrada. Iniciando descarga...[/]"
        )
        scraper.descargar_datos(CSV_FILE_PATH)
        history = loader.load_data()

    if not history or not history.dates:
        return None

    # LÓGICA DE ACTUALIZACIÓN INTELIGENTE (UX Refined)
    # Buscamos la fecha más reciente (el CSV es descendente, pero max() es infalible)
    try:
        dates_obj = []
        for d in history.dates:
            if isinstance(d, str):
                dates_obj.append(datetime.strptime(d, "%d/%m/%Y").date())
            else:
                dates_obj.append(d)

        ultima_fecha = max(dates_obj)
        dias_desde_ultimo = (datetime.now().date() - ultima_fecha).days

        # Si han pasado más de 3 días, intentamos sincronizar (Melate Retro: Mar, Jue, Sáb)
        if dias_desde_ultimo > 3:
            ui.console.print(
                f"[cyan]🌐 Sincronizando nuevos sorteos (Último: {ultima_fecha})...[/]"
            )
            if scraper.actualizar_csv():
                history = loader.load_data()
    except Exception as e:
        ui.console.print(
            f"[dim red]Aviso: No se pudo verificar caducidad de datos ({e})[/]"
        )

    return history


def main():
    # 1. Preparación de Interfaz
    ui = ConsoleUI()
    ui.clear_screen()
    ui.console.print("[bold white]📡 INICIANDO SISTEMA MRPRO V15...[/]", justify="left")

    # 2. Inicialización de Datos (Carga Silenciosa)
    loader = MelateLoader(CSV_FILE_PATH)
    history = initialize_data_layer(ui)

    if not history:
        ui.console.print(
            "[bold red]❌ ERROR CRÍTICO:[/] No se pudo establecer conexión con el histórico."
        )
        return

    # 3. Inyección de dependencias al controlador
    controller = MissionController(ui, history)

    # 4. BUCLE OPERATIVO (Command Center)
    while True:
        # UX: Refresco visual de bienvenida
        ui.show_welcome()

        # Sincronización de estado para el HUD (Heads-Up Display)
        # Buscamos el máximo real para evitar el error del ID incorrecto
        ultimo_id = max(history.concursos)
        proximo_id = ultimo_id + 1

        # Verificamos si ya hay apuestas en el Ledger para el HUD
        apuestas_bloqueadas = report.tiene_apuestas_pendientes(proximo_id)

        # Renderizado de la barra de estado superior
        ui.show_status_bar(history, tiene_apuestas=apuestas_bloqueadas)

        # Despliegue de Menú y captura de orden
        opcion = ui.show_main_menu()

        if opcion == "0":
            ui.clear_screen()
            ui.console.print(
                f"\n[bold cyan]🔌 Sesión finalizada correctamente. Sistema fuera de línea.[/]\n"
            )
            break

        # Ejecución de Misiones
        try:
            controller.run_mission(opcion)

            # UX: Si hubo sincronización o liquidación, refrescamos la memoria del sistema
            if opcion.upper() in ["8", "9"]:
                history = loader.load_data()
                controller.history = history

        except Exception as e:
            ui.console.print(f"\n[bold red]⚠️ ERROR EN MISIÓN:[/] {e}")
            ui.console.input(
                f"\n[yellow]Presiona ENTER para reestablecer consola...[/]"
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.RED}🛑 Interrupción forzada por el usuario.{Style.RESET_ALL}")
        sys.exit()
