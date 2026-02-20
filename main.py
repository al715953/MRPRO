# main.py

import sys
import os
from datetime import datetime
from colorama import Fore, Style

import src.data_access.report as report
import src.data_access.scraper as scraper
from src.data_access.loader import LotteryLoader
from src.data_access.config import LOTTERY_PROFILES
from src.interface.cli import ConsoleUI
from src.interface.mission_controller import MissionController

# Asegurar que el sistema reconozca las rutas del proyecto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def initialize_data_layer(ui, loader, profile):
    """
    Carga de datos genérica basada en el perfil seleccionado.
    """
    try:
        history = loader.load_data()
        ui.console.print(
            f"[green]✅ Datos de {profile.display_name} cargados correctamente.[/]"
        )
    except (FileNotFoundError, OSError, ValueError) as e:
        ui.console.print(f"[yellow]⚠️ Error al cargar base de datos: {e}[/]")
        ui.console.print("[cyan]Iniciando descarga de emergencia...[/]")

        if scraper.actualizar_csv(profile.code):
            history = loader.load_data()
        else:
            ui.console.print(
                "[bold red]❌ No se pudieron obtener datos para este sorteo.[/]"
            )
            return None

    if not history or not history.dates:
        return None

    return history


def select_lottery_profile(ui):
    """
    Interfaz de selección de sorteo al inicio.
    """
    ui.console.print("\n[bold cyan]🛸 MRPRO SYSTEM - SELECCIÓN DE MISIÓN[/]")
    ui.console.print("=" * 45)
    ui.console.print("1. Melate Retro (Tradicional)")
    ui.console.print("2. Tris Multiplicador (Alta Frecuencia)")
    ui.console.print("0. Salir")

    try:
        choice = ui.console.input("\n[bold yellow]Selecciona el objetivo: [/]")
    except KeyboardInterrupt:
        ui.console.print(
            "\n[yellow]Interrupción detectada (Ctrl+C). "
            "Ingresa 1, 2 o 0 para continuar.[/]"
        )
        return select_lottery_profile(ui)
    except EOFError:
        ui.console.print(
            "\n[bold red]No hay entrada interactiva disponible (EOF). Cerrando.[/]"
        )
        return None

    if choice == "1":
        return LOTTERY_PROFILES["melate_retro"]
    elif choice == "2":
        return LOTTERY_PROFILES["tris_multiplicador"]
    elif choice == "0":
        sys.exit()
    else:
        ui.console.print("[red]Opción inválida. Reintentando...[/]")
        return select_lottery_profile(ui)


def main():
    ui = ConsoleUI()
    ui.clear_screen()

    # 1. Selección del Perfil de Lotería
    profile = select_lottery_profile(ui)
    if profile is None:
        return

    # 2. Inicialización del Loader con el perfil inyectado
    loader = LotteryLoader(profile)

    # 3. Inicialización de Capa de Datos
    history = initialize_data_layer(ui, loader, profile)
    if not history:
        ui.console.print("[bold red]FATAL ERROR: Capa de datos no inicializada.[/]")
        return

    # 4. Configuración del Controlador de Misión
    # Inyectamos el historial y el perfil para que las estrategias sepan qué reglas usar
    controller = MissionController(ui, history, profile)

    # 5. Loop Principal de Operaciones
    while True:
        ui.clear_screen()

        # Metadata de HUD
        ultimo_id = max(history.concursos)
        proximo_id = ultimo_id + 1
        apuestas_bloqueadas = report.tiene_apuestas_pendientes(proximo_id)

        # Renderizado de la barra de estado superior
        ui.show_status_bar(
            history, tiene_apuestas=apuestas_bloqueadas, profile=profile
        )

        # Despliegue de Menú y captura de orden
        opcion = ui.show_main_menu(profile)

        if opcion == "0":
            ui.clear_screen()
            ui.console.print(
                f"\n[bold cyan]🔌 Sesión finalizada correctamente. Sistema fuera de línea.[/]\n"
            )
            break

        # Ejecución de Misiones
        try:
            controller.run_mission(opcion)

            # UX: Si hubo sincronización o liquidación, refrescamos la memoria
            if opcion.upper() in ["5", "8"]:
                history = loader.load_data()
                controller.history = history

        except Exception as e:
            ui.console.print(f"\n[bold red]⚠️ ERROR EN MISIÓN:[/] {e}")
            import traceback

            # Solo para debug de arquitectura en desarrollo:
            # ui.console.print(traceback.format_exc())
            ui.console.input(
                f"\n[yellow]Presiona ENTER para reestablecer consola...[/]"
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            f"\n{Fore.RED}Interrupción forzada por el usuario. Cerrando...{Style.RESET_ALL}"
        )
