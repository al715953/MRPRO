import sys
import os
from datetime import datetime
from colorama import Fore, Style
from src.data_access.loader import MelateLoader
from src.data_access.config import CSV_FILE_PATH
import src.data_access.scraper as scraper
from src.interface.cli import ConsoleUI
from src.interface.mission_controller import MissionController

# Configuración de rutas
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def initialize_data_layer():
    """Valida integridad y actualiza vía scraping si es necesario."""
    loader = MelateLoader(CSV_FILE_PATH)
    print(
        f"{Fore.CYAN}🔍 Verificando integridad de la base de datos...{Style.RESET_ALL}"
    )

    needs_update = False
    try:
        history = loader.load_data()
        if not history.dates:
            needs_update = True
        else:
            # Si el último sorteo tiene más de 4 días, actualizar
            last_date = history.dates[-1]
            if isinstance(last_date, str):
                last_date = datetime.strptime(last_date, "%d/%m/%Y").date()
            if (datetime.now().date() - last_date).days > 4:
                needs_update = True
    except:
        needs_update = True

    if needs_update:
        print(
            f"{Fore.YELLOW}📥 Datos obsoletos. Iniciando Web Scraping...{Style.RESET_ALL}"
        )
        scraper.descargar_datos(CSV_FILE_PATH)

    return loader.load_data()


def main():
    ui = ConsoleUI()
    ui.show_welcome()

    # 1. Fase de Carga e Integridad
    history = initialize_data_layer()
    if not history.dates:
        print(
            f"{Fore.RED}❌ ERROR CRÍTICO: No se pudieron cargar datos.{Style.RESET_ALL}"
        )
        return

    print(
        f"{Fore.GREEN}✅ Sistema listo. {len(history.winning_numbers)} sorteos en memoria.{Style.RESET_ALL}"
    )

    # 2. Inyección de dependencias al controlador
    controller = MissionController(ui, history)

    # 3. Bucle de ejecución
    while True:
        opcion = ui.show_main_menu()
        if opcion == "0":
            print(
                f"{Fore.CYAN}👋 Misión finalizada. ¡Hasta la próxima!{Style.RESET_ALL}"
            )
            break
        controller.run_mission(opcion)


if __name__ == "__main__":
    main()
