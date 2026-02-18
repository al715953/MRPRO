# src/data_access/scraper.py
import io
import os
import requests
import pandas as pd
import warnings
from rich.console import Console

from src.data_access.config import CSV_FILE_PATH, get_lottery_profile

warnings.simplefilter(
    "ignore", requests.packages.urllib3.exceptions.InsecureRequestWarning
)
console = Console()


def actualizar_csv(game_code: str = "melate_retro"):
    """Sincronización de históricos para el juego indicado."""
    profile = get_lottery_profile(game_code)
    destination_path = os.path.join(
        os.path.dirname(CSV_FILE_PATH), profile.csv_filename
    )

    console.print(
        f"\n[bold blue]📡 INICIANDO RECOLECCIÓN ({profile.display_name})[/bold blue]"
    )

    try:
        raw_content = _download_historical_data(profile.source_url)
        if not raw_content:
            console.print(
                "[bold red]❌ ERROR: El servidor no devolvió bytes de datos.[/bold red]"
            )
            return False

        try:
            df = pd.read_csv(io.StringIO(raw_content), sep=None, engine="python")
        except Exception:
            df = pd.read_csv(io.BytesIO(raw_content.encode("latin-1")), sep=",")

        if df.empty or len(df.columns) < 5:
            console.print(
                "[bold yellow]⚠️ ADVERTENCIA: Estructura de CSV inválida o vacía.[/bold yellow]"
            )
            return False

        df.columns = [c.strip().replace('"', "") for c in df.columns]

        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        df.to_csv(destination_path, index=False, encoding="utf-8")

        console.print(
            f"[bold green]✅ SINCRONIZACIÓN EXITOSA:[/bold green] {len(df)} registros en local."
        )
        return True

    except Exception as e:
        console.print(f"[bold red]⚠️ FALLO CRÍTICO EN SCRAPER:[/bold red] {str(e)}")
        return False


def _download_historical_data(url: str):
    """Descarga usando Session y emulación de navegador de alto nivel."""
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "es-MX,es;q=0.8,en-US;q=0.5,en;q=0.3",
        "Referer": "https://www.google.com/",
    }

    try:
        response = session.get(
            url, headers=headers, timeout=20, verify=False, allow_redirects=True
        )

        if response.status_code == 200:
            if "<html" in response.text.lower():
                console.print(
                    "[bold red]❌ ERROR: El servidor devolvió HTML en lugar de CSV (Posible bloqueo).[/bold red]"
                )
                return None
            return response.text

        console.print(
            f"[bold red]❌ STATUS CODE ERROR: {response.status_code}[/bold red]"
        )
    except Exception as e:
        console.print(f"[bold red]❌ ERROR DE CONEXIÓN: {str(e)}[/bold red]")

    return None
