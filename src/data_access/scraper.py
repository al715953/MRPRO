# src/data_access/scraper.py
import os
import requests
import pandas as pd
import io
import warnings
from rich.console import Console
from src.data_access.config import URL_MELATE, CSV_FILE_PATH

warnings.simplefilter(
    "ignore", requests.packages.urllib3.exceptions.InsecureRequestWarning
)
console = Console()


def actualizar_csv():
    """EntryPoint V15.1: Sincronización con inspección de integridad."""
    console.print(
        f"\n[bold blue]📡 INICIANDO RECOLECCIÓN DE INTELIGENCIA (V15.1)[/bold blue]"
    )

    try:
        raw_content = _download_historical_data()
        if not raw_content:
            console.print(
                "[bold red]❌ ERROR: El servidor no devolvió bytes de datos.[/bold red]"
            )
            return False

        # --- MEJORA: Detección Dinámica de Encoding ---
        # Intentamos leer con 'utf-8', si falla saltamos a 'latin-1' (Común en MX)
        try:
            df = pd.read_csv(io.StringIO(raw_content), sep=None, engine="python")
        except Exception:
            df = pd.read_csv(io.BytesIO(raw_content.encode("latin-1")), sep=",")

        if df.empty or len(df.columns) < 5:
            console.print(
                "[bold yellow]⚠️ ADVERTENCIA: Estructura de CSV inválida o vacía.[/bold yellow]"
            )
            return False

        # --- LIMPIEZA DE CABECERAS ---
        # A veces el scraper baja basura en los nombres de columnas
        df.columns = [c.strip().replace('"', "") for c in df.columns]

        os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
        df.to_csv(CSV_FILE_PATH, index=False, encoding="utf-8")

        console.print(
            f"[bold green]✅ SINCRONIZACIÓN EXITOSA:[/bold green] {len(df)} registros en local."
        )
        return True

    except Exception as e:
        console.print(f"[bold red]⚠️ FALLO CRÍTICO EN SCRAPER:[/bold red] {str(e)}")
        return False


def _download_historical_data():
    """Descarga usando Session y emulación de navegador de alto nivel."""
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "es-MX,es;q=0.8,en-US;q=0.5,en;q=0.3",
        "Referer": "https://www.google.com/",
    }

    try:
        # Forzamos allow_redirects=True para seguir el rastro del CSV
        response = session.get(
            URL_MELATE, headers=headers, timeout=20, verify=False, allow_redirects=True
        )

        if response.status_code == 200:
            # Si el contenido es HTML en lugar de CSV, algo anda mal (posible redirección a login/mantenimiento)
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
