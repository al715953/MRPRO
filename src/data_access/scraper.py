# src/data_access/scraper.py
import os
import requests
import pandas as pd
import io
import warnings
from urllib3.exceptions import InsecureRequestWarning
from rich.console import Console
from src.data_access.config import URL_MELATE, CSV_FILE_PATH

# Desactivar advertencias de seguridad al usar verify=False para mantener el log profesional
warnings.simplefilter("ignore", InsecureRequestWarning)
console = Console()


def actualizar_csv():
    """EntryPoint V15: Sincronización de datos con bypass de SSL."""
    console.print(f"\n[bold blue]📡 INICIANDO RECOLECCIÓN DE INTELIGENCIA[/bold blue]")

    try:
        raw_content = _download_historical_data()
        if not raw_content:
            console.print(
                "[bold red]❌ ERROR: No se recibió contenido del servidor.[/bold red]"
            )
            return False

        # Usamos io.StringIO para mayor compatibilidad con versiones modernas de pandas
        df = pd.read_csv(io.StringIO(raw_content))

        if df.empty:
            console.print(
                "[bold yellow]⚠️ ADVERTENCIA: El historial descargado está vacío.[/bold yellow]"
            )
            return False

        # Persistencia en la ruta configurada
        os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
        df.to_csv(CSV_FILE_PATH, index=False, encoding="utf-8")

        console.print(
            f"[bold green]✅ DATOS SINCRONIZADOS:[/bold green] {len(df)} sorteos listos."
        )
        return True

    except Exception as e:
        console.print(f"[bold red]⚠️ FALLO CRÍTICO EN SCRAPER:[/bold red] {str(e)}")
        return False


def _download_historical_data():
    """Descarga con reintento automático y bypass de certificado."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
    }

    try:
        # Intento 1: Con verificación (Seguridad estándar)
        response = requests.get(URL_MELATE, headers=headers, timeout=15, verify=True)
        response.raise_for_status()
        return response.text
    except (requests.exceptions.SSLError, requests.exceptions.ConnectionError):
        # Intento 2: Bypass de SSL (Misión Crítica: Los datos son prioridad)
        console.print(
            "[bold yellow]⚠️ Alerta de SSL detectada. Activando protocolo de bypass...[/bold yellow]"
        )
        response = requests.get(URL_MELATE, headers=headers, timeout=15, verify=False)
        if response.status_code == 200:
            return response.text

    return None
