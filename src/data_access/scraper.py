# src/data_access/scraper.py
import os
import requests
import pandas as pd
import io
import warnings
from rich.console import Console
from src.data_access.config import DATA_FOLDER, get_lottery_profile

warnings.simplefilter(
    "ignore", requests.packages.urllib3.exceptions.InsecureRequestWarning
)
console = Console()


def actualizar_csv(game_code: str = "melate_retro"):
    """Sincroniza historico para el juego solicitado."""
    profile = get_lottery_profile(game_code)
    csv_path = os.path.join(DATA_FOLDER, profile.csv_filename)

    console.print(
        f"\n[bold blue]📡 INICIANDO RECOLECCION DE INTELIGENCIA: {profile.display_name}[/bold blue]"
    )

    try:
        raw_content = _download_historical_data(profile.source_url)
        if not raw_content:
            console.print(
                "[bold red]❌ ERROR: El servidor no devolvió bytes de datos.[/bold red]"
            )
            return False

        df = _parse_csv_content(raw_content)

        if df.empty or len(df.columns) < 5:
            console.print(
                "[bold yellow]⚠️ ADVERTENCIA: Estructura de CSV inválida o vacía.[/bold yellow]"
            )
            return False

        # Limpieza base de headers
        df.columns = [c.strip().replace('"', "") for c in df.columns]

        if "tris" in profile.code:
            df = _normalize_tris_columns(df)

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False, encoding="utf-8")

        console.print(
            f"[bold green]✅ SINCRONIZACION EXITOSA:[/bold green] {len(df)} registros en local.\n[dim]{csv_path}[/]"
        )
        return True

    except Exception as e:
        console.print(f"[bold red]⚠️ FALLO CRÍTICO EN SCRAPER:[/bold red] {str(e)}")
        return False


def _download_historical_data(source_url: str):
    """Descarga usando Session y emulación de navegador de alto nivel."""
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "es-MX,es;q=0.8,en-US;q=0.5,en;q=0.3",
        "Referer": "https://www.google.com/",
    }

    try:
        # Forzamos allow_redirects=True para seguir el rastro del CSV.
        response = session.get(
            source_url, headers=headers, timeout=20, verify=False, allow_redirects=True
        )

        if response.status_code == 200:
            sample = response.text[:512].lower()
            if "<html" in sample:
                console.print(
                    "[bold red]❌ ERROR: El servidor devolvió HTML en lugar de CSV (Posible bloqueo).[/bold red]"
                )
                return None
            return response.content

        console.print(
            f"[bold red]❌ STATUS CODE ERROR: {response.status_code}[/bold red]"
        )
    except Exception as e:
        console.print(f"[bold red]❌ ERROR DE CONEXIÓN: {str(e)}[/bold red]")

    return None


def _parse_csv_content(raw_content: bytes) -> pd.DataFrame:
    """Parsea CSV con estrategias de encoding/separador tolerantes."""
    decode_candidates = ("utf-8-sig", "utf-8", "latin-1")
    last_error = None

    for encoding in decode_candidates:
        try:
            text = raw_content.decode(encoding, errors="strict")
            return pd.read_csv(io.StringIO(text), sep=None, engine="python")
        except Exception as e:
            last_error = e

    # Ultimo intento mas permisivo
    try:
        return pd.read_csv(io.BytesIO(raw_content), sep=",", encoding="latin-1")
    except Exception:
        raise ValueError(f"No se pudo parsear CSV descargado: {last_error}")


def _normalize_tris_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza encabezados de Tris a las columnas consumidas por LotteryLoader:
    CONCURSO, FECHA y R1..R5 (o NUMEROS como fallback).
    """
    rename_map = {}
    upper_cols = {c.upper(): c for c in df.columns}

    # IDs y fechas
    for source in ("CONCURSO", "SORTEO", "ID", "DRAW_ID"):
        if source in upper_cols:
            rename_map[upper_cols[source]] = "CONCURSO"
            break
    for source in ("FECHA", "DATE", "FEC"):
        if source in upper_cols:
            rename_map[upper_cols[source]] = "FECHA"
            break

    # Digitos en variantes comunes
    for idx in range(1, 6):
        candidates = (f"R{idx}", f"D{idx}", f"F{idx}", f"DIGITO{idx}", f"DIGIT{idx}")
        for source in candidates:
            if source in upper_cols:
                rename_map[upper_cols[source]] = f"R{idx}"
                break

    df = df.rename(columns=rename_map)

    # Si viene una sola columna de numero completo, la conservamos.
    if "NUMEROS" not in df.columns and "NUMERO" in df.columns:
        df = df.rename(columns={"NUMERO": "NUMEROS"})

    # Validacion minima para que el loader no falle despues.
    required_base = {"CONCURSO", "FECHA"}
    has_digits = set(f"R{i}" for i in range(1, 6)).issubset(df.columns)
    has_compact = "NUMEROS" in df.columns
    if not required_base.issubset(df.columns) or not (has_digits or has_compact):
        missing = sorted(list(required_base - set(df.columns)))
        raise ValueError(
            "CSV de Tris sin columnas esperadas. "
            f"Faltan base={missing}; detectadas={list(df.columns)}"
        )

    return df
