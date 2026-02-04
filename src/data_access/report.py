# src/data_access/report.py

import csv
import os
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

# Importación de rutas y etiquetas desde la configuración global
from src.data_access.config import FILE_APUESTAS, VERSION_TAG, MASTER_LOG_PATH

console = Console()

def guardar_prediccion(tickets):
    """
    Guarda las combinaciones finales generadas en el archivo de apuestas.
    """
    os.makedirs(os.path.dirname(FILE_APUESTAS), exist_ok=True)
    file_exists = os.path.isfile(FILE_APUESTAS)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(FILE_APUESTAS, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Fecha", "Version", "T1", "T2", "T3", "T4", "T5", "T6"])
            for t in tickets:
                writer.writerow([timestamp, VERSION_TAG] + sorted(t))
        console.print(f"\n[bold green]✅ ESTATUS:[/bold green] {len(tickets)} tickets guardados en {FILE_APUESTAS}")
    except Exception as e:
        console.print(f"[bold red]❌ ERROR al guardar tickets:[/bold red] {e}")

def guardar_forensic_csv(forensic_data_list):
    """
    Persistencia V7.17: Guarda el log detallado incluyendo el tamaño del universo (univ_size).
    """
    os.makedirs(os.path.dirname(MASTER_LOG_PATH), exist_ok=True)
    file_exists = os.path.isfile(MASTER_LOG_PATH)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    try:
        with open(MASTER_LOG_PATH, mode="a", newline="") as f:
            # Columnas alineadas para análisis de datos posterior
            fieldnames = ['timestamp', 'tag', 'draw_id', 'hits', 'rank', 'proximity', 'ai_score', 'geo_score', 'univ_size']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            for data in forensic_data_list:
                # Aseguramos que los metadatos de sesión estén presentes
                data_to_write = {
                    'timestamp': timestamp,
                    'tag': VERSION_TAG,
                    'draw_id': data.get('draw_id'),
                    'hits': data.get('hits'),
                    'rank': data.get('rank'),
                    'proximity': data.get('proximity'),
                    'ai_score': data.get('ai_score'),
                    'geo_score': data.get('geo_score'),
                    'univ_size': data.get('univ_size')
                }
                writer.writerow(data_to_write)
    except Exception as e:
        console.print(f"[bold red]❌ ERROR al guardar log forense:[/bold red] {e}")

def render_forensic_line(d_id, hits, ai_val, geo_val, audit_data):
    """
    LOG SNIPER V7.17: Telemetría meritocrática en consola.
    Diferencia entre aciertos reales (Top 20) vs Fantasmas (Rank alto).
    """
    univ_size = audit_data.get('univ_size', 0)
    rank = audit_data.get('rank', 99999)
    prox = audit_data.get('proximity', 999)
    
    # --- DETERMINACIÓN DEL STATUS MERITOCRÁTICO ---
    if prox == 0:
        if rank <= 20:
            status = Text("🎯 JACKPOT", style="bold green")
        elif rank <= 100:
            status = Text("🔥 TOP 100", style="bold yellow")
        else:
            status = Text("👻 GHOST", style="dim white") # Estaba en el pajar, pero muy hondo
    else:
        status = Text("❌ MISS", style="dim red")

    # Colores dinámicos para hits y proximidad
    potential_color = "bold green" if hits >= 4 else "bold yellow" if hits == 3 else "dim"
    prox_color = "bold cyan" if prox == 0 else "bold magenta" if prox < 15 else "dim"

    # Ensamblado de la línea (Respetando tu estética original)
    line = Text.assemble(
        (f"#{d_id:4d} ", "bold white"),
        ("| ", "white"),
        (f"U: {univ_size:7,d} ", "dim cyan"),
        ("| ", "white"),
        (f" {hits}/6 ", potential_color),
        ("| ", "white"),
        ("AI: ", "dim"),
        (f"{ai_val:.4f} ", "yellow"),
        ("| Geo: ", "dim"),
        (f"{geo_val:.4f}", "yellow"),
        (f" | Rank: #{rank:,}", "cyan"),
        (f" | Dist: {prox:,} ", prox_color),
        ("| ", "white"),
        status,
    )
    console.print(line)

def render_final_dashboard(size, invest, earn, funnel, dist):
    """
    DASHBOARD V7.17: Resumen financiero y análisis de Jackpots capturados.
    funnel: Diccionario con la distribución de mejores hits en el universo total.
    dist: Diccionario con la distribución de hits en el Top 20 (compra real).
    """
    balance = earn - invest
    
    # 1. Tabla de Hits Reales (Lo que cayó en el Top 20 de compra)
    dist_table = Table(title="[bold cyan]🎯 DISTRIBUCIÓN DE HITS REALES (ZONA DE COMPRA)[/]", box=box.SIMPLE, expand=True)
    dist_table.add_column("Categoría"), dist_table.add_column("Tickets", justify="right")
    
    for h in range(6, -1, -1):
        count = dist.get(h, 0)
        color = "green" if h >= 4 else "yellow" if h == 3 else "white"
        dist_table.add_row(f"{h}/6 Aciertos", f"[{color}]{count}[/]")

    # 2. Tabla de Potencial (Lo que los filtros lograron meter al pajar)
    pot_table = Table(title="[bold yellow]🏆 RESUMEN DE POTENCIAL (JACKPOTS EN UNIVERSO)[/]", box=box.ROUNDED, expand=True)
    pot_table.add_column("Premio", style="cyan")
    pot_table.add_column("Detectados en Pajar", justify="right", style="bold white")
    
    for h in range(6, 1, -1):
        count = funnel.get(h, 0)
        color = "bold green" if h >= 4 else "yellow"
        pot_table.add_row(f"{h}/6 Aciertos", f"[{color}]{count}[/]")

    # 3. Panel Financiero
    summary_text = Text.assemble(
        ("\n💰 BALANCE DE MISIÓN\n", "bold white"),
        (f"Sorteos Testeados: {size}\n", "dim"),
        (f"Inversión Total:  ${invest:,.2f}\n", "white"),
        (f"Ganancia Total:   ${earn:,.2f}\n", "green" if earn > invest else "white"),
        (f"Resultado Neto:   ${balance:,.2f}\n", "bold green" if balance >= 0 else "bold red")
    )

    console.print(Panel(summary_text, box=box.DOUBLE, title="[bold white]MRPRO V7.17 - DASHBOARD FINAL[/]"))
    console.print(dist_table)
    console.print(pot_table)