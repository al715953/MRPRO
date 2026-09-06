# src/data_access/report.py

import csv
import os
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from src.data_access.config import FILE_APUESTAS, VERSION_TAG
from src.core.rules import MelateRetroRules

console = Console()

LEDGER_FIELDS = [
    "Fecha",
    "Concurso",
    "Version",
    "T1",
    "T2",
    "T3",
    "T4",
    "T5",
    "T6",
    "Status",
    "Premio",
    "AciertosNaturales",
    "Adicional",
    "CategoriaPremio",
]


def _atomic_write_ledger(path, rows, fieldnames):
    temporary_path = f"{path}.tmp"
    try:
        with open(temporary_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _ensure_ledger_schema(path):
    """Add prize-detail columns to an existing ledger without losing rows."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return list(LEDGER_FIELDS)
    with open(path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_fields = list(reader.fieldnames or [])
        rows = list(reader)
    fieldnames = existing_fields + [
        field for field in LEDGER_FIELDS if field not in existing_fields
    ]
    if fieldnames != existing_fields:
        _atomic_write_ledger(path, rows, fieldnames)
    return fieldnames

def tiene_apuestas_pendientes(concurso_id: int) -> bool:
    """
    Regla de Oro: Evita la sobre-escritura de apuestas aceptadas.
    Verifica si el concurso ya tiene registros en el Ledger.
    """
    if not os.path.exists(FILE_APUESTAS):
        return False
    
    try:
        with open(FILE_APUESTAS, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Concurso") == str(concurso_id):
                    return True
    except Exception:
        return False
    return False

def guardar_prediccion(tickets, concurso_id: int):
    """
    Persiste las apuestas en el Ledger oficial con el ID del concurso.
    """
    os.makedirs(os.path.dirname(FILE_APUESTAS), exist_ok=True)
    file_exists = os.path.isfile(FILE_APUESTAS) and os.path.getsize(FILE_APUESTAS) > 0
    fieldnames = _ensure_ledger_schema(FILE_APUESTAS)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    try:
        with open(FILE_APUESTAS, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            
            for t in tickets:
                numbers = sorted(t)
                row = {
                    "Fecha": timestamp,
                    "Concurso": concurso_id,
                    "Version": VERSION_TAG,
                    "Status": "Pendiente",
                    "Premio": 0.0,
                    "AciertosNaturales": "",
                    "Adicional": "",
                    "CategoriaPremio": "",
                }
                row.update({f"T{i}": number for i, number in enumerate(numbers, 1)})
                writer.writerow(row)
                
        console.print(f"\n[bold green]✅ LEDGER ACTUALIZADO:[/bold green] {len(tickets)} tickets bloqueados para Sorteo #{concurso_id}")
    except Exception as e:
        console.print(f"[bold red]❌ ERROR CRÍTICO al acceder al Ledger:[/bold red] {e}")

def generar_ticket_limpio(tickets, concurso_id: int):
    """
    Genera el archivo minimalista para el punto de venta (UX de Producción).
    """
    path_txt = os.path.join(os.path.dirname(FILE_APUESTAS), f"tickets_sorteo_{concurso_id}.txt")
    release_name = VERSION_TAG.split("_", 1)[0]
    
    try:
        with open(path_txt, "w", encoding="utf-8") as f:
            f.write(f"--- MRPRO {release_name}: BALANCED EXPLORATION ---\n")
            f.write(f"SORTEO: #{concurso_id}\n")
            f.write(f"ESTRATEGIA: {VERSION_TAG}\n")
            f.write("-" * 30 + "\n")
            for i, t in enumerate(tickets, 1):
                t_str = " ".join(f"{n:02d}" for n in sorted(t))
                f.write(f"({i:02d})  {t_str}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total: {len(tickets)} tickets | Inversión: ${len(tickets)*10:.2f}\n")
            f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        
        console.print(Panel(f"📋 [bold cyan]ARCHIVO DE COMPRA LISTO:[/]\n{path_txt}", border_style="cyan"))
    except Exception as e:
        console.print(f"[red]Error al generar TXT de salida: {e}[/red]")

def liquidar_cartera(history):
    """
    Motor Forense de Liquidación: Cruza el Ledger contra el Historial Real.
    Calcula el ROI real basado en lo que realmente se jugó.
    """
    if not os.path.exists(FILE_APUESTAS):
        return None

    rules = MelateRetroRules()
    
    # Mapeo de resultados reales: {concurso_id: [n1, n2, n3, n4, n5, n6, ad]}
    dict_resultados = {str(c): n for c, n in zip(history.concursos, history.winning_numbers)}
    
    rows_actualizadas = []
    totales = {
        "inversion": 0.0,
        "ganancia": 0.0,
        "hits": 0,
        "concursos": set(),
        "desglose_premios": {},
    }

    try:
        fieldnames = _ensure_ledger_schema(FILE_APUESTAS)
        with open(FILE_APUESTAS, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                c_id = row["Concurso"]
                totales["inversion"] += rules.ticket_cost
                totales["concursos"].add(c_id)

                if c_id in dict_resultados:
                    # Extraer ticket del CSV
                    ticket = [int(row[f"T{i}"]) for i in range(1, 7)]
                    target = dict_resultados[c_id]
                    
                    # Validar contra reglas oficiales
                    h_n, h_a = rules.validate_ticket(ticket, target)
                    premio = rules.calculate_prize(h_n, h_a)
                    categoria = rules.prize_category(h_n, h_a)

                    row["Status"] = "🏆 GANADOR" if premio > 0 else "Validado"
                    row["Premio"] = premio
                    row["AciertosNaturales"] = h_n
                    row["Adicional"] = "Sí" if h_a else "No"
                    row["CategoriaPremio"] = categoria
                    totales["ganancia"] += premio
                    bucket = totales["desglose_premios"].setdefault(
                        categoria, {"tickets": 0, "ganancia": 0.0}
                    )
                    bucket["tickets"] += 1
                    bucket["ganancia"] += premio
                    if premio > 0:
                        totales["hits"] += 1
                
                rows_actualizadas.append(row)

        # Actualizamos el Ledger con los resultados validados
        if rows_actualizadas:
            _atomic_write_ledger(FILE_APUESTAS, rows_actualizadas, fieldnames)
        
        return totales

    except Exception as e:
        console.print(f"[red]Error en la liquidación de cartera: {e}[/red]")
        return None

def mostrar_resumen_roi(totales):
    """
    UI Dashboard para el CLI.
    """
    if not totales:
        console.print("[yellow]No hay apuestas registradas para liquidar.[/]")
        return

    neto = totales["ganancia"] - totales["inversion"]
    roi = (neto / totales["inversion"]) * 100 if totales["inversion"] > 0 else 0
    color_roi = "green" if neto >= 0 else "red"

    table = Table(title="📊 DASHBOARD DE RENDIMIENTO REAL (MRPRO)", box=box.DOUBLE_EDGE)
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", justify="right")

    table.add_row("Sorteos Participados", str(len(totales["concursos"])))
    table.add_row("Tickets Comprados", str(int(totales["inversion"]/10)))
    table.add_row("Inversión Total", f"${totales['inversion']:,.2f}")
    table.add_row("Ganancia Bruta", f"${totales['ganancia']:,.2f}")
    table.add_row("Balance Neto", f"[{color_roi}]${neto:,.2f}[/]")
    table.add_row("ROI Real", f"[{color_roi}]{roi:.2f}%[/]")
    table.add_row("Tickets Premiados", f"[bold yellow]{totales['hits']}[/]")

    console.print(table)

    breakdown = totales.get("desglose_premios", {})
    if breakdown:
        prize_table = Table(title="🎟️ DESGLOSE DE PREMIOS", box=box.SIMPLE_HEAVY)
        prize_table.add_column("Categoría", style="cyan")
        prize_table.add_column("Tickets", justify="right")
        prize_table.add_column("Ganancia", justify="right")
        for category in MelateRetroRules.PRIZE_CATEGORY_ORDER:
            bucket = breakdown.get(category)
            if not bucket:
                continue
            prize_table.add_row(
                category,
                str(int(bucket["tickets"])),
                f"${float(bucket['ganancia']):,.2f}",
            )
        console.print(prize_table)
