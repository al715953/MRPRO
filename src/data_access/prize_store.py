"""Persistencia y parsing de premios oficiales de Melate Retro."""

from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import unicodedata
from typing import Any

from src.data_access.config import MELATE_PRIZE_TABLE_PATH, URL_MELATE_RESULTS


PRIZE_CATEGORIES = ("6", "5+AD", "5", "4", "3", "2+AD", "1+AD")


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(self, tag: str, _attrs) -> None:
        tag = tag.lower()
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._cell_parts is not None:
            text = " ".join("".join(self._cell_parts).split())
            if self._row is not None:
                self._row.append(text)
            self._cell_parts = None
        elif tag == "tr" and self._row is not None:
            if self._table is not None and self._row:
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = None


def _normalized(value: str) -> str:
    raw = unicodedata.normalize("NFKD", str(value))
    return "".join(ch for ch in raw if not unicodedata.combining(ch)).lower()


def _category_from_description(description: str) -> str | None:
    text = _normalized(description)
    match = re.search(r"\b([1-6])\b", text)
    if not match:
        return None
    hits = match.group(1)
    if "adicional" in text and hits in {"1", "2", "5"}:
        return f"{hits}+AD"
    return hits


def _parse_money(value: str) -> float:
    match = re.search(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)", str(value))
    if not match:
        raise ValueError(f"Importe oficial inválido: {value!r}")
    return float(match.group(1).replace(",", ""))


def parse_melate_results_html(
    html: str,
    *,
    source_url: str = URL_MELATE_RESULTS,
) -> dict[str, Any]:
    """Extrae concurso, combinación y tabla individual del HTML oficial."""

    parser = _HTMLTableParser()
    parser.feed(html)
    for table in parser.tables:
        flattened = " ".join(cell for row in table for cell in row)
        if "Individual" not in flattened or "Aciertos" not in flattened:
            continue

        contest_match = re.search(r"Sorteo:\s*(\d+)", flattened, re.IGNORECASE)
        date_match = re.search(
            r"Fecha\s+(\d{2}/\d{2}/\d{4})", flattened, re.IGNORECASE
        )
        combination_match = re.search(
            r"\b(\d{2}(?:\s+\d{2}){5})\s*-\s*(\d{2})\b", flattened
        )
        if not contest_match or not combination_match:
            continue

        prizes: dict[str, float] = {}
        winners: dict[str, int] = {}
        for row in table:
            if len(row) < 4:
                continue
            category = _category_from_description(row[1])
            if category not in PRIZE_CATEGORIES:
                continue
            prizes[category] = _parse_money(row[3])
            winner_text = re.sub(r"[^0-9]", "", row[2])
            winners[category] = int(winner_text) if winner_text else 0

        missing = [category for category in PRIZE_CATEGORIES if category not in prizes]
        if missing:
            raise ValueError(f"Tabla oficial incompleta; faltan categorías: {missing}")

        naturals = [int(value) for value in combination_match.group(1).split()]
        additional = int(combination_match.group(2))
        return {
            "contest": int(contest_match.group(1)),
            "date": date_match.group(1) if date_match else None,
            "winning_numbers": naturals + [additional],
            "prizes": prizes,
            "winners": winners,
            "source": str(source_url),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    raise ValueError("No se encontró una tabla oficial de Melate Retro en el HTML")


def load_prize_catalog(path: str | os.PathLike[str] = MELATE_PRIZE_TABLE_PATH):
    source = Path(path)
    if not source.exists():
        return {"schema_version": 1, "contests": {}}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Catálogo de premios ilegible: {source}") from exc
    contests = payload.get("contests")
    if not isinstance(contests, dict):
        raise ValueError(f"Catálogo de premios sin concursos: {source}")
    return {"schema_version": 1, "contests": contests}


def save_official_prize_record(
    record: dict[str, Any],
    path: str | os.PathLike[str] = MELATE_PRIZE_TABLE_PATH,
) -> None:
    target = Path(path)
    payload = load_prize_catalog(target)
    contest = int(record["contest"])
    payload["contests"][str(contest)] = dict(record)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{target}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def official_prize_record(
    contest: int,
    path: str | os.PathLike[str] = MELATE_PRIZE_TABLE_PATH,
) -> dict[str, Any] | None:
    record = load_prize_catalog(path)["contests"].get(str(int(contest)))
    return dict(record) if isinstance(record, dict) else None
