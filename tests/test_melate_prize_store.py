import csv

import pandas as pd
import pytest

import src.data_access.report as report_module
import src.data_access.scraper as scraper_module
from src.data_access.prize_store import (
    load_prize_catalog,
    parse_melate_results_html,
    save_official_prize_record,
)
from src.domain.dtos import DrawHistoryDTO


OFFICIAL_HTML = """
<html><body><table>
  <tr><td colspan="8">Sorteo: 1666<br>Fecha 05/09/2026<br>Combinación ganadora</td></tr>
  <tr><td colspan="8"><h3>07 22 23 26 27 37-13</h3></td></tr>
  <tr><td>Lugar</td><td>Aciertos</td><td>Ganadores</td><td>Individual</td></tr>
  <tr><td>1er.</td><td>6 números naturales</td><td>2</td><td>$2,500,000.00</td></tr>
  <tr><td>2do.</td><td>5 números naturales y el adicional</td><td>1</td><td>$72,052.61</td></tr>
  <tr><td>3er.</td><td>5 números naturales</td><td>50</td><td>$546.39</td></tr>
  <tr><td>4to.</td><td>4 números naturales</td><td>1,605</td><td>$100.68</td></tr>
  <tr><td>5to.</td><td>3 números naturales</td><td>21,943</td><td>$21.51</td></tr>
  <tr><td>6to.</td><td>2 números naturales y el adicional</td><td>18,173</td><td>$16.13</td></tr>
  <tr><td>7mo.</td><td>1 número natural y el adicional</td><td>46,149</td><td>$10.00</td></tr>
</table></body></html>
"""


def test_official_results_parser_extracts_complete_prize_table():
    record = parse_melate_results_html(OFFICIAL_HTML)

    assert record["contest"] == 1666
    assert record["winning_numbers"] == [7, 22, 23, 26, 27, 37, 13]
    assert record["prizes"] == {
        "6": 2_500_000.0,
        "5+AD": 72_052.61,
        "5": 546.39,
        "4": 100.68,
        "3": 21.51,
        "2+AD": 16.13,
        "1+AD": 10.0,
    }


def test_prize_catalog_upserts_without_losing_previous_contests(tmp_path):
    path = tmp_path / "prizes.json"
    first = parse_melate_results_html(OFFICIAL_HTML)
    save_official_prize_record(first, path)
    second = dict(first, contest=1667)
    save_official_prize_record(second, path)

    catalog = load_prize_catalog(path)

    assert set(catalog["contests"]) == {"1666", "1667"}


def test_real_liquidation_replaces_estimate_with_contest_official_prize(
    tmp_path, monkeypatch
):
    ledger = tmp_path / "Mis_Apuestas.csv"
    catalog_path = tmp_path / "prizes.json"
    monkeypatch.setattr(report_module, "FILE_APUESTAS", str(ledger))
    report_module.guardar_prediccion([[7, 22, 23, 26, 30, 31]], 1666)
    record = parse_melate_results_html(OFFICIAL_HTML)
    save_official_prize_record(record, catalog_path)
    history = DrawHistoryDTO(
        dates=["2026-09-05"],
        concursos=[1666],
        winning_numbers=[[7, 22, 23, 26, 27, 37, 13]],
    )

    totals = report_module.liquidar_cartera(history, catalog_path)

    assert totals["ganancia"] == 100.68
    assert totals["concursos_premio_oficial"] == {"1666"}
    with ledger.open(newline="", encoding="utf-8") as source:
        row = next(csv.DictReader(source))
    assert row["Premio"] == "100.68"
    assert row["FuentePremio"].startswith("https://")


def test_historical_sync_also_persists_matching_official_prizes(
    tmp_path, monkeypatch
):
    raw_csv = (
        "NPRODUCTO,CONCURSO,F1,F2,F3,F4,F5,F6,F7,BOLSA,FECHA\n"
        "30,1666,7,22,23,26,27,37,13,5000000,05/09/2026\n"
    ).encode()
    record = parse_melate_results_html(OFFICIAL_HTML)
    saved = []
    monkeypatch.setattr(scraper_module, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        scraper_module, "_download_historical_data", lambda _url: raw_csv
    )
    monkeypatch.setattr(
        scraper_module, "_download_latest_melate_prize_record", lambda: record
    )
    monkeypatch.setattr(
        scraper_module, "save_official_prize_record", lambda value: saved.append(value)
    )

    assert scraper_module.actualizar_csv("melate_retro") is True
    assert saved == [record]
    assert (tmp_path / "Melate-Retro.csv").exists()


def test_prize_sync_rejects_a_contest_that_does_not_match_history():
    frame = pd.DataFrame(
        [
            {
                "CONCURSO": 1666,
                "F1": 7,
                "F2": 22,
                "F3": 23,
                "F4": 26,
                "F5": 27,
                "F6": 37,
                "F7": 13,
            }
        ]
    )
    wrong = dict(parse_melate_results_html(OFFICIAL_HTML), contest=1665)

    with pytest.raises(ValueError, match="no corresponden"):
        scraper_module._validate_prize_record_against_history(wrong, frame)
