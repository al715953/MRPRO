import csv

import src.data_access.report as report_module
from src.core.rules import MelateRetroRules
from src.data_access.report import LEDGER_FIELDS
from src.domain.dtos import DrawHistoryDTO


def test_melate_rules_classify_every_paid_category():
    rules = MelateRetroRules()
    winning = [1, 2, 3, 4, 5, 6, 7]
    cases = [
        ([1, 7, 20, 21, 22, 23], (1, True, "1+AD", 10.0)),
        ([1, 2, 7, 20, 21, 22], (2, True, "2+AD", 15.0)),
        ([1, 2, 3, 20, 21, 22], (3, False, "3", 20.0)),
        ([1, 2, 3, 7, 20, 21], (3, True, "3", 20.0)),
        ([1, 2, 3, 4, 20, 21], (4, False, "4", 150.0)),
        ([1, 2, 3, 4, 5, 20], (5, False, "5", 800.0)),
        ([1, 2, 3, 4, 5, 7], (5, True, "5+AD", 30_000.0)),
        ([1, 2, 3, 4, 5, 6], (6, False, "6", 4_650_000.0)),
        ([7, 20, 21, 22, 23, 24], (0, True, "SIN_PREMIO", 0.0)),
    ]

    for ticket, expected in cases:
        hits, additional = rules.validate_ticket(ticket, winning)
        category = rules.prize_category(hits, additional)
        prize = rules.calculate_prize(hits, additional)
        assert (hits, additional, category, prize) == expected


def test_legacy_prize_records_reconstruct_additional_categories():
    rules = MelateRetroRules()
    assert rules.category_from_recorded_result(1, 10) == "1+AD"
    assert rules.category_from_recorded_result(2, 15) == "2+AD"
    assert rules.category_from_recorded_result(5, 30_000) == "5+AD"
    assert rules.category_from_recorded_result(5, 800) == "5"
    assert rules.category_from_recorded_result(2, 0) == "SIN_PREMIO"


def test_real_ledger_persists_and_aggregates_prize_breakdown(tmp_path, monkeypatch):
    ledger = tmp_path / "Mis_Apuestas.csv"
    monkeypatch.setattr(report_module, "FILE_APUESTAS", str(ledger))
    report_module.guardar_prediccion(
        [[1, 7, 20, 21, 22, 23], [1, 2, 7, 20, 21, 22]],
        10,
    )
    history = DrawHistoryDTO(
        dates=["2026-08-20"],
        concursos=[10],
        winning_numbers=[[1, 2, 3, 4, 5, 6, 7]],
    )

    totals = report_module.liquidar_cartera(history)

    assert totals["ganancia"] == 25.0
    assert totals["hits"] == 2
    assert totals["desglose_premios"] == {
        "1+AD": {"tickets": 1, "ganancia": 10.0},
        "2+AD": {"tickets": 1, "ganancia": 15.0},
    }
    with ledger.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert list(rows[0]) == LEDGER_FIELDS
    assert [row["CategoriaPremio"] for row in rows] == ["1+AD", "2+AD"]
    assert all(row["Status"] == "🏆 GANADOR" for row in rows)


def test_existing_ledger_schema_is_upgraded_before_append(tmp_path, monkeypatch):
    ledger = tmp_path / "Mis_Apuestas.csv"
    old_fields = LEDGER_FIELDS[:11]
    with ledger.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=old_fields)
        writer.writeheader()
        writer.writerow(
            {
                "Fecha": "2026-08-01",
                "Concurso": 9,
                "Version": "legacy",
                **{f"T{i}": i for i in range(1, 7)},
                "Status": "Pendiente",
                "Premio": 0,
            }
        )
    monkeypatch.setattr(report_module, "FILE_APUESTAS", str(ledger))

    report_module.guardar_prediccion([[8, 9, 10, 11, 12, 13]], 10)

    with ledger.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert list(rows[0]) == LEDGER_FIELDS
    assert len(rows) == 2
    assert rows[0]["CategoriaPremio"] == ""
    assert rows[1]["Status"] == "Pendiente"
