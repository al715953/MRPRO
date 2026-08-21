import json

from src.data_access.shadow_ledger import (
    guardar_carteras_sombra,
    liquidar_carteras_sombra,
)
from src.domain.dtos import DrawHistoryDTO


def _variants():
    return [
        {
            "key": "principal_ai_adaptive",
            "label": "Principal",
            "official": True,
            "settings": {"resonance_blend_mode": "adaptive"},
            "tickets": [[1, 2, 3, 4, 5, 6], [1, 2, 3, 8, 9, 10]],
            "metadata": {"ai_signal_enabled": True, "selected_ranks": [1, 20]},
        },
        {
            "key": "challenger_ai10_geo90",
            "label": "IA 10 / Geo 90",
            "official": False,
            "settings": {"hybrid_alpha": 0.1, "hybrid_beta": 0.9},
            "tickets": [[1, 2, 3, 4, 5, 8]],
            "metadata": {"ai_signal_enabled": True},
        },
        {
            "key": "control_geo_only",
            "label": "Geo",
            "official": False,
            "settings": {"hybrid_alpha": 0.0, "hybrid_beta": 1.0},
            "tickets": [[1, 2, 3, 4, 8, 9]],
            "metadata": {"ai_signal_enabled": True},
        },
    ]


def test_shadow_ledger_saves_once_without_creating_real_bets(tmp_path):
    path = tmp_path / "Carteras_Sombra.json"

    assert guardar_carteras_sombra(1661, _variants(), 1660, str(path)) is True
    assert guardar_carteras_sombra(1661, _variants(), 1660, str(path)) is False

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["portfolios"]) == 1
    portfolio = payload["portfolios"][0]
    assert portfolio["concurso"] == 1661
    assert portfolio["dataset_through_concurso"] == 1660
    assert len(portfolio["variants"]) == 3
    assert portfolio["variants"][0]["official"] is True
    assert portfolio["variants"][1]["status"] == "Pendiente"
    assert list(tmp_path.iterdir()) == [path]


def test_shadow_liquidation_is_simulated_and_idempotent(tmp_path):
    path = tmp_path / "Carteras_Sombra.json"
    guardar_carteras_sombra(1661, _variants(), 1660, str(path))
    history = DrawHistoryDTO(
        dates=["2026-08-20"],
        concursos=[1661],
        winning_numbers=[[1, 2, 3, 4, 5, 6, 7]],
    )

    first = liquidar_carteras_sombra(history, str(path))
    second = liquidar_carteras_sombra(history, str(path))

    assert first["updated_contests"] == [1661]
    assert second["updated_contests"] == []
    assert first["pending_contests"] == []

    principal = first["variants"]["principal_ai_adaptive"]
    challenger = first["variants"]["challenger_ai10_geo90"]
    control = first["variants"]["control_geo_only"]
    assert principal["hits_6"] == 1
    assert principal["simulated_prize"] == 4_650_020.0
    assert challenger["hits_5"] == 1
    assert challenger["simulated_prize"] == 800.0
    assert control["hits_4"] == 1
    assert control["simulated_prize"] == 150.0
    assert second["variants"] == first["variants"]
    assert principal["tickets_per_contest"] == 2.0
    assert principal["avg_max_hits"] == 6.0
    assert principal["contest_rate_ge_4"] == 1.0
    assert principal["high_hit_tickets_per_1000"] == 500.0
    dashboard = tmp_path / "Tablero_Sombra.json"
    assert dashboard.exists()
    exported = json.loads(dashboard.read_text(encoding="utf-8"))
    assert exported["promotion"]["automatic_production_change"] is False


def test_shadow_liquidation_keeps_future_contest_pending(tmp_path):
    path = tmp_path / "Carteras_Sombra.json"
    guardar_carteras_sombra(1662, _variants(), 1661, str(path))
    history = DrawHistoryDTO(dates=[], concursos=[], winning_numbers=[])

    summary = liquidar_carteras_sombra(history, str(path))

    assert summary["pending_contests"] == [1662]
    assert summary["updated_contests"] == []
    assert summary["variants"] == {}
