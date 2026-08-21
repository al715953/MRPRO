from types import SimpleNamespace

from src.domain.dtos import (
    DrawHistoryDTO,
    PredictionResultDTO,
    sort_history_chronologically,
)
from src.interface import mission_controller as mission_module
from src.interface.mission_controller import MissionController


class _ConsoleStub:
    def print(self, *args, **kwargs):
        return None


class _UIStub:
    console = _ConsoleStub()

    def clear_screen(self):
        return None

    def show_prediction_results(self, prediction):
        return None


def _descending_history():
    return DrawHistoryDTO(
        concursos=[103, 101, 102],
        dates=["newest", "oldest", "middle"],
        winning_numbers=[
            [3, 3, 3, 3, 3, 0],
            [1, 1, 1, 1, 1, 0],
            [2, 2, 2, 2, 2, 0],
        ],
    )


def test_sort_history_chronologically_returns_aligned_copy():
    original = _descending_history()

    ordered = sort_history_chronologically(original)

    assert ordered.concursos == [101, 102, 103]
    assert ordered.dates == ["oldest", "middle", "newest"]
    assert [row[0] for row in ordered.winning_numbers] == [1, 2, 3]
    assert original.concursos == [103, 101, 102]


def test_melate_production_passes_chronological_history_to_both_stages(monkeypatch):
    received = []
    saved_shadow = []

    class ReducerStub:
        def predict(self, history, config):
            received.append(("reducer", list(history.concursos)))
            return PredictionResultDTO(
                strategy_name="reducer",
                tickets=[],
                metadata={"raw_ndarray": [[1, 2, 3, 4, 5, 6]]},
            )

    class SelectorStub:
        def predict(self, history, config):
            received.append(("selector", list(history.concursos)))
            return PredictionResultDTO(
                strategy_name="selector",
                tickets=[[1, 2, 3, 4, 5, 6]],
            )

    monkeypatch.setattr(mission_module, "UniverseReductionStrategy", ReducerStub)
    monkeypatch.setattr(mission_module, "GeneticSelectorStrategy", SelectorStub)
    monkeypatch.setattr(
        mission_module,
        "build_promoted_covering_shadows",
        lambda *_args, **_kwargs: [
            {
                "key": "cover_mixed_v20_m300",
                "label": "Cover V20",
                "official": False,
                "settings": {"shadow_family": "combinatorial_covering"},
                "tickets": [[1, 2, 3, 4, 5, 6]],
                "metadata": {"candidate_pool_size": 20},
            },
            {
                "key": "cover_mixed_v18_m300",
                "label": "Cover V18",
                "official": False,
                "settings": {"shadow_family": "combinatorial_covering"},
                "tickets": [[1, 2, 3, 4, 5, 6]],
                "metadata": {"candidate_pool_size": 18},
            },
        ],
    )
    monkeypatch.setattr(mission_module.report, "tiene_apuestas_pendientes", lambda _: False)
    monkeypatch.setattr(mission_module.report, "guardar_prediccion", lambda *_: None)
    monkeypatch.setattr(mission_module.report, "generar_ticket_limpio", lambda *_: None)
    monkeypatch.setattr(
        mission_module.shadow_ledger,
        "guardar_carteras_sombra",
        lambda **kwargs: saved_shadow.append(kwargs) or True,
    )
    monkeypatch.setattr("builtins.input", lambda *_: "")

    controller = MissionController(_UIStub(), _descending_history())
    controller._run_production()

    assert received == [
        ("reducer", [101, 102, 103]),
        ("selector", [101, 102, 103]),
        ("selector", [101, 102, 103]),
        ("selector", [101, 102, 103]),
        ("selector", [101, 102, 103]),
    ]
    variants = saved_shadow[0]["variants"]
    assert [variant["key"] for variant in variants] == [
        "principal_ai_adaptive",
        "benchmark_mrpro_native_m300",
        "challenger_ai10_geo90",
        "control_geo_only",
        "cover_mixed_v20_m300",
        "cover_mixed_v18_m300",
    ]
    assert [variant["official"] for variant in variants] == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert variants[1]["settings"]["shadow_family"] == "same_budget_benchmark"
    assert variants[2]["settings"]["hybrid_alpha"] == 0.10
    assert variants[3]["settings"]["hybrid_alpha"] == 0.0


def test_tris_production_passes_chronological_history_to_predictor(monkeypatch):
    received = []

    class PredictorStub:
        def predict(self, history, config):
            received.append(list(history.concursos))
            return PredictionResultDTO(
                strategy_name="tris",
                tickets=[],
                metadata={"pos_probs": []},
            )

    monkeypatch.setattr(mission_module, "TrisForecastV1A", PredictorStub)
    profile = SimpleNamespace(total_balls=10, ticket_size=5)
    controller = MissionController(_UIStub(), _descending_history(), profile)
    monkeypatch.setattr(controller, "_pause", lambda: None)

    controller._run_tris_production()

    assert received == [[101, 102, 103]]
