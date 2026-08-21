from types import SimpleNamespace

from src.domain.dtos import DrawHistoryDTO

import run_melate_ab_experiments as experiment


class _Artifacts:
    training_cutoff_contest = 10
    test_start_contest = 11
    test_end_contest = 12
    dataset_hash = "fixed-hash"
    context_model_path = "context-fixed.json"
    number_model_path = "number-fixed.json"

    def to_dict(self):
        return {"training_cutoff_contest": self.training_cutoff_contest}


def test_ab_runner_uses_fixed_origin_models_and_records_split(monkeypatch):
    history = DrawHistoryDTO(
        dates=["a", "b"],
        winning_numbers=[[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
        concursos=[11, 12],
    )
    captured = {}

    class LoaderStub:
        def __init__(self, _profile):
            pass

        def load_data(self):
            return history

    class EngineStub:
        def __init__(self):
            self.forensic_data = [
                {
                    "draw_id": 12,
                    "rank": 1,
                    "proximity": 0,
                    "ai_score": 0.5,
                    "geo_score": 0.5,
                    "hits": 4,
                    "event_id": "test",
                }
            ]

        def run(self, strategy, _history, config, **_kwargs):
            captured["context_path"] = strategy.context_path
            captured["number_path"] = strategy.number_path
            captured["settings"] = dict(config.filter_overrides)
            return SimpleNamespace(
                total_draws_tested=1,
                investment=24.0,
                earnings=0.0,
                net_balance=-24.0,
                hit_distribution={4: 1},
            )

    class SelectorStub:
        def __init__(self, model_path, number_model_path):
            self.context_path = model_path
            self.number_path = number_model_path

    monkeypatch.setattr(experiment, "LotteryLoader", LoaderStub)
    monkeypatch.setattr(experiment, "BacktestEngine", EngineStub)
    monkeypatch.setattr(experiment, "GeneticSelectorStrategy", SelectorStub)
    monkeypatch.setattr(
        experiment, "prepare_fixed_origin_models", lambda *_args: _Artifacts()
    )

    report = experiment.run_experiments(
        1,
        24,
        42,
        variants=experiment.CORE_VARIANTS[:1],
    )

    assert captured["context_path"] == "context-fixed.json"
    assert captured["number_path"] == "number-fixed.json"
    assert captured["settings"]["fixed_origin_training_cutoff"] == 10
    assert captured["settings"]["fixed_origin_test_start"] == 11
    assert report["fixed_origin"]["training_cutoff_contest"] == 10


def test_selector_shadow_suite_keeps_equal_ticket_budget_configuration():
    names = [variant["name"] for variant in experiment.SELECTOR_SHADOW_VARIANTS]

    assert names == [
        "A_contextual_geo_adaptive",
        "G_context50_number50_adaptive",
        "H_deep_rank_5000_same_budget",
    ]
    deep = experiment.SELECTOR_SHADOW_VARIANTS[2]["overrides"]
    assert deep["fitness_candidate_max_rank"] == 5000
