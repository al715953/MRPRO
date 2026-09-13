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
                    "metrics_json": {"selected_max_hits": 4},
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
    assert report["ledger_isolated"] is True
    assert report["variants"][0]["selected_max_hits_by_draw"] == [4]


def test_selector_shadow_suite_keeps_equal_ticket_budget_configuration():
    names = [variant["name"] for variant in experiment.SELECTOR_SHADOW_VARIANTS]

    assert names == [
        "A_contextual_geo_adaptive",
        "G_context50_number50_adaptive",
        "H_deep_rank_5000_same_budget",
    ]
    deep = experiment.SELECTOR_SHADOW_VARIANTS[2]["overrides"]
    assert deep["fitness_candidate_max_rank"] == 5000


def test_controlled_weight_suite_changes_only_ai_ensemble_weights():
    variants = experiment.CONTROLLED_WEIGHT_VARIANTS

    assert [variant["name"] for variant in variants] == [
        "A_contextual_geo_adaptive",
        "C_contextual_number15_geo_adaptive",
        "G_context50_number50_adaptive",
    ]
    assert [variant["overrides"]["ai_number_weight"] for variant in variants] == [
        0.0,
        0.15,
        0.50,
    ]
    assert all(
        variant["overrides"]["resonance_blend_mode"] == "adaptive"
        for variant in variants
    )


def test_deep_dispersion_suite_has_equal_30_ticket_reference():
    variants = experiment.DEEP_DISPERSION_VARIANTS

    assert [variant["name"] for variant in variants] == [
        "I_native_30_reference",
        "J_core20_deep10_equal_population",
    ]
    challenger = variants[1]["overrides"]
    assert challenger["fitness_selector_mode"] == "core_plus_deep"
    assert challenger["deep_dispersion_core_tickets"] == 20
    assert challenger["deep_dispersion_tickets"] == 10
    assert challenger["deep_dispersion_min_rank"] == 501


def test_elite_coverage_deep_suite_precommits_three_equal_budget_variants():
    variants = experiment.ELITE_COVERAGE_DEEP_VARIANTS

    assert [variant["name"] for variant in variants] == [
        "J_core20_deep10_equal_population",
        "K_elite10_cover10_deep10",
        "L_elite15_cover10_deep5",
        "M_elite10_cover15_deep5",
    ]
    allocations = [
        (
            variant["overrides"].get("portfolio_elite_tickets", 0),
            variant["overrides"].get("portfolio_coverage_tickets", 0),
            variant["overrides"].get("portfolio_deep_tickets", 0),
        )
        for variant in variants[1:]
    ]
    assert allocations == [(10, 10, 10), (15, 10, 5), (10, 15, 5)]
    assert all(sum(allocation) == 30 for allocation in allocations)
    assert all(
        variant["overrides"]["portfolio_triple_novelty_weight"] == 0.30
        and variant["overrides"]["portfolio_quad_novelty_weight"] == 0.30
        for variant in variants[1:]
    )


def test_universe_v17_suite_compares_legacy_control_with_production():
    variants = experiment.UNIVERSE_V17_VARIANTS

    assert [variant["name"] for variant in variants] == [
        "legacy_hard_geo_v16",
        "balanced_mixed50_v17",
    ]
    legacy = variants[0]["overrides"]
    assert legacy["candidate_selection_mode"] == "density"
    assert legacy["sniper_mode"] == "hard"
    assert legacy["radar_percentile"] == 50.0
    assert legacy["sum_filter_enabled"] is True
    assert variants[1]["overrides"] == {}
    assert experiment.VARIANT_SUITES["universe-v17"] is variants


def test_universe_dynamic_suite_freezes_v17_2_and_isolates_universe_axes():
    variants = experiment.UNIVERSE_DYNAMIC_VARIANTS
    by_name = {variant["name"]: variant["overrides"] for variant in variants}

    assert experiment.VARIANT_SUITES["universe-dynamic"] is variants
    assert by_name["UA_v17_2_balanced45_reference"]["universe_ticket_limit"] == 45000
    assert by_name["UB_balanced30"]["universe_ticket_limit"] == 30000
    assert by_name["UC_balanced60"]["universe_ticket_limit"] == 60000
    assert by_name["UD_balanced90"]["universe_ticket_limit"] == 90000
    assert by_name["UE_core100_45"]["universe_exploration_fraction"] == 0.0
    assert by_name["UH_random100_45"]["universe_exploration_fraction"] == 1.0
    assert by_name["UJ_topology_hard_current_scorer"]["sum_filter_enabled"] is True
    assert by_name["UJ_topology_hard_current_scorer"]["sniper_mode"] == "soft"
    assert by_name["UO_topology_hard_radar50"]["sum_filter_enabled"] is True
    assert by_name["UO_topology_hard_radar50"]["radar_percentile"] == 50.0
    assert by_name["UP_balanced45_radar50"]["sum_filter_enabled"] is False
    assert by_name["UP_balanced45_radar50"]["radar_percentile"] == 50.0
    assert by_name["UI_legacy_hard_complete"]["sniper_mode"] == "hard"
    assert by_name["UI_legacy_hard_complete"]["radar_percentile"] == 50.0
    assert all(
        overrides["fitness_selector_mode"] == "core_plus_deep"
        and overrides["deep_dispersion_core_tickets"] == 16
        and overrides["deep_dispersion_tickets"] == 8
        and overrides["resonance_blend_mode"] == "fixed"
        and overrides["hybrid_alpha"] == 1.0
        for overrides in by_name.values()
    )


def test_universe_hybrid_suite_precommits_three_24_ticket_allocations():
    variants = experiment.UNIVERSE_HYBRID_VARIANTS
    challengers = variants[3:]

    assert experiment.VARIANT_SUITES["universe-hybrid"] is variants
    assert [variant["name"] for variant in challengers] == [
        "HQ_v17_2_18_topology_6",
        "HR_v17_2_16_topology_8",
        "HS_v17_2_12_topology_12",
    ]
    allocations = [
        (
            variant["overrides"]["hybrid_primary_tickets"],
            variant["overrides"]["hybrid_topology_tickets"],
        )
        for variant in challengers
    ]
    assert allocations == [(18, 6), (16, 8), (12, 12)]
    assert all(sum(allocation) == 24 for allocation in allocations)
    assert all(
        variant["overrides"]["fitness_selector_mode"] == "hybrid_dual_lane"
        and variant["overrides"]["universe_strategy_mode"]
        == "hybrid_soft_topology"
        for variant in challengers
    )


def test_universe_nested_cap_suite_uses_ordered_deterministic_limits():
    variants = experiment.UNIVERSE_NESTED_CAP_VARIANTS

    assert experiment.VARIANT_SUITES["universe-nested-cap"] is variants
    assert [
        variant["overrides"]["hybrid_primary_universe_limit"]
        for variant in variants
    ] == [20000, 25000, 30000, 35000, 40000, 45000]
    assert all(
        variant["overrides"]["hybrid_primary_selection_mode"]
        == "nested_balanced"
        and variant["overrides"]["fitness_selector_mode"]
        == "hybrid_dual_lane"
        for variant in variants
    )


def test_summary_records_dynamic_universe_size_and_exact_winner_coverage():
    result = SimpleNamespace(
        total_draws_tested=2,
        investment=48.0,
        earnings=0.0,
        net_balance=-48.0,
        hit_distribution={},
    )
    rows = [
        {
            "draw_id": 1,
            "rank": 1,
            "proximity": 0,
            "hits": 3,
            "univ_size": 39864,
            "metrics_json": {
                "selected_max_hits": 3,
                "winner_in_universe": 1,
            },
        },
        {
            "draw_id": 2,
            "rank": 2,
            "proximity": 1,
            "hits": 4,
            "univ_size": 45000,
            "metrics_json": {
                "selected_max_hits": 4,
                "winner_in_universe": 0,
            },
        },
    ]

    summary = experiment._summarize(result, rows)

    assert summary["universe_size_min"] == 39864
    assert summary["universe_size_mean"] == 42432.0
    assert summary["universe_size_max"] == 45000
    assert summary["winner_exact_in_universe"] == 1
    assert summary["winner_exact_in_universe_rate"] == 0.5


def test_paired_comparison_reports_mcnemar_and_reproducible_permutation():
    rows = [
        {
            "name": "reference",
            "selected_max_hits_by_draw": [3, 4, 4, 5, 2],
            "earnings_by_draw": [20, 150, 150, 800, 0],
        },
        {
            "name": "challenger",
            "selected_max_hits_by_draw": [4, 4, 3, 5, 4],
            "earnings_by_draw": [150, 150, 20, 800, 150],
        },
    ]

    first = experiment._paired_comparisons(rows, seed=7, resamples=500)
    second = experiment._paired_comparisons(rows, seed=7, resamples=500)

    assert first == second
    assert first[0]["max_hits_wins"] == 2
    assert first[0]["max_hits_losses"] == 1
    assert first[0]["max_hits_ties"] == 2
    assert first[0]["mcnemar_ge_4"]["challenger_only"] == 2
    assert first[0]["mcnemar_ge_4"]["reference_only"] == 1
    assert first[0]["paired_earnings"]["wins"] == 2
    assert first[0]["paired_earnings"]["losses"] == 1
    assert first[0]["paired_earnings"]["total_delta"] == 150.0
