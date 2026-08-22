import json
import math

import numpy as np
import pytest

from run_melate_full_random_benchmark import (
    build_benchmark_report,
    build_outcome_population,
    simulate_random_portfolios,
)


def test_full_outcome_partition_is_exact_and_includes_jackpot_once():
    cells = build_outcome_population()

    assert sum(cell["population"] for cell in cells) == math.comb(39, 6)
    jackpot = [
        cell
        for cell in cells
        if cell["natural_hits"] == 6 and not cell["has_additional"]
    ]
    assert len(jackpot) == 1
    assert jackpot[0]["population"] == 1
    assert jackpot[0]["prize"] == 4_650_000.0


def test_random_simulation_uses_exact_budget_and_is_reproducible():
    kwargs = dict(draws=4, tickets=30, trials=25, seed=1234, chunk_size=7)
    first, first_categories, _ = simulate_random_portfolios(**kwargs)
    second, second_categories, _ = simulate_random_portfolios(**kwargs)

    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    for name in first_categories:
        np.testing.assert_array_equal(first_categories[name], second_categories[name])

    total_selected = sum(first_categories.values())
    np.testing.assert_array_equal(total_selected, np.full(25, 4 * 30))
    assert np.all(first["draws_ge_4"] <= 4)
    assert np.all(first["tickets_hits_6"] <= first["draws_eq_6"])


def test_benchmark_rejects_mismatched_mrpro_budget(tmp_path):
    source = tmp_path / "mrpro.json"
    source.write_text(
        json.dumps(
            {
                "tickets_per_draw": 24,
                "variants": [
                    {
                        "name": "candidate",
                        "draws": 2,
                        "selected_max_hits_by_draw": [2, 3],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="24 boletos"):
        build_benchmark_report(
            mrpro_report_path=source,
            variant_name="candidate",
            draws=2,
            tickets=30,
            trials=2,
        )


def test_small_benchmark_report_compares_every_primary_metric(tmp_path):
    source = tmp_path / "mrpro.json"
    source.write_text(
        json.dumps(
            {
                "experiment": "test",
                "tickets_per_draw": 30,
                "evaluated_draw_ids": [10, 11],
                "variants": [
                    {
                        "name": "candidate",
                        "draws": 2,
                        "investment": 600.0,
                        "earnings": 100.0,
                        "gross_return_ratio": 1 / 6,
                        "net_roi": -5 / 6,
                        "selected_jackpots": 0,
                        "selected_max_hits_by_draw": [3, 4],
                        "ticket_hit_distribution": {
                            "4": 1,
                            "5": 0,
                            "6": 0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_benchmark_report(
        mrpro_report_path=source,
        variant_name="candidate",
        draws=2,
        tickets=30,
        trials=10,
        seed=99,
        chunk_size=3,
    )

    assert report["configuration"]["full_universe_size"] == math.comb(39, 6)
    assert report["configuration"]["uses_history_or_rules"] is False
    assert report["mrpro"]["values"]["draws_ge_4"] == 1
    assert set(report["mrpro_vs_random"]) == {
        "earnings",
        "earnings_without_jackpot",
        "avg_max_hits",
        "draws_ge_4",
        "draws_ge_5",
        "draws_eq_6",
        "tickets_hits_4",
        "tickets_hits_5",
        "tickets_hits_6",
    }
