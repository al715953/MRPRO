import numpy as np

from run_melate_ticket_budget_sweep import summarize_budget_curve
from src.strategies.genetic.fitness import select_tickets_v16


def test_selector_portfolios_are_nested_from_24_to_36():
    rng = np.random.default_rng(42)
    tickets = np.asarray(
        [sorted(rng.choice(np.arange(1, 40), size=6, replace=False)) for _ in range(600)],
        dtype=np.uint8,
    )
    tickets = np.unique(tickets, axis=0)
    scores = np.linspace(1.0, 0.0, len(tickets), dtype=np.float32)
    maximum, _ = select_tickets_v16(tickets, scores, n_tickets=36, xp=np)

    for budget in range(24, 37, 2):
        selected, _ = select_tickets_v16(tickets, scores, n_tickets=budget, xp=np)
        assert selected == maximum[:budget]


def test_budget_summary_uses_nested_prefixes_and_exact_marginals():
    rows = [
        {
            "draw_id": 1,
            "metrics_json": {
                "selected_ticket_hits": [3, 2, 4, 1, 5, 0],
                "selected_ticket_prizes": [20, 15, 150, 10, 30000, 0],
            },
        },
        {
            "draw_id": 2,
            "metrics_json": {
                "selected_ticket_hits": [2, 3, 1, 4, 2, 6],
                "selected_ticket_prizes": [0, 20, 10, 150, 15, 4650000],
            },
        },
    ]

    summaries, comparisons = summarize_budget_curve(
        rows,
        (4, 6),
        ticket_cost=10.0,
        seed=7,
    )

    assert summaries[0]["draws_ge_4"] == 2
    assert summaries[0]["draws_ge_5"] == 0
    assert summaries[1]["draws_ge_5"] == 2
    assert summaries[1]["draws_eq_6"] == 1
    assert summaries[1]["marginal_vs_previous"]["investment_added"] == 40.0
    assert summaries[1]["marginal_vs_previous"]["additional_draws_ge_5"] == 2
    assert summaries[0]["prize_breakdown"]["1+AD"]["tickets"] == 2
    assert summaries[0]["prize_breakdown"]["2+AD"]["earnings"] == 15.0
    assert summaries[1]["prize_breakdown"]["5+AD"]["earnings"] == 30000.0
    assert summaries[1]["prize_breakdown"]["6"]["tickets"] == 1
    assert summaries[1]["gross_return_ratio"] > 0
    assert summaries[1]["net_roi"] > 0
    assert comparisons[0]["improved_draw_ids"] == [1, 2]
