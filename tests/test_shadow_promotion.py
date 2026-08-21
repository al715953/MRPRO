from src.core.shadow_promotion import (
    PromotionRules,
    STATUS_ELIGIBLE,
    STATUS_INSUFFICIENT,
    STATUS_UNMATCHED,
    evaluate_shadow_promotions,
)


def _validation(max_hits: int, tickets: int) -> dict:
    distribution = {str(hit): 0 for hit in range(7)}
    distribution[str(max_hits)] = 1
    return {
        "ticket_count": tickets,
        "max_hits": max_hits,
        "hit_distribution": distribution,
    }


def _portfolio(contest: int, challenger_hits: int, *, include_benchmark=True):
    variants = [
        {
            "key": "principal_ai_adaptive",
            "label": "Principal",
            "official": True,
            "settings": {},
            "validation": _validation(3, 24),
        }
    ]
    if include_benchmark:
        variants.append(
            {
                "key": "benchmark_mrpro_native_m300",
                "label": "Benchmark 300",
                "official": False,
                "settings": {"shadow_family": "same_budget_benchmark"},
                "validation": _validation(3, 300),
            }
        )
    variants.append(
        {
            "key": "cover_mixed_v20_m300",
            "label": "Cover V20",
            "official": False,
            "settings": {
                "promotion_reference_key": "benchmark_mrpro_native_m300"
            },
            "validation": _validation(challenger_hits, 300),
        }
    )
    return {"concurso": contest, "variants": variants}


def test_gate_requires_precommitted_minimum_sample():
    payload = {"portfolios": [_portfolio(100 + idx, 4) for idx in range(10)]}
    result = evaluate_shadow_promotions(
        payload,
        PromotionRules(permutation_trials=100, bootstrap_trials=100),
    )

    evaluation = result["evaluations"]["cover_mixed_v20_m300"]
    assert evaluation["status"] == STATUS_INSUFFICIENT
    assert evaluation["paired_draws"] == 10
    assert result["automatic_production_change"] is False


def test_gate_can_mark_strong_same_budget_challenger_for_pilot():
    payload = {"portfolios": [_portfolio(100 + idx, 4) for idx in range(60)]}
    rules = PromotionRules(
        permutation_trials=500,
        bootstrap_trials=500,
        random_seed=7,
    )

    result = evaluate_shadow_promotions(payload, rules)
    evaluation = result["evaluations"]["cover_mixed_v20_m300"]

    assert evaluation["status"] == STATUS_ELIGIBLE
    assert evaluation["reference_key"] == "benchmark_mrpro_native_m300"
    assert evaluation["ticket_budget"] == 300
    assert evaluation["avg_max_hits_delta"] == 1.0
    assert evaluation["bootstrap_max_hits"]["ci_low"] == 1.0


def test_gate_refuses_comparison_without_equal_budget_reference():
    payload = {
        "portfolios": [
            _portfolio(100 + idx, 4, include_benchmark=False) for idx in range(25)
        ]
    }

    result = evaluate_shadow_promotions(
        payload,
        PromotionRules(permutation_trials=50, bootstrap_trials=50),
    )

    assert (
        result["evaluations"]["cover_mixed_v20_m300"]["status"]
        == STATUS_UNMATCHED
    )
