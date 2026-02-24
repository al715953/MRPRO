import numpy as np

from src.strategies.tris.ablation_analyzer import ablation_study
from src.strategies.tris.structural_filters import StructuralFilterConfig


def test_ablation_relaxing_filter_does_not_reduce_avgu():
    rng = np.random.default_rng(20260224)
    history = rng.integers(0, 10, size=(180, 5), endpoint=False).tolist()

    base_cfg = StructuralFilterConfig(
        enabled=True,
        sum_min=15,
        sum_max=30,
        allowed_even_counts=(2, 3),
        min_unique_digits=3,
        max_consecutive_run=3,
        max_positional_repeats_vs_prev=2,
        hard_filter=True,
        soft_penalties=None,
    )

    rows = ablation_study(history, base_cfg, start=30, end=120)
    by_name = {r["variant"]: r for r in rows}

    assert by_name["no_parity"]["AvgU"] >= by_name["base"]["AvgU"]
    assert by_name["no_mirror_prev"]["AvgU"] >= by_name["base"]["AvgU"]
    assert by_name["no_sum"]["AvgU"] >= by_name["base"]["AvgU"]
