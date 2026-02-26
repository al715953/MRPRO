import numpy as np

from src.strategies.tris.structural_filters import StructuralFilterConfig
from src.strategies.tris.universe_5d import (
    get_universe_and_static_mask,
    get_universe_with_positional_mask,
)


def _relaxed_cfg() -> StructuralFilterConfig:
    return StructuralFilterConfig(
        enable_global_sum_filter=False,
        enable_global_parity_filter=False,
        min_unique_digits=1,
        max_consecutive_run=5,
    )


def test_masked_universe_top2_each_position_has_32_tickets():
    mask = np.zeros((5, 10), dtype=bool)
    mask[:, :2] = True
    cfg = _relaxed_cfg()

    tickets, features, static_mask = get_universe_with_positional_mask(cfg, mask)

    assert tickets.shape == (32, 5)
    assert tickets.dtype == np.uint8
    assert static_mask.shape == (32,)
    assert int(np.sum(static_mask)) == 32
    assert np.asarray(features["sum_digits"]).shape == (32,)


def test_masked_universe_empty_when_any_position_has_no_allowed_digits():
    mask = np.ones((5, 10), dtype=bool)
    mask[2, :] = False
    cfg = _relaxed_cfg()

    tickets, features, static_mask = get_universe_with_positional_mask(cfg, mask)

    assert tickets.shape == (0, 5)
    assert tickets.dtype == np.uint8
    for key in ("sum_digits", "even_count", "unique_count", "consecutive_run_ge4"):
        assert np.asarray(features[key]).shape == (0,)
    assert static_mask.shape == (0,)


def test_positional_mask_none_keeps_full_universe_size():
    cfg = StructuralFilterConfig()
    tickets, features, static_mask = get_universe_with_positional_mask(cfg, None)
    tickets_ref, features_ref, static_ref = get_universe_and_static_mask(cfg)

    assert tickets.shape == (100000, 5)
    assert tickets is tickets_ref
    assert features is features_ref
    assert static_mask is static_ref


def test_masked_universe_cache_returns_same_objects_for_same_inputs():
    mask = np.zeros((5, 10), dtype=bool)
    mask[:, [1, 7]] = True
    cfg = _relaxed_cfg()

    t1, f1, s1 = get_universe_with_positional_mask(cfg, mask)
    t2, f2, s2 = get_universe_with_positional_mask(cfg, mask)

    assert t1 is t2
    assert f1 is f2
    assert s1 is s2


def test_masked_universe_return_diag_exposes_mask_support_summary():
    mask = np.zeros((5, 10), dtype=bool)
    mask[:, [2, 8]] = True
    cfg = _relaxed_cfg()

    tickets, features, static_mask, diag = get_universe_with_positional_mask(
        cfg,
        mask,
        return_diag=True,
    )

    assert tickets.shape == (32, 5)
    assert np.asarray(features["sum_digits"]).shape == (32,)
    assert static_mask.shape == (32,)
    assert isinstance(diag, dict)
    assert isinstance(diag.get("mask_hash"), str)
    assert int(diag.get("masked_universe_size_raw", -1)) == 32
    allowed = diag.get("allowed_digits_per_pos")
    assert isinstance(allowed, list)
    assert len(allowed) == 5
    assert all(int(row.get("count", -1)) == 2 for row in allowed)
