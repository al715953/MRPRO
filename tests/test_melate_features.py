import numpy as np

from src.core.melate_features import (
    FEATURE_NAMES,
    build_candidate_features,
    build_context_state,
)
from src.core.train_static_model import build_walk_forward_dataset


def _draws(n=140):
    rng = np.random.default_rng(123)
    return np.asarray(
        [np.sort(rng.choice(np.arange(1, 40), size=6, replace=False)) for _ in range(n)],
        dtype=np.uint8,
    )


def test_contextual_features_are_finite_and_have_stable_schema():
    draws = _draws()
    candidates = draws[-3:]

    features = build_candidate_features(candidates, draws[:-3])

    assert features.shape == (3, len(FEATURE_NAMES))
    assert np.all(np.isfinite(features))
    assert np.all(features >= 0.0)
    assert np.all(features <= 1.0)


def test_target_draw_is_not_part_of_its_own_context():
    draws = _draws()
    target_idx = 120
    target = draws[target_idx : target_idx + 1]

    expected = build_candidate_features(target, draws[:target_idx])
    altered = draws.copy()
    altered[target_idx] = np.array([1, 2, 3, 4, 5, 39], dtype=np.uint8)
    observed = build_candidate_features(target, altered[:target_idx])

    assert np.array_equal(expected, observed)


def test_walk_forward_dataset_has_one_positive_per_draw():
    draws = _draws()

    features, labels = build_walk_forward_dataset(
        draws,
        start_idx=120,
        end_idx=125,
        rng=np.random.default_rng(9),
    )

    assert features.shape == (5 * 11, len(FEATURE_NAMES))
    assert int(np.sum(labels == 1)) == 5
    assert int(np.sum(labels == 0)) == 50


def test_context_state_changes_only_after_new_draw_becomes_history():
    draws = _draws()

    before = build_context_state(draws[:120])
    after = build_context_state(draws[:121])

    assert before.n_draws == 120
    assert after.n_draws == 121
    assert not np.array_equal(before.last_draw_mask, after.last_draw_mask)
