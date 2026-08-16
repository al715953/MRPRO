import numpy as np

from src.core.melate_number_model import (
    NUMBER_FEATURE_NAMES,
    build_number_features,
    build_number_walk_forward_dataset,
    number_topk_metrics,
    score_tickets_from_number_probs,
)


def _draws(n=130):
    rng = np.random.default_rng(987)
    return np.asarray(
        [np.sort(rng.choice(np.arange(1, 40), size=6, replace=False)) for _ in range(n)],
        dtype=np.uint8,
    )


def test_number_features_have_one_finite_row_per_ball():
    features = build_number_features(_draws())

    assert features.shape == (39, len(NUMBER_FEATURE_NAMES))
    assert np.all(np.isfinite(features))


def test_number_walk_forward_labels_exactly_six_winners_per_draw():
    features, labels = build_number_walk_forward_dataset(
        _draws(),
        start_idx=120,
        end_idx=125,
    )

    assert features.shape == (5 * 39, len(NUMBER_FEATURE_NAMES))
    assert labels.shape == (5 * 39,)
    assert np.array_equal(labels.reshape(5, 39).sum(axis=1), np.full(5, 6.0))


def test_number_ticket_score_rewards_higher_probability_numbers():
    probs = np.linspace(0.01, 0.39, 39, dtype=np.float32)
    tickets = np.asarray(
        [[1, 2, 3, 4, 5, 6], [34, 35, 36, 37, 38, 39]], dtype=np.uint8
    )

    scores = score_tickets_from_number_probs(tickets, probs)

    assert scores[1] > scores[0]


def test_topk_metrics_are_exact_for_perfect_number_ranking():
    labels = np.zeros((2, 39), dtype=np.float32)
    labels[:, :6] = 1.0
    scores = labels.copy()

    metrics = number_topk_metrics(scores.ravel(), labels.ravel())

    assert metrics["mean_hits_at_6"] == 6.0
    assert metrics["recall_at_6"] == 1.0
