import numpy as np

from src.strategies.tris.feature_lr_model import FeatureLRModel
from src.strategies.tris.universe_5d import get_universe_and_static_mask


def test_feature_lr_model_prefers_sum_zero_over_sum_45_when_trained_on_zeros():
    digits_list = [[0, 0, 0, 0, 0] for _ in range(300)]
    model = FeatureLRModel(
        alpha=1.0,
        short_window=200,
        long_window=2000,
        mix_lambda=0.7,
        use_mirror=True,
    ).fit(digits_list)

    all_tickets, features_cache, _ = get_universe_and_static_mask(None)
    scores = model.score_all(all_tickets, features_cache, prev_digits=[0, 0, 0, 0, 0])

    idx_00000 = 0
    idx_99999 = 99999
    assert scores.shape == (100000,)
    assert scores.dtype == np.float64
    assert float(scores[idx_00000]) > float(scores[idx_99999])


def test_feature_lr_model_shrinkage_without_evidence_scores_near_zero():
    model = FeatureLRModel(
        alpha=1.0,
        short_window=200,
        long_window=2000,
        mix_lambda=0.7,
        use_mirror=True,
        shrink_c=3000.0,
    ).fit([])

    all_tickets, features_cache, _ = get_universe_and_static_mask(None)
    scores = model.score_all(all_tickets, features_cache, prev_digits=[1, 2, 3, 4, 5])

    assert scores.shape == (100000,)
    assert float(np.max(np.abs(scores))) < 1e-10
