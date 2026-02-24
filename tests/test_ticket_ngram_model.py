import numpy as np

from src.strategies.tris.ticket_ngram_model import TicketNgramModel


def test_ticket_ngram_model_ranks_pattern_01234_above_random():
    digits_list = [[0, 1, 2, 3, 4] for _ in range(500)]
    model = TicketNgramModel(
        alpha=0.5,
        window=2000,
        short_window=200,
        long_window=2000,
        mix_lambda=0.7,
        uniform_mix=0.0,
    )
    model.fit(digits_list)

    candidates = np.array(
        [
            [0, 1, 2, 3, 4],
            [9, 8, 7, 6, 5],
            [0, 1, 2, 3, 5],
            [1, 2, 3, 4, 5],
        ],
        dtype=np.uint8,
    )
    scores = model.score_all(candidates)

    assert int(np.argmax(scores)) == 0
    assert float(scores[0]) > float(scores[1])
    assert model.score_ticket([0, 1, 2, 3, 4]) > model.score_ticket([9, 8, 7, 6, 5])
