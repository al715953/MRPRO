import pandas as pd

from src.core.rules import TrisMultiplicadorRules
from src.data_access.loader import LotteryLoader
from src.data_access.config import get_lottery_profile
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO
from src.strategies.tris.tris_forecast import TrisForecastV1A


def test_tris_loader_reads_digit_columns():
    df = pd.DataFrame(
        {
            "CONCURSO": [1001],
            "FECHA": ["01/01/2025"],
            "R1": [1],
            "R2": [2],
            "R3": [3],
            "R4": [4],
            "R5": [5],
            "Multiplicador": ["SI"],
        }
    )

    profile = get_lottery_profile("tris_multiplicador")
    loader = LotteryLoader(profile)
    history = loader._process_tris(df)

    assert history.concursos == [1001]
    assert history.winning_numbers == [[1, 2, 3, 4, 5, 1]]


def test_tris_rules_exact_match_prize_and_profile_exists():
    rules = TrisMultiplicadorRules(base_prize=500)
    assert rules.validate_ticket([1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 2]) == (5, True)
    assert rules.calculate_prize(5, True) == 1000
    assert get_lottery_profile("tris_multiplicador").ticket_size == 5


def test_tris_forecast_v1a_predict_output_shape():
    draws = []
    concursos = []
    dates = []
    for i in range(100):
        d1 = i % 10
        d2 = (i + 3) % 10
        d3 = (i * 2) % 10
        d4 = (i * 3 + 1) % 10
        d5 = (i * 4 + 2) % 10
        mult = 1 if i % 4 == 0 else 0
        draws.append([d1, d2, d3, d4, d5, mult])
        concursos.append(1000 + i)
        dates.append(f"2025-01-{(i % 28) + 1:02d}")

    history = DrawHistoryDTO(dates=dates, winning_numbers=draws, concursos=concursos)
    config = PredictionConfigDTO(total_balls=10, ticket_size=5, num_tickets=12)

    pred = TrisForecastV1A().predict(history, config)

    assert len(pred.tickets) == config.num_tickets
    for t in pred.tickets:
        assert len(t) == 5
        assert all(0 <= d <= 9 for d in t)

    pos_probs = pred.metadata["pos_probs"]
    assert len(pos_probs) == 5
    assert all(len(row) == 10 for row in pos_probs)
    for row in pos_probs:
        assert abs(sum(row) - 1.0) < 1e-6
