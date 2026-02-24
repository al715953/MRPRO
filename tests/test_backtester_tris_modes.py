from src.core.backtester import BacktestEngine
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO


class _DummyRules:
    max_hits = 5
    ticket_cost = 0.0

    def validate_ticket(self, ticket, target):
        return 0, False

    def calculate_prize(self, hits, has_additional):
        return 0.0


class _UniverseStrategyNoVerbose:
    def predict(self, history, config):
        k = 7
        raw = [[0, 0, 0, 0, 0] for _ in range(k)]
        return PredictionResultDTO(
            strategy_name="dummy_universe_strategy",
            tickets=[],
            metadata={"raw_ndarray": raw},
        )


def test_backtester_universe_strategy_uses_raw_ndarray_size_for_audit():
    draws = []
    concursos = []
    dates = []
    for i in range(8):
        draws.append([i % 10, (i + 1) % 10, (i + 2) % 10, (i + 3) % 10, (i + 4) % 10, 0])
        concursos.append(7000 + i)
        dates.append(f"2025-07-{(i % 28) + 1:02d}")

    history = DrawHistoryDTO(dates=dates, winning_numbers=draws, concursos=concursos)
    config = PredictionConfigDTO(
        total_balls=10,
        ticket_size=5,
        num_tickets=1,
        backtest_size=4,
        filter_overrides={
            "tris_backtest_mode": "universe_strategy",
            "structural_enabled": False,
        },
    )

    engine = BacktestEngine(rules=_DummyRules())
    engine.run(_UniverseStrategyNoVerbose(), history, config, verbose=False)

    assert len(engine.forensic_data) == 4
    assert all(int(row.get("univ_size", -1)) == 7 for row in engine.forensic_data)
