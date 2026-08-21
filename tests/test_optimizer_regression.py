import numpy as np

from src.core.optimizer import StrategyOptimizer
from src.data_access.config import BEST_SETTINGS
from src.domain.dtos import DrawHistoryDTO


class _DummyReducer:
    backend_name = "test"

    def reduce(self, history, config, verbose=False):
        universe = np.tile(np.array([[1, 2, 3, 4, 5, 6]], dtype=np.uint8), (15000, 1))
        return universe, "Sniper:-7(0.90)"


class _InactiveSniperFilters:
    def get_sniper_exclusion(self, *args, **kwargs):
        return [], ""


class _InactiveSniperReducer:
    backend_name = "test"
    filters = _InactiveSniperFilters()


def test_optimize_filters_accepts_reduce_tuple_return():
    optimizer = StrategyOptimizer.__new__(StrategyOptimizer)
    optimizer.reducer = _DummyReducer()
    optimizer.xp = np
    optimizer._print_progress = lambda *args, **kwargs: None

    history = DrawHistoryDTO(
        dates=["2024-01-01", "2024-01-02"],
        winning_numbers=[[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 8, 9, 10]],
        concursos=[1, 2],
    )

    best = optimizer.optimize_filters(
        history,
        draws_to_test=1,
        custom_grid={
            "e_min": [2.1],
            "e_max": [2.5],
            "s_min": [22],
            "s_max": [38],
            "ac": [7],
            "std_min": [8.2],
            "std_max": [12.4],
        },
    )

    assert isinstance(best, dict)
    assert "entropy_min" in best
    assert best["optimizer_split"]["validation_range"] == [1, 1]
    assert best["optimizer_split"]["test_range"] == [2, 2]
    assert best["optimizer_validation_metrics"]["draws"] == 1
    assert best["optimizer_test_metrics"]["draws"] == 1
    assert best["sniper_mode"] == "off"
    assert best["std_filter_enabled"] is True


def test_voter_optimizer_keeps_current_weights_without_validation_signal():
    optimizer = StrategyOptimizer.__new__(StrategyOptimizer)
    optimizer.reducer = _InactiveSniperReducer()
    optimizer.xp = np
    optimizer._print_progress = lambda *args, **kwargs: None
    history = DrawHistoryDTO(
        dates=["2024-01-01", "2024-01-02"],
        winning_numbers=[[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
        concursos=[1, 2],
    )

    result = optimizer.optimize_voter_weights(history, n_draws=2)

    assert result["selection_inconclusive"] is True
    assert result["w_gap"] == BEST_SETTINGS["w_gap"]
    assert result["w_term"] == BEST_SETTINGS["w_term"]
    assert result["w_freq"] == BEST_SETTINGS["w_freq"]
