from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class CoveringExperimentConfig:
    """Configuration for one reproducible covering-design experiment."""

    candidate_pool_size: int = 15
    target_subset_size: Optional[int] = None
    secondary_target_subset_size: Optional[int] = None
    primary_target_weight: float = 0.5
    secondary_target_weight: float = 0.5
    ticket_budget: int = 300
    coverage_algorithm: str = "greedy_local"
    random_trials: int = 100
    random_seed: int = 20260816
    local_search_iterations: int = 100
    coverage_target: float = 1.0
    candidate_method: str = "oracle_candidate_set"
    explicit_candidates: Optional[Sequence[int]] = None
    backtest_draws: int = 108
    candidate_rank_depth: int = 500
    include_current_mrpro: bool = True
    current_mrpro_ticket_count: int = 24
    max_candidate_tickets: int = 200_000
    max_target_subsets: int = 250_000
    max_incidences: int = 8_000_000
    permutation_trials: int = 10_000
    temporal_folds: int = 3

    def resolved_target_size(self, ticket_size: int) -> int:
        return (
            int(self.target_subset_size)
            if self.target_subset_size is not None
            else int(ticket_size) - 1
        )

    def validate(self, total_balls: int, ticket_size: int) -> None:
        v = int(self.candidate_pool_size)
        t = self.resolved_target_size(ticket_size)
        if not ticket_size <= v <= total_balls:
            raise ValueError(
                f"candidate_pool_size debe estar entre {ticket_size} y {total_balls}"
            )
        if not 1 <= t <= ticket_size:
            raise ValueError(f"target_subset_size debe estar entre 1 y {ticket_size}")
        secondary_t = self.secondary_target_subset_size
        if secondary_t is not None:
            secondary_t = int(secondary_t)
            if not 1 <= secondary_t <= ticket_size:
                raise ValueError(
                    "secondary_target_subset_size debe estar entre 1 y "
                    f"{ticket_size}"
                )
            if secondary_t == t:
                raise ValueError("Los targets primario y secundario deben ser distintos")
            weights = (
                float(self.primary_target_weight),
                float(self.secondary_target_weight),
            )
            if any(weight <= 0 for weight in weights):
                raise ValueError("Los pesos multiobjetivo deben ser positivos")
        if int(self.ticket_budget) <= 0:
            raise ValueError("ticket_budget debe ser positivo")
        if int(self.current_mrpro_ticket_count) <= 0:
            raise ValueError("current_mrpro_ticket_count debe ser positivo")
        if int(self.candidate_rank_depth) <= 0:
            raise ValueError("candidate_rank_depth debe ser positivo")
        if int(self.random_trials) <= 0:
            raise ValueError("random_trials debe ser positivo")
        if int(self.local_search_iterations) < 0:
            raise ValueError("local_search_iterations no puede ser negativo")
        if int(self.temporal_folds) <= 0:
            raise ValueError("temporal_folds debe ser positivo")
        if not 0.0 < float(self.coverage_target) <= 1.0:
            raise ValueError("coverage_target debe estar en (0, 1]")
        allowed = {
            "mrpro_candidate_set",
            "random_candidate_set",
            "oracle_candidate_set",
            "explicit_candidate_set",
        }
        if self.candidate_method not in allowed:
            raise ValueError(f"candidate_method desconocido: {self.candidate_method}")
        if self.candidate_method == "explicit_candidate_set":
            explicit = tuple(self.explicit_candidates or ())
            if len(set(int(number) for number in explicit)) != v:
                raise ValueError("explicit_candidates debe contener exactamente v únicos")
            if any(int(number) < 1 or int(number) > total_balls for number in explicit):
                raise ValueError("explicit_candidates contiene números fuera del juego")

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.explicit_candidates is not None:
            payload["explicit_candidates"] = [
                int(number) for number in self.explicit_candidates
            ]
        return payload
