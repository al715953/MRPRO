"""Experimental covering-design tools for lottery ticket portfolios."""

from .config import CoveringExperimentConfig
from .covering import (
    CombinatorialProblem,
    ProblemEstimate,
    ProblemTooLargeError,
    estimate_problem,
)
from .greedy import DesignSolution, greedy_maximum_coverage
from .local_search import improve_by_local_search
from .multiobjective import (
    MultiObjectiveResult,
    improve_weighted_local_search,
    weighted_greedy_maximum_coverage,
)
from .shadow import (
    CoveringShadowSpec,
    PROMOTED_COVERING_SHADOWS,
    build_promoted_covering_shadows,
)

__all__ = [
    "CombinatorialProblem",
    "CoveringExperimentConfig",
    "DesignSolution",
    "ProblemEstimate",
    "ProblemTooLargeError",
    "estimate_problem",
    "greedy_maximum_coverage",
    "improve_by_local_search",
    "MultiObjectiveResult",
    "improve_weighted_local_search",
    "weighted_greedy_maximum_coverage",
    "CoveringShadowSpec",
    "PROMOTED_COVERING_SHADOWS",
    "build_promoted_covering_shadows",
]
