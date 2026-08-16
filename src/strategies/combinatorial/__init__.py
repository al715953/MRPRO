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

__all__ = [
    "CombinatorialProblem",
    "CoveringExperimentConfig",
    "DesignSolution",
    "ProblemEstimate",
    "ProblemTooLargeError",
    "estimate_problem",
    "greedy_maximum_coverage",
    "improve_by_local_search",
]
