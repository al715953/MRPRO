from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Sequence

import numpy as np


class ProblemTooLargeError(RuntimeError):
    """Raised when a requested combinatorial problem exceeds configured guards."""


@dataclass(frozen=True)
class ProblemEstimate:
    v: int
    k: int
    t: int
    candidate_tickets: int
    target_subsets: int
    subsets_per_ticket: int
    incidences: int
    estimated_bytes: int

    def to_dict(self) -> dict:
        return {
            "v": self.v,
            "k": self.k,
            "t": self.t,
            "candidate_tickets": self.candidate_tickets,
            "target_subsets": self.target_subsets,
            "subsets_per_ticket": self.subsets_per_ticket,
            "incidences": self.incidences,
            "estimated_bytes": self.estimated_bytes,
        }


def estimate_problem(v: int, k: int, t: int) -> ProblemEstimate:
    v, k, t = int(v), int(k), int(t)
    if not 1 <= t <= k <= v:
        raise ValueError("Se requiere 1 <= t <= k <= v")
    candidate_tickets = math.comb(v, k)
    target_subsets = math.comb(v, t)
    subsets_per_ticket = math.comb(k, t)
    incidences = candidate_tickets * subsets_per_ticket
    estimated_bytes = (
        candidate_tickets * k * np.dtype(np.int16).itemsize
        + target_subsets * t * np.dtype(np.int16).itemsize
        + incidences * np.dtype(np.int32).itemsize
    )
    return ProblemEstimate(
        v=v,
        k=k,
        t=t,
        candidate_tickets=candidate_tickets,
        target_subsets=target_subsets,
        subsets_per_ticket=subsets_per_ticket,
        incidences=incidences,
        estimated_bytes=estimated_bytes,
    )


@dataclass(frozen=True)
class CombinatorialProblem:
    candidate_numbers: tuple[int, ...]
    k: int
    t: int
    ticket_positions: np.ndarray
    tickets: np.ndarray
    target_positions: np.ndarray
    ticket_target_indices: np.ndarray
    estimate: ProblemEstimate

    @property
    def v(self) -> int:
        return len(self.candidate_numbers)

    @property
    def n_tickets(self) -> int:
        return int(self.ticket_positions.shape[0])

    @property
    def n_targets(self) -> int:
        return int(self.target_positions.shape[0])

    @classmethod
    def build(
        cls,
        candidate_numbers: Sequence[int],
        k: int,
        t: int,
        *,
        max_candidate_tickets: int = 200_000,
        max_target_subsets: int = 250_000,
        max_incidences: int = 8_000_000,
    ) -> "CombinatorialProblem":
        numbers = tuple(int(number) for number in candidate_numbers)
        if len(numbers) != len(set(numbers)):
            raise ValueError("candidate_numbers contiene duplicados")
        if any(number <= 0 for number in numbers):
            raise ValueError("candidate_numbers debe contener enteros positivos")
        estimate = estimate_problem(len(numbers), k, t)
        violations = []
        if estimate.candidate_tickets > int(max_candidate_tickets):
            violations.append(
                f"tickets={estimate.candidate_tickets:,}>{int(max_candidate_tickets):,}"
            )
        if estimate.target_subsets > int(max_target_subsets):
            violations.append(
                f"targets={estimate.target_subsets:,}>{int(max_target_subsets):,}"
            )
        if estimate.incidences > int(max_incidences):
            violations.append(
                f"incidencias={estimate.incidences:,}>{int(max_incidences):,}"
            )
        if violations:
            raise ProblemTooLargeError("Configuración omitida: " + ", ".join(violations))

        positions = range(len(numbers))
        ticket_positions = np.asarray(
            list(itertools.combinations(positions, int(k))), dtype=np.int16
        )
        target_positions = np.asarray(
            list(itertools.combinations(positions, int(t))), dtype=np.int16
        )
        target_index = {
            tuple(int(value) for value in target): idx
            for idx, target in enumerate(target_positions)
        }
        subsets_per_ticket = estimate.subsets_per_ticket
        incidence = np.empty(
            (estimate.candidate_tickets, subsets_per_ticket), dtype=np.int32
        )
        for ticket_idx, ticket in enumerate(ticket_positions):
            incidence[ticket_idx] = [
                target_index[tuple(int(value) for value in target)]
                for target in itertools.combinations(ticket.tolist(), int(t))
            ]

        number_array = np.asarray(numbers, dtype=np.int16)
        tickets = np.sort(number_array[ticket_positions], axis=1)
        return cls(
            candidate_numbers=numbers,
            k=int(k),
            t=int(t),
            ticket_positions=ticket_positions,
            tickets=tickets,
            target_positions=target_positions,
            ticket_target_indices=incidence,
            estimate=estimate,
        )

    def selected_tickets(self, indices: Sequence[int]) -> np.ndarray:
        selected = np.asarray(indices, dtype=np.int64)
        if selected.size == 0:
            return np.empty((0, self.k), dtype=np.int16)
        return self.tickets[selected]
