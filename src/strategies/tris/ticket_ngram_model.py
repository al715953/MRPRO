from __future__ import annotations

from typing import Sequence

import numpy as np


class TicketNgramModel:
    """
    Modelo n-grama por ticket (bigrama posicional intra-ticket).

    IMPORTANT: No-leakage.
    El caller debe entrenar este modelo solo con historia disponible hasta t-1
    para puntuar candidatos del sorteo t.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        window: int = 2000,
        short_window: int = 200,
        long_window: int = 2000,
        mix_lambda: float = 0.7,
        uniform_mix: float = 0.0,
    ):
        self.alpha = float(alpha)
        self.window = int(window)
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.mix_lambda = float(mix_lambda)
        self.uniform_mix = float(uniform_mix)

        self.short_counts = np.zeros((4, 10, 10), dtype=np.float64)
        self.long_counts = np.zeros((4, 10, 10), dtype=np.float64)
        self.short_d0_counts = np.zeros(10, dtype=np.float64)
        self.long_d0_counts = np.zeros(10, dtype=np.float64)

        self.trans_probs = np.full((4, 10, 10), 0.1, dtype=np.float64)
        self.d0_probs = np.full(10, 0.1, dtype=np.float64)

    @staticmethod
    def _coerce_digit(value) -> int:
        try:
            d = int(float(value))
        except Exception:
            d = 0
        return d % 10

    @classmethod
    def _coerce_rows(cls, digits_list: Sequence[Sequence[int]]) -> np.ndarray:
        rows = []
        for row in digits_list or []:
            if row is None or len(row) < 5:
                continue
            rows.append([cls._coerce_digit(row[i]) for i in range(5)])
        if not rows:
            return np.empty((0, 5), dtype=np.int16)
        return np.asarray(rows, dtype=np.int16)

    @staticmethod
    def _coerce_tickets_array(all_tickets: np.ndarray) -> np.ndarray:
        tickets = np.asarray(all_tickets)
        if tickets.ndim != 2 or tickets.shape[1] < 5:
            raise ValueError("all_tickets debe tener shape (N, 5).")
        tickets = np.asarray(tickets[:, :5], dtype=np.int16)
        tickets = np.mod(tickets, 10)
        return tickets.astype(np.int16, copy=False)

    @staticmethod
    def _count_transitions(rows: np.ndarray) -> np.ndarray:
        counts = np.zeros((4, 10, 10), dtype=np.float64)
        if rows.shape[0] == 0:
            return counts
        for pos in range(4):
            prev_digits = rows[:, pos].astype(np.int64, copy=False)
            next_digits = rows[:, pos + 1].astype(np.int64, copy=False)
            np.add.at(counts[pos], (prev_digits, next_digits), 1.0)
        return counts

    @staticmethod
    def _count_d0(rows: np.ndarray) -> np.ndarray:
        counts = np.zeros(10, dtype=np.float64)
        if rows.shape[0] == 0:
            return counts
        d0 = rows[:, 0].astype(np.int64, copy=False)
        np.add.at(counts, d0, 1.0)
        return counts

    def _smoothed_transitions(self, counts: np.ndarray) -> np.ndarray:
        numer = np.asarray(counts, dtype=np.float64) + self.alpha
        probs = numer / np.clip(np.sum(numer, axis=2, keepdims=True), 1e-12, None)
        u = min(max(self.uniform_mix, 0.0), 1.0)
        if u > 0.0:
            probs = (1.0 - u) * probs + u * 0.1
            probs /= np.clip(np.sum(probs, axis=2, keepdims=True), 1e-12, None)
        return probs

    def _smoothed_d0(self, counts: np.ndarray) -> np.ndarray:
        numer = np.asarray(counts, dtype=np.float64) + self.alpha
        return numer / max(float(np.sum(numer)), 1e-12)

    def fit(self, digits_list: Sequence[Sequence[int]]) -> None:
        rows_all = self._coerce_rows(digits_list)

        if rows_all.shape[0] == 0:
            self.short_counts.fill(0.0)
            self.long_counts.fill(0.0)
            self.short_d0_counts.fill(0.0)
            self.long_d0_counts.fill(0.0)
            self.trans_probs.fill(0.1)
            self.d0_probs.fill(0.1)
            return

        max_w = max(0, self.window)
        rows_w = rows_all[-min(rows_all.shape[0], max_w) :] if max_w > 0 else rows_all[:0]

        short_n = min(rows_w.shape[0], max(0, self.short_window))
        long_n = min(rows_w.shape[0], max(0, self.long_window))
        rows_short = rows_w[-short_n:] if short_n > 0 else rows_w[:0]
        rows_long = rows_w[-long_n:] if long_n > 0 else rows_w[:0]

        self.short_counts = self._count_transitions(rows_short)
        self.long_counts = self._count_transitions(rows_long)
        self.short_d0_counts = self._count_d0(rows_short)
        self.long_d0_counts = self._count_d0(rows_long)

        short_trans_probs = self._smoothed_transitions(self.short_counts)
        long_trans_probs = self._smoothed_transitions(self.long_counts)
        short_d0_probs = self._smoothed_d0(self.short_d0_counts)
        long_d0_probs = self._smoothed_d0(self.long_d0_counts)

        lam = min(max(self.mix_lambda, 0.0), 1.0)
        self.trans_probs = lam * short_trans_probs + (1.0 - lam) * long_trans_probs
        self.trans_probs /= np.clip(
            np.sum(self.trans_probs, axis=2, keepdims=True), 1e-12, None
        )
        self.d0_probs = lam * short_d0_probs + (1.0 - lam) * long_d0_probs
        self.d0_probs /= max(float(np.sum(self.d0_probs)), 1e-12)

    def score_all(self, all_tickets: np.ndarray) -> np.ndarray:
        tickets = self._coerce_tickets_array(all_tickets)
        n = tickets.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float64)

        d0 = tickets[:, 0].astype(np.int64, copy=False)
        scores = np.log(np.clip(self.d0_probs[d0], 1e-12, None))
        for pos in range(4):
            a = tickets[:, pos].astype(np.int64, copy=False)
            b = tickets[:, pos + 1].astype(np.int64, copy=False)
            scores += np.log(np.clip(self.trans_probs[pos, a, b], 1e-12, None))
        return scores.astype(np.float64, copy=False)

    def score_ticket(self, ticket: Sequence[int]) -> float:
        if ticket is None or len(ticket) < 5:
            raise ValueError("ticket debe tener al menos 5 digitos.")
        arr = np.asarray([ticket[:5]], dtype=np.int16)
        return float(self.score_all(arr)[0])
