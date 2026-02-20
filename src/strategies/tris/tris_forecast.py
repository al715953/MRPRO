from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO
from src.strategies.tris.topk import beam_search, select_diverse
from src.strategies.tris.v1a_model import TrisV1AModel, _extract_tris_series


class TrisForecastV1A:
    def __init__(self):
        self.strategy_name = "Tris Forecast V1-A"
        self.model_version = "tris_v1a_bayes_markov_001"

    @staticmethod
    def _get_override(cfg: Dict, key: str, default):
        return cfg.get(key, default) if isinstance(cfg, dict) else default

    @staticmethod
    def _uniform_probs() -> np.ndarray:
        return np.full((5, 10), 0.1, dtype=np.float64)

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        overrides = config.filter_overrides or {}

        short_window = int(self._get_override(overrides, "short_window", 200))
        long_window = int(self._get_override(overrides, "long_window", 2000))
        alpha_bayes = float(self._get_override(overrides, "alpha_bayes", 0.5))
        mix_lambda = float(self._get_override(overrides, "mix_lambda", 0.7))
        markov_window = int(self._get_override(overrides, "markov_window", 2000))
        alpha_markov = float(self._get_override(overrides, "alpha_markov", 0.2))
        blend_markov = float(self._get_override(overrides, "blend_markov", 0.35))
        topk_k = int(self._get_override(overrides, "topk_k", 2000))
        per_pos_topm = int(self._get_override(overrides, "per_pos_topm", 6))
        beam_width = int(self._get_override(overrides, "beam_width", 2500))
        diversity_min_hamming = int(
            self._get_override(overrides, "diversity_min_hamming", 2)
        )
        topk_preview = int(self._get_override(overrides, "topk_preview", 50))

        digits_list, mult_list = _extract_tris_series(history)
        n_draws = len(digits_list)

        if n_draws > 0:
            context_last_digits = digits_list[-1]
        else:
            context_last_digits = [0, 0, 0, 0, 0]

        if n_draws < 50:
            pos_probs = self._uniform_probs()
            positives = sum(1 for v in mult_list if v)
            p_multiplier = float((positives + 1.0) / (len(mult_list) + 2.0)) if mult_list else 0.5
            entropy_pos = -np.sum(pos_probs * np.log(pos_probs), axis=1)
            entropy_mean = float(np.mean(entropy_pos))
        else:
            model = TrisV1AModel(
                blend_markov=blend_markov,
                bayes_params={
                    "alpha": alpha_bayes,
                    "short_window": short_window,
                    "long_window": long_window,
                    "mix_lambda": mix_lambda,
                },
                markov_params={
                    "alpha": alpha_markov,
                    "window": markov_window,
                },
            )
            model.fit(digits_list, mult_list)
            pos_probs, p_multiplier, entropy_pos, entropy_mean = model.predict(
                context_last_digits
            )

        candidates = beam_search(
            pos_probs,
            k=topk_k,
            per_pos_topm=per_pos_topm,
            beam_width=beam_width,
        )
        tickets = select_diverse(
            candidates,
            n=config.num_tickets,
            min_hamming=diversity_min_hamming,
        )

        if len(tickets) < config.num_tickets:
            fallback = [int(np.argmax(pos_probs[pos])) for pos in range(5)]
            while len(tickets) < config.num_tickets:
                tickets.append(fallback[:])

        metadata = {
            "pos_probs": pos_probs.tolist(),
            "p_multiplier": float(p_multiplier),
            "entropy_pos": entropy_pos.tolist(),
            "entropy_mean": float(entropy_mean),
            "model_version": self.model_version,
            "topk_preview": [
                {"digits": d, "logp": float(lp)}
                for d, lp in candidates[: max(0, topk_preview)]
            ],
        }

        return PredictionResultDTO(
            strategy_name=self.strategy_name,
            tickets=tickets[: config.num_tickets],
            metadata=metadata,
        )
