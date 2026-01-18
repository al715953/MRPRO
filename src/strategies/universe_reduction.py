import numpy as np
import pandas as pd
import os
import itertools
from collections import Counter
from typing import List, Tuple
from numba import jit
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO
from src.data_access.config import BEST_SETTINGS

# --- CONFIGURACIÓN HPC OPTIMIZADA ---
RAW_GENERATION_SIZE = 5_000_000
GPU_CHUNK_SIZE = 1_000_000
BATCH_BUFFER_RATE = 2.5  # Aumentado para compensar filtros


@jit(nopython=True, fastmath=True, cache=True)
def check_ac_original(candidates, ac_min):
    n_rows, n_cols = candidates.shape
    keep_mask = np.empty(n_rows, dtype=np.bool_)
    for i in range(n_rows):
        diffs = np.zeros(15, dtype=np.int16)
        count = 0
        for j in range(n_cols):
            for k in range(j + 1, n_cols):
                d = candidates[i, k] - candidates[i, j]
                exists = False
                for x in range(count):
                    if diffs[x] == d:
                        exists = True
                        break
                if not exists:
                    diffs[count] = d
                    count += 1
        keep_mask[i] = (count - 5) >= ac_min
    return keep_mask


@jit(nopython=True, fastmath=True, cache=True)
def check_interval_harmony(candidates, max_gap=22):
    """
    Filtro Relaxed Harmony V9.5.1.
    - max_gap subido de 15 a 22 para permitir dispersión natural.
    """
    n_rows, n_cols = candidates.shape
    keep_mask = np.empty(n_rows, dtype=np.bool_)
    for i in range(n_rows):
        is_harmonious = True
        consecutive_count = 0
        for j in range(n_cols - 1):
            gap = candidates[i, j + 1] - candidates[i, j]
            if gap == 1:
                consecutive_count += 1
            else:
                consecutive_count = 0
            # Solo bloquea si hay 4 números seguidos (3 gaps de 1) o gap extremo
            if consecutive_count >= 3 or gap > max_gap:
                is_harmonious = False
                break
        keep_mask[i] = is_harmonious
    return keep_mask


def generate_hybrid_batch(
    target_total_raw, ticket_size, total_balls, weights_np, filter_cfg, verbose=False
):
    pool_nums_gpu = cp.arange(1, total_balls + 1, dtype=cp.uint8)
    weights_gpu = cp.asarray(weights_np, dtype=cp.float32)
    weights_gpu /= weights_gpu.sum()
    survivors_gpu_list = []
    generated_count = 0
    total_raw_needed = int(target_total_raw * BATCH_BUFFER_RATE)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]⚡ Harmony Engine V9.5.1...[/]"),
        BarColumn(),
        disable=not verbose,
    ) as progress:
        task = progress.add_task("GPU", total=total_raw_needed)
        while generated_count < total_raw_needed:
            chunk = min(total_raw_needed - generated_count, GPU_CHUNK_SIZE)
            raw_batch = cp.random.choice(
                pool_nums_gpu, size=(chunk, ticket_size), replace=True, p=weights_gpu
            ).astype(cp.uint8)
            raw_batch.sort(axis=1)
            mask = (
                (cp.min(cp.diff(raw_batch, axis=1), axis=1) > 0)
                & (cp.sum(raw_batch, axis=1) >= filter_cfg["sum_min"])
                & (cp.sum(raw_batch, axis=1) <= filter_cfg["sum_max"])
            )
            if cp.any(mask):
                survivors_gpu_list.append(cp.asnumpy(raw_batch[mask]))
            generated_count += chunk
            progress.update(task, advance=chunk)

    if not survivors_gpu_list:
        return np.array([]), 0
    candidates = np.unique(np.concatenate(survivors_gpu_list, axis=0), axis=0)

    # Aplicar filtros con Válvula de Seguridad
    filtered = candidates[check_interval_harmony(candidates, max_gap=22)]
    if len(filtered) < 1000:  # Si el filtro mató casi todo, recuperamos originales
        filtered = candidates

    filtered = filtered[check_ac_original(filtered, filter_cfg["ac_min"])]
    return filtered, len(filtered)


class UniverseReductionStrategy(ILotteryStrategy):
    def _calculate_geo_scores(self, candidates, history):
        matrix = np.zeros((41, 41), dtype=np.uint16)
        for draw in history.winning_numbers:
            for a, b in itertools.combinations(sorted(draw[:6]), 2):
                matrix[a, b] += 1
                matrix[b, a] += 1
        scores = np.zeros(len(candidates), dtype=np.int32)
        for i in range(len(candidates)):
            r = candidates[i]
            for j in range(6):
                for k in range(j + 1, 6):
                    scores[i] += matrix[r[j], r[k]]
        return scores

    def predict(self, history, config):
        final_cfg = BEST_SETTINGS.copy()
        final_cfg.update(getattr(config, "filter_overrides", {}))
        freq = Counter([n for d in history.winning_numbers for n in d[:6]])
        weights = np.array(
            [freq.get(n, 1) + 1 for n in range(1, config.total_balls + 1)], dtype=float
        )

        candidates, _ = generate_hybrid_batch(
            RAW_GENERATION_SIZE,
            6,
            39,
            weights,
            final_cfg,
            final_cfg.get("verbose", False),
        )
        if len(candidates) == 0:
            return PredictionResultDTO("Empty", [])

        scores = self._calculate_geo_scores(candidates, history)
        # Rescate Equilibrado
        d1 = np.sum((candidates >= 1) & (candidates <= 9), axis=1)
        d2 = np.sum((candidates >= 10) & (candidates <= 19), axis=1)
        d3 = np.sum((candidates >= 20) & (candidates <= 29), axis=1)
        d4 = np.sum((candidates >= 30) & (candidates <= 39), axis=1)
        balanced_mask = (d1 >= 1) & (d2 >= 1) & (d3 >= 1) & (d4 >= 1)

        threshold = np.percentile(scores, 85.0)
        final_universe = candidates[
            (scores >= threshold)
            | (balanced_mask & (scores >= np.percentile(scores, 60.0)))
        ]

        res = PredictionResultDTO("Universe V9.5.1", [tuple(x) for x in final_universe])
        res.metadata = {
            "final_size": len(final_universe),
            "raw_ndarray": final_universe,
        }
        return res
