import numpy as np
import pandas as pd
import os
import itertools
from collections import Counter
from numba import jit
from colorama import Fore, Style
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from src.domain.dtos import PredictionResultDTO
from src.data_access.config import BEST_SETTINGS

# --- HPC CONFIG ---
RAW_GENERATION_SIZE = 5_000_000
GPU_CHUNK_SIZE = 1_000_000
BATCH_BUFFER_RATE = 2.5


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
            if consecutive_count >= 3 or gap > max_gap:
                is_harmonious = False
                break
        keep_mask[i] = is_harmonious
    return keep_mask


def generate_hybrid_batch(
    target_total_raw, ticket_size, total_balls, weights_np, filter_cfg, verbose=False
):
    xp = cp if HAS_CUPY else np
    pool_nums = xp.arange(1, total_balls + 1, dtype=xp.uint8)
    weights = xp.asarray(weights_np, dtype=xp.float32)
    weights /= weights.sum()

    survivors_list = []
    total_raw_needed = int(target_total_raw * BATCH_BUFFER_RATE)

    with Progress(
        SpinnerColumn(),
        TextColumn(
            f"[bold green]⚡ Harmony Engine ({'GPU' if HAS_CUPY else 'CPU'})...[/]"
        ),
        BarColumn(),
        disable=not verbose,  # SILENCIO INTELIGENTE
    ) as progress:
        task = progress.add_task("Gen", total=total_raw_needed)
        generated = 0
        while generated < total_raw_needed:
            chunk = min(total_raw_needed - generated, GPU_CHUNK_SIZE)
            batch = xp.random.choice(
                pool_nums, size=(chunk, ticket_size), replace=True, p=weights
            ).astype(xp.uint8)
            batch.sort(axis=1)
            mask = (
                (xp.min(xp.diff(batch, axis=1), axis=1) > 0)
                & (xp.sum(batch, axis=1) >= filter_cfg["sum_min"])
                & (xp.sum(batch, axis=1) <= filter_cfg["sum_max"])
            )
            if xp.any(mask):
                data = batch[mask]
                survivors_list.append(data.get() if HAS_CUPY else data)
            generated += chunk
            progress.update(task, advance=chunk)

    if not survivors_list:
        return np.array([]), 0
    candidates = np.unique(np.concatenate(survivors_list, axis=0), axis=0)
    filtered = candidates[check_interval_harmony(candidates)]
    filtered = filtered[check_ac_original(filtered, filter_cfg["ac_min"])]
    return filtered, len(filtered)


class UniverseReductionStrategy:
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
        final_cfg.update(config.filter_overrides)
        is_verbose = final_cfg.get("verbose", False)

        freq = Counter([n for d in history.winning_numbers for n in d[:6]])
        weights = np.array(
            [freq.get(n, 1) + 1 for n in range(1, config.total_balls + 1)], dtype=float
        )

        candidates, _ = generate_hybrid_batch(
            RAW_GENERATION_SIZE, 6, 39, weights, final_cfg, is_verbose
        )
        if len(candidates) == 0:
            return PredictionResultDTO("Empty", [])

        scores = self._calculate_geo_scores(candidates, history)
        threshold = np.percentile(scores, 85.0)
        final_universe = candidates[scores >= threshold]

        # PERSISTENCIA INTELIGENTE (Solo en modo Verbose/Producción)
        if is_verbose:
            try:
                from src.data_access.config import DATA_FOLDER

                os.makedirs(DATA_FOLDER, exist_ok=True)
                out_path = os.path.join(DATA_FOLDER, "universo_reducido.csv")
                pd.DataFrame(final_universe).to_csv(out_path, index=False, header=False)
                print(
                    f"   {Fore.GREEN}💾 Universo persistido en: {out_path}{Style.RESET_ALL}"
                )
            except Exception as e:
                print(f"   {Fore.RED}⚠ Error I/O: {e}{Style.RESET_ALL}")

        res = PredictionResultDTO("Universe V9.5.2", [tuple(x) for x in final_universe])
        res.metadata = {
            "final_size": len(final_universe),
            "raw_ndarray": final_universe,
        }
        return res
