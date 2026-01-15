import numpy as np
import pandas as pd
import os
import itertools
import time
import gc
from collections import Counter
from typing import List, Tuple, Dict, Any

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
    MofNCompleteColumn,
)
from rich.console import Console

try:
    import cupy as cp

    HAS_CUPY = True
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)
    print("🚀 NVIDIA GPU DETECTADA: Motor CuPy Activado (Float32 Turbo).")
except ImportError:
    HAS_CUPY = False
    print("⚠ CuPy no detectado. Usando modo CPU estándar.")

try:
    from numba import jit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("⚠ Numba no instalado. Modo lento activado.")

from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO

# --- CONFIGURACIÓN ---
RAW_GENERATION_SIZE = 5_000_000
GPU_CHUNK_SIZE = 1_000_000
QUALITY_PERCENTILE = 77
BATCH_BUFFER_RATE = 1.9

# --- NUMBA KERNELS ---
if HAS_NUMBA:

    @jit(nopython=True, fastmath=True, cache=True)
    def check_ac_vectorized(candidates, ac_min):
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
            ac_value = count - (n_cols - 1)
            keep_mask[i] = ac_value >= ac_min
        return keep_mask

    @jit(nopython=True, fastmath=True, cache=True)
    def score_clusters_fast(candidates, heatmap_matrix):
        n_rows, n_cols = candidates.shape
        scores = np.zeros(n_rows, dtype=np.int32)
        for i in range(n_rows):
            row_score = 0
            for j in range(n_cols):
                for k in range(j + 1, n_cols):
                    a = candidates[i, j]
                    b = candidates[i, k]
                    row_score += heatmap_matrix[a, b]
            scores[i] = row_score
        return scores

else:

    def check_ac_vectorized(candidates, ac_min):
        return np.ones(len(candidates), dtype=bool)

    def score_clusters_fast(candidates, heatmap_matrix):
        return np.zeros(len(candidates), dtype=int)


# --- GPU WORKER ---


def generate_on_gpu_batched(
    target_total_raw: int,
    ticket_size: int,
    total_balls: int,
    weights_np: np.ndarray,
    filter_cfg: Dict[str, Any],
    force_progress: bool = True,
    context_label: str = "GPU Mining",  # Parametro nuevo para etiqueta
) -> np.ndarray:

    console = Console()
    pool_nums_gpu = cp.arange(1, total_balls + 1, dtype=cp.uint8)
    weights_gpu = cp.asarray(weights_np, dtype=cp.float32)
    weights_gpu /= weights_gpu.sum()

    primes_cpu = np.array(
        [
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
        ],
        dtype=bool,
    )
    if len(primes_cpu) < total_balls + 1:
        primes_cpu = np.pad(
            primes_cpu, (0, (total_balls + 1) - len(primes_cpu)), constant_values=0
        )
    primes_lookup_gpu = cp.asarray(primes_cpu[: total_balls + 1])

    survivors_list = []
    generated_count = 0
    total_raw_needed = int(target_total_raw * BATCH_BUFFER_RATE)

    # --- WARM-UP ---
    # Solo mostramos el texto de warm-up si NO estamos en una fase intensiva repetitiva
    # para no spammear la consola.
    if force_progress and "Batch" not in context_label:
        console.print("[dim cyan]   🔥 Calentando núcleos CUDA...[/]")

    try:
        warmup = cp.random.choice(pool_nums_gpu, size=(100, ticket_size), p=weights_gpu)
        cp.cuda.Stream.null.synchronize()
        del warmup
    except Exception as e:
        pass

    # --- BARRA DE PROGRESO MEJORADA ---
    progress = None
    if force_progress:
        progress = Progress(
            SpinnerColumn("dots", style="bold yellow"),
            TextColumn(f"[bold cyan]{context_label}[/]"),  # Etiqueta Dinámica
            BarColumn(bar_width=30, style="dim white", complete_style="green"),
            MofNCompleteColumn(),
            TransferSpeedColumn(),
            TimeElapsedColumn(),  # Cambiado a Tiempo Transcurrido (más estable)
            console=console,
        )
        task_id = progress.add_task("Gen", total=total_raw_needed)
        progress.start()

    try:
        while generated_count < total_raw_needed:
            remaining = total_raw_needed - generated_count
            current_chunk_size = min(remaining, GPU_CHUNK_SIZE)

            raw_batch_gpu = cp.random.choice(
                pool_nums_gpu,
                size=(current_chunk_size, ticket_size),
                replace=True,
                p=weights_gpu,
            ).astype(cp.uint8)
            raw_batch_gpu.sort(axis=1)

            diffs = cp.diff(raw_batch_gpu, axis=1)
            mask_unique = cp.min(diffs, axis=1) > 0
            candidates_gpu = raw_batch_gpu[mask_unique]

            if len(candidates_gpu) > 0:
                sums = cp.sum(candidates_gpu, axis=1)
                mask = (sums >= filter_cfg["sum_min"]) & (sums <= filter_cfg["sum_max"])
                if cp.any(mask):
                    evens = (candidates_gpu % 2 == 0).sum(axis=1)
                    sub_mask = (evens >= filter_cfg["even_min"]) & (
                        evens <= filter_cfg["even_max"]
                    )
                    mask = mask & sub_mask
                if cp.any(mask):
                    p_counts = primes_lookup_gpu[candidates_gpu].sum(axis=1)
                    sub_mask = (p_counts >= filter_cfg["prime_min"]) & (
                        p_counts <= filter_cfg["prime_max"]
                    )
                    mask = mask & sub_mask
                batch_survivors = candidates_gpu[mask]
                if len(batch_survivors) > 0:
                    survivors_list.append(cp.asnumpy(batch_survivors))

            del raw_batch_gpu, candidates_gpu
            if "batch_survivors" in locals():
                del batch_survivors
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

            generated_count += current_chunk_size
            if progress:
                progress.update(task_id, advance=current_chunk_size)

    finally:
        if progress:
            progress.stop()

    if not survivors_list:
        return np.array([])
    return np.concatenate(survivors_list, axis=0)


class UniverseReductionStrategy(ILotteryStrategy):
    """
    Estrategia 'Red de Pesca' V6.5 (UX Fixed).
    """

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        overrides = getattr(config, "filter_overrides", {})
        verbose = overrides.get("verbose", False)

        # --- DETECCIÓN DE CONTEXTO ---
        # Si venimos del Optimizador, overrides tendrá info extra
        opt_iter = overrides.get("opt_iter", None)
        opt_total = overrides.get("opt_total", None)

        if opt_iter and opt_total:
            # Estamos en bucle de optimización
            label = f"⚡ GPU Mining (Batch {opt_iter}/{opt_total})"
            force_ui = True
        else:
            # Ejecución normal única
            label = "⚡ GPU Mining"
            force_ui = True  # Opcional: Podrías poner 'verbose' aquí si prefieres ocultarla en normal

        freq_counter = Counter()
        for draw in history.winning_numbers:
            freq_counter.update(draw[:6])
        weights = [freq_counter.get(n, 1) + 1 for n in range(1, config.total_balls + 1)]
        weights_np = np.array(weights, dtype=float)
        weights_np /= weights_np.sum()

        cluster_matrix = np.zeros(
            (config.total_balls + 1, config.total_balls + 1), dtype=np.uint16
        )
        for draw in history.winning_numbers:
            for a, b in itertools.combinations(sorted(draw[:6]), 2):
                cluster_matrix[a, b] += 1
                cluster_matrix[b, a] += 1

        filter_config = {
            "sum_min": overrides.get("sum_min", 108),
            "sum_max": overrides.get("sum_max", 180),
            "even_min": overrides.get("even_min", 2),
            "even_max": overrides.get("even_max", 4),
            "prime_min": overrides.get("prime_min", 1),
            "prime_max": overrides.get("prime_max", 4),
            "ac_min": overrides.get("ac_min", 5),
        }

        final_candidates_np = np.array([])
        if HAS_CUPY:
            try:
                survivors = generate_on_gpu_batched(
                    RAW_GENERATION_SIZE,
                    config.ticket_size,
                    config.total_balls,
                    weights_np,
                    filter_config,
                    force_progress=force_ui,
                    context_label=label,  # Pasamos la etiqueta dinámica
                )
                final_candidates_np = survivors
            except Exception as e:
                print(f"⚠ Error GPU: {e}. Fallback CPU.")
                # Fallback simple
                pool_nums = np.arange(1, config.total_balls + 1)
                final_candidates_np = np.random.choice(
                    pool_nums,
                    size=(RAW_GENERATION_SIZE, config.ticket_size),
                    p=weights_np,
                )

        if len(final_candidates_np) == 0:
            if verbose:
                print("⚠ Universo vacío.")
            return PredictionResultDTO("Empty", [])

        mask_ac = check_ac_vectorized(final_candidates_np, filter_config["ac_min"])
        final_candidates_np = final_candidates_np[mask_ac]
        scores = score_clusters_fast(final_candidates_np, cluster_matrix)

        valid_indices = np.where(scores > 0)[0]
        valid_scores = scores[valid_indices]
        valid_tickets = final_candidates_np[valid_indices]

        if len(valid_scores) > 0:
            threshold = np.percentile(valid_scores, QUALITY_PERCENTILE)
            mask_quality = valid_scores >= threshold
            final_tickets_np = valid_tickets[mask_quality]
            final_universe = list(map(tuple, final_tickets_np))
        else:
            final_universe = []

        return PredictionResultDTO("Universe V6.5", final_universe)
