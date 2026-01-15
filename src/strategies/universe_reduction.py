import numpy as np
import pandas as pd
import os
import itertools
import time
import gc
from collections import Counter
from typing import List, Tuple, Dict, Any

# --- RICH UI ---
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)

# --- GPU (CuPy) ---
try:
    import cupy as cp

    HAS_CUPY = True
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)
    print("🚀 GPU DETECTADA: Fase 1 (Fuerza Bruta) en CuPy.")
except ImportError:
    HAS_CUPY = False
    print("⚠ CuPy no detectado. Usando modo CPU lento.")

# --- CPU (Numba) ---
try:
    from numba import jit

    HAS_NUMBA = True
    print("🧠 CPU DETECTADA: Fase 2 (Lógica AC) en Numba.")
except ImportError:
    HAS_NUMBA = False
    print("⚠ Numba no instalado.")

from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO

# Importamos tu configuración ORIGINAL (la del backup)
from src.data_access.config import BEST_SETTINGS

# --- CONFIGURACIÓN RESTAURADA (BACKUP VALUES) ---
RAW_GENERATION_SIZE = 5_000_000
GPU_CHUNK_SIZE = 1_000_000
QUALITY_PERCENTILE = 77  # VOLVEMOS AL ORIGINAL
BATCH_BUFFER_RATE = 1.9

# =============================================================================
# 🧠 KERNELS DE CPU (RESTAURADOS DEL BACKUP)
# =============================================================================
if HAS_NUMBA:

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
            ac_value = count - (n_cols - 1)
            keep_mask[i] = ac_value >= ac_min
        return keep_mask

else:

    def check_ac_original(candidates, ac_min):
        return np.ones(len(candidates), dtype=bool)


# =============================================================================
# 🚀 MOTOR HÍBRIDO (GPU -> CPU)
# =============================================================================


def generate_hybrid_batch(
    target_total_raw: int,
    ticket_size: int,
    total_balls: int,
    weights_np: np.ndarray,
    filter_cfg: Dict[str, Any],
) -> np.ndarray:

    # 1. PREPARACIÓN GPU
    pool_nums_gpu = cp.arange(1, total_balls + 1, dtype=cp.uint8)
    weights_gpu = cp.asarray(weights_np, dtype=cp.float32)
    weights_gpu /= weights_gpu.sum()

    # Primos Lookup en GPU
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

    survivors_gpu_list = []
    generated_count = 0
    total_raw_needed = int(target_total_raw * BATCH_BUFFER_RATE)

    # UI
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold green]⚡ GPU Generating...[/]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    )
    task = progress.add_task("GPU", total=total_raw_needed)
    progress.start()

    try:
        # --- FASE 1: FUERZA BRUTA EN GPU ---
        while generated_count < total_raw_needed:
            remaining = total_raw_needed - generated_count
            current_chunk = min(remaining, GPU_CHUNK_SIZE)

            # A. Generar
            raw_batch = cp.random.choice(
                pool_nums_gpu,
                size=(current_chunk, ticket_size),
                replace=True,
                p=weights_gpu,
            ).astype(cp.uint8)
            raw_batch.sort(axis=1)

            # B. Unicidad (Local del batch)
            diffs = cp.diff(raw_batch, axis=1)
            mask = cp.min(diffs, axis=1) > 0
            candidates = raw_batch[mask]

            if len(candidates) > 0:
                # C. Filtros Vectoriales
                sums = cp.sum(candidates, axis=1)
                mask_f = (sums >= filter_cfg["sum_min"]) & (
                    sums <= filter_cfg["sum_max"]
                )

                if cp.any(mask_f):
                    evens = (candidates % 2 == 0).sum(axis=1)
                    sub = (evens >= filter_cfg["even_min"]) & (
                        evens <= filter_cfg["even_max"]
                    )
                    mask_f = mask_f & sub

                if cp.any(mask_f):
                    p_c = primes_lookup_gpu[candidates].sum(axis=1)
                    sub = (p_c >= filter_cfg["prime_min"]) & (
                        p_c <= filter_cfg["prime_max"]
                    )
                    mask_f = mask_f & sub

                valid_gpu = candidates[mask_f]

                if len(valid_gpu) > 0:
                    survivors_gpu_list.append(cp.asnumpy(valid_gpu))

            del raw_batch, candidates
            cp.get_default_memory_pool().free_all_blocks()
            generated_count += current_chunk
            progress.update(task, advance=current_chunk)

    finally:
        progress.stop()

    if not survivors_gpu_list:
        return np.array([])

    # Unimos todo en RAM
    candidates_cpu = np.concatenate(survivors_gpu_list, axis=0)
    print(f"   📥 GPU Raw: {len(candidates_cpu):,} candidatos (con duplicados).")

    # --- FASE CRÍTICA: DEDUPLICACIÓN (V9 FIX) ---
    # Esto elimina los clones generados por la saturación del universo
    candidates_cpu = np.unique(candidates_cpu, axis=0)
    print(f"   ♻️  Deduplicado: {len(candidates_cpu):,} únicos.")

    # --- FASE 2: FILTRO AC (LÓGICA CPU ORIGINAL) ---
    if len(candidates_cpu) > 0:
        mask_ac = check_ac_original(candidates_cpu, filter_cfg["ac_min"])
        candidates_cpu = candidates_cpu[mask_ac]
        print(f"   📉 Filtro AC: {len(candidates_cpu):,} sobreviven.")

    return candidates_cpu


class UniverseReductionStrategy(ILotteryStrategy):
    """
    Estrategia Híbrida V9 (Deduplication Fix).
    """

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        overrides = getattr(config, "filter_overrides", {})
        verbose = overrides.get("verbose", False)

        # 1. Configuración: Usamos BEST_SETTINGS como base y overrides encima
        final_cfg = BEST_SETTINGS.copy()
        final_cfg.update(overrides)

        # Validar rangos críticos del backup
        final_cfg["sum_min"] = final_cfg.get("sum_min", 115)
        final_cfg["sum_max"] = final_cfg.get("sum_max", 172)
        final_cfg["ac_min"] = final_cfg.get("ac_min", 5)

        if verbose:
            print(f"🌌 Generando Universo V9...")
            print(
                f"   ⚙️ Config: Suma[{final_cfg['sum_min']}-{final_cfg['sum_max']}] AC[{final_cfg['ac_min']}] Percentil[{QUALITY_PERCENTILE}]"
            )

        # 2. Pesos
        freq_counter = Counter()
        for draw in history.winning_numbers:
            freq_counter.update(draw[:6])
        weights = [freq_counter.get(n, 1) + 1 for n in range(1, config.total_balls + 1)]
        weights_np = np.array(weights, dtype=float)
        weights_np /= weights_np.sum()

        # 3. Matriz de Clusters
        cluster_matrix = np.zeros(
            (config.total_balls + 1, config.total_balls + 1), dtype=np.uint16
        )
        for draw in history.winning_numbers:
            for a, b in itertools.combinations(sorted(draw[:6]), 2):
                cluster_matrix[a, b] += 1
                cluster_matrix[b, a] += 1

        # 4. EJECUCIÓN
        if HAS_CUPY:
            try:
                candidates = generate_hybrid_batch(
                    RAW_GENERATION_SIZE,
                    config.ticket_size,
                    config.total_balls,
                    weights_np,
                    final_cfg,
                )
            except Exception as e:
                print(f"⚠ Error GPU: {e}.")
                return PredictionResultDTO("Error", [])
        else:
            return PredictionResultDTO("Error", [])

        if len(candidates) == 0:
            return PredictionResultDTO("Empty", [])

        # 5. SCORING & PERCENTIL
        n = len(candidates)
        scores = np.zeros(n, dtype=int)
        for i in range(n):
            row = candidates[i]
            s = 0
            for j in range(config.ticket_size):
                for k in range(j + 1, config.ticket_size):
                    s += cluster_matrix[row[j], row[k]]
            scores[i] = s

        threshold = np.percentile(scores, QUALITY_PERCENTILE)
        mask_quality = scores >= threshold
        final_universe_np = candidates[mask_quality]

        final_universe = list(map(tuple, final_universe_np))

        if verbose:
            print(f"   🏆 Universo Final: {len(final_universe):,} combinaciones.")

        return PredictionResultDTO("Universe V9", final_universe)
