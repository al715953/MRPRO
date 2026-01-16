import numpy as np
import pandas as pd
import os
import itertools
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

# --- GPU (CuPy) - Engine V9.3 ---
try:
    import cupy as cp

    HAS_CUPY = True
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)
except ImportError:
    HAS_CUPY = False

# --- CPU (Numba JIT) ---
try:
    from numba import jit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO
from src.data_access.config import BEST_SETTINGS

# --- CONFIGURACIÓN DE ALTA DENSIDAD ---
RAW_GENERATION_SIZE = 5_000_000
GPU_CHUNK_SIZE = 1_000_000
BATCH_BUFFER_RATE = 1.9

if HAS_NUMBA:

    @jit(nopython=True, fastmath=True, cache=True)
    def check_ac_original(candidates, ac_min):
        """Capa Numba: Filtro de Complejidad Aritmética de alto rendimiento."""
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


def generate_hybrid_batch(
    target_total_raw, ticket_size, total_balls, weights_np, filter_cfg, verbose=False
):
    """Motor de generación acelerado por GPU CuPy para reducción de universo."""
    pool_nums_gpu = cp.arange(1, total_balls + 1, dtype=cp.uint8)
    weights_gpu = cp.asarray(weights_np, dtype=cp.float32)
    weights_gpu /= weights_gpu.sum()

    survivors_gpu_list = []
    generated_count = 0
    total_raw_needed = int(target_total_raw * BATCH_BUFFER_RATE)

    progress = None
    if verbose:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold green]⚡ GPU Generating (Engine V9.3)...[/]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        )
        task = progress.add_task("GPU", total=total_raw_needed)
        progress.start()

    try:
        while generated_count < total_raw_needed:
            remaining = total_raw_needed - generated_count
            current_chunk = min(remaining, GPU_CHUNK_SIZE)

            # Generación aleatoria ponderada en el espacio de la GPU
            raw_batch = cp.random.choice(
                pool_nums_gpu,
                size=(current_chunk, ticket_size),
                replace=True,
                p=weights_gpu,
            ).astype(cp.uint8)
            raw_batch.sort(axis=1)

            # Filtro de duplicados internos (vectorización CuPy)
            diffs = cp.diff(raw_batch, axis=1)
            mask = cp.min(diffs, axis=1) > 0
            candidates = raw_batch[mask]

            if len(candidates) > 0:
                # Filtro de Suma optimizado
                sums = cp.sum(candidates, axis=1)
                mask_f = (sums >= filter_cfg["sum_min"]) & (
                    sums <= filter_cfg["sum_max"]
                )
                valid_gpu = candidates[mask_f]
                if len(valid_gpu) > 0:
                    survivors_gpu_list.append(cp.asnumpy(valid_gpu))

            generated_count += current_chunk
            if verbose and progress:
                progress.update(task, advance=current_chunk)
    finally:
        if verbose and progress:
            progress.stop()

    if not survivors_gpu_list:
        return np.array([]), 0

    # Consolidación en memoria CPU
    candidates_cpu = np.concatenate(survivors_gpu_list, axis=0)
    candidates_cpu = np.unique(candidates_cpu, axis=0)

    # Filtrado AC vía Numba
    ac_survivors_count = 0
    if len(candidates_cpu) > 0:
        mask_ac = check_ac_original(candidates_cpu, filter_cfg["ac_min"])
        candidates_cpu = candidates_cpu[mask_ac]
        ac_survivors_count = len(candidates_cpu)

    return candidates_cpu, ac_survivors_count


class UniverseReductionStrategy(ILotteryStrategy):
    """
    Fase 1: Reducción de Universo V9.3 (Zero-Copy HPC).
    Implementa un Corte Técnico P80 para garantizar agilidad en la Fase 3.
    """

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        overrides = getattr(config, "filter_overrides", {})
        verbose = overrides.get("verbose", False)

        # 1. Configuración Táctica
        final_cfg = BEST_SETTINGS.copy()
        final_cfg.update(overrides)

        # 2. Análisis de Pesos (Frecuencia Histórica)
        freq_counter = Counter()
        for draw in history.winning_numbers:
            freq_counter.update(draw[:6])

        weights_np = np.array(
            [freq_counter.get(n, 1) + 1 for n in range(1, config.total_balls + 1)],
            dtype=float,
        )
        weights_np /= weights_np.sum()

        # 3. Pre-cálculo de Matriz de Adyacencia para Scoring Geométrico
        cluster_matrix = np.zeros(
            (config.total_balls + 1, config.total_balls + 1), dtype=np.uint16
        )
        for draw in history.winning_numbers:
            for a, b in itertools.combinations(sorted(draw[:6]), 2):
                cluster_matrix[a, b] += 1
                cluster_matrix[b, a] += 1

        # 4. Ejecución del Motor Híbrido (GPU + Numba)
        candidates, ac_surv = generate_hybrid_batch(
            RAW_GENERATION_SIZE,
            config.ticket_size,
            config.total_balls,
            weights_np,
            final_cfg,
            verbose,
        )

        if len(candidates) == 0:
            return PredictionResultDTO("Empty", [])

        # 5. Scoring Geométrico Vectorizado
        n = len(candidates)
        scores = np.zeros(n, dtype=int)
        for i in range(n):
            row = candidates[i]
            s = 0
            for j in range(config.ticket_size):
                for k in range(j + 1, config.ticket_size):
                    s += cluster_matrix[row[j], row[k]]
            scores[i] = s

        # 6. CORTE TÉCNICO V9.3 (Punto de Equilibrio Rendimiento/Recall)
        # Fijamos P80 para mantener un universo manejable de ~300k candidatos.
        # Esto evita la saturación de memoria en el scoring de la IA.
        TECHNICAL_REDUCTION_P = 92.0
        threshold = np.percentile(scores, TECHNICAL_REDUCTION_P)
        final_universe_np = candidates[scores >= threshold]

        # 7. PERSISTENCIA Y TRANSFERENCIA ZERO-COPY
        # Guardamos en disco para análisis forense, pero usamos RAM para ejecución activa.
        os.makedirs("data", exist_ok=True)
        pd.DataFrame(final_universe_np).to_csv(
            os.path.join("data", "universo_reducido.csv"), index=False
        )

        # Inyectamos el ndarray en metadatos para evitar lecturas de disco en el Sniper
        res = PredictionResultDTO(
            "Universe V9.3 - Zero-Copy", [tuple(x) for x in final_universe_np]
        )
        res.metadata = {
            "ac_survivors": int(ac_surv),
            "final_size": int(len(final_universe_np)),
            "raw_ndarray": final_universe_np,  # <--- PUNTERO CRÍTICO HPC
        }
        return res
