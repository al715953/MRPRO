import numpy as np
import pandas as pd
import os
import itertools
import time
from multiprocessing import Pool, cpu_count
from collections import Counter
from typing import List, Tuple, Dict, Any

# --- NUEVO: JIT Compilation ---
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("⚠ ADVERTENCIA: Numba no instalado. Modo lento activado.")

from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import (
    DrawHistoryDTO,
    PredictionConfigDTO,
    PredictionResultDTO,
)

# --- CONFIGURACIÓN ---
RAW_GENERATION_SIZE = 5_000_000
QUALITY_PERCENTILE = 77
BATCH_BUFFER_RATE = 1.9 

# --- NUMBA KERNELS (Código Máquina) ---

if HAS_NUMBA:
    @jit(nopython=True)
    def check_ac_vectorized(candidates, ac_min):
        """
        Calcula AC Value para un batch entero de tickets usando bucles C-level.
        Retorna una máscara booleana.
        AC = UniqueDiffs - (TicketSize - 1)
        """
        n_rows, n_cols = candidates.shape
        keep_mask = np.empty(n_rows, dtype=np.bool_)
        
        # Iteramos sobre cada ticket (fila)
        for i in range(n_rows):
            # Calcular diferencias únicas manualmente para evitar sets (lento en GPU/Numba)
            # Max diferencias posibles para 6 bolas: 15 (5+4+3+2+1)
            # Usamos un array fijo pequeño como buffer de diffs
            diffs = np.zeros(15, dtype=np.int32)
            count = 0
            
            for j in range(n_cols):
                for k in range(j + 1, n_cols):
                    d = candidates[i, k] - candidates[i, j]
                    
                    # Verificar si ya existe en diffs
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
    # Fallback lento si no hay Numba
    def check_ac_vectorized(candidates, ac_min):
        n_rows = candidates.shape[0]
        mask = np.zeros(n_rows, dtype=bool)
        for i in range(n_rows):
            nums = candidates[i]
            diffs = {b - a for a, b in itertools.combinations(nums, 2)}
            ac = len(diffs) - (len(nums) - 1)
            mask[i] = ac >= ac_min
        return mask

# --- WORKER ---

def worker_weighted_generation_optimized(args: Tuple) -> List[Tuple[float, Tuple[int, ...]]]:
    (
        target_batch_size,
        ticket_size,
        total_balls,
        weights_array,
        top_clusters_dict, # Ahora pasamos dict, no set, para velocidad
        filter_cfg,
    ) = args

    # 1. Generación Vectorial (Numpy)
    pool_nums = np.arange(1, total_balls + 1)
    raw_size = int(target_batch_size * BATCH_BUFFER_RATE)

    raw_batch = np.random.choice(
        pool_nums, size=(raw_size, ticket_size), replace=True, p=weights_array
    )
    raw_batch.sort(axis=1)

    # 2. Filtro Unicidad
    diffs = np.diff(raw_batch, axis=1)
    mask_unique = np.min(diffs, axis=1) > 0
    candidates = raw_batch[mask_unique]
    diffs = diffs[mask_unique] # Sincronizar diffs

    if len(candidates) == 0: return []

    # 3. Filtros Vectorizados Básicos
    # A. Suma
    sums = candidates.sum(axis=1)
    mask = (sums >= filter_cfg["sum_min"]) & (sums <= filter_cfg["sum_max"])

    # B. Pares
    if np.any(mask):
        evens = (candidates[mask] % 2 == 0).sum(axis=1)
        sub_mask = (evens >= filter_cfg["even_min"]) & (evens <= filter_cfg["even_max"])
        mask[mask] = sub_mask

    # C. Primos
    if np.any(mask):
        primes_lookup = np.array([False] * (total_balls + 2))
        primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        primes_lookup[primes_list] = True
        
        subset = candidates[mask]
        p_counts = primes_lookup[subset].sum(axis=1)
        sub_mask = (p_counts >= filter_cfg["prime_min"]) & (p_counts <= filter_cfg["prime_max"])
        mask[mask] = sub_mask

    # Aplicar reducción drástica inicial
    survivors = candidates[mask]

    if len(survivors) == 0: return []

    # 4. FILTRO PESADO: AC VALUE (Ahora con Numba)
    # Aquí estaba el cuello de botella. Ahora es C-speed.
    mask_ac = check_ac_vectorized(survivors, filter_cfg["ac_min"])
    final_candidates_np = survivors[mask_ac]

    # 5. Scoring (Clusters) - Lógica pura Python (necesaria por el Dict lookup)
    # Como ya filtramos el 99% de basura, este loop es rápido.
    valid_candidates = []
    
    # Pre-cálculo para loop
    # Convertimos a lista de tuplas para iteración rápida
    for row in final_candidates_np:
        tup = tuple(row)
        score = 0
        
        # Itertools es muy rápido en C
        for pair in itertools.combinations(tup, 2):
            if pair in top_clusters_dict:
                score += top_clusters_dict[pair]
        
        if score > 0:
            valid_candidates.append((score, tup))

    # Recorte al target
    return valid_candidates[:target_batch_size]


class UniverseReductionStrategy(ILotteryStrategy):
    """
    Estrategia 'Red de Pesca' V4 (Numba Accelerated).
    """

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        overrides = getattr(config, "filter_overrides", {})
        verbose = overrides.get("verbose", False)

        start_time = time.time()

        if verbose:
            print(f"🌌 Generando Universo V4 (Numba Engine: {'ON' if HAS_NUMBA else 'OFF'})...")

        # --- PREPARACIÓN ---
        freq_counter = Counter()
        for draw in history.winning_numbers:
            freq_counter.update(draw[:6])

        weights = [freq_counter.get(n, 1) + 1 for n in range(1, config.total_balls + 1)]
        weights_np = np.array(weights, dtype=float)
        weights_np /= weights_np.sum()

        # Clusters (Hot Pairs)
        cluster_counter = Counter()
        for draw in history.winning_numbers:
            for pair in itertools.combinations(sorted(draw[:6]), 2):
                cluster_counter[pair] += 1
        # Convertimos a dict normal para serialización rápida en multiproceso
        clusters_dict = dict(cluster_counter)

        filter_config = {
            "sum_min": overrides.get("sum_min", 108),
            "sum_max": overrides.get("sum_max", 180),
            "even_min": overrides.get("even_min", 2),
            "even_max": overrides.get("even_max", 4),
            "prime_min": overrides.get("prime_min", 1),
            "prime_max": overrides.get("prime_max", 4),
            "ac_min": overrides.get("ac_min", 5),
        }

        # --- MULTIPROCESSING ---
        num_cores = max(1, cpu_count() - 1)
        chunk_size = RAW_GENERATION_SIZE // num_cores

        args = [
            (
                chunk_size,
                config.ticket_size,
                config.total_balls,
                weights_np,
                clusters_dict,
                filter_config,
            )
            for _ in range(num_cores)
        ]

        global_pool = []
        with Pool(processes=num_cores) as pool:
            results = pool.map(worker_weighted_generation_optimized, args)
            for res in results:
                global_pool.extend(res)

        # --- CIERRE ---
        if not global_pool:
            if verbose: print("⚠ Universo vacío. Relaja filtros.")
            return PredictionResultDTO("Empty", [])

        # Deduplicación rápida
        unique_map = {t: s for s, t in global_pool}
        pool_list = list(unique_map.items())

        # Corte por Calidad
        scores = np.array([x[1] for x in pool_list])
        threshold = np.percentile(scores, QUALITY_PERCENTILE)
        final_universe = [t for t, s in pool_list if s >= threshold]

        # Guardar solo si es ejecución real (no backtest loop interno)
        if verbose:
            output_folder = "data"
            os.makedirs(output_folder, exist_ok=True)
            filename = os.path.join(output_folder, "universo_reducido.csv")
            df = pd.DataFrame(final_universe, columns=[f"B{i}" for i in range(1, 7)])
            df.to_csv(filename, index=False)
            
            elapsed = time.time() - start_time
            print(
                f"⏱️  Tiempo: {elapsed:.2f}s | 📥 Bruto: {len(pool_list):,} | 📤 Neto: {len(final_universe):,}"
            )

        return PredictionResultDTO("Universe V4", [list(t) for t in final_universe])