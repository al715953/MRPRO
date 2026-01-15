import pandas as pd
import numpy as np
import os
import itertools
from collections import Counter
from typing import List, Tuple, Dict, Optional, Any
from rich.console import Console

# --- JIT COMPILATION (Mejora de Rendimiento 50x) ---
try:
    from numba import jit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO
from src.core.ai_scorer import LotteryAIModel

console = Console()

# --- KERNELS NUMBA (CÓDIGO MÁQUINA) ---
if HAS_NUMBA:

    @jit(nopython=True, fastmath=True, cache=True)
    def calc_heuristics_vectorized(
        candidates, cluster_matrix, hotness_vector, total_balls
    ):
        """
        Calcula Cluster Score y Hotness Score para todo el array de candidatos a la vez.
        candidates: (N, 6) uint8/int
        cluster_matrix: (40, 40) uint16
        hotness_vector: (40,) uint16
        """
        n_rows, n_cols = candidates.shape
        cluster_scores = np.zeros(n_rows, dtype=np.float32)
        hotness_scores = np.zeros(n_rows, dtype=np.float32)

        for i in range(n_rows):
            # 1. Cluster Score (Sumatoria de pesos de pares)
            c_score = 0
            for j in range(n_cols):
                for k in range(j + 1, n_cols):
                    a = candidates[i, j]
                    b = candidates[i, k]
                    c_score += cluster_matrix[a, b]
            cluster_scores[i] = c_score

            # 2. Hotness Score (Sumatoria de frecuencia individual)
            h_score = 0
            for j in range(n_cols):
                val = candidates[i, j]
                # Safety check por si hay bolas fuera de rango
                if val <= total_balls:
                    h_score += hotness_vector[val]
            hotness_scores[i] = h_score

        return cluster_scores, hotness_scores

else:
    # Fallback lento
    def calc_heuristics_vectorized(
        candidates, cluster_matrix, hotness_vector, total_balls
    ):
        n_rows = len(candidates)
        c_scores = np.zeros(n_rows)
        h_scores = np.zeros(n_rows)
        for i in range(n_rows):
            row = candidates[i]
            # Cluster
            c = 0
            for a, b in itertools.combinations(row, 2):
                c += cluster_matrix[a, b]
            c_scores[i] = c
            # Hotness
            h = 0
            for val in row:
                if val <= total_balls:
                    h += hotness_vector[val]
            h_scores[i] = h
        return c_scores, h_scores


class GeneticSelectorStrategy(ILotteryStrategy):
    """
    ESTRATEGIA 'CENTAURO' V12 (Vectorized & Floating Thresholds).

    MEJORAS V12:
    - Motor Numba: Cálculo de scores heurísticos masivo (Matrix Lookup).
    - Bucketing Vectorial: Numpy Boolean Masking para clasificación instantánea.
    - Retro-compatibilidad: Mantiene APIs para Optimizer y Backtester.
    """

    def __init__(self):
        self.ai_model = LotteryAIModel()
        self._last_trained_date = None

        # Cache optimizado (Matrices Numpy en lugar de Dicts)
        self._matrix_cache = {
            "cluster_matrix": None,  # (40, 40)
            "hotness_vector": None,  # (40,)
            "max_cluster": 1.0,
            "max_hotness": 1.0,
        }

    def _train_model(self, history: DrawHistoryDTO, total_balls: int):
        last_date = history.dates[-1] if history.dates else "None"

        # Re-entrenamos solo si hay datos nuevos
        if self._last_trained_date != last_date:
            # 1. Entrenar XGBoost
            self.ai_model.train(history.winning_numbers, total_balls)

            # 2. Actualizar Matrices Heurísticas (V12)
            self._update_heuristic_matrices(history, total_balls)

            self._last_trained_date = last_date

    def _update_heuristic_matrices(self, history: DrawHistoryDTO, total_balls: int):
        """Construye las matrices de lookup para Numba."""
        # A. Cluster Matrix (Mapa de calor de pares)
        # Usamos uint16 para ahorrar memoria pero permitir conteos > 255
        matrix = np.zeros((total_balls + 2, total_balls + 2), dtype=np.uint16)

        # Llenado rápido
        for draw in history.winning_numbers:
            # Asumimos draw ordenado, o lo ordenamos
            sorted_draw = sorted(draw[:6])
            for a, b in itertools.combinations(sorted_draw, 2):
                matrix[a, b] += 1
                matrix[b, a] += 1  # Simetría

        max_cluster_val = np.max(matrix) if np.max(matrix) > 0 else 1

        # B. Hotness Vector (Frecuencia reciente)
        # Analizamos últimos 15 sorteos para "Hotness" (tendencia corta)
        recent_draws = history.winning_numbers[-15:]
        freq_vec = np.zeros(total_balls + 2, dtype=np.uint16)

        for draw in recent_draws:
            for num in draw[:6]:
                freq_vec[num] += 1

        max_hot_val = np.max(freq_vec) if np.max(freq_vec) > 0 else 1

        self._matrix_cache["cluster_matrix"] = matrix
        self._matrix_cache["hotness_vector"] = freq_vec
        self._matrix_cache["max_cluster"] = float(max_cluster_val)
        self._matrix_cache["max_hotness"] = float(max_hot_val)

    def _compute_v7_score_legacy(
        self, ticket: Tuple[int, ...], ai_score: float, weights: Dict[str, float]
    ) -> Tuple[float, float, float]:
        """
        Wrapper para mantener compatibilidad con Optimizer.py y Forensic.
        Calcula el score de un SOLO ticket usando las matrices cacheadas.
        """
        if self._matrix_cache["cluster_matrix"] is None:
            # Safety init si se llama fuera de flujo normal
            return 0.5, 0.5, 0.5

        # Extraer datos de cache
        mat = self._matrix_cache["cluster_matrix"]
        vec = self._matrix_cache["hotness_vector"]
        mc = self._matrix_cache["max_cluster"]
        mh = self._matrix_cache["max_hotness"]

        # 1. Cluster Score
        raw_c = 0
        for a, b in itertools.combinations(ticket, 2):
            raw_c += mat[a, b]
        # Normalizar: Max posible por ticket es 15 pares * max_val_pair
        # Ajustamos normalización para que sea relativa al maximo histórico
        norm_c = raw_c / (15 * mc) if mc > 0 else 0

        # 2. Hotness Score
        raw_h = sum(vec[n] for n in ticket if n < len(vec))
        norm_h = raw_h / (6 * mh) if mh > 0 else 0

        # Penalización por exceso de hotness (Mean Reversion)
        if norm_h > 0.75:
            norm_h *= 0.80

        # 3. Fusión
        w_cluster = weights.get("w_cluster", 0.6)
        w_hotness = weights.get("w_hotness", 0.4)
        w_ai = weights.get("w_ai", 0.3)

        heur_score = (norm_c * w_cluster) + (norm_h * w_hotness)

        # Balance Global: (Heuristica * (1-w_ai)) + (AI * w_ai)
        w_heur_global = 1.0 - w_ai
        # Ajuste legacy para mantener comportamiento de configs viejas donde w_ai=0.3
        if w_ai == 0.3:
            w_heur_global = 0.7

        final_score = (heur_score * w_heur_global) + (ai_score * w_ai)

        return final_score, heur_score, ai_score

    # Alias para compatibilidad con llamadas externas que busquen _compute_v7_score
    _compute_v7_score = _compute_v7_score_legacy

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        overrides = getattr(config, "filter_overrides", {})
        verbose = overrides.get("verbose", True)

        # --- UMBRALES DINÁMICOS ---
        th_elite = overrides.get("threshold_elite", 0.70)
        th_mid = overrides.get("threshold_mid", 0.60)

        if verbose:
            console.print(
                f"\n[bold yellow]🧬 CENTAURO V12 (Vectorized Numba Engine E:{th_elite} M:{th_mid})...[/]"
            )

        # 1. Cargar Universo (Optimizado con Pandas Engine C)
        csv_path = os.path.join("data", "universo_reducido.csv")
        if not os.path.exists(csv_path):
            return PredictionResultDTO("Error: No Universe", [])

        try:
            # Leemos directo a Numpy int8/uint8 para velocidad
            df = pd.read_csv(csv_path)
            # Asumimos columnas B1..B6
            candidates_np = df.iloc[:, :6].values.astype(np.uint8)
        except Exception as e:
            return PredictionResultDTO(f"Error reading CSV: {e}", [])

        if len(candidates_np) == 0:
            return PredictionResultDTO("Empty Universe", [])

        # 2. Entrenar y Preparar Matrices
        self._train_model(history, config.total_balls)

        # 3. SCORING MASIVO (Fase Vectorizada)
        # A. AI Score (XGBoost ya es vectorizado)
        # Convertimos a tuplas solo si el modelo lo exige, o pasamos array si lo soporta.
        # (AIModel.score_tickets espera tuplas actualmente, optimizaremos eso luego.
        #  Por ahora el cuello de botella era el loop python heurístico).
        tuples_list = [tuple(x) for x in candidates_np]
        raw_ai_scores = self.ai_model.score_tickets(tuples_list)

        # B. Heurística Vectorizada (Numba)
        raw_c_scores, raw_h_scores = calc_heuristics_vectorized(
            candidates_np,
            self._matrix_cache["cluster_matrix"],
            self._matrix_cache["hotness_vector"],
            config.total_balls,
        )

        # C. Normalización y Fusión (Numpy Broadcasting)
        mc = self._matrix_cache["max_cluster"]
        mh = self._matrix_cache["max_hotness"]

        # Normalización vectorial
        norm_c = raw_c_scores / (15 * mc)
        norm_h = raw_h_scores / (6 * mh)

        # Penalización hotness (Mean Reversion) vectorial
        norm_h = np.where(norm_h > 0.75, norm_h * 0.80, norm_h)

        # Pesos
        wc = overrides.get("w_cluster", 0.6)
        wh = overrides.get("w_hotness", 0.4)
        wai = overrides.get("w_ai", 0.3)

        # Fórmula Final Vectorizada
        heur_val = (norm_c * wc) + (norm_h * wh)

        w_heur_global = 1.0 - wai
        if wai == 0.3:
            w_heur_global = 0.7

        final_scores = (heur_val * w_heur_global) + (raw_ai_scores * wai)

        # 4. BUCKETING VECTORIZADO (Numpy Masks)
        # Creamos indices para rastrear los tickets originales
        indices = np.arange(len(final_scores))

        # Unimos score + ticket en una estructura manejable si necesitamos ordenar
        # Pero es más rápido filtrar indices

        mask_elite = final_scores >= th_elite
        mask_mid = (final_scores >= th_mid) & (final_scores < th_elite)
        # Low es un rango específico, no todo el resto
        mask_low = (final_scores >= (th_mid - 0.10)) & (final_scores < th_mid)

        # Extraer indices y ordenar por score descendente
        def get_sorted_indices(mask):
            idx = indices[mask]
            s = final_scores[idx]
            # argsort es ascendente, invertimos con [::-1]
            sorted_args = np.argsort(s)[::-1]
            return idx[sorted_args]

        idx_elite = get_sorted_indices(mask_elite)
        idx_mid = get_sorted_indices(mask_mid)
        idx_low = get_sorted_indices(mask_low)

        # 5. SELECCIÓN TÁCTICA
        q_elite = overrides.get("quota_elite", 1)
        q_mid = overrides.get("quota_mid", 12)
        q_low = overrides.get("quota_low", 2)

        final_selection = []
        seen_tickets = []  # Lista de conjuntos para chequeo rapido

        # Función helper para selección segura
        def fill_bucket(sorted_indices, quota):
            count = 0
            for idx in sorted_indices:
                if count >= quota:
                    break

                # Recuperamos ticket del numpy array original
                ticket_tup = tuple(candidates_np[idx])
                ticket_set = set(ticket_tup)

                # Chequeo anti-clon (Overlap >= 5)
                # Lamentablemente esto sigue siendo secuencial, difícil de vectorizar
                # sin una matriz de distancia N*N masiva.
                is_clone = False
                for existing in seen_tickets:
                    if len(ticket_set.intersection(existing)) >= 5:
                        is_clone = True
                        break

                if not is_clone:
                    final_selection.append(ticket_tup)
                    seen_tickets.append(ticket_set)
                    count += 1
            return count

        c_e = fill_bucket(idx_elite, q_elite)
        c_m = fill_bucket(idx_mid, q_mid)
        c_l = fill_bucket(idx_low, q_low)

        # 6. RELLENO DE EMERGENCIA (Vectorizado)
        # Si faltan tickets, tomamos del universo general ordenado
        if len(final_selection) < config.num_tickets:
            if verbose:
                console.print("   ⚠ Relleno de Emergencia...")

            # Ordenar todo el universo por score
            all_sorted_args = np.argsort(final_scores)[::-1]

            for idx in all_sorted_args:
                if len(final_selection) >= config.num_tickets:
                    break

                tup = tuple(candidates_np[idx])
                t_set = set(tup)

                is_clone = False
                for existing in seen_tickets:
                    if len(t_set.intersection(existing)) >= 5:
                        is_clone = True
                        break

                if not is_clone:
                    final_selection.append(tup)
                    seen_tickets.append(t_set)

        if verbose:
            console.print(
                f"[bold green]✅ CENTAURO V12 COMPLETADO: {len(final_selection)} tickets.[/]"
            )

        return PredictionResultDTO("Centaur V12", final_selection)

    def audit_winner(
        self,
        history: DrawHistoryDTO,
        config: PredictionConfigDTO,
        winning_ticket: List[int],
    ) -> str:
        """
        Forensic Tool V8 (Compatible con V12 Vectorizado).
        Usa la infraestructura V12 para analizar dónde cayó el ganador.
        """
        target_tuple = tuple(sorted(winning_ticket[:6]))
        overrides = getattr(config, "filter_overrides", {})

        # Cargar configuraciones
        th_elite = overrides.get("threshold_elite", 0.70)
        th_mid = overrides.get("threshold_mid", 0.60)

        # 1. Cargar Universo (Optimizado)
        csv_path = os.path.join("data", "universo_reducido.csv")
        try:
            df = pd.read_csv(csv_path)
            candidates_np = df.iloc[:, :6].values.astype(np.uint8)
            candidates_tuples = [tuple(x) for x in candidates_np]
        except:
            return "[red]Error leyendo universo[/]"

        # 2. Calcular Scores V12 (Vectorizado)
        self._train_model(history, config.total_balls)

        # Score IA
        raw_ai = self.ai_model.score_tickets(candidates_tuples)

        # Score Heurístico (Numba)
        raw_c, raw_h = calc_heuristics_vectorized(
            candidates_np,
            self._matrix_cache["cluster_matrix"],
            self._matrix_cache["hotness_vector"],
            config.total_balls,
        )

        # Replicar fórmula de fusión
        mc = self._matrix_cache["max_cluster"]
        mh = self._matrix_cache["max_hotness"]
        wc = overrides.get("w_cluster", 0.6)
        wh = overrides.get("w_hotness", 0.4)
        wai = overrides.get("w_ai", 0.3)

        norm_c = raw_c / (15 * mc)
        norm_h = raw_h / (6 * mh)
        norm_h = np.where(norm_h > 0.75, norm_h * 0.80, norm_h)

        heur_val = (norm_c * wc) + (norm_h * wh)
        w_heur = 1.0 - wai
        if wai == 0.3:
            w_heur = 0.7

        final_scores = (heur_val * w_heur) + (raw_ai * wai)

        # 3. Buscar al ganador
        try:
            # Pandas/Numpy lookup es rápido
            idx_list = df.index[
                (df["B1"] == target_tuple[0])
                & (df["B2"] == target_tuple[1])
                & (df["B3"] == target_tuple[2])
                & (df["B4"] == target_tuple[3])
                & (df["B5"] == target_tuple[4])
                & (df["B6"] == target_tuple[5])
            ].tolist()

            winner_idx = idx_list[0] if idx_list else -1
        except:
            winner_idx = -1

        status_msg = ""
        subject_score = 0.0

        if winner_idx != -1:
            subject_score = final_scores[winner_idx]
            status_msg = f"[bold green]PRESENTE (Jackpot)[/]"
        else:
            # Buscar el más cercano (Hits) - Esto es lento pero es forense (no critico)
            # Simplificación: Retornamos status Ausente
            status_msg = "[bold red]AUSENTE (Filtrado en Fase 1)[/]"
            subject_score = 0.0  # No podemos saber su score si no pasó el filtro

        # Determinar Bucket Teórico
        bucket = "Low/Trash"
        if subject_score >= th_elite:
            bucket = "Elite"
        elif subject_score >= th_mid:
            bucket = "Mid"
        elif subject_score >= (th_mid - 0.10):
            bucket = "Low"

        # Generar Reporte Visual
        msg = f"\n   🕵️‍♂️ [bold cyan]REPORTE FORENSE V12:[/]\n"
        msg += f"   🎯 [bold]Sujeto:[/bold] {target_tuple}\n"
        if winner_idx != -1:
            msg += f"   📊 [bold]Score V12:[/bold] {subject_score:.5f}\n"
            msg += f"   📦 [bold]Bucket:[/bold] {bucket} (E>={th_elite}, M>={th_mid})\n"
        msg += f"   🌌 [bold]Status Universo:[/bold] {status_msg}\n"

        return msg

    def _analyze_tier_distribution(self, *args, **kwargs):
        # Placeholder para evitar errores si alguien lo llama,
        # pero la lógica principal ya está en audit_winner
        return ""
