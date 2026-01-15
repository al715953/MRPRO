import pandas as pd
import numpy as np
import os
import itertools
from typing import List, Tuple, Dict, Optional, Any
from rich.console import Console

# --- CAPA HPC (Numba JIT) ---
try:
    from numba import jit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO
from src.core.ai_scorer import LotteryAIModel

console = Console()

# --- KERNELS VECTORIZADOS ---
if HAS_NUMBA:

    @jit(nopython=True, fastmath=True, cache=True)
    def calc_heuristics_vectorized(
        candidates, cluster_matrix, hotness_vector, total_balls
    ):
        n_rows, n_cols = candidates.shape
        cluster_scores = np.zeros(n_rows, dtype=np.float32)
        hotness_scores = np.zeros(n_rows, dtype=np.float32)

        for i in range(n_rows):
            c_score = 0
            for j in range(n_cols):
                for k in range(j + 1, n_cols):
                    a, b = candidates[i, j], candidates[i, k]
                    c_score += cluster_matrix[a, b]
            cluster_scores[i] = c_score

            h_score = 0
            for j in range(n_cols):
                val = candidates[i, j]
                if val <= total_balls:
                    h_score += hotness_vector[val]
            hotness_scores[i] = h_score
        return cluster_scores, hotness_scores

else:

    def calc_heuristics_vectorized(
        candidates, cluster_matrix, hotness_vector, total_balls
    ):
        n_rows = len(candidates)
        c_scores, h_scores = np.zeros(n_rows), np.zeros(n_rows)
        for i in range(n_rows):
            row = candidates[i]
            c = sum(cluster_matrix[a, b] for a, b in itertools.combinations(row, 2))
            h = sum(hotness_vector[val] for val in row if val <= total_balls)
            c_scores[i], h_scores[i] = c, h
        return c_scores, h_scores


class GeneticSelectorStrategy(ILotteryStrategy):
    """
    ESTRATEGIA 'SNIPER' V29 (Deep Dive Protocol).
    Arquitectura Híbrida: GPU para Universo -> Numba para Heurística -> Sniper para Selección.
    """

    def __init__(self):
        self.ai_model = LotteryAIModel()
        self._last_trained_date = None
        self._matrix_cache = {
            "cluster_matrix": None,
            "hotness_vector": None,
            "max_cluster": 1.0,
            "max_hotness": 1.0,
        }
        # Coherencia Forense: Snapshot de la última ejecución
        self._forensic_snapshot = {
            "universe": None,
            "ai_scores": None,
            "geo_scores": None,
            "thresholds": {},
        }

    def _train_model(self, history: DrawHistoryDTO, total_balls: int):
        last_date = history.dates[-1] if history.dates else "None"
        if self._last_trained_date != last_date:
            self.ai_model.train(history.winning_numbers, total_balls)
            self._update_heuristic_matrices(history, total_balls)
            self._last_trained_date = last_date

    def _update_heuristic_matrices(self, history: DrawHistoryDTO, total_balls: int):
        matrix = np.zeros((total_balls + 2, total_balls + 2), dtype=np.uint16)
        for draw in history.winning_numbers:
            sorted_draw = sorted(draw[:6])
            for a, b in itertools.combinations(sorted_draw, 2):
                matrix[a, b] += 1
                matrix[b, a] += 1

        flat_matrix = matrix.flatten()
        max_cluster_val = (
            np.percentile(flat_matrix[flat_matrix > 0], 99) if np.sum(matrix) > 0 else 1
        )

        recent_draws = history.winning_numbers[-12:]
        freq_vec = np.zeros(total_balls + 2, dtype=np.uint16)
        for draw in recent_draws:
            for num in draw[:6]:
                freq_vec[num] += 1

        self._matrix_cache.update(
            {
                "cluster_matrix": matrix,
                "hotness_vector": freq_vec,
                "max_cluster": float(max_cluster_val),
                "max_hotness": float(np.max(freq_vec)) if np.max(freq_vec) > 0 else 1,
            }
        )

    def _calculate_v29_scores(self, candidates_np, total_balls):
        raw_c, raw_h = calc_heuristics_vectorized(
            candidates_np,
            self._matrix_cache["cluster_matrix"],
            self._matrix_cache["hotness_vector"],
            total_balls,
        )
        norm_c = np.clip(raw_c / (15 * self._matrix_cache["max_cluster"]), 0, 1.0)
        norm_h = np.clip(raw_h / (6 * self._matrix_cache["max_hotness"]), 0, 1.0)
        return (norm_c * 0.70) + (norm_h * 0.30)

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        verbose = getattr(config, "filter_overrides", {}).get("verbose", True)

        # --- PARÁMETROS V29 DEEP DIVE ---
        AI_THRESHOLD = 0.84
        GEO_P_FLOOR = 35.0

        if verbose:
            console.print(
                f"\n[bold magenta]🎯 SNIPER V29 (Deep Dive Protocol) Activo...[/]"
            )

        # 1. Cargar Universo Reducido
        csv_path = os.path.join("data", "universo_reducido.csv")
        try:
            df = pd.read_csv(csv_path)
            candidates_np = df.iloc[:, :6].values.astype(np.uint8)
        except Exception as e:
            return PredictionResultDTO(f"Error cargando universo: {str(e)}", [])

        # 2. Motor de Scoring Híbrido
        self._train_model(history, config.total_balls)
        tuples_list = [tuple(x) for x in candidates_np]

        raw_ai_scores = np.array(
            self.ai_model.score_tickets(tuples_list), dtype=np.float32
        )
        final_geo_scores = self._calculate_v29_scores(candidates_np, config.total_balls)

        # 3. Definición de Bandas (P35 Floor)
        global_upper = np.percentile(final_geo_scores, 99.0)
        global_lower = np.percentile(final_geo_scores, GEO_P_FLOOR)
        mid_point = (global_lower + global_upper) / 2

        # Registro Snapshot para Auditoría Posterior
        self._forensic_snapshot = {
            "universe": candidates_np,
            "ai_scores": raw_ai_scores,
            "geo_scores": final_geo_scores,
            "thresholds": {
                "ai_limit": AI_THRESHOLD,
                "geo_low": global_lower,
                "geo_mid": mid_point,
                "geo_high": global_upper,
            },
        }

        if verbose:
            console.print(
                f"   📊 Floor P35: {global_lower:.4f} | AI Override: {AI_THRESHOLD}"
            )

        # 4. SELECCIÓN QUIRÚRGICA
        final_selection, seen_tickets = [], []
        indices = np.arange(len(final_geo_scores))

        def add_ticket(idx):
            tup = tuple(candidates_np[idx])
            t_set = set(tup)
            # Filtro de Diversidad (Overlap < 5)
            if any(len(t_set.intersection(s)) >= 5 for s in seen_tickets):
                return False
            final_selection.append(tup)
            seen_tickets.append(t_set)
            return True

        # A. AI OVERRIDE (Super Stars)
        mask_override = raw_ai_scores >= AI_THRESHOLD
        idx_override = indices[mask_override]
        sorted_override = idx_override[np.argsort(raw_ai_scores[mask_override])[::-1]]

        c_stars = 0
        for idx in sorted_override:
            if c_stars >= 4:
                break  # Cupo expandido V29
            if add_ticket(idx):
                c_stars += 1

        # B. ESTRATIFICACIÓN (High / Low Band)
        remaining = config.num_tickets - len(final_selection)
        quota_high = int(remaining * 0.50)

        # High Band
        mask_high = (final_geo_scores >= mid_point) & (final_geo_scores <= global_upper)
        idx_high = indices[mask_high]
        sorted_high = idx_high[np.argsort(raw_ai_scores[mask_high])[::-1]]

        c = 0
        for idx in sorted_high:
            if c >= quota_high:
                break
            if add_ticket(idx):
                c += 1

        # Low Band (Captura de ganadores "feos" como #1595)
        mask_low = (final_geo_scores >= global_lower) & (final_geo_scores < mid_point)
        idx_low = indices[mask_low]
        sorted_low = idx_low[np.argsort(raw_ai_scores[mask_low])[::-1]]

        for idx in sorted_low:
            if len(final_selection) >= config.num_tickets:
                break
            add_ticket(idx)

        # Relleno Final
        if len(final_selection) < config.num_tickets:
            for idx in sorted_high:
                if len(final_selection) >= config.num_tickets:
                    break
                add_ticket(idx)

        return PredictionResultDTO(
            f"Sniper V29 (P35/AI{AI_THRESHOLD})", final_selection
        )

    def audit_winner(self, history, config, winning_ticket):
        snap = self._forensic_snapshot
        if snap["universe"] is None:
            return "[red]No hay snapshot de memoria disponible.[/]"

        target = np.array(sorted(winning_ticket[:6]))
        hits = np.sum(np.isin(snap["universe"], target), axis=1)
        idx_6 = np.where(hits == 6)[0]

        msg = f"\n   🕵️‍♂️ [bold magenta]INFORME FORENSE V29 (Deep Dive):[/]\n"
        if len(idx_6) > 0:
            idx = idx_6[0]
            ai_val = snap["ai_scores"][idx]
            geo_val = snap["geo_scores"][idx]
            th = snap["thresholds"]

            status = "[bold red]ELIMINADO[/]"
            if ai_val >= th["ai_limit"]:
                status = "[bold magenta]🌟 CAPTURADO POR AI OVERRIDE 🌟[/]"
            elif geo_val >= th["geo_low"]:
                status = "[bold green]DENTRO DE BANDAS (CAPTURED)[/]"

            msg += f"   🎯 [bold]Sujeto:[/bold] {tuple(target)}\n"
            msg += f"   📊 [bold]GeoScore:[/bold] {geo_val:.5f} (P35: {th['geo_low']:.5f})\n"
            msg += (
                f"   📈 [bold]AI Score:[/bold] {ai_val:.5f} (Corte: {th['ai_limit']})\n"
            )
            msg += f"   📦 [bold]Status Final:[/bold] {status}\n"
        else:
            msg += f"   ❌ [red]Fallo Crítico: El ticket no sobrevivió a la Fase 1 (GPU Filters).[/]\n"

        return msg
