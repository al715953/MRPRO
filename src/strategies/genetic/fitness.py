# src/strategies/genetic/fitness.py
from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None


# ============================================================
# Configs
# ============================================================


@dataclass(frozen=True)
class StrataConfig:
    """
    Estratos de rank (1 = mejor).
    Ej: edges=(10,30,60,100,150,200,500) genera buckets:
      [1..10], [11..30], [31..60], [61..100], [101..150], [151..200], [201..500], [501..inf]
    """

    rank_edges: Tuple[int, ...] = (10, 30, 60, 100, 150, 200, 500)
    # Pesos por bucket (mismo número de buckets = len(rank_edges)+1)
    # Debe ser decreciente pero NO demasiado (queremos cubrir 39,47,150 sin morir en top10)
    bucket_weights: Tuple[float, ...] = (1.00, 0.90, 0.80, 0.70, 0.62, 0.55, 0.35, 0.18)

    # Binning estructural (sum/std) para cobertura por diversidad
    n_sum_bins: int = 8
    n_std_bins: int = 8
    binning_mode: str = "quantile"  # "quantile" o "linspace"


@dataclass(frozen=True)
class FitnessConfig:
    """
    Fitness “portfolio-style”:
      - Exploit: utilidad por rank/score
      - Coverage: cubrir buckets de rank con rendimientos decrecientes
      - Structure coverage: cubrir bins (sum/std) y perfiles de décadas
      - Similarity penalty: castigar tickets demasiado parecidos (overlap 4/5/6)
    """

    # Pool objetivo principal
    focus_max_rank: int = 200  # énfasis total en top200
    candidate_max_rank: int = 500  # permitimos un pequeño hedge fuera del top200
    bucket_plan: Tuple[Tuple[int, int, int], ...] = (
        (21, 40, 4),
        (41, 60, 3),
        (61, 90, 2),
        (91, 120, 3),
        (121, 160, 3),
        (161, 200, 3),
        (201, 500, 1),
    )

    # Utilidad por rank (power law)
    rank_alpha: float = 0.70  # más alto => más presión al top
    w_rank: float = 0.75
    w_score: float = 0.25  # mezcla del score continuo (AI+Geo)

    # Peso de componentes (tuning principal)
    w_exploit: float = 1.00
    w_bucket_cov: float = 1.10
    w_struct_cov: float = 0.55
    w_decade_cov: float = 0.30
    w_similarity: float = 0.90

    # Penalizaciones por overlap de números (matches entre tickets)
    pen_match_3: float = 0.12
    pen_match_4: float = 0.60
    pen_match_5: float = 3.50
    pen_match_6: float = 12.0

    # Estabilidad numérica
    eps: float = 1e-9


@dataclass(frozen=True)
class DeepDispersionConfig:
    """Precommitted hedge that distributes tickets throughout the score depth."""

    core_tickets: int = 20
    deep_tickets: int = 10
    min_deep_rank: int = 501
    max_overlap_preferred: int = 3
    w_pair_novelty: float = 0.40
    w_number_rarity: float = 0.25
    w_dissimilarity: float = 0.20
    w_local_quality: float = 0.15


@dataclass(frozen=True)
class EliteCoverageDeepConfig:
    """Three-zone portfolio: exact elites, combinatorial cover and depth hedge."""

    elite_tickets: int = 10
    coverage_tickets: int = 10
    deep_tickets: int = 10
    coverage_max_rank: int = 500
    min_deep_rank: int = 501
    max_overlap_preferred: int = 3
    w_pair_novelty: float = 0.15
    w_triple_novelty: float = 0.30
    w_quad_novelty: float = 0.30
    w_number_rarity: float = 0.05
    w_dissimilarity: float = 0.05
    w_local_quality: float = 0.15


# ============================================================
# Backend helpers
# ============================================================


def _get_xp(arr, xp=None):
    if xp is not None:
        return xp
    if cp is None:
        return np
    try:
        return cp.get_array_module(arr)
    except Exception:
        # fallback
        return np


def compute_ranks_desc(scores, xp=None):
    """
    ranks[i] = 1 si scores[i] es el máximo, 2 si es el segundo, etc.
    """
    xp = _get_xp(scores, xp)
    order = xp.argsort(-scores)
    ranks = xp.empty_like(order, dtype=xp.int32)
    ranks[order] = xp.arange(1, scores.shape[0] + 1, dtype=xp.int32)
    return ranks


def _minmax01(x, xp, eps=1e-9):
    x_min = xp.min(x)
    x_max = xp.max(x)
    return (x - x_min) / (x_max - x_min + eps)


# ============================================================
# Feature building (GPU-friendly)
# ============================================================


def _safe_edges(values, xp, n_bins: int, mode: str, eps: float):
    """
    Construye edges para binning (n_bins bins => n_bins+1 edges).
    Preferimos quantiles por robustez temporal, con fallback a linspace.
    """
    if n_bins <= 1:
        vmin = xp.min(values)
        vmax = xp.max(values)
        return xp.asarray([vmin, vmax + eps], dtype=xp.float32)

    if mode == "quantile":
        qs = xp.linspace(0.0, 100.0, n_bins + 1, dtype=xp.float32)
        edges = xp.percentile(values, qs)
        # Si se colapsan edges (muchos repetidos), fallback
        try:
            uniq = xp.unique(edges)
            if int(uniq.size) < (n_bins + 1):
                raise ValueError("collapsed quantile edges")
        except Exception:
            vmin = xp.min(values)
            vmax = xp.max(values)
            edges = xp.linspace(vmin, vmax + eps, n_bins + 1, dtype=xp.float32)
        return edges.astype(xp.float32)

    # linspace
    vmin = xp.min(values)
    vmax = xp.max(values)
    return xp.linspace(vmin, vmax + eps, n_bins + 1, dtype=xp.float32)


def build_ticket_features(
    tickets_6, xp=None, strata: Optional[StrataConfig] = None, eps: float = 1e-9
):
    """
    tickets_6: (N,6) uint8 sorted ascending
    Return:
      sums, stds, struct_bin, decade_hash, one_hot
    """
    xp = _get_xp(tickets_6, xp)
    strata = strata or StrataConfig()

    t = tickets_6.astype(xp.float32)
    sums = xp.sum(t, axis=1)
    stds = xp.std(t, axis=1)

    # decade profile hash
    # decades: 0=1..10, 1=11..20, 2=21..30, 3=31..39
    dec = (tickets_6.astype(xp.int32) - 1) // 10
    counts = xp.zeros((tickets_6.shape[0], 4), dtype=xp.int32)
    for d in range(4):
        counts[:, d] = xp.sum(dec == d, axis=1)
    decade_hash = (
        counts[:, 0] * 1000 + counts[:, 1] * 100 + counts[:, 2] * 10 + counts[:, 3]
    )

    # sum/std binning
    sum_edges = _safe_edges(sums, xp, strata.n_sum_bins, strata.binning_mode, eps)
    std_edges = _safe_edges(stds, xp, strata.n_std_bins, strata.binning_mode, eps)

    sum_bin = xp.searchsorted(sum_edges, sums, side="right") - 1
    std_bin = xp.searchsorted(std_edges, stds, side="right") - 1

    sum_bin = xp.clip(sum_bin, 0, strata.n_sum_bins - 1).astype(xp.int32)
    std_bin = xp.clip(std_bin, 0, strata.n_std_bins - 1).astype(xp.int32)

    struct_bin = (sum_bin * strata.n_std_bins + std_bin).astype(xp.int32)
    n_struct = int(strata.n_sum_bins * strata.n_std_bins)

    # one_hot para overlaps (N, 41) usando índices 1..39
    one_hot = xp.zeros((tickets_6.shape[0], 41), dtype=xp.float32)
    rows = xp.arange(tickets_6.shape[0], dtype=xp.int32)[:, None]
    one_hot[rows, tickets_6.astype(xp.int32)] = 1.0

    return {
        "sums": sums.astype(xp.float32),
        "stds": stds.astype(xp.float32),
        "struct_bin": struct_bin,
        "n_struct": n_struct,
        "decade_hash": decade_hash.astype(xp.int32),
        "one_hot": one_hot,
        "sum_edges": sum_edges,
        "std_edges": std_edges,
    }


def build_draw_context(
    tickets_6,
    scores,
    xp=None,
    cfg: Optional[FitnessConfig] = None,
    strata: Optional[StrataConfig] = None,
):
    """
    Construye todo lo necesario para evaluar fitness/selección.
    """
    cfg = cfg or FitnessConfig()
    xp = _get_xp(scores, xp)
    strata = strata or StrataConfig()

    ranks = compute_ranks_desc(scores, xp=xp)

    feats = build_ticket_features(tickets_6, xp=xp, strata=strata, eps=cfg.eps)

    # utility: mezcla rank-based + score-based
    s01 = _minmax01(scores.astype(xp.float32), xp, cfg.eps)
    r = ranks.astype(xp.float32)
    rank_u = 1.0 / (xp.power(r, cfg.rank_alpha) + cfg.eps)
    rank_u = rank_u / (xp.max(rank_u) + cfg.eps)

    utility = cfg.w_rank * rank_u + cfg.w_score * s01
    utility = utility.astype(xp.float32)

    # buckets por rank
    edges = xp.asarray(strata.rank_edges, dtype=xp.int32)
    bucket_id = xp.searchsorted(edges, ranks, side="right").astype(xp.int32)
    n_buckets = int(len(strata.rank_edges) + 1)

    bucket_w = np.array(strata.bucket_weights, dtype=np.float32)
    if bucket_w.shape[0] != n_buckets:
        # fallback seguro
        bucket_w = np.linspace(1.0, 0.2, n_buckets, dtype=np.float32)
    bucket_w_xp = xp.asarray(bucket_w, dtype=xp.float32)

    # Candidato permitido (para no “irse” del objetivo top200)
    cand_ok = ranks <= int(cfg.candidate_max_rank)

    return {
        "xp": xp,
        "cfg": cfg,
        "strata": strata,
        "tickets": tickets_6,
        "scores": scores.astype(xp.float32),
        "ranks": ranks,
        "utility": utility,
        "bucket_id": bucket_id,
        "bucket_w": bucket_w_xp,
        "n_buckets": n_buckets,
        "cand_ok": cand_ok,
        **feats,
    }


# ============================================================
# Fitness (set function) + Greedy portfolio optimizer
# ============================================================


def _pair_penalty(matches, xp, cfg: FitnessConfig):
    """
    matches: vector con valores 0..6 (overlap de números)
    """
    m = matches.astype(xp.int32)
    pen = xp.zeros_like(m, dtype=xp.float32)
    pen = xp.where(m == 3, cfg.pen_match_3, pen)
    pen = xp.where(m == 4, cfg.pen_match_4, pen)
    pen = xp.where(m == 5, cfg.pen_match_5, pen)
    pen = xp.where(m >= 6, cfg.pen_match_6, pen)
    return pen.astype(xp.float32)


def portfolio_fitness(ctx: Dict[str, Any], selected_idx):
    """
    Evalúa fitness de un set seleccionado (para GA o auditoría).
    selected_idx: iterable de índices (en el pool ctx["tickets"])
    """
    xp = ctx["xp"]
    cfg: FitnessConfig = ctx["cfg"]

    sel = xp.asarray(list(selected_idx), dtype=xp.int32)
    if sel.size == 0:
        return xp.float32(0.0)

    util = ctx["utility"][sel]
    ranks = ctx["ranks"][sel]
    bucket_id = ctx["bucket_id"][sel]
    struct_bin = ctx["struct_bin"][sel]
    decade_hash = ctx["decade_hash"][sel]

    # Exploit puro
    exploit = xp.sum(util)

    # Cobertura por buckets (rendimientos decrecientes)
    bucket_mass = xp.zeros(ctx["n_buckets"], dtype=xp.float32)
    xp.add.at(bucket_mass, bucket_id, util)
    bucket_cov = xp.sum(ctx["bucket_w"] * xp.log1p(bucket_mass))

    # Cobertura estructural (sum/std bins)
    n_struct = int(ctx["n_struct"])
    struct_counts = xp.zeros(n_struct, dtype=xp.float32)
    xp.add.at(struct_counts, struct_bin, 1.0)
    struct_cov = xp.sum(xp.log1p(struct_counts))

    # Cobertura por perfiles de décadas
    # (se premia variedad; equivalente a log1p(count) por hash)
    dec_u, dec_counts = xp.unique(decade_hash, return_counts=True)
    decade_cov = xp.sum(xp.log1p(dec_counts.astype(xp.float32)))

    # Penalidad por similitud interna
    # Nota: esto es O(k^2). k=20 => ok para auditoría.
    one_hot = ctx["one_hot"][sel]  # (k,41)
    sim = xp.float32(0.0)
    k = int(one_hot.shape[0])
    for i in range(k):
        matches = xp.dot(one_hot, one_hot[i])  # (k,)
        # excluye self
        matches = matches - (xp.arange(k, dtype=xp.int32) == i).astype(xp.int32) * 6
        sim += xp.sum(_pair_penalty(matches, xp, cfg))

    return (
        cfg.w_exploit * exploit
        + cfg.w_bucket_cov * bucket_cov
        + cfg.w_struct_cov * struct_cov
        + cfg.w_decade_cov * decade_cov
        - cfg.w_similarity * sim
    ).astype(xp.float32)


def select_tickets_v16(
    tickets_6,
    scores,
    n_tickets: int = 20,
    xp=None,
    cfg: Optional[FitnessConfig] = None,
    strata: Optional[StrataConfig] = None,
):
    """
    Selector determinista tipo "portfolio optimizer" (greedy):
      - maximiza exploit + cobertura (rank buckets + estructura)
      - penaliza similitud incrementalmente
    Retorna: (tickets_list, debug_dict)
    """
    cfg = cfg or FitnessConfig()
    ctx = build_draw_context(tickets_6, scores, xp=xp, cfg=cfg, strata=strata)
    xp = ctx["xp"]

    N = int(ctx["tickets"].shape[0])
    if N == 0:
        return [], {"selected_idx": [], "selected_ranks": []}
    if N <= n_tickets:
        t_cpu = (
            ctx["tickets"].get().tolist()
            if hasattr(ctx["tickets"], "get")
            else ctx["tickets"].tolist()
        )
        ranks_cpu = (
            ctx["ranks"].get().tolist()
            if hasattr(ctx["ranks"], "get")
            else ctx["ranks"].tolist()
        )
        return t_cpu, {"selected_idx": list(range(N)), "selected_ranks": ranks_cpu}

    # Máscaras
    available = ctx["cand_ok"].copy()
    # Siempre permitir top focus (<=focus_max_rank) aunque cand_ok falle por config
    available = available | (ctx["ranks"] <= int(cfg.focus_max_rank))

    # Estado incremental
    bucket_mass = xp.zeros(ctx["n_buckets"], dtype=xp.float32)
    struct_counts = xp.zeros(int(ctx["n_struct"]), dtype=xp.float32)

    # Penalidad por similitud acumulada contra el set seleccionado
    sim_accum = xp.zeros(N, dtype=xp.float32)

    # Para acelerar la penalidad, necesitamos one_hot completo (N,41)
    one_hot = ctx["one_hot"]

    selected = []
    selected_ranks = []

    # Precomputo: término de exploit base por candidato
    base_exploit = cfg.w_exploit * ctx["utility"]

    # Penaliza candidatos no disponibles
    neg_inf = xp.float32(-1e9)

    # =========================
    # V16 Patch: 24 tickets + anchors Top-5 + bucket quotas (Opción 2)
    # =========================

    reserve_top_m = 5

    # cuotas: (lo, hi, count)
    bucket_plan = cfg.bucket_plan

    def _pick_one(mask):
        # Score base
        score = base_exploit.copy()

        # Cobertura de buckets (marginal): w(bucket) * (log1p(m+u)-log1p(m))
        b = ctx["bucket_id"]  # (N,)
        bw = ctx["bucket_w"]  # (n_buckets,)
        prev_mass = bucket_mass[b]  # (N,)
        w_cand = bw[b]  # (N,)
        score += (
            cfg.w_bucket_cov
            * w_cand
            * (xp.log1p(prev_mass + ctx["utility"]) - xp.log1p(prev_mass))
        )

        # Cobertura estructural (marginal): log1p(c+1)-log1p(c)
        sb = ctx["struct_bin"]
        prev_c = struct_counts[sb]
        score += cfg.w_struct_cov * (xp.log1p(prev_c + 1.0) - xp.log1p(prev_c))

        # Penalidad de similitud incremental
        score -= cfg.w_similarity * sim_accum

        # Máscara de disponibilidad + bucket
        score = xp.where(mask, score, neg_inf)

        bi = int(score.argmax())
        if float(score[bi]) <= float(neg_inf) / 2:
            return None
        return bi

    def _accept(idx_i: int):
        selected.append(idx_i)
        selected_ranks.append(int(ctx["ranks"][idx_i]))

        available[idx_i] = False

        # update bucket mass
        bi = int(ctx["bucket_id"][idx_i])
        bucket_mass[bi] += ctx["utility"][idx_i]

        # update struct
        sbi = int(ctx["struct_bin"][idx_i])
        struct_counts[sbi] += 1.0

        # update similarity penalties (1 dot product)
        matches = xp.dot(one_hot, one_hot[idx_i])  # (N,)
        sim_accum[:] = sim_accum + _pair_penalty(matches, xp, cfg)

    # ---- 1) Anchors: fuerza Top-5 por rank (exploit duro) ----
    if int(n_tickets) >= reserve_top_m:
        order = xp.argsort(ctx["ranks"])  # rank ascendente => mejores primero
        anchors = order[:reserve_top_m]

        for a in anchors.tolist() if hasattr(anchors, "tolist") else list(anchors):
            a = int(a)
            if not bool(available[a]):
                continue
            _accept(a)

    # ---- 2) Buckets con cuotas fijas ----
    for lo, hi, cnt in bucket_plan:
        for _ in range(int(cnt)):
            if len(selected) >= int(n_tickets):
                break

            mask = available & (ctx["ranks"] >= lo) & (ctx["ranks"] <= hi)
            idx_i = _pick_one(mask)
            if idx_i is None:
                break
            _accept(int(idx_i))

    # ---- 3) Fill: si faltan picks, rellena con lo mejor disponible <= candidate_max_rank ----
    while len(selected) < int(n_tickets):
        mask = available  # ya incluye cand_ok OR focus<=200 (por tu lógica previa)
        idx_i = _pick_one(mask)
        if idx_i is None:
            break
        _accept(int(idx_i))

    # Export CPU
    t_sel = ctx["tickets"][xp.asarray(selected, dtype=xp.int32)]
    t_cpu = t_sel.get().tolist() if hasattr(t_sel, "get") else t_sel.tolist()

    return t_cpu, {"selected_idx": selected, "selected_ranks": selected_ranks}


def select_core_plus_deep_tickets(
    tickets_6,
    scores,
    n_tickets: int = 30,
    xp=None,
    cfg: Optional[FitnessConfig] = None,
    strata: Optional[StrataConfig] = None,
    deep_cfg: Optional[DeepDispersionConfig] = None,
):
    """Keep the native core and add a deterministic depth-diversity hedge.

    The eligible depth (``min_deep_rank..N``) is divided into equal-population
    strata. One ticket is selected from each stratum, so the bands adapt to the
    candidate-pool size without being fitted to historical winning ranks.
    """
    cfg = cfg or FitnessConfig()
    deep_cfg = deep_cfg or DeepDispersionConfig()
    xp = _get_xp(scores, xp)

    tickets_cpu = (
        tickets_6.get() if hasattr(tickets_6, "get") else np.asarray(tickets_6)
    )
    scores_cpu = scores.get() if hasattr(scores, "get") else np.asarray(scores)
    tickets_cpu = np.asarray(tickets_cpu, dtype=np.uint8)
    scores_cpu = np.asarray(scores_cpu, dtype=np.float64)
    candidate_count = int(tickets_cpu.shape[0])
    target = min(max(0, int(n_tickets)), candidate_count)
    if target <= 0:
        return [], {
            "selected_idx": [],
            "selected_ranks": [],
            "core_selected_ranks": [],
            "deep_selected_ranks": [],
            "deep_rank_bands": [],
        }

    requested_core = min(max(0, int(deep_cfg.core_tickets)), target)
    requested_deep = min(
        max(0, int(deep_cfg.deep_tickets)), target - requested_core
    )
    core_tickets, core_debug = select_tickets_v16(
        tickets_6,
        scores,
        n_tickets=requested_core,
        xp=xp,
        cfg=cfg,
        strata=strata,
    )

    order = np.argsort(-scores_cpu, kind="stable")
    ranks = np.empty(candidate_count, dtype=np.int32)
    ranks[order] = np.arange(1, candidate_count + 1, dtype=np.int32)
    key_to_idx = {tuple(int(v) for v in row): idx for idx, row in enumerate(tickets_cpu)}
    selected = [key_to_idx[tuple(int(v) for v in row)] for row in core_tickets]
    selected_set = set(selected)
    core_selected_ranks = [int(ranks[idx]) for idx in selected]

    max_number = max(1, int(tickets_cpu.max()) if tickets_cpu.size else 1)
    one_hot = np.zeros((candidate_count, max_number + 1), dtype=np.uint8)
    row_idx = np.arange(candidate_count)[:, None]
    one_hot[row_idx, tickets_cpu.astype(np.int32)] = 1
    pair_positions = tuple(itertools.combinations(range(6), 2))
    pair_base = max_number + 1
    pair_codes = np.stack(
        [
            tickets_cpu[:, left].astype(np.int32) * pair_base
            + tickets_cpu[:, right].astype(np.int32)
            for left, right in pair_positions
        ],
        axis=1,
    )
    covered_pairs = np.zeros(pair_base * pair_base, dtype=bool)
    number_counts = np.zeros(max_number + 1, dtype=np.int32)

    def _register(idx: int) -> None:
        selected.append(int(idx))
        selected_set.add(int(idx))
        covered_pairs[pair_codes[int(idx)]] = True
        number_counts[tickets_cpu[int(idx)]] += 1

    # Register native core coverage before evaluating the deep reserve.
    native_core = list(selected)
    selected = []
    selected_set = set()
    for idx in native_core:
        _register(idx)

    def _minmax_cpu(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        low = float(values.min())
        high = float(values.max())
        if high <= low:
            return np.ones(values.shape, dtype=np.float64)
        return (values - low) / (high - low)

    def _pick(candidates: np.ndarray) -> int | None:
        candidates = np.asarray(
            [int(idx) for idx in candidates if int(idx) not in selected_set],
            dtype=np.int32,
        )
        if candidates.size == 0:
            return None
        selected_matrix = one_hot[np.asarray(selected, dtype=np.int32)]
        overlap = one_hot[candidates] @ selected_matrix.T
        max_overlap = overlap.max(axis=1) if overlap.size else np.zeros(candidates.size)
        preferred = max_overlap <= int(deep_cfg.max_overlap_preferred)
        if not np.any(preferred):
            for allowed in range(int(deep_cfg.max_overlap_preferred) + 1, 7):
                preferred = max_overlap <= allowed
                if np.any(preferred):
                    break

        novelty = np.mean(~covered_pairs[pair_codes[candidates]], axis=1)
        rarity = np.mean(
            1.0 / (1.0 + number_counts[tickets_cpu[candidates]]), axis=1
        )
        dissimilarity = 1.0 - (max_overlap.astype(np.float64) / 6.0)
        local_quality = 1.0 - _minmax_cpu(ranks[candidates].astype(np.float64))
        objective = (
            float(deep_cfg.w_pair_novelty) * _minmax_cpu(novelty)
            + float(deep_cfg.w_number_rarity) * _minmax_cpu(rarity)
            + float(deep_cfg.w_dissimilarity) * dissimilarity
            + float(deep_cfg.w_local_quality) * local_quality
        )
        objective = np.where(preferred, objective, -np.inf)
        return int(candidates[int(np.argmax(objective))])

    deep_eligible = order[ranks[order] >= max(1, int(deep_cfg.min_deep_rank))]
    band_count = min(requested_deep, int(deep_eligible.size))
    deep_rank_bands = []
    deep_selected_ranks = []
    if band_count > 0:
        for band in np.array_split(deep_eligible, band_count):
            if band.size == 0:
                continue
            chosen = _pick(band)
            if chosen is None:
                continue
            _register(chosen)
            deep_selected_ranks.append(int(ranks[chosen]))
            deep_rank_bands.append(
                {
                    "rank_min": int(ranks[band[0]]),
                    "rank_max": int(ranks[band[-1]]),
                    "chosen_rank": int(ranks[chosen]),
                    "candidates": int(band.size),
                }
            )

    # Defensive fill for undersized pools; it does not normally execute for Melate.
    if len(selected) < target:
        for idx in order:
            idx = int(idx)
            if idx in selected_set:
                continue
            _register(idx)
            if len(selected) >= target:
                break

    chosen = selected[:target]
    tickets = tickets_cpu[np.asarray(chosen, dtype=np.int32)].tolist()
    selected_ranks = [int(ranks[idx]) for idx in chosen]
    return tickets, {
        "selected_idx": chosen,
        "selected_ranks": selected_ranks,
        "core_selected_ranks": core_selected_ranks,
        "deep_selected_ranks": deep_selected_ranks,
        "deep_rank_bands": deep_rank_bands,
        "deep_dispersion_weights": {
            "pair_novelty": float(deep_cfg.w_pair_novelty),
            "number_rarity": float(deep_cfg.w_number_rarity),
            "dissimilarity": float(deep_cfg.w_dissimilarity),
            "local_quality": float(deep_cfg.w_local_quality),
        },
    }


def select_elite_coverage_deep_tickets(
    tickets_6,
    scores,
    n_tickets: int = 30,
    xp=None,
    cfg: Optional[FitnessConfig] = None,
    strata: Optional[StrataConfig] = None,
    portfolio_cfg: Optional[EliteCoverageDeepConfig] = None,
):
    """Build a deterministic elite + coverage + depth portfolio.

    The elite tranche contains the exact best-ranked tickets and cannot be
    displaced by diversity penalties. The middle tranche greedily maximizes
    new pairs, triples and quadruples among the leading ranks. The final tranche
    applies the same objective inside equal-population depth bands.

    This is a portfolio construction rule, not a claim that any number is more
    likely to be drawn. Its purpose is to avoid losing strong ranked candidates
    while reducing combinatorial redundancy among the remaining tickets.
    """
    del cfg, strata  # Reserved for a compatible selector interface.
    portfolio_cfg = portfolio_cfg or EliteCoverageDeepConfig()
    xp = _get_xp(scores, xp)
    tickets_cpu = (
        tickets_6.get() if hasattr(tickets_6, "get") else np.asarray(tickets_6)
    )
    scores_cpu = scores.get() if hasattr(scores, "get") else np.asarray(scores)
    tickets_cpu = np.asarray(tickets_cpu, dtype=np.uint8)
    scores_cpu = np.asarray(scores_cpu, dtype=np.float64)
    candidate_count = int(tickets_cpu.shape[0])
    target = min(max(0, int(n_tickets)), candidate_count)
    if target <= 0:
        return [], {
            "selected_idx": [],
            "selected_ranks": [],
            "elite_selected_ranks": [],
            "coverage_selected_ranks": [],
            "deep_selected_ranks": [],
            "deep_rank_bands": [],
            "phase_by_ticket": [],
        }

    order = np.argsort(-scores_cpu, kind="stable")
    ranks = np.empty(candidate_count, dtype=np.int32)
    ranks[order] = np.arange(1, candidate_count + 1, dtype=np.int32)
    requested_elite = min(max(0, int(portfolio_cfg.elite_tickets)), target)
    requested_coverage = min(
        max(0, int(portfolio_cfg.coverage_tickets)), target - requested_elite
    )
    requested_deep = min(
        max(0, int(portfolio_cfg.deep_tickets)),
        target - requested_elite - requested_coverage,
    )

    max_number = max(1, int(tickets_cpu.max()) if tickets_cpu.size else 1)
    code_base = max_number + 1
    one_hot = np.zeros((candidate_count, code_base), dtype=np.uint8)
    rows = np.arange(candidate_count)[:, None]
    one_hot[rows, tickets_cpu.astype(np.int32)] = 1

    def _subset_codes(subset_size: int) -> np.ndarray:
        position_sets = tuple(itertools.combinations(range(6), subset_size))
        encoded = []
        for positions in position_sets:
            code = np.zeros(candidate_count, dtype=np.int32)
            for position in positions:
                code = code * code_base + tickets_cpu[:, position].astype(np.int32)
            encoded.append(code)
        return np.stack(encoded, axis=1)

    pair_codes = _subset_codes(2)
    triple_codes = _subset_codes(3)
    quad_codes = _subset_codes(4)
    covered_pairs = np.zeros(code_base**2, dtype=bool)
    covered_triples = np.zeros(code_base**3, dtype=bool)
    covered_quads = np.zeros(code_base**4, dtype=bool)
    number_counts = np.zeros(code_base, dtype=np.int32)
    selected: list[int] = []
    selected_set: set[int] = set()
    phase_by_ticket: list[str] = []

    def _register(idx: int, phase: str) -> None:
        idx = int(idx)
        selected.append(idx)
        selected_set.add(idx)
        phase_by_ticket.append(str(phase))
        covered_pairs[pair_codes[idx]] = True
        covered_triples[triple_codes[idx]] = True
        covered_quads[quad_codes[idx]] = True
        number_counts[tickets_cpu[idx]] += 1

    for idx in order[:requested_elite]:
        _register(int(idx), "elite")
    elite_selected_ranks = [int(ranks[idx]) for idx in selected]

    def _minmax(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return values
        low, high = float(values.min()), float(values.max())
        if high <= low:
            return np.ones(values.shape, dtype=np.float64)
        return (values - low) / (high - low)

    def _pick(candidates: np.ndarray) -> int | None:
        candidates = np.asarray(
            [int(idx) for idx in candidates if int(idx) not in selected_set],
            dtype=np.int32,
        )
        if candidates.size == 0:
            return None
        selected_matrix = one_hot[np.asarray(selected, dtype=np.int32)]
        overlap = one_hot[candidates] @ selected_matrix.T
        max_overlap = (
            overlap.max(axis=1) if overlap.size else np.zeros(candidates.size)
        )
        preferred = max_overlap <= int(portfolio_cfg.max_overlap_preferred)
        if not np.any(preferred):
            for allowed in range(int(portfolio_cfg.max_overlap_preferred) + 1, 7):
                preferred = max_overlap <= allowed
                if np.any(preferred):
                    break

        pair_novelty = np.mean(~covered_pairs[pair_codes[candidates]], axis=1)
        triple_novelty = np.mean(
            ~covered_triples[triple_codes[candidates]], axis=1
        )
        quad_novelty = np.mean(~covered_quads[quad_codes[candidates]], axis=1)
        rarity = np.mean(
            1.0 / (1.0 + number_counts[tickets_cpu[candidates]]), axis=1
        )
        dissimilarity = 1.0 - max_overlap.astype(np.float64) / 6.0
        local_quality = 1.0 - _minmax(ranks[candidates].astype(np.float64))
        objective = (
            float(portfolio_cfg.w_pair_novelty) * _minmax(pair_novelty)
            + float(portfolio_cfg.w_triple_novelty) * _minmax(triple_novelty)
            + float(portfolio_cfg.w_quad_novelty) * _minmax(quad_novelty)
            + float(portfolio_cfg.w_number_rarity) * _minmax(rarity)
            + float(portfolio_cfg.w_dissimilarity) * dissimilarity
            + float(portfolio_cfg.w_local_quality) * local_quality
        )
        objective = np.where(preferred, objective, -np.inf)
        return int(candidates[int(np.argmax(objective))])

    coverage_min_rank = requested_elite + 1
    coverage_pool = order[
        (ranks[order] >= coverage_min_rank)
        & (ranks[order] <= max(coverage_min_rank, int(portfolio_cfg.coverage_max_rank)))
    ]
    coverage_selected_ranks = []
    for _ in range(requested_coverage):
        chosen = _pick(coverage_pool)
        if chosen is None:
            break
        _register(chosen, "coverage")
        coverage_selected_ranks.append(int(ranks[chosen]))

    deep_eligible = order[
        ranks[order] >= max(1, int(portfolio_cfg.min_deep_rank))
    ]
    deep_rank_bands = []
    deep_selected_ranks = []
    band_count = min(requested_deep, int(deep_eligible.size))
    if band_count > 0:
        for band in np.array_split(deep_eligible, band_count):
            if band.size == 0:
                continue
            chosen = _pick(band)
            if chosen is None:
                continue
            _register(chosen, "deep")
            deep_selected_ranks.append(int(ranks[chosen]))
            deep_rank_bands.append(
                {
                    "rank_min": int(ranks[band[0]]),
                    "rank_max": int(ranks[band[-1]]),
                    "chosen_rank": int(ranks[chosen]),
                    "candidates": int(band.size),
                }
            )

    if len(selected) < target:
        for idx in order:
            if int(idx) in selected_set:
                continue
            _register(int(idx), "fill")
            if len(selected) >= target:
                break

    chosen = selected[:target]
    return tickets_cpu[np.asarray(chosen, dtype=np.int32)].tolist(), {
        "selected_idx": chosen,
        "selected_ranks": [int(ranks[idx]) for idx in chosen],
        "elite_selected_ranks": elite_selected_ranks,
        "coverage_selected_ranks": coverage_selected_ranks,
        "deep_selected_ranks": deep_selected_ranks,
        "deep_rank_bands": deep_rank_bands,
        "phase_by_ticket": phase_by_ticket[:target],
        "coverage_unique_pairs": int(covered_pairs.sum()),
        "coverage_unique_triples": int(covered_triples.sum()),
        "coverage_unique_quads": int(covered_quads.sum()),
        "coverage_weights": {
            "pair_novelty": float(portfolio_cfg.w_pair_novelty),
            "triple_novelty": float(portfolio_cfg.w_triple_novelty),
            "quad_novelty": float(portfolio_cfg.w_quad_novelty),
            "number_rarity": float(portfolio_cfg.w_number_rarity),
            "dissimilarity": float(portfolio_cfg.w_dissimilarity),
            "local_quality": float(portfolio_cfg.w_local_quality),
        },
    }
