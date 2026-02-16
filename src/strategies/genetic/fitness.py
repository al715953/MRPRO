# src/strategies/genetic/fitness.py
from __future__ import annotations

from dataclasses import dataclass
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
    bucket_plan = [
        (21, 40, 4),
        (41, 60, 3),
        (61, 90, 2),
        (91, 120, 3),
        (121, 160, 3),
        (161, 200, 3),
        (201, 500, 1),
    ]

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
