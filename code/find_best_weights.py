# find_best_weights.py
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import metrics as mc
import numpy as np
import pandas as pd
from meta_classes import ProductsMeta, QueryStore, get_base_dirs
from method_comparison import (
    Evaluation,
    VectorSearchMethodStore,
    VectorSearchMethodStoreStatic,
)
from tqdm import tqdm

# ------------------------------------------------------------
# Logging / Seeds
# ------------------------------------------------------------
logger = logging.getLogger("find_best_weights")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def set_all_seeds(seed: int) -> None:
    import os
    import random

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        "Weight search (Methods 2 & 3) with coarse→fine; candidates fast-path + catalog, multi-metric tie-break"
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="all",
        help="Model name(s) or 'all' to run on all embedding dirs in data/embeddings/.",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["candidates", "catalog", "both"],
        default=["both"],
        help="Which modes to evaluate this run.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["2", "3"],
        default=["2", "3"],
        help="Which methods to optimize (2=combine+renorm, 3=score-mix).",
    )
    p.add_argument(
        "--frac_queries",
        type=float,
        default=0.05,
        help="Fraction of queries to sample (0<frac<=1).",
    )
    p.add_argument(
        "--ndcg_k", type=int, default=10, help="Primary k for NDCG@k optimization."
    )
    p.add_argument(
        "--topk_candidates",
        type=int,
        default=40,
        help="Top-k used for metrics extraction.",
    )
    p.add_argument(
        "--grid_step", type=float, default=0.05, help="Coarse simplex grid step."
    )
    p.add_argument(
        "--optimize_for_stat",
        choices=["mean", "median"],
        default="mean",
        help="Aggregate statistic.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    # Fine grid params
    p.add_argument(
        "--max_stages", type=int, default=4, help="Number of fine stages after coarse."
    )
    p.add_argument(
        "--min_step", type=float, default=0.0125, help="Stop when step < min_step."
    )
    p.add_argument(
        "--local_radius_steps",
        type=int,
        default=2,
        help="Radius (in step units) around best for fine grid.",
    )
    p.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Minimal per-field weight (enforce w_f>=epsilon). Default: epsilon=grid_step.",
    )
    return p


# ------------------------------------------------------------
# Utilities: simplex grids (global and local)
# ------------------------------------------------------------
def _round_to_grid(x: float, step: float) -> int:
    return int(round(x / step))


def generate_simplex_grid(
    field_names: List[str], step: float, *, epsilon: float
) -> List[Dict[str, float]]:
    """
    Globales Simplex-Grid (Summe=1). Erzwingt w_f >= epsilon (damit alle Felder berücksichtigt werden).
    """
    assert 0 < step <= 1.0
    L = int(round(1.0 / step))  # Gesamt-"Masse"
    F = len(field_names)
    min_count = max(1, _round_to_grid(epsilon, step))

    result: List[Dict[str, float]] = []
    cur = [0] * F

    def rec(i: int, remaining: int):
        if i == F - 1:
            cur[i] = remaining
            counts = [c + min_count for c in cur]
            if sum(counts) == L:
                w = {field_names[j]: round(counts[j] * step, 8) for j in range(F)}
                if abs(sum(w.values()) - 1.0) < 1e-6 and all(
                    w[f] >= epsilon - 1e-12 for f in field_names
                ):
                    result.append(w)
            return
        for v in range(remaining + 1):
            cur[i] = v
            rec(i + 1, remaining - v)

    rec(0, L - F * min_count)
    return result


def _generate_bounded_compositions(
    total: int, lows: Sequence[int], highs: Sequence[int]
) -> List[List[int]]:
    """Alle integer Vektoren v mit sum(v)=total und lows[i] <= v[i] <= highs[i]."""
    n = len(lows)
    out: List[List[int]] = []
    cur = [0] * n

    def rec(i: int, remaining: int):
        if i == n - 1:
            v = max(lows[i], min(highs[i], remaining))
            if v == remaining:
                cur[i] = v
                out.append(cur.copy())
            return
        lo = lows[i]
        hi = min(highs[i], remaining - sum(lows[j] for j in range(i + 1, n)))
        for v in range(lo, max(lo, hi) + 1):
            cur[i] = v
            rec(i + 1, remaining - v)

    rec(0, total)
    return out


def generate_local_simplex_around_best(
    field_names: List[str],
    best_w: Dict[str, float],
    step: float,
    radius_steps: int,
    epsilon: float,
) -> List[Dict[str, float]]:
    """
    Lokales Simplex-Feingrid um best_w (Summe=1; Vielfache von 'step'; w_f >= epsilon).
    """
    F = len(field_names)
    L = int(round(1.0 / step))
    min_count = max(1, _round_to_grid(epsilon, step))

    k_star = [_round_to_grid(best_w[f], step) for f in field_names]
    diff = L - sum(k_star)
    if diff != 0:
        for i in range(abs(diff)):
            k_star[i % F] += 1 if diff > 0 else -1

    lows = [max(min_count, k_star[i] - radius_steps) for i in range(F)]
    highs = [min(L - (F - 1) * min_count, k_star[i] + radius_steps) for i in range(F)]

    low_sum, high_sum = sum(lows), sum(highs)
    if low_sum > L:
        overflow = low_sum - L
        for i in range(F):
            take = min(overflow, lows[i] - min_count)
            lows[i] -= take
            overflow -= take
            if overflow == 0:
                break
    elif high_sum < L:
        deficit = L - high_sum
        for i in range(F):
            add = min(deficit, L - highs[i])
            highs[i] += add
            deficit -= add
            if deficit == 0:
                break

    vecs = _generate_bounded_compositions(L, lows, highs)
    out: List[Dict[str, float]] = []
    for v in vecs:
        w = {f: round(v[i] * step, 8) for i, f in enumerate(field_names)}
        if (
            all(w[f] >= epsilon - 1e-12 for f in field_names)
            and abs(sum(w.values()) - 1.0) < 1e-6
        ):
            out.append(w)
    out.sort(key=lambda w: sum(abs(w[f] - best_w[f]) for f in field_names))
    return out


# ------------------------------------------------------------
# Data model (FAST candidates)
# ------------------------------------------------------------
@dataclass
class PrecomputePack:
    field_names: List[str]
    gains: List[np.ndarray]  # per-query (Pi,)
    scores_per_field: Dict[str, List[np.ndarray]]  # f -> list over queries of (Pi,)
    grams: List[np.ndarray]  # list over queries of (Pi, F, F)
    q_ids: List[str]
    top_k: int


# ------------------------------------------------------------
# Loading & sampling
# ------------------------------------------------------------
def load_products_and_fields(
    base_dir: Path,
) -> Tuple[Dict[str, "ProductFieldStore"], Dict[str, int], ProductsMeta]:
    fields, pid_to_row, products_meta = VectorSearchMethodStoreStatic.vars_for_vsms(
        base_dir
    )
    return fields, pid_to_row, products_meta


def load_query_store(base_dir: Path) -> QueryStore:
    return QueryStore.load(base_dir)


def sample_query_ids(qs: QueryStore, frac: float, seed: int) -> List[str]:
    assert 0 < frac <= 1.0
    n_total = len(qs.id_to_row)
    n = max(1, int(round(n_total * frac)))
    _, ids = qs.get_query_sample(n=n, random_state=seed)
    return ids


# ------------------------------------------------------------
# Precompute (FAST candidates)
# ------------------------------------------------------------
def precompute_candidates(
    q_ids: List[str],
    qs: QueryStore,
    fields: Dict[str, "ProductFieldStore"],
    pid2row: Dict[str, int],
    top_k: int,
) -> PrecomputePack:
    field_names = [f for f in fields.keys() if f != "product_summary"]
    F = len(field_names)
    if F == 0:
        raise ValueError(
            "No fields found (expected title/description/bullet_point/brand)."
        )

    q_vecs = qs.get_vectors_for_ids(q_ids, as_float32=True)  # (Q, D) normalized

    gains_all: List[np.ndarray] = []
    scores_per_field: Dict[str, List[np.ndarray]] = {f: [] for f in field_names}
    grams_all: List[np.ndarray] = []
    field_stores = {f: fields[f].store for f in field_names}
    row2pid = {v: k for k, v in pid2row.items()}

    for qi, qid in enumerate(tqdm(q_ids, desc="Precompute (candidates)")):
        q = q_vecs[qi : qi + 1]  # (1, D)
        pid_to_gain = qs.get_pids_esci_gain_for_id(qid)
        cand_rows = [
            pid2row[str(pid)] for pid in pid_to_gain.keys() if str(pid) in pid2row
        ]
        if not cand_rows:
            gains_all.append(np.zeros((0,), dtype=np.float32))
            for f in field_names:
                scores_per_field[f].append(np.zeros((0,), dtype=np.float32))
            grams_all.append(np.zeros((0, F, F), dtype=np.float32))
            continue

        # Gains in Reihenfolge der cand_rows

        ordered_pids = [row2pid[r] for r in cand_rows]
        gains = np.array(
            [pid_to_gain.get(pid, 0.0) for pid in ordered_pids], dtype=np.float32
        )
        gains_all.append(gains)

        Pi = len(cand_rows)
        gram = np.empty((Pi, F, F), dtype=np.float32)
        field_mats = {
            f: field_stores[f].get_rows_by_indices(cand_rows, as_float32=True)
            for f in field_names
        }

        # Scores je Feld + Gram
        for fi, fname in enumerate(field_names):
            E = field_mats[fname]
            s = (q @ E.T).astype(np.float32).ravel()
            scores_per_field[fname].append(s)
            gram[:, fi, fi] = np.einsum("ij,ij->i", E, E, dtype=np.float32)

        for fi in range(F):
            Ei = field_mats[field_names[fi]]
            for fj in range(fi + 1, F):
                Ej = field_mats[field_names[fj]]
                gij = np.einsum("ij,ij->i", Ei, Ej, dtype=np.float32)
                gram[:, fi, fj] = gij
                gram[:, fj, fi] = gij

        grams_all.append(gram)

    return PrecomputePack(
        field_names=field_names,
        gains=gains_all,
        scores_per_field=scores_per_field,
        grams=grams_all,
        q_ids=q_ids,
        top_k=top_k,
    )


# ------------------------------------------------------------
# Scores (FAST candidates)
# ------------------------------------------------------------
def scores_method3(pack: PrecomputePack, weight: Dict[str, float]) -> List[np.ndarray]:
    # S = Sum_f w_f * s_f
    return [
        sum(weight[f] * pack.scores_per_field[f][i] for f in pack.field_names)
        for i in range(len(pack.q_ids))
    ]


def scores_method2(pack: PrecomputePack, weight: Dict[str, float]) -> List[np.ndarray]:
    # cos(q, (Σ w_f e_f)/||Σ w_f e_f||) = (Σ w_f s_f) / sqrt(w^T C_p w)
    w = np.array([weight[f] for f in pack.field_names], dtype=np.float32)  # (F,)
    scores: List[np.ndarray] = []
    for i in range(len(pack.q_ids)):
        num = sum(weight[f] * pack.scores_per_field[f][i] for f in pack.field_names)
        Cp = pack.grams[i]  # (Pi, F, F)
        tmp = Cp @ w  # (Pi, F)
        den = np.sqrt((tmp * w).sum(axis=1) + 1e-12)
        scores.append(num / den)
    return scores


# ------------------------------------------------------------
# Metrics & tie-break
# ------------------------------------------------------------
def topk_order(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.size)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def describe_metrics_over_queries(
    gains_lists: List[np.ndarray],
    score_lists: List[np.ndarray],
    ks: Iterable[int],
) -> Dict[str, Dict[str, float]]:
    """
    Für jede Query: Top-k Ranking -> gains extrahieren -> metrics.get_all_metrics
    Danach describe() über alle Queries und als Dict (orient='index') zurückgeben.
    """
    assert len(gains_lists) == len(score_lists)
    rows = []
    for gains, scores in zip(gains_lists, score_lists):
        if gains.size == 0:
            rows.append({f"NDCG@{k}": 0.0 for k in ks})
            continue
        row = {}
        for k in ks:
            order = topk_order(scores, k)
            gains_from_search = gains[order].tolist()
            ideal_gains = gains.tolist()
            m = mc.get_all_metrics(gains_from_search, ideal_gains, ks=[k])
            row.update(m)
        rows.append(row)
    df = pd.DataFrame(rows)
    desc = df.describe().T
    desc = desc.round(4)  # do not overfit numbers
    return desc.to_dict(orient="index")


def build_tie_break_ks(primary_k: int, max_k: int) -> List[int]:
    """Erzeuge priorisierte K-Liste: 5,10,15,20,25,... bis max_k; stelle sicher, dass primary_k enthalten ist."""
    ks = []
    for k in [5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100]:
        if k <= max_k:
            ks.append(k)
    if primary_k not in ks and primary_k <= max_k:
        ks.insert(0, primary_k)
    # eindeutige Reihenfolge, primary_k möglichst vorne
    seen = set()
    ordered = []
    for k in [primary_k] + ks:
        if k <= max_k and k not in seen:
            seen.add(k)
            ordered.append(k)
    return ordered


def score_vector_from_desc(
    desc: Dict[str, Dict[str, float]],
    *,
    ks_priority: List[int],
    stat: str,
    include_other_metrics: bool = True,
) -> Tuple[float, ...]:
    """
    Baue einen lexikografischen Score-Vektor:
      1) NDCG@k für k in ks_priority
      2) (optional) MAP@k, MRR@k, Precision@k, Recall@k, F1@k für dieselben k
    Fehlende Werte -> -inf.
    """
    vect: List[float] = []
    # 1) NDCG
    for k in ks_priority:
        vect.append(float(desc.get(f"NDCG@{k}", {}).get(stat, float("-inf"))))
    if include_other_metrics:
        for metric in ["MAP", "MRR", "Precision", "Recall", "F1"]:
            for k in ks_priority:
                vect.append(
                    float(desc.get(f"{metric}@{k}", {}).get(stat, float("-inf")))
                )
    return tuple(vect)


def better_score(vec_a: Tuple[float, ...], vec_b: Tuple[float, ...]) -> bool:
    """True, wenn vec_a lexikografisch besser als vec_b ist."""
    return vec_a > vec_b  # Python tupelvergleich ist lexikografisch, max-first


# ------------------------------------------------------------
# Optimization (FAST candidates): coarse→fine + tie-break
# ------------------------------------------------------------
def optimize_fast_candidates(
    methode: str,
    pack: PrecomputePack,
    *,
    coarse_step: float,
    epsilon: float,
    ndcg_k: int,
    optimize_for_stat: str,
    max_stages: int,
    min_step: float,
    local_radius_steps: int,
) -> Tuple[Dict[str, float], float]:
    field_names = pack.field_names
    get_scores = scores_method3 if methode == "3" else scores_method2

    ks_priority = build_tie_break_ks(primary_k=ndcg_k, max_k=pack.top_k)

    def eval_grid(
        weight_grid: List[Dict[str, float]], step_for_log: float
    ) -> Tuple[Dict[str, float], Tuple[float, ...]]:
        best_w: Dict[str, float] = {}
        best_vec: Tuple[float, ...] = tuple(
            [float("-inf")] * (len(ks_priority) * (1 + 5))
        )  # ndcg + 5 more metrics
        for w in tqdm(
            weight_grid, desc=f"Method {methode} (FAST; step={step_for_log:.4f})"
        ):
            S = get_scores(pack, w)
            desc = describe_metrics_over_queries(pack.gains, S, ks=ks_priority)
            vec = score_vector_from_desc(
                desc,
                ks_priority=ks_priority,
                stat=optimize_for_stat,
                include_other_metrics=True,
            )
            if better_score(vec, best_vec):
                best_vec, best_w = vec, w
                logger.info(
                    f"New best NDCG@{ndcg_k}: {best_vec[0]:.3f} with weights {best_w}"
                )

        return best_w, best_vec

    # Coarse
    coarse_grid = generate_simplex_grid(field_names, step=coarse_step, epsilon=epsilon)
    best_w, best_vec = eval_grid(coarse_grid, coarse_step)

    # Fine
    step = coarse_step
    for _ in range(max_stages):
        step /= 2.0
        if step < min_step:
            break
        local_grid = generate_local_simplex_around_best(
            field_names=field_names,
            best_w=best_w,
            step=step,
            radius_steps=local_radius_steps,
            epsilon=epsilon,
        )
        if not local_grid:
            continue
        bw, bvec = eval_grid(local_grid, step)
        if better_score(bvec, best_vec):
            best_w, best_vec = bw, bvec

    # Primärer Score (für Logging/Speichern) = NDCG@ndcg_k (erste Komponente im Vektor)
    primary_ndcg = best_vec[0]
    return best_w, float(primary_ndcg)


# ------------------------------------------------------------
# Catalog (langsam): coarse→fine + tie-break via VSMS/Evaluation
# ------------------------------------------------------------
def evaluate_catalog_coarse_fine(
    base_dir: Path,
    methode: str,
    qs: QueryStore,
    vsms: VectorSearchMethodStore,
    q_ids: List[str],
    *,
    coarse_step: float,
    epsilon: float,
    ndcg_k: int,
    optimize_for_stat: str,
    max_stages: int,
    min_step: float,
    local_radius_steps: int,
    top_k: int,
) -> Tuple[Dict[str, float], float]:
    canonical = [
        "product_title",
        "product_description",
        "product_bullet_point",
        "product_brand",
    ]
    field_names = [f for f in canonical if f in vsms.fields]
    extra = [
        f for f in vsms.fields.keys() if f not in field_names and f != "product_summary"
    ]
    if extra:
        logger.warning(
            "Catalog has extra fields ignored for weight search: %s", ", ".join(extra)
        )
    ks_priority = build_tie_break_ks(primary_k=ndcg_k, max_k=top_k)

    def set_weights_and_eval(w: Dict[str, float]) -> Tuple[float, ...]:
        if methode == "2":
            if "methode_2" in vsms.init_methods:
                vsms.reinit_methode_2(w)
            else:
                vsms.init_methode_2(w)
        elif methode == "3":
            if "methode_3" in vsms.init_methods:
                vsms.reinit_methode_3(w)
            else:
                vsms.init_methode_3(w)
        else:
            raise ValueError("Unknown method.")

        evaluator = Evaluation(vsms, qs)
        all_metrics = evaluator.get_all_query_metrics_for_query_ids(
            q_ids, top_k=top_k, eval_on="catalog", methodes=[methode]
        )
        df = pd.DataFrame(all_metrics.get(f"methode_{methode}", {})).T
        if df.empty:
            # vollständiger -inf Vektor
            return tuple([float("-inf")] * (len(ks_priority) * (1 + 5)))

        desc = df.describe().T.to_dict(orient="index")
        vec = score_vector_from_desc(
            desc,
            ks_priority=ks_priority,
            stat=optimize_for_stat,
            include_other_metrics=True,
        )
        return vec

    # Coarse
    coarse_grid = generate_simplex_grid(
        field_names,
        step=coarse_step,
        epsilon=epsilon,
    )
    best_w: Dict[str, float] = {}
    best_vec: Tuple[float, ...] = tuple([float("-inf")] * (len(ks_priority) * (1 + 5)))

    for w in tqdm(
        coarse_grid,
        desc=f"Method {methode} (CATALOG; step={coarse_step:.4f})",
    ):
        vec = set_weights_and_eval(w)
        if better_score(vec, best_vec):
            best_vec, best_w = vec, w
            logger.info(
                f"New best NDCG@{ndcg_k}: {best_vec[0]:.3f} with weights {best_w}"
            )

    # Fine
    step = coarse_step
    for _ in range(max_stages):
        step /= 2.0
        if step < min_step:
            break
        local_grid = generate_local_simplex_around_best(
            field_names=field_names,
            best_w=best_w,
            step=step,
            radius_steps=local_radius_steps,
            epsilon=epsilon,
        )
        if not local_grid:
            continue
        for w in tqdm(
            local_grid, desc=f"Method {methode} (CATALOG fine; step={step:.4f})"
        ):
            vec = set_weights_and_eval(w)
            if better_score(vec, best_vec):
                best_vec, best_w = vec, w
                logger.info(
                    f"New best NDCG@{ndcg_k}: {best_vec[0]:.3f} with weights {best_w}"
                )

    primary_ndcg = best_vec[0]
    return best_w, float(primary_ndcg)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
def save_best(
    base_dir: Path,
    mode: str,
    methode: str,
    best_weight: Dict[str, float],
    best_score: float,
    args: argparse.Namespace,
) -> None:
    obj = {
        "type": "weight_search",
        "strategy": "coarse_fine_grid_with_tie_break",
        "mode": mode,
        "method": methode,
        "ndcg_k": int(args.ndcg_k),
        "optimize_for_stat": args.optimize_for_stat,
        "topk_candidates": int(args.topk_candidates),
        "frac_queries": float(args.frac_queries),
        "grid_step": float(args.grid_step),
        "max_stages": int(args.max_stages),
        "min_step": float(args.min_step),
        "local_radius_steps": int(args.local_radius_steps),
        "epsilon": float(args.epsilon if args.epsilon is not None else args.grid_step),
        "seed": int(args.seed),
        "weights": best_weight,
        "score": best_score,
    }
    fname = (
        f"best_weights_method_fast{methode}.json"
        if mode == "catalog"
        else f"best_weights_method_fast{methode}_on_candidates.json"
    )
    out_path = base_dir / fname
    out_path.write_text(json.dumps(obj, indent=2))
    logger.info("Saved best weights to %s", out_path)


def run_base_dir(base_dir: Path) -> None:
    global args
    assert base_dir.exists(), f"base_dir not found: {base_dir}"

    # epsilon default = grid_step (erzwingt alle Felder strikt >0)
    if args.epsilon is None:
        args.epsilon = args.min_step
    epsilon: float = float(args.epsilon)

    # Laden
    fields, pid2row, _products_meta = load_products_and_fields(base_dir)
    qs = load_query_store(base_dir)

    # Query-Stichprobe
    q_ids = sample_query_ids(qs, frac=args.frac_queries, seed=args.seed)
    logger.info(f"Sampled {len(q_ids)}/{len(qs.id_to_row)} queries.")

    # Felder (zur Info)
    field_names = [f for f in fields.keys() if f != "product_summary"]
    logger.info("Fields for mixing: %s", ", ".join(field_names))

    # Modi
    run_candidates = "both" in args.modes or "candidates" in args.modes
    run_catalog = "both" in args.modes or "catalog" in args.modes

    # Precompute (candidates)
    pack = None
    if run_candidates:
        pack = precompute_candidates(
            q_ids=q_ids,
            qs=qs,
            fields=fields,
            pid2row=pid2row,
            top_k=args.topk_candidates,
        )

    # VSMS für Catalog
    vsms = None
    if run_catalog:
        vsms = VectorSearchMethodStore.load(base_dir)  # type: ignore

    # Optimierung je Methode
    for methode in args.methods:
        if run_candidates:
            best_w_cand, best_s_cand = optimize_fast_candidates(
                methode=methode,
                pack=pack,
                # type: ignore[arg-type]
                coarse_step=args.grid_step,
                epsilon=epsilon,
                ndcg_k=args.ndcg_k,
                optimize_for_stat=args.optimize_for_stat,
                max_stages=args.max_stages,
                min_step=args.min_step,
                local_radius_steps=args.local_radius_steps,
            )
            logger.info(
                "Best (method %s, candidates): NDCG@%d %s=%.6f, weights=%s",
                methode,
                args.ndcg_k,
                args.optimize_for_stat,
                best_s_cand,
                json.dumps(best_w_cand),
            )
            save_best(base_dir, "candidates", methode, best_w_cand, best_s_cand, args)

        if run_catalog:
            best_w_cat, best_s_cat = evaluate_catalog_coarse_fine(
                base_dir=base_dir,
                methode=methode,
                qs=qs,
                vsms=vsms,
                # type: ignore[arg-type]
                q_ids=q_ids,
                coarse_step=args.grid_step,
                epsilon=epsilon,
                ndcg_k=args.ndcg_k,
                optimize_for_stat=args.optimize_for_stat,
                max_stages=args.max_stages,
                min_step=args.min_step,
                local_radius_steps=args.local_radius_steps,
                top_k=args.topk_candidates,
            )
            logger.info(
                "Best (method %s, catalog): NDCG@%d %s=%.6f, weights=%s",
                methode,
                args.ndcg_k,
                args.optimize_for_stat,
                best_s_cat,
                json.dumps(best_w_cat),
            )
            save_best(base_dir, "catalog", methode, best_w_cat, best_s_cat, args)


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------
if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    base_dirs = get_base_dirs(args.model_name)

    for base_dir in base_dirs:
        logger.info(f"Working on {base_dir}")
        run_base_dir(base_dir)

"""
poetry run python find_best_weights.py --model_name jinaai_jina-embeddings-v2-small-en_cpu_314ce109
poetry run python find_best_weights.py --model_name Qwen_Qwen3-Embedding-0.6B_cpu
"""
