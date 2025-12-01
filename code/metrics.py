from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Dict, List

ESCI_TO_GAIN_DEFAULT: Dict[str, float] = {
    "E": 1.0,
    "S": 0.1,
    "C": 0.01,
    "I": 0,
}  # see https://github.com/amazon-science/esci-data/blob/7916cdf6ab75a462e77f20ab40428a10923998d5/ranking/train.py#L49
RELEVANCE_THRESHOLD = ESCI_TO_GAIN_DEFAULT[
    "S"
]  # ab welchem gain ein Dokument als relevant gilt≈

GAIN_LIST_TYPE = List[float | int]


def _validate_and_prepare(
    gains: Sequence[float | int], ideal_gains: Sequence[float | int], k: int
) -> tuple[list[float | int], list[float | int], int]:
    if not isinstance(gains, Sequence) or not isinstance(ideal_gains, Sequence):
        raise ValueError(
            "gains und ideal_gains müssen Sequenzen sein (z.B. list/tuple)."
        )
    if not all(isinstance(g, float | int) for g in gains):
        raise ValueError("Alle gains müssen int oder float sein.")
    if not all(isinstance(g, float | int) for g in ideal_gains):
        raise ValueError("Alle ideal_gains müssen int oder float sein.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k muss eine positive ganze Zahl sein.")
    g = list(gains)
    ideal_gains_sorted = list(sorted(ideal_gains, reverse=True))
    if len(g) < k:
        g = g + [0] * (k - len(g))
    if len(ideal_gains_sorted) < k:
        ideal_gains_sorted = ideal_gains_sorted + [0] * (k - len(ideal_gains_sorted))
    return g[:k], ideal_gains_sorted[:k], k


def _is_relevant(g: float | int) -> bool:
    return g >= RELEVANCE_THRESHOLD


def dcg(gains: GAIN_LIST_TYPE, _ideal_gains_unused: GAIN_LIST_TYPE, k: int) -> float:
    gains_pre, _, k = _validate_and_prepare(gains, [], k)
    val = 0.0
    for i, gain in enumerate(gains_pre[:k]):
        val += float(gain) / math.log2(i + 2)
    return val


def ndcg(gains: List[float | int], ideal_gains: List[float | int], k: int) -> float:
    gains_pre, ideal_gains_pre, k = _validate_and_prepare(gains, ideal_gains, k)
    actual = dcg(gains_pre, [], k)
    ideal = dcg(ideal_gains_pre, [], k)
    return (actual / ideal) if ideal > 0 else 0.0


def dcg_group(
    gains: GAIN_LIST_TYPE, _ideal_gains_unused: GAIN_LIST_TYPE, k: int
) -> float:
    GROUP_RANGE = 3
    gains_pre, _, k = _validate_and_prepare(gains, [], k)
    val = 0.0
    for i, gain in enumerate(gains_pre[:k]):
        i_group = i // GROUP_RANGE
        val += float(gain) / math.log2(i_group + 2)
    return val


def ndcg_group(
    gains: List[float | int], ideal_gains: List[float | int], k: int
) -> float:
    gains_pre, ideal_gains_pre, k = _validate_and_prepare(gains, ideal_gains, k)
    actual = dcg_group(gains_pre, [], k)
    ideal = dcg_group(ideal_gains_pre, [], k)
    return (actual / ideal) if ideal > 0 else 0.0


def precision(gains: GAIN_LIST_TYPE, ideal_gains: GAIN_LIST_TYPE, k: int) -> float:
    gains_pre, _, k = _validate_and_prepare(gains, [], k)
    rel = sum(_is_relevant(x) for x in gains_pre)
    return rel / k


def recall(gains: GAIN_LIST_TYPE, ideal_gains: GAIN_LIST_TYPE, k: int) -> float:
    gains_pre, _, k = _validate_and_prepare(gains, ideal_gains, k)
    rel_found = sum(_is_relevant(x) for x in gains_pre)
    rel_total = sum(_is_relevant(x) for x in ideal_gains)  # volle Liste!
    return rel_found / rel_total if rel_total > 0 else 0.0


def mean_average_precision(
    gains: GAIN_LIST_TYPE, ideal_gains: GAIN_LIST_TYPE, k: int
) -> float:
    g, _, k = _validate_and_prepare(gains, ideal_gains, k)
    rel = 0
    s = 0.0
    for i, gain in enumerate(g[:k], start=1):
        if _is_relevant(gain):
            rel += 1
            s += rel / i
    return s / rel if rel > 0 else 0.0


def mrr(gains: GAIN_LIST_TYPE, ideal_gains: GAIN_LIST_TYPE, k: int) -> float:
    g, _, k = _validate_and_prepare(gains, ideal_gains, k)
    for i, gain in enumerate(g[:k], start=1):
        if _is_relevant(gain):
            return 1.0 / i
    return 0.0


def f1(gains: GAIN_LIST_TYPE, ideal_gains: GAIN_LIST_TYPE, k: int) -> float:
    p = precision(gains, ideal_gains, k)
    r = recall(gains, ideal_gains, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def get_all_metrics(
    list_of_gains: GAIN_LIST_TYPE,
    list_of_ideal_gains: GAIN_LIST_TYPE,
    ks: GAIN_LIST_TYPE | None = None,
) -> Dict[str, float]:
    """
    Berechnet gängige Ranking-Metriken für mehrere k.
    Beachte: Gains werden als binär (>0 relevant) oder abgestuft (E/S/C/I->3/2/1/0) interpretiert.
    """
    ks = ks or [5, 10, 15]
    all_metrics: Dict[str, float] = {}
    metrics = {
        "DCG": dcg,
        "NDCG": ndcg,
        "Precision": precision,
        "Recall": recall,
        "MAP": mean_average_precision,
        "MRR": mrr,
        "F1": f1,
    }
    for k in ks:
        for name, fn in metrics.items():
            all_metrics[f"{name}@{k}"] = fn(list_of_gains, list_of_ideal_gains, k)
    return all_metrics


def example_usage():
    E, S, C, I = ESCI_TO_GAIN_DEFAULT.values()
    gains = [E, C, S, I]  # System-Ranking
    ideal_gains = [E, S, C, I]  # Ground Truth gesamt, absteigend sortiert

    assert round(dcg(gains, [], 4), 6) == round(
        E / 1 + C / math.log2(3) + S / math.log2(4), 6
    )
    assert round(ndcg(gains, ideal_gains, 4), 6) <= 1.0
    assert (
        precision(gains, ideal_gains, 3) == 2 / 3
    )  # bei Threshold=S sind E und S relevant


if __name__ == "__main__":
    example_usage()
