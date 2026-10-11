"""Evaluation metrics. `ranked` is a list of doc ids, best first.

`qrels` maps doc_id -> relevance grade (0 = not relevant). Binary metrics
treat any grade > 0 as relevant; nDCG uses the grade itself.
"""
import numpy as np


def precision_at_k(ranked: list[int], qrels: dict[int, int], k: int = 10) -> float:
    return sum(qrels.get(d, 0) > 0 for d in ranked[:k]) / k


def recall_at_k(ranked: list[int], qrels: dict[int, int], k: int = 100) -> float:
    n_rel = sum(g > 0 for g in qrels.values())
    return sum(qrels.get(d, 0) > 0 for d in ranked[:k]) / n_rel if n_rel else 0.0


def average_precision(ranked: list[int], qrels: dict[int, int]) -> float:
    """Mean of precision values at each rank where a relevant doc appears."""
    n_rel = sum(g > 0 for g in qrels.values())
    hits, total = 0, 0.0
    for i, d in enumerate(ranked, start=1):
        if qrels.get(d, 0) > 0:
            hits += 1
            total += hits / i
    return total / n_rel if n_rel else 0.0


def ndcg_at_k(ranked: list[int], qrels: dict[int, int], k: int = 10) -> float:
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    gains = np.array([2 ** qrels.get(d, 0) - 1 for d in ranked[:k]], dtype=np.float64)
    dcg = float((gains * discounts[: len(gains)]).sum())
    ideal = np.array(sorted((2 ** g - 1 for g in qrels.values()), reverse=True)[:k], dtype=np.float64)
    idcg = float((ideal * discounts[: len(ideal)]).sum())
    return dcg / idcg if idcg else 0.0


RECALL_LEVELS = np.linspace(0, 1, 11)


def interpolated_precision(ranked: list[int], qrels: dict[int, int]) -> np.ndarray:
    """11-point interpolated precision: best precision at any recall >= r."""
    n_rel = sum(g > 0 for g in qrels.values())
    if not n_rel:
        return np.zeros(11)
    hits, prec, rec = 0, [], []
    for i, d in enumerate(ranked, start=1):
        hits += qrels.get(d, 0) > 0
        prec.append(hits / i)
        rec.append(hits / n_rel)
    prec, rec = np.array(prec), np.array(rec)
    return np.array([prec[rec >= r].max() if (rec >= r).any() else 0.0 for r in RECALL_LEVELS])
