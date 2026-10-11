"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009)."""
from collections import defaultdict


def rrf(rankings: list[list[int]], k: int = 60, top_n: int | None = None) -> list[tuple[int, float]]:
    """Fuse ranked lists of doc ids.

    RRF(d) = sum over rankers m of 1 / (k + rank_m(d)), rank is 1-based.
    Only ranks are used, so the rankers' score scales never need normalising.
    """
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (k + rank)
    out = sorted(fused.items(), key=lambda x: (-x[1], x[0]))
    return out[:top_n] if top_n else out
