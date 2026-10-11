"""Shared evaluation machinery for tune.py and evaluate.py."""
import json

import numpy as np

from .data import PROCESSED
from .engine import SearchEngine
from .fusion import rrf
from .metrics import average_precision, ndcg_at_k, precision_at_k, recall_at_k
from .preprocess import preprocess
from .rankers import BM25Ranker, top_k

DEPTH = 1000  # ranking depth used for MAP and the precision-recall curve
METRICS = ["Precision@10", "Recall@100", "MAP", "nDCG@10"]
TYPE_NAMES = {"title": "Known-item (title)", "typo": "Known-item (misspelled)", "author": "Author search",
              "topic": "Topic search", "cbf": "Content-based filtering"}
SEARCH_TYPES = ["title", "typo", "author", "topic"]


def load_queries() -> list[dict]:
    """Test collection with a fixed 30/70 dev/test split inside every query type."""
    queries = json.loads((PROCESSED / "testset.json").read_text())
    seen = {}
    for q in queries:
        i = seen[q["type"]] = seen.get(q["type"], -1) + 1
        q["split"] = "dev" if i % 10 < 3 else "test"
        q["qrels"] = {int(d): g for d, g in q["qrels"].items()}
    return queries


def score(ranked: list[int], qrels: dict[int, int]) -> dict[str, float]:
    return {"Precision@10": precision_at_k(ranked, qrels, 10), "Recall@100": recall_at_k(ranked, qrels, 100),
            "MAP": average_precision(ranked, qrels), "nDCG@10": ndcg_at_k(ranked, qrels, 10)}


def fuse(run: dict[str, list[int]], members=("TF-IDF", "BM25", "Dense"), k: int = 60) -> list[int]:
    return [d for d, _ in rrf([run[m] for m in members], k=k, top_n=DEPTH)]


class Runner:
    def __init__(self, engine: SearchEngine, queries: list[dict]):
        self.engine, self.queries = engine, queries
        self.tokens = [preprocess(q["query"]) for q in queries]
        self.q_emb = engine.dense.encode([q["query"] for q in queries])
        # Topic qrels only exist for books with Open Library subjects, so topic
        # queries are ranked over that judged pool only (condensed-list evaluation).
        self.unjudged = np.flatnonzero((engine.catalog["subjects"].map(len) == 0).values)

    def _rank(self, i: int, scores: np.ndarray, floor: float = 0.0) -> list[int]:
        q = self.queries[i]
        if q["exclude"]:
            scores[q["exclude"]] = floor
        if q["type"] == "topic":
            scores[self.unjudged] = floor
        return [d for d, _ in top_k(scores, DEPTH)]

    def tfidf(self, idx: list[int], w_desc: float | None = None) -> list[list[int]]:
        return [self._rank(i, self.engine.tfidf_scores(self.tokens[i], w_desc)) for i in idx]

    def bm25(self, idx: list[int], w_desc: float | None = None) -> list[list[int]]:
        ranker = self.engine.bm25 if w_desc is None else BM25Ranker(
            self.engine.index, extra_fields=((self.engine.desc_index, w_desc),))
        return [self._rank(i, ranker.scores(self.tokens[i])) for i in idx]

    def dense(self, idx: list[int], use_desc: bool = True) -> list[list[int]]:
        return [self._rank(i, self.engine.dense.scores_from_vector(self.q_emb[i], use_desc), floor=-1e9) for i in idx]

    def base_runs(self, idx: list[int], descriptions: bool = True) -> list[dict[str, list[int]]]:
        w = None if descriptions else 0.0
        return [{"TF-IDF": a, "BM25": b, "Dense": c} for a, b, c in
                zip(self.tfidf(idx, w), self.bm25(idx, w), self.dense(idx, descriptions))]

    def mean_by_type(self, idx: list[int], rankings: list[list[int]], metric: str = "MAP") -> float:
        """Macro-average over query types, so no single type dominates tuning."""
        by_type = {}
        for i, ranked in zip(idx, rankings):
            by_type.setdefault(self.queries[i]["type"], []).append(score(ranked, self.queries[i]["qrels"])[metric])
        return float(np.mean([np.mean(v) for v in by_type.values()]))
