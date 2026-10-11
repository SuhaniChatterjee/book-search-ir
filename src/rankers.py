"""Sparse rankers (TF-IDF, BM25) scored term-at-a-time over the inverted index."""
from collections import Counter

import numpy as np

from .index import InvertedIndex


def top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return the k highest-scoring (doc_id, score) pairs, ignoring zero scores."""
    k = min(k, int(np.count_nonzero(scores)))
    if k == 0:
        return []
    cand = np.argpartition(-scores, k - 1)[:k]
    # Sort by score descending, doc_id ascending so ties are deterministic.
    cand = cand[np.lexsort((cand, -scores[cand]))]
    return [(int(d), float(scores[d])) for d in cand]


class TfidfRanker:
    """Vector space model with cosine similarity.

    weight(t, d) = (1 + log tf) * idf(t),  idf(t) = log(N / df)
    """

    def __init__(self, index: InvertedIndex):
        self.index = index
        self.idf = {t: np.log(index.N / len(ids)) for t, (ids, _) in index.postings.items()}
        # Precompute each document's vector length for cosine normalisation.
        sq = np.zeros(index.N, dtype=np.float64)
        for t, (ids, tfs) in index.postings.items():
            sq[ids] += ((1 + np.log(tfs)) * self.idf[t]) ** 2
        self.doc_norm = np.sqrt(sq)
        self.doc_norm[self.doc_norm == 0] = 1.0

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.index.N, dtype=np.float64)
        q_weights = {
            t: (1 + np.log(tf)) * self.idf[t]
            for t, tf in Counter(query_tokens).items()
            if t in self.idf
        }
        q_norm = np.sqrt(sum(w * w for w in q_weights.values())) or 1.0
        for t, qw in q_weights.items():
            ids, tfs = self.index.postings[t]
            scores[ids] += qw * (1 + np.log(tfs)) * self.idf[t]
        return scores / (self.doc_norm * q_norm)

    def search(self, query_tokens: list[str], k: int = 10):
        return top_k(self.scores(query_tokens), k)


class BM25Ranker:
    """Okapi BM25, extended to weighted fields (BM25F, Robertson & Zaragoza 2009).

    Each field f has its own inverted index and length normalisation. A term's
    frequencies are combined into one pseudo-frequency before saturation:

        tf'(t, d) = sum_f  w_f * tf_f / (1 - b + b * len_f(d) / avglen_f)
        score(q, d) = sum_t  idf(t) * tf' * (k1 + 1) / (tf' + k1)
        idf(t) = log(1 + (N - df + 0.5) / (df + 0.5)),  df counted over any field

    With a single field of weight 1 this is exactly classic BM25.
    """

    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75,
                 extra_fields: tuple[tuple[InvertedIndex, float], ...] = ()):
        self.N, self.k1 = index.N, k1
        fields = [(index, 1.0), *[(ix, w) for ix, w in extra_fields if w > 0]]
        self.fields = [(ix, w, 1 - b + b * ix.doc_len / ix.avgdl) for ix, w in fields]
        self._buf = np.zeros(self.N, dtype=np.float64)

    def _pseudo_tf(self, term: str):
        """Doc ids containing the term in any field, and their combined tf'."""
        hits = []
        for ix, w, norm in self.fields:
            if term in ix.postings:
                ids, tfs = ix.postings[term]
                self._buf[ids] += w * tfs / norm[ids]
                hits.append(ids)
        if not hits:
            return None, None
        ids = hits[0] if len(hits) == 1 else np.unique(np.concatenate(hits))
        tf = self._buf[ids].copy()
        self._buf[ids] = 0.0
        return ids, tf

    def _idf(self, df: int) -> float:
        return float(np.log(1 + (self.N - df + 0.5) / (df + 0.5)))

    def df(self, term: str) -> int:
        ids, _ = self._pseudo_tf(term)
        return 0 if ids is None else len(ids)

    def idf(self, term: str) -> float:
        return self._idf(self.df(term))

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float64)
        for t in set(query_tokens):
            ids, tf = self._pseudo_tf(t)
            if ids is not None:
                scores[ids] += self._idf(len(ids)) * tf * (self.k1 + 1) / (tf + self.k1)
        return scores

    def search(self, query_tokens: list[str], k: int = 10):
        return top_k(self.scores(query_tokens), k)
