import math

import numpy as np

from src.fusion import rrf
from src.index import InvertedIndex
from src.metrics import (average_precision, interpolated_precision, ndcg_at_k,
                         precision_at_k, recall_at_k)
from src.preprocess import preprocess
from src.rankers import BM25Ranker, TfidfRanker

DOCS = [["harri", "potter", "stone"], ["harri", "potter", "potter"], ["hobbit"], ["potter", "field"]]


def test_preprocess():
    assert preprocess("The Sorcerer's Stones &amp; Café") == ["sorcer", "stone", "cafe"]


def test_index():
    idx = InvertedIndex(DOCS)
    assert idx.N == 4 and idx.df("potter") == 3 and idx.df("zzz") == 0
    ids, tfs = idx.postings["potter"]
    assert ids.tolist() == [0, 1, 3] and tfs.tolist() == [1, 2, 1]
    assert idx.avgdl == 9 / 4


def test_bm25_matches_hand_computation():
    idx = InvertedIndex(DOCS)
    bm25 = BM25Ranker(idx, k1=1.5, b=0.75)
    idf = math.log(1 + (4 - 2 + 0.5) / (2 + 0.5))  # 'harri' is in 2 of 4 docs
    expected = idf * 1 * 2.5 / (1 + 1.5 * (1 - 0.75 + 0.75 * 3 / 2.25))
    assert math.isclose(bm25.scores(["harri"])[0], expected, rel_tol=1e-6)


def test_tfidf_is_cosine():
    idx = InvertedIndex(DOCS)
    s = TfidfRanker(idx).scores(["hobbit"])
    assert math.isclose(s[2], 1.0, rel_tol=1e-9) and np.count_nonzero(s) == 1


def test_rrf():
    fused = rrf([[1, 2, 3], [2, 1, 4]], k=60)
    assert [d for d, _ in fused] == [1, 2, 3, 4]
    assert math.isclose(fused[0][1], 1 / 61 + 1 / 62)


def test_metrics():
    ranked, qrels = [1, 9, 2, 8], {1: 2, 2: 1, 3: 1}
    assert precision_at_k(ranked, qrels, 4) == 0.5
    assert math.isclose(recall_at_k(ranked, qrels, 4), 2 / 3)
    assert math.isclose(average_precision(ranked, qrels), (1 / 1 + 2 / 3) / 3)
    dcg = 3 / math.log2(2) + 1 / math.log2(4)
    idcg = 3 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert math.isclose(ndcg_at_k(ranked, qrels, 4), dcg / idcg)
    ip = interpolated_precision(ranked, qrels)
    assert ip[0] == 1.0 and ip[-1] == 0.0
