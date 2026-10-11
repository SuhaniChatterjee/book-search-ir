"""SearchEngine: ties catalog, the fielded index, the three rankers and RRF together."""
import json
import pickle

import numpy as np

from .data import PROCESSED, ROOT, load_catalog
from .dense import DenseRanker
from .fusion import rrf
from .index import InvertedIndex
from .preprocess import preprocess
from .rankers import BM25Ranker, TfidfRanker, top_k

MODELS = ["TF-IDF", "BM25", "Dense", "Hybrid RRF"]
BASE_MODELS = MODELS[:3]
DEPTH = 100  # how many results each ranker contributes to the fusion in the UI
PARAMS_PATH = ROOT / "params.json"  # written by tune.py


class SearchEngine:
    def __init__(self, params: dict | None = None):
        self.params = params or json.loads(PARAMS_PATH.read_text())
        self.catalog = cat = load_catalog()

        index_path = PROCESSED / "index.pkl"
        if index_path.exists():
            with open(index_path, "rb") as f:
                self.index, self.desc_index = pickle.load(f)
        else:
            # Field 1: title + author + publisher. Field 2: description (often empty).
            self.index = InvertedIndex([preprocess(t) for t in cat["text"]])
            self.desc_index = InvertedIndex([preprocess(t) if t else [] for t in cat["description"]])
            with open(index_path, "wb") as f:
                pickle.dump((self.index, self.desc_index), f)

        self.tfidf = TfidfRanker(self.index)
        self.tfidf_desc = TfidfRanker(self.desc_index)
        self.bm25 = BM25Ranker(self.index, extra_fields=((self.desc_index, self.params["bm25_w_desc"]),))

        desc_rows = np.flatnonzero((cat["description"] != "").values)
        desc_texts = (cat["dense_text"].iloc[desc_rows] + ". " + cat["description"].iloc[desc_rows].str[:1200]).tolist()
        self.dense = DenseRanker(cat["dense_text"].tolist(), desc_rows, desc_texts, PROCESSED)

    def tfidf_scores(self, tokens: list[str], w_desc: float | None = None) -> np.ndarray:
        """Fielded TF-IDF: cosine on the metadata field plus weighted cosine on the description."""
        w = self.params["tfidf_w_desc"] if w_desc is None else w_desc
        scores = self.tfidf.scores(tokens)
        return scores + w * self.tfidf_desc.scores(tokens) if w > 0 else scores

    def search_all(self, query: str, k: int = DEPTH, rrf_k: int | None = None) -> dict[str, list[tuple[int, float]]]:
        """Run every model once; returns {model name: [(doc_id, score), ...]}."""
        q = preprocess(query)
        out = {
            "TF-IDF": top_k(self.tfidf_scores(q), k),
            "BM25": self.bm25.search(q, k),
            "Dense": self.dense.search(query, k),
        }
        out["Hybrid RRF"] = rrf([[d for d, _ in r] for r in out.values()],
                                k=rrf_k or self.params["rrf_k"], top_n=k)
        return out
