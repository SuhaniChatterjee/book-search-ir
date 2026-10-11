"""Dense retrieval: Sentence Transformer embeddings + exact cosine search.

Every book has a metadata vector (title, author, publisher). Books with an
Open Library description get a second vector that includes the description.
A book's score is the larger of its two cosine similarities, so a short title
query and a descriptive topic query can each match the vector that suits them.
"""
from pathlib import Path

import numpy as np

from .rankers import top_k

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class DenseRanker:
    def __init__(self, texts: list[str], desc_rows: np.ndarray, desc_texts: list[str], cache_dir: Path):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL_NAME)
        self.desc_rows = desc_rows
        self.emb = self._load_or_encode(texts, cache_dir / "embeddings.npy")
        self.emb_desc = self._load_or_encode(desc_texts, cache_dir / "embeddings_desc.npy")

    def _load_or_encode(self, texts: list[str], cache: Path) -> np.ndarray:
        if cache.exists():
            emb = np.load(cache)
        else:
            # Unit-length vectors, so a dot product equals cosine similarity.
            emb = self.model.encode(
                texts, batch_size=128, normalize_embeddings=True,
                show_progress_bar=True, convert_to_numpy=True,
            ).astype(np.float32)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, emb)
        assert len(emb) == len(texts), f"{cache.name} does not match the catalog; delete it to rebuild"
        return emb

    def encode(self, queries: list[str]) -> np.ndarray:
        return self.model.encode(queries, normalize_embeddings=True, convert_to_numpy=True)

    def scores_from_vector(self, q: np.ndarray, use_desc: bool = True) -> np.ndarray:
        scores = self.emb @ q
        if use_desc and len(self.desc_rows):
            scores[self.desc_rows] = np.maximum(scores[self.desc_rows], self.emb_desc @ q)
        return scores

    def scores(self, query: str) -> np.ndarray:
        return self.scores_from_vector(self.encode([query])[0])

    def search(self, query: str, k: int = 10):
        return top_k(self.scores(query), k)
