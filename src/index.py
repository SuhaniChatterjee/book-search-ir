"""Inverted index built from scratch.

For every term we store a postings list: which documents contain it and how
often. Ranking then only touches documents that share a term with the query,
instead of scanning the whole catalog.
"""
from collections import Counter, defaultdict

import numpy as np


class InvertedIndex:
    def __init__(self, docs_tokens: list[list[str]]):
        self.N = len(docs_tokens)
        self.doc_len = np.array([len(d) for d in docs_tokens], dtype=np.float32)
        # Average over non-empty documents: most books have no description field.
        non_empty = self.doc_len[self.doc_len > 0]
        self.avgdl = float(non_empty.mean()) if len(non_empty) else 1.0

        ids, tfs = defaultdict(list), defaultdict(list)
        for doc_id, tokens in enumerate(docs_tokens):
            for term, tf in Counter(tokens).items():
                ids[term].append(doc_id)
                tfs[term].append(tf)

        # term -> (doc_ids, term_frequencies), doc_ids sorted ascending
        self.postings = {
            t: (np.array(ids[t], dtype=np.int32), np.array(tfs[t], dtype=np.float32))
            for t in ids
        }

    def df(self, term: str) -> int:
        """Document frequency: number of documents containing the term."""
        return len(self.postings[term][0]) if term in self.postings else 0

    @property
    def vocab_size(self) -> int:
        return len(self.postings)
