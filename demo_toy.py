"""Walk through every stage of the pipeline on a 10-book toy catalog.

Run:  python demo_toy.py "harry potter"
"""
import sys

from src.fusion import rrf
from src.index import InvertedIndex
from src.metrics import average_precision, ndcg_at_k, precision_at_k
from src.preprocess import clean, preprocess, tokenize
from src.rankers import BM25Ranker, TfidfRanker

BOOKS = [
    "Harry Potter and the Sorcerer's Stone | J. K. Rowling | Scholastic",
    "Harry Potter and the Chamber of Secrets | J. K. Rowling | Scholastic",
    "The Fellowship of the Ring | J. R. R. Tolkien | Houghton Mifflin",
    "The Hobbit | J. R. R. Tolkien | Houghton Mifflin",
    "Pride and Prejudice | Jane Austen | Penguin",
    "The Shining | Stephen King | Signet",
    "Pet Sematary | Stephen King | Signet",
    "The Potter's Field | Ellis Peters | Mysterious Press",
    "Sense and Sensibility | Jane Austen | Penguin",
    "A Brief History of Time | Stephen Hawking | Bantam",
]

query = sys.argv[1] if len(sys.argv) > 1 else "harry potter"

print(f"QUERY: {query!r}\n")

print("1. PREPROCESSING (book 0)")
print("   raw      :", BOOKS[0])
print("   cleaned  :", clean(BOOKS[0]))
print("   tokens   :", tokenize(clean(BOOKS[0])))
print("   no stops :", preprocess(BOOKS[0], do_stem=False))
print("   stemmed  :", preprocess(BOOKS[0]))

docs = [preprocess(b) for b in BOOKS]
index = InvertedIndex(docs)
q = preprocess(query)

print(f"\n2. INVERTED INDEX  ({index.N} docs, {index.vocab_size} terms, avg length {index.avgdl:.1f})")
for t in q:
    if t in index.postings:
        ids, tfs = index.postings[t]
        print(f"   {t!r:10} df={index.df(t)}  postings={[(int(i), int(f)) for i, f in zip(ids, tfs)]}")
    else:
        print(f"   {t!r:10} not in index")

tfidf, bm25 = TfidfRanker(index), BM25Ranker(index)
r_tfidf, r_bm25 = tfidf.search(q, 5), bm25.search(q, 5)

for name, res in [("3. TF-IDF", r_tfidf), ("4. BM25", r_bm25)]:
    print(f"\n{name}")
    for rank, (d, s) in enumerate(res, 1):
        print(f"   {rank}. [{d}] {s:.3f}  {BOOKS[d]}")

fused = rrf([[d for d, _ in r_tfidf], [d for d, _ in r_bm25]], top_n=5)
print("\n5. RRF (k=60) of TF-IDF + BM25")
for rank, (d, s) in enumerate(fused, 1):
    print(f"   {rank}. [{d}] {s:.4f}  {BOOKS[d]}")

if query == "harry potter":
    qrels = {0: 1, 1: 1}  # the two Harry Potter books are the relevant ones
    ranked = [d for d, _ in r_bm25]
    print("\n6. METRICS for BM25 (relevant = books 0 and 1)")
    print(f"   P@5  = {precision_at_k(ranked, qrels, 5):.3f}")
    print(f"   AP   = {average_precision(ranked, qrels):.3f}")
    print(f"   nDCG = {ndcg_at_k(ranked, qrels, 5):.3f}")
