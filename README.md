# Book Search on Book-Crossing

An information retrieval system for book catalog search over the 271,061 books of the
Book-Crossing dataset. It compares four ranking models:

- **TF-IDF** (vector space model, cosine similarity)
- **BM25F** (Okapi BM25 with weighted fields)
- **Sentence Transformer** (`all-MiniLM-L6-v2`, dense retrieval)
- **Hybrid RRF** (Reciprocal Rank Fusion of the three)

The inverted index, both sparse rankers, the fusion and the metrics are implemented from
scratch with NumPy.

## Results

Held-out test queries (493 search queries: exact title, misspelled title, author, topic).

| Model | Precision@10 | Recall@100 | MAP | nDCG@10 |
|---|---|---|---|---|
| TF-IDF | 0.361 | 0.794 | 0.643 | 0.718 |
| BM25 | 0.376 | 0.802 | 0.665 | 0.742 |
| Sentence Transformer | 0.281 | 0.700 | 0.486 | 0.601 |
| Hybrid RRF | 0.373 | 0.808 | 0.661 | 0.745 |

BM25 and Hybrid RRF are statistically tied overall. The hybrid is best for misspelled and
topic queries; BM25 is best for exact title and author look-ups. Per-query-type tables,
ablations and figures are in `reports/`.

## Data

The data is not in the repository.

1. **Book-Crossing.** Download the dataset from Kaggle and place `BX-Books.csv`,
   `BX-Book-Ratings.csv` and `BX-Users.csv` in `data/raw/`.
2. **Open Library descriptions and subjects** (optional). `python fetch_openlibrary.py`
   fetches them for the 20,000 most-rated works into `data/openlibrary/`. It takes several
   hours and can be stopped and resumed. Without it the system runs on title, author and
   publisher only, and topic queries cannot be built.

Book-Crossing has no queries or relevance judgements, so `build_testset.py` simulates
1,009 queries of five types and derives the relevance labels.

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```

## Running

Run these in order. Each step caches its output in `data/processed/`.

| Command | What it does |
|---|---|
| `python demo_toy.py "harry potter"` | Walks through every pipeline stage on a 10-book toy catalog |
| `python eda.py` | Dataset statistics and charts |
| `python build_testset.py` | Builds the simulated queries and relevance labels |
| `python tune.py` | Tunes field weights and the RRF constant on the dev queries, writes `params.json` |
| `python evaluate.py` | Evaluates all models on the test queries, writes `reports/results.json` and figures |
| `streamlit run app.py` | Search interface with the four models side by side |
| `python build_report.py` | Fills the report template with the measured numbers (author rows come from an optional, git-ignored `students.json`) |
| `python -m pytest tests` | Unit tests |

The first run of `tune.py`, `evaluate.py` or `app.py` builds the index (seconds) and
encodes the catalog with the Sentence Transformer (a few minutes on a laptop).

## Layout

```
src/preprocess.py   cleaning, tokenization, stop-word removal, Porter stemming
src/data.py         loading and cleaning the CSVs, grouping editions into works
src/index.py        inverted index
src/rankers.py      TF-IDF and BM25F
src/dense.py        Sentence Transformer ranker
src/fusion.py       Reciprocal Rank Fusion
src/metrics.py      Precision@k, Recall@k, MAP, nDCG@k, precision-recall curve
src/engine.py       SearchEngine combining all of the above
src/evaluation.py   dev/test split and shared evaluation code
```
