"""Tune the description-field weights and the RRF constant on the dev queries.

Only the 30% dev split is used here; evaluate.py reports on the other 70%.
Run:  python tune.py
"""
import json

from src.engine import PARAMS_PATH, SearchEngine
from src.evaluation import SEARCH_TYPES, Runner, fuse, load_queries

W_GRID = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
K_GRID = [1, 5, 10, 20, 40, 60, 100]

engine = SearchEngine()
queries = load_queries()
runner = Runner(engine, queries)
dev = [i for i, q in enumerate(queries) if q["split"] == "dev" and q["type"] in SEARCH_TYPES]
print(f"tuning on {len(dev)} dev queries (macro-averaged MAP over {len(SEARCH_TYPES)} query types)\n")

log, params = {}, {}
for name, run in (("tfidf_w_desc", runner.tfidf), ("bm25_w_desc", runner.bm25)):
    log[name] = {w: runner.mean_by_type(dev, run(dev, w)) for w in W_GRID}
    params[name] = max(log[name], key=log[name].get)
    print(name, {w: round(v, 4) for w, v in log[name].items()}, "-> best", params[name])

engine.params.update(params)
engine.bm25 = None  # rebuilt below with the tuned weight
runs = [{"TF-IDF": a, "BM25": b, "Dense": c} for a, b, c in
        zip(runner.tfidf(dev, params["tfidf_w_desc"]), runner.bm25(dev, params["bm25_w_desc"]), runner.dense(dev))]
log["rrf_k"] = {k: runner.mean_by_type(dev, [fuse(r, k=k) for r in runs]) for k in K_GRID}
params["rrf_k"] = max(log["rrf_k"], key=log["rrf_k"].get)
print("rrf_k", {k: round(v, 4) for k, v in log["rrf_k"].items()}, "-> best", params["rrf_k"])

PARAMS_PATH.write_text(json.dumps(params, indent=2) + "\n")
(PARAMS_PATH.parent / "reports" / "tuning_log.json").write_text(json.dumps(log, indent=2))
print("\nsaved", PARAMS_PATH)
