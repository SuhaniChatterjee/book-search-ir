"""Evaluate TF-IDF, BM25, Dense and Hybrid RRF on the held-out test queries.

Parameters come from params.json (chosen by tune.py on the dev split).
Writes reports/results.json and the report figures.
Run:  python evaluate.py
"""
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.data import ROOT
from src.engine import MODELS, SearchEngine
from src.evaluation import METRICS, SEARCH_TYPES, TYPE_NAMES, Runner, fuse, load_queries, score
from src.metrics import RECALL_LEVELS, interpolated_precision
from src.plots import INK2, SERIES, use_style

engine = SearchEngine()
RRF_K = engine.params["rrf_k"]
queries = load_queries()
runner = Runner(engine, queries)
test = [i for i, q in enumerate(queries) if q["split"] == "test"]
qtype = [queries[i]["type"] for i in test]
qrels = [queries[i]["qrels"] for i in test]
search_pos = [j for j, t in enumerate(qtype) if t in SEARCH_TYPES]


def evaluate(runs):
    """runs: per test query {model: ranking}. Returns {model: [metric dict per query]}."""
    out = defaultdict(list)
    for run, rel in zip(runs, qrels):
        run["Hybrid RRF"] = fuse(run, k=RRF_K)
        for model in MODELS:
            out[model].append(score(run[model], rel))
    return out


def table(per_query, positions):
    return {m: {k: float(np.mean([per_query[m][j][k] for j in positions])) for k in METRICS} for m in MODELS}


def tables_for(per_query):
    out = {f"All search queries (n={len(search_pos)})": table(per_query, search_pos)}
    for t, name in TYPE_NAMES.items():
        pos = [j for j, x in enumerate(qtype) if x == t]
        out[f"{name} (n={len(pos)})"] = table(per_query, pos)
    return out


# ---- 1. main results (with descriptions) and the metadata-only comparison
runs = runner.base_runs(test)
per_query = evaluate(runs)
tables = tables_for(per_query)
tables_meta_only = tables_for(evaluate(runner.base_runs(test, descriptions=False)))

# ---- 2. ablation on the search queries: which rankers to fuse, and the RRF constant k
def mean_fused(members, k):
    s = [score(fuse(runs[j], members, k), qrels[j]) for j in search_pos]
    return {m: float(np.mean([x[m] for x in s])) for m in ("MAP", "nDCG@10", "Recall@100")}


ablation = {
    f"members (k={RRF_K})": {" + ".join(m): mean_fused(m, RRF_K) for m in
                             [("TF-IDF", "BM25"), ("TF-IDF", "Dense"), ("BM25", "Dense"), ("TF-IDF", "BM25", "Dense")]},
    "k (all three rankers)": {str(k): mean_fused(("TF-IDF", "BM25", "Dense"), k) for k in (1, 10, 30, 60, 100)},
}

# ---- 3. paired t-test on per-query AP: Hybrid RRF against each single model
ap = {m: np.array([per_query[m][j]["MAP"] for j in search_pos]) for m in MODELS}
significance = {}
for m in MODELS[:3]:
    t, p = stats.ttest_rel(ap["Hybrid RRF"], ap[m])
    significance[m] = {"mean_diff": float((ap["Hybrid RRF"] - ap[m]).mean()), "t": float(t), "p": float(p)}

results = {"n_test_queries": len(test), "depth": 1000, "params": engine.params, "tables": tables,
           "tables_metadata_only": tables_meta_only, "ablation": ablation, "significance_vs_hybrid": significance}
(ROOT / "reports" / "results.json").write_text(json.dumps(results, indent=2))


def show(tabs, metrics=METRICS):
    for name, tab in tabs.items():
        print(f"\n{name}\n{'':12}" + "".join(f"{m:>14}" for m in metrics))
        for model, row in tab.items():
            print(f"{model:12}" + "".join(f"{row[m]:14.3f}" for m in metrics))


print(f"params: {engine.params}")
show(tables)
print("\n\n===== WITHOUT descriptions (metadata fields only), MAP / nDCG@10 =====")
show(tables_meta_only, ["MAP", "nDCG@10"])
print("\nAblation:")
for group, rows in ablation.items():
    print(" ", group)
    for name, row in rows.items():
        print(f"    {name:24}" + "  ".join(f"{k} {v:.3f}" for k, v in row.items()))
print("\nHybrid RRF minus model, per-query AP on search queries:")
for m, s in significance.items():
    print(f"    {m:8} diff {s['mean_diff']:+.3f}  t={s['t']:.2f}  p={s['p']:.2g}")

# ---- 4. figures
use_style()
FIG = ROOT / "reports" / "figures"

fig, ax = plt.subplots(figsize=(5.2, 3.6))
for model, color in zip(MODELS, SERIES):
    curve = np.mean([interpolated_precision(runs[j][model], qrels[j]) for j in search_pos], axis=0)
    ax.plot(RECALL_LEVELS, curve, color=color, marker="o", markersize=4, label=model)
ax.set(xlabel="Recall", ylabel="Interpolated precision", xlim=(-0.02, 1.02), ylim=(0, 1),
       title=f"Precision-recall, {len(search_pos)} test search queries")
ax.legend(loc="upper right")
fig.savefig(FIG / "fig3_pr_curve.png")

fig, ax = plt.subplots(figsize=(7.6, 3.4))
width = 0.2
type_tables = list(tables.values())[1:]
for j, (model, color) in enumerate(zip(MODELS, SERIES)):
    vals = [t[model]["nDCG@10"] for t in type_tables]
    bars = ax.bar(np.arange(len(vals)) + (j - 1.5) * width, vals, width * 0.9, color=color, label=model)
    ax.bar_label(bars, fmt="%.2f", fontsize=6, color=INK2, padding=2)
ax.set(xticks=np.arange(len(TYPE_NAMES)), xticklabels=[n.replace(" (", "\n(").replace("based ", "based\n") for n in TYPE_NAMES.values()],
       ylabel="nDCG@10", ylim=(0, 1.2), yticks=np.arange(0, 1.01, 0.2), title="nDCG@10 by query type (test queries)")
ax.grid(axis="x", visible=False)
ax.legend(ncol=4, loc="upper center")
fig.savefig(FIG / "fig4_by_query_type.png")
print("\nsaved figures to", FIG)
