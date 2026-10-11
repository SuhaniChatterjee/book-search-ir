"""Exploratory data analysis: prints dataset statistics and saves Figure 2 of the report.

Run:  python eda.py
"""
import json

import matplotlib.pyplot as plt

from src.data import ROOT, load_catalog, load_raw
from src.plots import SERIES, use_style
from src.preprocess import preprocess

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

books, ratings, users = load_raw()
cat = load_catalog()
explicit = ratings[ratings["rating"] > 0]

stats = {
    "raw_books": len(books), "raw_ratings": len(ratings), "raw_users": len(users),
    "catalog_books": len(cat), "works": int(cat["work_id"].nunique()),
    "works_multi_edition": int((cat.groupby("work_id").size() > 1).sum()),
    "authors": int(cat["author_key"].nunique()), "publishers": int(cat["publisher"].nunique()),
    "implicit_ratings": int((ratings["rating"] == 0).sum()), "explicit_ratings": len(explicit),
    "books_with_explicit_rating": int((cat["n_ratings"] > 0).sum()),
    "year_unknown": int((cat["year"] == 0).sum()),
    "mean_title_words": round(float(cat["title"].str.split().str.len().mean()), 2),
    "mean_doc_terms": round(float(cat["text"].head(20000).map(lambda t: len(preprocess(t))).mean()), 2),
}
print(json.dumps(stats, indent=2))
(ROOT / "reports" / "dataset_stats.json").write_text(json.dumps(stats, indent=2))

use_style()
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.2))
blue = SERIES[0]

rc = explicit["rating"].value_counts().sort_index()
ax[0, 0].bar(rc.index, rc.values / 1000, color=blue, width=0.75)
ax[0, 0].set(title="(a) Explicit ratings by score", xlabel="Rating", ylabel="Ratings (thousands)", xticks=range(1, 11))
ax[0, 0].grid(axis="x", visible=False)

yrs = cat.loc[cat["year"] >= 1950, "year"]
ax[0, 1].hist(yrs, bins=range(1950, 2007), color=blue)
ax[0, 1].set(title="(b) Books by publication year", xlabel="Year", ylabel="Books")
ax[0, 1].grid(axis="x", visible=False)

top = cat[cat["author_key"] != ""].groupby("author_key").agg(n=("isbn", "size"), name=("author", "first"))
top = top.sort_values("n").tail(10)
ax[1, 0].barh(top["name"].str.title(), top["n"], color=blue, height=0.7)
ax[1, 0].set(title="(c) Ten most frequent authors", xlabel="Catalog entries")
ax[1, 0].grid(axis="y", visible=False)

tl = cat["title"].str.split().str.len().clip(upper=20)
ax[1, 1].hist(tl, bins=range(1, 22), color=blue, rwidth=0.85, align="left")
ax[1, 1].set(title="(d) Title length", xlabel="Words in title (20 = 20 or more)", ylabel="Books",
             xticks=range(2, 21, 2))
ax[1, 1].grid(axis="x", visible=False)

fig.tight_layout()
fig.savefig(FIG / "fig2_dataset.png")
print("saved", FIG / "fig2_dataset.png")
