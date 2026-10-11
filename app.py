"""Streamlit search interface.

Run:  streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st

from src.data import ROOT
from src.engine import MODELS, SearchEngine
from src.preprocess import clean, preprocess, tokenize

st.set_page_config(page_title="Book-Crossing Search", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading catalog, index and embeddings…")
def get_engine():
    return SearchEngine()


engine = get_engine()
cat = engine.catalog

st.title("Book-Crossing Search")
n_desc = int((cat["description"] != "").sum())
st.caption(f"{len(cat):,} books ({n_desc:,} with an Open Library description) · {engine.index.vocab_size:,} index terms · "
           "TF-IDF, BM25, Sentence Transformer and Hybrid RRF")

with st.sidebar:
    st.header("Settings")
    top_n = st.slider("Results per model", 5, 20, 10)
    rrf_k = st.slider("RRF constant k", 1, 100, engine.params["rrf_k"])
    st.caption("Smaller k rewards top ranks more; larger k flattens the difference between ranks.")

search_tab, eval_tab = st.tabs(["Search", "Evaluation"])

with search_tab:
    query = st.text_input("Search by title, author, publisher or topic", "harry potter")
    if query.strip():
        results = engine.search_all(query, rrf_k=rrf_k)
        ranks = {m: {d: i for i, (d, _) in enumerate(r, 1)} for m, r in results.items()}

        with st.expander("How the query is processed"):
            q_terms = preprocess(query)
            st.write("**Tokens:**", tokenize(clean(query)))
            st.write("**After stop-word removal and stemming:**", q_terms)
            rows = [
                {"term": t, "books containing it (df)": engine.bm25.df(t), "BM25 idf": round(engine.bm25.idf(t), 3)}
                for t in dict.fromkeys(q_terms)
            ]
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True)
                st.caption("Rare terms get a high idf, so they count for more in the sparse scores.")
            else:
                st.warning("Every query word is a stop word, so TF-IDF and BM25 have nothing to match. Only the dense model can answer.")

        cols = st.columns(len(MODELS))
        for col, model in zip(cols, MODELS):
            with col:
                st.subheader(model)
                if not results[model]:
                    st.write("No results")
                for rank, (doc, score) in enumerate(results[model][:top_n], 1):
                    b = cat.iloc[doc]
                    year = f" · {b.year}" if b.year else ""
                    st.markdown(f"**{rank}. {b.title}**  \n{b.author}{year}")
                    if model == "Hybrid RRF":
                        src = " · ".join(f"{m} #{ranks[m][doc]}" for m in MODELS[:3] if doc in ranks[m])
                        st.caption(f"RRF {score:.4f} — {src}")
                    else:
                        st.caption(f"score {score:.3f}")
                    if b.description:
                        with st.expander("Description"):
                            st.write(b.description[:600] + ("…" if len(b.description) > 600 else ""))

with eval_tab:
    res_path = ROOT / "reports" / "results.json"
    if not res_path.exists():
        st.info("Run `python evaluate.py` to produce the evaluation results.")
    else:
        res = json.loads(res_path.read_text())
        for name, table in res["tables"].items():
            st.subheader(name)
            st.dataframe(pd.DataFrame(table).T.style.format("{:.3f}").highlight_max(axis=0, props="font-weight:bold"))
        st.caption(f"Held-out test queries only. Tuned on the dev split: {res['params']}")
        for fig in ("fig3_pr_curve.png", "fig4_by_query_type.png"):
            if (ROOT / "reports" / "figures" / fig).exists():
                st.image(str(ROOT / "reports" / "figures" / fig))
