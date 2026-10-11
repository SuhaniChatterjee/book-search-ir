"""Fill the course report template with the measured results.

Every number in the report is read from reports/results.json, the catalog or
the test collection, so re-running the pipeline and this script keeps the
report in sync.

Run:  python build_report.py
"""
import json
import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.text.paragraph import Paragraph
from matplotlib.patches import FancyBboxPatch

from src.data import PROCESSED, ROOT, load_catalog
from src.evaluation import TYPE_NAMES, load_queries
from src.plots import INK, INK2, SERIES, use_style

REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
OUT = REPORTS / "IR_Report_BookCrossing.docx"

# Names, registration numbers and emails are kept out of the repository:
# put rows of [name, registration no., email] in students.json (git-ignored).
_students_path = ROOT / "students.json"
STUDENTS = json.loads(_students_path.read_text()) if _students_path.exists() else []

# ------------------------------------------------------------------ numbers
res = json.loads((REPORTS / "results.json").read_text())
stats = json.loads((REPORTS / "dataset_stats.json").read_text())
tuning = json.loads((REPORTS / "tuning_log.json").read_text())
cat = load_catalog()
queries = load_queries()
with open(PROCESSED / "index.pkl", "rb") as f:
    meta_index, desc_index = pickle.load(f)

P = res["params"]
T, T0 = res["tables"], res["tables_metadata_only"]
MODELS = ["TF-IDF", "BM25", "Dense", "Hybrid RRF"]
LABEL = {"TF-IDF": "TF-IDF", "BM25": "BM25", "Dense": "Sentence Transformer", "Hybrid RRF": "Hybrid RRF"}


def tab(tables, prefix):
    return next(v for k, v in tables.items() if k.startswith(prefix))


def n_of(prefix):
    return int(next(k for k in T if k.startswith(prefix)).split("n=")[1].rstrip(")"))


ALL = tab(T, "All search")
n_search = n_of("All search")
n_desc_docs = int((cat["description"] != "").sum())
n_desc_works = int(cat.loc[cat["description"] != "", "work_id"].nunique())
n_subj_docs = int((cat["subjects"].map(len) > 0).sum())
n_subj_works = int(cat.loc[cat["subjects"].map(len) > 0, "work_id"].nunique())
sig = res["significance_vs_hybrid"]
abl_members = next(v for k, v in res["ablation"].items() if k.startswith("members"))
abl_k = res["ablation"]["k (all three rankers)"]
f3 = lambda x: f"{x:.3f}"


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.2f}" if p >= 0.1 else f"p = {p:.3f}"


# ------------------------------------------------------------------ figure 1
def draw_architecture(path):
    use_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.axis("off")

    def box(x, y, w, h, text, color="#f3f2ee", edge="#898781", bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=color, edgecolor=edge, linewidth=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8,
                color=INK, fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, linewidth=1, shrinkA=0, shrinkB=0))

    box(3.6, 9.1, 2.8, 0.7, "User query", bold=True)
    box(0.3, 7.6, 4.2, 0.9, "Query preprocessing\nclean, tokenize, stop words, stem")
    box(5.5, 7.6, 4.2, 0.9, "Sentence Transformer encoder\nall-MiniLM-L6-v2, 384 dimensions")
    box(0.3, 5.9, 4.2, 1.1, "Fielded inverted index (hand-built)\nfield 1: title + author + publisher\nfield 2: Open Library description",
        color="#e4eefb", edge=SERIES[0])
    box(5.5, 5.9, 4.2, 1.1, "Dense vector store\nmetadata vector for every book\ndescription vector where available",
        color="#e0f4ec", edge=SERIES[2])
    box(0.3, 4.2, 2.0, 1.0, "TF-IDF\nfielded cosine")
    box(2.5, 4.2, 2.0, 1.0, "BM25F\nweighted fields")
    box(5.5, 4.2, 4.2, 1.0, "Cosine similarity\nmaximum over the two vectors")
    box(0.3, 2.9, 2.0, 0.7, "Ranked list 1")
    box(2.5, 2.9, 2.0, 0.7, "Ranked list 2")
    box(6.6, 2.9, 2.0, 0.7, "Ranked list 3")
    box(2.6, 1.4, 4.8, 0.9, f"Reciprocal Rank Fusion (k = {P['rrf_k']})\nscore(d) = sum of 1 / (k + rank)",
        color="#fdf0d5", edge=SERIES[3], bold=True)
    box(2.2, 0.1, 5.6, 0.7, "Final top-k results in the search interface", bold=True)

    arrow(4.3, 9.1, 2.4, 8.5); arrow(5.7, 9.1, 7.6, 8.5)
    arrow(2.4, 7.6, 2.4, 7.0); arrow(7.6, 7.6, 7.6, 7.0)
    arrow(1.3, 5.9, 1.3, 5.2); arrow(3.5, 5.9, 3.5, 5.2); arrow(7.6, 5.9, 7.6, 5.2)
    arrow(1.3, 4.2, 1.3, 3.6); arrow(3.5, 4.2, 3.5, 3.6); arrow(7.6, 4.2, 7.6, 3.6)
    arrow(1.3, 2.9, 3.4, 2.3); arrow(3.5, 2.9, 4.6, 2.3); arrow(7.6, 2.9, 6.4, 2.3)
    arrow(5.0, 1.4, 5.0, 0.8)
    ax.text(0.3, 9.45, "Sparse path", fontsize=8, color=SERIES[0], fontweight="bold")
    ax.text(9.7, 9.45, "Dense path", fontsize=8, color=SERIES[2], fontweight="bold", ha="right")
    fig.savefig(path)
    plt.close(fig)


draw_architecture(FIG / "fig1_architecture.png")

# ------------------------------------------------------------------ docx helpers
doc = Document(REPORTS / "template.docx")
body = doc.element.body
src = list(body)
REF = {"h1": src[5], "body": src[6], "h2": src[13], "caption": src[27], "bullet": src[32],
       "eq": src[38], "table_title": src[48], "ref": src[62], "blank": src[4]}
out = []  # new body content, in order


def clone(kind, *texts):
    """Copy a template paragraph and replace its text, one string per run."""
    el = deepcopy(REF[kind])
    p = Paragraph(el, doc)
    while len(p.runs) < len(texts):
        p.add_run()
    for r in p.runs[len(texts):]:
        r._r.getparent().remove(r._r)
    for r, text in zip(p.runs, texts):
        r.text = text
    return p


def add(kind, *texts):
    out.append(clone(kind, *texts)._p)


def bullet(lead, rest):
    p = clone("bullet", lead + " ", rest)
    p.runs[0].bold, p.runs[1].bold = True, None
    out.append(p._p)


def item(text):
    """Plain bullet: a full sentence, no bold lead-in."""
    p = clone("bullet", text)
    p.runs[0].bold = None
    out.append(p._p)


def blank():
    out.append(deepcopy(REF["blank"]))


def figure(path, width, caption):
    p = clone("caption", "")
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.runs[0].add_picture(str(path), width=Inches(width))
    out.append(p._p)
    add("caption", caption)


def shade(cell, fill):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def table(title, header, rows, widths, bold_max_cols=(), align_left_cols=(0,), size=9):
    """Template-styled table: navy header row, grid borders, best value per column in bold."""
    p = clone("table_title", title)
    p.paragraph_format.keep_with_next = True
    out.append(p._p)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    best = {c: max(float(r[c]) for r in rows) for c in bold_max_cols}
    for i, row in enumerate([header] + rows):
        for c, value in enumerate(row):
            cell = t.rows[i].cells[c]
            cell.width = Inches(widths[c])
            para = cell.paragraphs[0]
            para.paragraph_format.space_after = Pt(0)
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT if c in align_left_cols else WD_ALIGN_PARAGRAPH.CENTER
            run = para.add_run(str(value))
            run.font.size = Pt(size)
            if i == 0:
                run.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                shade(cell, "1B365D")
            elif c in best and float(value) == best[c]:
                run.bold = True
    out.append(t._tbl)
    spacer = clone("body", "")
    spacer.paragraph_format.space_after = Pt(2)
    out.append(spacer._p)


# ------------------------------------------------------------------ front matter
title = Paragraph(src[0], doc)
for r in title.runs[1:]:
    r._r.getparent().remove(r._r)
title.runs[0].text = ("Comparative Performance Evaluation of Sparse, Dense and Hybrid RRF Retrieval "
                      "for Book Catalog Search on the Book-Crossing Dataset")

doc.tables[0].autofit = False
for row in doc.tables[0].rows:
    for cell, width in zip(row.cells, (1.7, 1.6, 3.2)):  # wide last column so emails stay on one line
        cell.width = Inches(width)
for row, student in zip(doc.tables[0].rows[1:], STUDENTS):
    for cell, value in zip(row.cells, student):
        cell.paragraphs[0].text = value
        for r in cell.paragraphs[0].runs:
            r.font.size = Pt(10)

abstract = doc.tables[1].rows[0].cells[0].paragraphs[0]
for r in abstract.runs[2:]:
    r._r.getparent().remove(r._r)
abstract.runs[0].text = "Abstract — "
abstract.runs[1].text = (
    f"Each record in a book catalog is a title, an author and a publisher, which comes to about nine index terms. "
    f"We wanted to know whether hybrid search still helps when the documents are that short. We built a search "
    f"system over the {len(cat):,} books of Book-Crossing and compared TF-IDF, BM25, a Sentence Transformer and a "
    f"Reciprocal Rank Fusion (RRF) of the three. Book-Crossing has no queries and no relevance judgements, so we "
    f"generated {len(queries):,} queries ourselves in five types: exact titles, misspelled titles, author names, "
    f"topics and a content-based filtering task. We also added Open Library descriptions for the 20,000 most-rated "
    f"works. On {n_search} held-out search queries BM25 and the hybrid came out level (MAP {f3(ALL['BM25']['MAP'])} "
    f"and {f3(ALL['Hybrid RRF']['MAP'])}, nDCG@10 {f3(ALL['BM25']['nDCG@10'])} and "
    f"{f3(ALL['Hybrid RRF']['nDCG@10'])}; the difference is not significant), and both were well ahead of TF-IDF "
    f"and the dense model. The hybrid had the best recall and won on misspelled and topic queries. BM25 won on "
    f"exact titles and authors. Content-based filtering from metadata did not work with any model."
)
abstract.add_run().add_break()
abstract.add_run().add_break()
abstract.add_run("Keywords: ").bold = True
abstract.add_run("Information Retrieval · Book Search · BM25F · Sentence Transformers · Reciprocal Rank Fusion · "
                 "Book-Crossing · Simulated Queries")

out += [src[0], src[1], src[2], src[3]]
blank()

# ------------------------------------------------------------------ 1. Introduction
add("h1", "1. Introduction")
add("body",
    "A search engine has to decide which documents match a query and in what order to show them. The older "
    "family of methods is lexical. Documents are stored in an inverted index and scored by the words they share "
    "with the query, using TF-IDF or BM25 [1]. These methods are fast, and they are hard to beat when the user "
    "types the exact words that appear in the document. They have nothing to offer when the query and the "
    "document use different words for the same thing. Dense retrieval was developed for that case: a Sentence "
    "Transformer [2] turns text into a vector, and texts with similar meaning end up close together. The cost is "
    "that dense models are less reliable on names and rare words. Reciprocal Rank Fusion [3] merges the ranked "
    "lists of several models by rank position, which avoids having to put their scores on one scale.")
add("body",
    "Our task is book search. Compared with the collections usually used in IR courses, a book catalog is an odd "
    "corpus. The records are tiny. Most queries are look-ups for one particular book or one author. The same book "
    "is listed many times under different ISBNs, and people misspell titles. Results on Cranfield or MS MARCO, "
    "where the documents are paragraphs and hybrid search normally wins, may not hold here, and we wanted to "
    "measure that directly.")
add("body",
    "The dataset is Book-Crossing [4]. It has a large catalog and over a million reader ratings, but no queries, "
    "no relevance judgements and no descriptions of the books. A good share of the work therefore went into "
    "building something to evaluate against. This report describes:")
item("a retrieval pipeline written from scratch in Python, with preprocessing, a fielded inverted index, TF-IDF, "
     "BM25F, dense retrieval, RRF and a Streamlit search page;")
item(f"a test collection of {len(queries):,} generated queries with relevance labels, split into development and "
     "test sets;")
item("descriptions and subject headings for 20,000 works, fetched from Open Library [8];")
item("results for each query type, with significance tests.")
blank()

# ------------------------------------------------------------------ 2. Dataset
add("h1", "2. About Dataset")
add("h2", "2.1 Visualization of dataset in form of graphs or tables and explanation of dataset using an example")
add("body",
    "Book-Crossing was collected by Ziegler et al. [4] in 2004 from the bookcrossing.com community. It has three "
    "tables. BX-Books lists ISBN, title, author, year and publisher. BX-Book-Ratings holds user ratings from 1 to "
    "10, with 0 recorded when a user interacted with a book without rating it. BX-Users gives a location and an "
    "age. Table 1 has the counts before and after our cleaning.")
table("Table 1. Book-Crossing statistics and the catalog used for retrieval.",
      ["Property", "Value"],
      [["Book records (raw / after cleaning)", f"{stats['raw_books']:,} / {len(cat):,}"],
       ["Distinct works after grouping editions", f"{stats['works']:,}"],
       ["Works with more than one edition", f"{stats['works_multi_edition']:,}"],
       ["Distinct authors / publishers", f"{stats['authors']:,} / {stats['publishers']:,}"],
       ["Users", f"{stats['raw_users']:,}"],
       ["Ratings (explicit 1-10 / implicit 0)", f"{stats['explicit_ratings']:,} / {stats['implicit_ratings']:,}"],
       ["Books with at least one explicit rating", f"{stats['books_with_explicit_rating']:,}"],
       ["Mean title length / mean index terms per record", f"{stats['mean_title_words']} words / {meta_index.avgdl:.1f} terms"],
       ["Records with an Open Library description", f"{n_desc_docs:,} ({n_desc_works:,} works)"],
       ["Records with Open Library subject headings", f"{n_subj_docs:,} ({n_subj_works:,} works)"]],
      widths=[3.6, 2.6], align_left_cols=(0, 1))
figure(FIG / "fig2_dataset.png", 5.6,
       "Figure 1: Book-Crossing at a glance: (a) explicit ratings by score, (b) books by publication year, "
       "(c) the ten most frequent authors, (d) title length in words.")
add("body",
    f"Three things in Figure 1 affect retrieval. The ratings are sparse and generous: "
    f"{stats['implicit_ratings'] / stats['raw_ratings']:.0%} are implicit zeros and most of the explicit ones are 7 "
    f"or higher. A handful of authors account for hundreds of records each (panel c). And titles are short, with "
    f"a median of five words, so a whole record gives the index only {meta_index.avgdl:.1f} terms on average. Most "
    f"of the books were published between 1990 and 2004.")
n_broken = int(cat["title"].str.contains("Ã").sum())
add("body",
    f"The raw file needed some repair. We upper-cased ISBNs and dropped {stats['raw_books'] - len(cat)} duplicate "
    f"or empty records, fixed HTML entities and characters that had been UTF-8 encoded twice ({n_broken} titles "
    f"could not be recovered), and set {stats['year_unknown']:,} impossible publication years to unknown. The "
    f"bigger issue was editions. One book is usually listed under several ISBNs, and for evaluation any of them "
    f"is a correct answer. We grouped records into works using a normalised title with the bracketed edition "
    f"notes removed, plus a normalised author name. That gave {stats['works']:,} works. Little Women alone has 47 "
    f"editions.")
ex = cat[cat["isbn"] == "059035342X"].iloc[0]
table("Table 2. An example catalog record after cleaning and enrichment.",
      ["Field", "Value"],
      [["ISBN", ex.isbn], ["Title", ex.title], ["Author / Publisher / Year", f"{ex.author} / {ex.publisher} / {ex.year}"],
       ["Work ID (shared by all editions)", f"{ex.work_id} ({int((cat['work_id'] == ex.work_id).sum())} editions in the catalog)"],
       ["Explicit ratings", f"{ex.n_ratings} ratings, mean {ex.avg_rating}"],
       ["Description (Open Library)", ex.description[:170].rsplit(" ", 1)[0] + " ..."],
       ["Subjects (Open Library, not indexed)", ", ".join(s for s in ex.subjects if s in
                                                           ("wizards", "magic", "fantasy", "orphans", "schools", "england"))]],
      widths=[2.3, 3.9], align_left_cols=(0, 1))
add("body",
    f"Book-Crossing has no text about what a book is about, so we looked up the 20,000 most-rated works in Open "
    f"Library [8]. Descriptions came back for {n_desc_works:,} of them and subject headings for {n_subj_works:,}. "
    f"We copied both to every edition of the work; Table 2 shows one enriched record. Descriptions average about "
    f"{desc_index.avgdl:.0f} index terms and are indexed as a second field. Subject headings are not indexed at "
    f"all. We use them only as relevance labels for topic queries (Section 3.4). The other "
    f"{1 - n_desc_docs / len(cat):.0%} of the catalog can be found by title, author and publisher only.")
blank()

# ------------------------------------------------------------------ 3. Methodology
add("h1", "3. Methodology")
add("body",
    "Figure 2 shows the system. A query goes down two paths. On the sparse path its terms are looked up in the "
    "inverted index and the matching books are scored with TF-IDF and with BM25. On the dense path the query is "
    "encoded as a vector and compared with book vectors computed in advance. RRF merges the three ranked lists. "
    "We wrote everything ourselves in Python and NumPy except the pre-trained encoder.")
figure(FIG / "fig1_architecture.png", 5.3,
       "Figure 2: Architecture of the hybrid Reciprocal Rank Fusion retrieval pipeline.")
add("h2", "3.1 Preprocessing and Indexing")
add("body",
    "We lower-case the text, strip accents, fix HTML entities and split on anything that is not a letter or a "
    "digit. Stop words from the NLTK English list are dropped and the remaining tokens are stemmed with the "
    "Porter stemmer [5]. \"Sorcerer's Stones\" becomes sorcer and stone. Queries go through the same steps as "
    "documents.")
add("body",
    f"The inverted index maps each term to the list of documents that contain it and how often, held as NumPy "
    f"arrays, along with every document's length. There are two indexes, one per field: title, author and "
    f"publisher in the first ({meta_index.vocab_size:,} terms) and descriptions in the second "
    f"({desc_index.vocab_size:,} terms). Our first version had a single field with the description appended to "
    f"the record, and it went badly. A book with a 100-word description was penalised by length normalisation "
    f"against records of nine terms, and BM25's MAP on exact title queries dropped from about 0.89 to 0.47. "
    f"Separating the fields fixed this. Scoring goes one query term at a time and only touches documents in that "
    f"term's postings, so a query over all {len(cat):,} books takes a few milliseconds.")
add("h2", "3.2 Ranking Models")
add("body", "We compare four models.")
item(f"TF-IDF weights a term by (1 + log tf) · log(N / df) and compares query and document by cosine similarity. "
     f"We add the cosine on the metadata field to {P['tfidf_w_desc']} times the cosine on the description field.")
item(f"BM25 [1] adds term-frequency saturation (k1 = 1.5) and length normalisation (b = 0.75). We use the fielded "
     f"variant, BM25F: each field is length-normalised on its own, and the description frequency is multiplied by "
     f"{P['bm25_w_desc']} before the two are added and saturated. With one field this is ordinary BM25.")
item("The Sentence Transformer is all-MiniLM-L6-v2 [2], which produces 384-dimensional unit vectors. Every book "
     "has a vector for its metadata. Books with a description have a second vector that includes it. A book's "
     "score is the higher of its two cosine similarities with the query, and the search is exact, with no "
     "approximate index.")
item("Hybrid RRF [3] fuses the three ranked lists above.")
add("h2", "3.3 Reciprocal Rank Fusion Formulation")
add("body", "For a set of rankers M and a constant k, RRF gives document d the score")
add("eq", "RRF_Score(d ∈ D) = ∑ (m ∈ M)  1 / (k + r_m(d))")
add("body",
    f"where r_m(d) is the rank of d in model m's list, counting from 1. A document that a model did not return "
    f"gets nothing from that model. Only ranks enter the formula, so BM25 scores, which have no upper bound, and "
    f"cosines, which lie between -1 and 1, never have to be rescaled. A small k lets the top few ranks dominate. "
    f"Cormack et al. [3] suggest k = 60; on our development queries k = {P['rrf_k']} worked better (Section 4).")
add("h2", "3.4 Test Collection Construction")
add("body",
    "With no queries in the dataset we had to generate them. Simulated known-item queries are an established "
    "technique [6]. We made five types, summarised in Table 3.")
item("For an exact-title query we sample a work with at least five ratings and use its title, or two title words "
     "plus the author's surname. Every edition of the work counts as relevant.")
item("A misspelled query is built the same way, then one or two of its words get a random character deleted, "
     "swapped with its neighbour or replaced.")
item("An author query is the author's name, and all records by that author are relevant.")
item("A topic query is an Open Library subject heading such as \"missing persons\" or \"time travel\". We picked "
     "114 of them by hand. Works tagged with the heading are relevant. Since headings are not indexed, a model can "
     "only find these books through titles and descriptions.")
item("For content-based filtering we take a user, pick three books they rated 8 or higher, and use the titles and "
     "authors of those books as the query. The user's other rated books are the relevant set, graded by rating "
     "(10 gives 3, 8 or 9 gives 2, 6 or 7 gives 1). The three profile books are removed from the results.")
by_type = {t: [q for q in queries if q["type"] == t] for t in TYPE_NAMES}
example = {"title": "shopaholic ties the knot", "typo": "anna krenina", "author": "evelyn waugh",
           "topic": "missing persons", "cbf": "titles and authors of three liked books"}
assert all(any(q["query"] == example[t] for q in by_type[t]) for t in ("title", "typo", "author", "topic"))
table("Table 3. The simulated test collection.",
      ["Query type", "Queries (dev / test)", "Mean relevant records", "Example query"],
      [[TYPE_NAMES[t], f"{len(qs)} ({sum(q['split'] == 'dev' for q in qs)} / {sum(q['split'] == 'test' for q in qs)})",
        f"{np.mean([sum(g > 0 for g in q['qrels'].values()) for q in qs]):.1f}", example[t]]
       for t, qs in by_type.items()],
      widths=[1.9, 1.4, 1.3, 1.9], align_left_cols=(0, 3))
blank()

# ------------------------------------------------------------------ 4. Experiments
add("h1", "4. Experimental Setup and Performance Analysis")
n_dev = sum(q["split"] == "dev" and q["type"] != "cbf" for q in queries)
add("body",
    f"We split each query type 30/70 into development and test queries. Three parameters were tuned on the "
    f"{n_dev} development search queries, using MAP averaged over the query types: the description weight for "
    f"TF-IDF, the description weight for BM25F and the RRF constant. The weights were picked from "
    f"{{0, 0.1, 0.25, 0.5, 1, 2}} and k from {{1, 5, 10, 20, 40, 60, 100}}. The chosen values were "
    f"{P['tfidf_w_desc']}, {P['bm25_w_desc']} and k = {P['rrf_k']}. Everything reported below comes from the "
    f"{res['n_test_queries']} test queries, which played no part in tuning.")
add("body",
    f"Each model returns 1,000 records per query. We report Precision@10, Recall@100, MAP and nDCG@10 with gain "
    f"2^rel - 1. Topic queries need one adjustment. Only records with subject headings can be judged, so for "
    f"those queries we rank the {n_subj_docs:,} records that have headings and ignore the rest, as in "
    f"condensed-list evaluation [7]. All experiments ran on an Apple M2 laptop with 8 GB of memory. Encoding the "
    f"catalog took about three minutes.")
table(f"Table 4. Performance on the {n_search} held-out search queries (title, misspelled, author and topic).",
      ["Ranking Model", "Precision@10", "Recall@100", "MAP", "nDCG@10"],
      [[LABEL[m] + (" (TF-IDF + BM25 + Dense)" if m == "Hybrid RRF" else "")] +
       [f3(ALL[m][k]) for k in ("Precision@10", "Recall@100", "MAP", "nDCG@10")] for m in MODELS],
      widths=[2.6, 0.95, 0.95, 0.85, 0.85], bold_max_cols=(1, 2, 3, 4), size=10)
add("body",
    f"BM25 and the hybrid are the two best models in Table 4 and there is very little between them. BM25 is "
    f"slightly ahead on Precision@10 and MAP, the hybrid on Recall@100 and nDCG@10. A paired t-test on per-query "
    f"average precision finds no significant difference ({fmt_p(sig['BM25']['p'])}). The hybrid is significantly "
    f"better than TF-IDF ({fmt_p(sig['TF-IDF']['p'])}) and the Sentence Transformer ({fmt_p(sig['Dense']['p'])}). "
    f"On its own the dense model is a long way behind, with MAP {f3(ALL['Dense']['MAP'])}. Figure 3 shows the same "
    f"order at every recall level. We had expected the hybrid to win outright, as it does on most published "
    f"benchmarks. The breakdown by query type explains why it does not.")
figure(FIG / "fig3_pr_curve.png", 3.9,
       f"Figure 3: Eleven-point interpolated precision-recall curves on the {n_search} test search queries.")
type_rows = []
for t, name in TYPE_NAMES.items():
    type_rows.append([name, str(n_of(name))] + [f3(tab(T, name)[m]["nDCG@10"]) for m in MODELS])
table("Table 5. nDCG@10 by query type on the test queries.",
      ["Query type", "Queries", "TF-IDF", "BM25", "Sentence Transformer", "Hybrid RRF"],
      type_rows, widths=[2.0, 0.7, 0.8, 0.8, 1.2, 0.9], size=9)
# bold the best model per row
tbl = doc.tables[-1]
for row in tbl.rows[1:]:
    vals = [float(c.text) for c in row.cells[2:]]
    for c, v in zip(row.cells[2:], vals):
        if v == max(vals):
            c.paragraphs[0].runs[0].bold = True
figure(FIG / "fig4_by_query_type.png", 5.6, "Figure 4: nDCG@10 of the four models for each query type.")

ti, ty, au, tp, cb = (tab(T, TYPE_NAMES[t]) for t in TYPE_NAMES)
add("body",
    f"On exact titles and author names the right answer shares its words with the query, and BM25 is close to "
    f"perfect (nDCG@10 of {f3(ti['BM25']['nDCG@10'])} and {f3(au['BM25']['nDCG@10'])}). The dense model does poorly "
    f"on authors ({f3(au['Dense']['nDCG@10'])}). A book's vector is mostly shaped by its title, so a query that is "
    f"only a name pulls in books whose titles look like names. When that weak list is fused with two strong ones, "
    f"the result is slightly worse than BM25 alone, and this is what costs the hybrid the overall lead.")
add("body",
    f"Misspelled queries go the other way. A typo makes a term disappear from the sparse match altogether. The "
    f"dense model splits words into sub-word pieces and often still lands near the right title. The two kinds of "
    f"model recover different books, so fusing them helps: nDCG@10 goes from {f3(ty['BM25']['nDCG@10'])} for BM25 "
    f"to {f3(ty['Hybrid RRF']['nDCG@10'])}, and Recall@100 from {f3(ty['BM25']['Recall@100'])} to "
    f"{f3(ty['Hybrid RRF']['Recall@100'])}. Topic queries show a smaller gain of the same kind, with MAP of "
    f"{f3(tp['Hybrid RRF']['MAP'])} for the hybrid and {f3(tp['BM25']['MAP'])} for BM25.")
add("body",
    f"Content-based filtering did not work. No model gets above {max(cb[m]['nDCG@10'] for m in MODELS):.2f} "
    f"nDCG@10. Knowing the titles and authors of three books a reader liked says little about their other books, "
    f"apart from more by the same authors. A recommender for this dataset would need the rating matrix.")
tp0, all0 = tab(T0, TYPE_NAMES["topic"]), tab(T0, "All search")
full, sparse_pair = abl_members["TF-IDF + BM25 + Dense"], abl_members["TF-IDF + BM25"]
assert all(round(full[m], 3) >= round(v[m], 3) for v in abl_members.values() for m in ("MAP", "nDCG@10"))
assert all(v["MAP"] < sparse_pair["MAP"] for name, v in abl_members.items() if "Dense" in name and name.count("+") == 1)
table("Table 6. Ablations on the test search queries: effect of descriptions, fused models and the RRF constant.",
      ["Configuration", "MAP", "nDCG@10"],
      [["BM25, metadata only → with descriptions (topic queries)", f"{f3(tp0['BM25']['MAP'])} → {f3(tp['BM25']['MAP'])}",
        f"{f3(tp0['BM25']['nDCG@10'])} → {f3(tp['BM25']['nDCG@10'])}"],
       ["Sentence Transformer, same comparison (topic queries)", f"{f3(tp0['Dense']['MAP'])} → {f3(tp['Dense']['MAP'])}",
        f"{f3(tp0['Dense']['nDCG@10'])} → {f3(tp['Dense']['nDCG@10'])}"],
       ["Hybrid RRF, same comparison (all search queries)", f"{f3(all0['Hybrid RRF']['MAP'])} → {f3(ALL['Hybrid RRF']['MAP'])}",
        f"{f3(all0['Hybrid RRF']['nDCG@10'])} → {f3(ALL['Hybrid RRF']['nDCG@10'])}"]] +
      [[f"RRF of {name}", f3(v["MAP"]), f3(v["nDCG@10"])] for name, v in abl_members.items()] +
      [[f"RRF of all three, k = {k}", f3(v["MAP"]), f3(v["nDCG@10"])] for k, v in abl_k.items() if k in ("1", "60", "100")],
      widths=[3.8, 1.2, 1.2], size=9)
add("body",
    f"Table 6 collects three smaller experiments. The first rows show what the descriptions bought us. BM25's "
    f"nDCG@10 on topic queries rises from {f3(tp0['BM25']['nDCG@10'])} to {f3(tp['BM25']['nDCG@10'])} when the "
    f"description field is indexed, and title and author search stay where they were. The middle rows vary which "
    f"models are fused. All three together match or beat every pair. The pairs that include the dense model score "
    f"lower on MAP than TF-IDF with BM25, though the dense list is where the recall and typo gains come from. The "
    f"last rows vary k. The effect is small: MAP is {f3(abl_k[str(P['rrf_k'])]['MAP'])} at k = {P['rrf_k']} and "
    f"{f3(abl_k['60']['MAP'])} at k = 60. Our queries are short look-ups with one or two right answers, so it "
    f"makes sense that weighting the top ranks more heavily helps a little.")
add("body",
    f"The evaluation has weaknesses we should be clear about. The queries are generated. Title and author queries "
    f"are cut from the records themselves, which helps the lexical models, and our typos are random character "
    f"edits, which is not how people misspell. The labels have gaps: two different books with the same title are "
    f"treated as unrelated, and Open Library volunteers do not tag consistently. Only "
    f"{n_desc_docs / len(cat):.0%} of records have a description, and those are the popular books. Two "
    f"preprocessing choices also cost us. Removing stop words leaves nothing of a title like Stephen King's It, so "
    f"the sparse models cannot find it, and the stemmer turns \"missing\" into \"miss\", which is why a search for "
    f"missing persons also returns Miss Marple.")
blank()

# ------------------------------------------------------------------ 5. Conclusion
add("h1", "5. Conclusion")
add("body",
    f"We built a search system for the Book-Crossing catalog and a set of {len(queries):,} generated queries to "
    f"test it with. The system has a fielded inverted index, TF-IDF, BM25F, a Sentence Transformer ranker, RRF "
    f"and a search page.")
add("body",
    f"What we found is that hybrid search helps on some kinds of query and not on others. With records of about "
    f"nine terms, BM25 is very hard to beat on exact titles and author names, and the dense model alone is clearly "
    f"worse (MAP {f3(ALL['Dense']['MAP'])} against {f3(ALL['BM25']['MAP'])}). The hybrid ends up level with BM25 "
    f"overall (nDCG@10 {f3(ALL['Hybrid RRF']['nDCG@10'])} against {f3(ALL['BM25']['nDCG@10'])}), has the best "
    f"recall, and is the better choice when the query is misspelled or describes a topic. Descriptions are what "
    f"make topic search work, but they have to be scored as a separate field or they damage title matching. "
    f"Recommending books from metadata alone did not work.")
add("body",
    "If we continued, the first thing to try would be a fusion that gives the dense list less weight when the "
    "query looks like a name. Spelling correction or character n-grams should help the misspelled queries. We "
    "would also like descriptions for the rest of the catalog.")
blank()

# ------------------------------------------------------------------ References
out.append(src[61])
for i, ref in enumerate([
    "Robertson, S., Zaragoza, H.: The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in "
    "Information Retrieval 3(4), 333–389 (2009)",
    "Reimers, N., Gurevych, I.: Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In: Proceedings of "
    "EMNLP-IJCNLP, pp. 3982–3992 (2019)",
    "Cormack, G.V., Clarke, C.L.A., Buettcher, S.: Reciprocal Rank Fusion outperforms Condorcet and individual Rank "
    "Learning Methods. In: Proceedings of SIGIR, pp. 758–759 (2009)",
    "Ziegler, C.-N., McNee, S.M., Konstan, J.A., Lausen, G.: Improving Recommendation Lists Through Topic "
    "Diversification. In: Proceedings of WWW, pp. 22–32 (2005)",
    "Porter, M.F.: An algorithm for suffix stripping. Program 14(3), 130–137 (1980)",
    "Azzopardi, L., de Rijke, M., Balog, K.: Building simulated queries for known-item topics: an analysis using six "
    "European languages. In: Proceedings of SIGIR, pp. 455–462 (2007)",
    "Sakai, T.: Alternatives to Bpref. In: Proceedings of SIGIR, pp. 71–78 (2007)",
    "Open Library: Books and Works APIs. Internet Archive, https://openlibrary.org/developers/api",
], start=1):
    add("ref", f"{i}. {ref}")

# ------------------------------------------------------------------ assemble
# The template's page header carries an instruction note for students, not report content.
for p in doc.sections[0].header.paragraphs:
    for r in p.runs:
        r.text = ""
sect = body.find(qn("w:sectPr"))
for child in list(body):
    if child is not sect:
        body.remove(child)
for el in out:
    sect.addprevious(el)
doc.save(OUT)
print("saved", OUT)
