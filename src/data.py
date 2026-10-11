"""Load and clean the Book-Crossing CSVs and build the searchable catalog."""
import json
import re
from pathlib import Path

import pandas as pd

from .preprocess import clean

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

_CSV = dict(sep=";", encoding="latin-1", quotechar='"', escapechar="\\", dtype=str)

# Titles were UTF-8 encoded twice, e.g. 'ErzÃ?Â¤hlungen' should be 'Erzählungen'.
_mojibake_re = re.compile(r"Ã.Â(.)|Ã([\x80-\xbf])")
_parens_re = re.compile(r"\([^()]*\)")
_non_alnum_re = re.compile(r"[^a-z0-9]+")
_ol_noise_re = re.compile(r"\(?\[[^\]]*\]\[\d+\]\)?|https?://\S+|-{3,}")
OPENLIBRARY = ROOT / "data" / "openlibrary"


def fix_mojibake(text: str) -> str:
    def repair(m):
        try:
            return bytes([0xC3, ord(m.group(1) or m.group(2))]).decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            return m.group(0)

    return _mojibake_re.sub(repair, text)


def title_key(title: str) -> str:
    """Normalised title with edition notes like '(Book 1, Large Print)' removed."""
    t = clean(title)
    while _parens_re.search(t):  # loop handles nested parentheses
        t = _parens_re.sub(" ", t)
    return _non_alnum_re.sub(" ", t).strip()


def author_key(author: str) -> str:
    """'J. K. Rowling' and 'J.K. ROWLING' both become 'jkrowling'."""
    return _non_alnum_re.sub("", clean(author))


def clean_description(text: str) -> str:
    """Drop Open Library's markdown source links and footnotes."""
    text = text.split("[1]:")[0]
    return " ".join(_ol_noise_re.sub(" ", text).split())


def load_openlibrary() -> dict[str, dict]:
    """ISBN -> {'description', 'subjects'}; the work record wins over the edition."""
    if not (OPENLIBRARY / "editions.jsonl").exists():
        return {}
    works = {w["work_key"]: w for w in map(json.loads, open(OPENLIBRARY / "works.jsonl"))}
    out = {}
    for e in map(json.loads, open(OPENLIBRARY / "editions.jsonl")):
        w = works.get(e["work_key"], {})
        subjects = w.get("subjects") or e["subjects"]
        out[e["isbn"]] = {
            "description": clean_description(w.get("description") or e["description"]),
            "subjects": sorted({s.strip().lower() for s in subjects if s.strip()}),
        }
    return out


def load_raw():
    books = pd.read_csv(RAW / "BX-Books.csv", **_CSV).iloc[:, :5]
    books.columns = ["isbn", "title", "author", "year", "publisher"]
    ratings = pd.read_csv(RAW / "BX-Book-Ratings.csv", **_CSV)
    ratings.columns = ["user", "isbn", "rating"]
    ratings["rating"] = ratings["rating"].astype(int)
    users = pd.read_csv(RAW / "BX-Users.csv", **_CSV)
    users.columns = ["user", "location", "age"]
    return books, ratings, users


def build_catalog(books: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """One row per ISBN (= one retrievable document), with a work_id that
    groups different editions of the same book."""
    b = books.copy()
    b["isbn"] = b["isbn"].str.strip().str.upper()
    b = b.drop_duplicates("isbn")
    for col in ("title", "author", "publisher"):
        b[col] = b[col].fillna("").map(fix_mojibake).str.replace("&amp;", "&").str.strip()
    b["year"] = pd.to_numeric(b["year"], errors="coerce").fillna(0).astype(int)
    b.loc[(b["year"] < 1800) | (b["year"] > 2005), "year"] = 0  # 0 = unknown

    b["title_key"] = b["title"].map(title_key)
    b["author_key"] = b["author"].map(author_key)
    b = b[b["title_key"] != ""]
    b["work_id"] = pd.factorize(b["title_key"] + "|" + b["author_key"])[0]

    r = ratings.assign(isbn=ratings["isbn"].str.strip().str.upper())
    explicit = r[r["rating"] > 0].groupby("isbn")["rating"].agg(["size", "mean"])
    b["n_ratings"] = b["isbn"].map(explicit["size"]).fillna(0).astype(int)
    b["avg_rating"] = b["isbn"].map(explicit["mean"]).round(2)

    # Open Library enrichment is fetched per work, so share it across all editions.
    ol = load_openlibrary()
    by_work = {wid: ol[isbn] for isbn, wid in zip(b["isbn"], b["work_id"]) if isbn in ol}
    b["description"] = b["work_id"].map(lambda w: by_work[w]["description"] if w in by_work else "")
    b["subjects"] = b["work_id"].map(lambda w: by_work[w]["subjects"] if w in by_work else [])

    # Two fields: "text" holds title/author/publisher, "description" is indexed
    # separately. Subjects are never indexed: they are the relevance labels
    # for topic queries.
    b["text"] = b["title"] + " " + b["author"] + " " + b["publisher"]
    b["dense_text"] = b["title"] + " by " + b["author"] + ", " + b["publisher"]
    return b.reset_index(drop=True)


def load_catalog() -> pd.DataFrame:
    path = PROCESSED / "catalog.pkl"
    if path.exists():
        return pd.read_pickle(path)
    books, ratings, _ = load_raw()
    catalog = build_catalog(books, ratings)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    catalog.to_pickle(path)
    return catalog
