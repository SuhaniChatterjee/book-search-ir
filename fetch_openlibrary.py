"""Fetch descriptions and subjects from Open Library for the most-rated works.

Phase 1 looks up ISBNs in batches (edition record + work key).
Phase 2 fetches each work record, which holds the description and subjects.
Both phases append to JSONL files and skip what is already there, so the
script can be stopped and restarted.

Run:  python fetch_openlibrary.py [n_works]
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from src.data import ROOT, load_catalog

N_WORKS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
OUT = ROOT / "data" / "openlibrary"
OUT.mkdir(parents=True, exist_ok=True)
EDITIONS, WORKS = OUT / "editions.jsonl", OUT / "works.jsonl"
HEADERS = {"User-Agent": "book-search-ir-course-project/0.1 (Book-Crossing IR coursework)"}
lock = threading.Lock()


def get(url, retries=5):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            return json.load(urllib.request.urlopen(req, timeout=90))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(15 * (attempt + 1))  # back off on 429 / 5xx
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None


def text_of(value):
    """Open Library stores text either as a string or as {'value': ...}."""
    if isinstance(value, dict):
        value = value.get("value", "")
    return " ".join(str(value or "").split())


def subject_names(subjects):
    return [s["name"] if isinstance(s, dict) else str(s) for s in subjects or []]


def read_jsonl(path):
    return [json.loads(line) for line in open(path)] if path.exists() else []


def append(path, rows):
    with lock, open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def fetch_editions(batch):
    data = get("https://openlibrary.org/api/books?format=json&jscmd=details&bibkeys="
               + ",".join("ISBN:" + i for i in batch)) or {}
    rows = []
    for isbn in batch:
        d = data.get("ISBN:" + isbn, {}).get("details", {})
        works = d.get("works") or [{}]
        rows.append({"isbn": isbn, "work_key": works[0].get("key"),
                     "description": text_of(d.get("description")), "subjects": subject_names(d.get("subjects"))})
    append(EDITIONS, rows)
    time.sleep(0.5)


def fetch_work(key):
    w = get(f"https://openlibrary.org{key}.json") or {}
    append(WORKS, [{"work_key": key, "description": text_of(w.get("description")),
                    "subjects": subject_names(w.get("subjects"))}])
    time.sleep(0.6)


def run(fn, items, workers, label):
    done, t0 = 0, time.time()
    with ThreadPoolExecutor(workers) as pool:
        for _ in pool.map(fn, items):
            done += 1
            if done % 50 == 0 or done == len(items):
                print(f"{label}: {done}/{len(items)}  ({time.time() - t0:.0f}s)", flush=True)


cat = load_catalog()
# One ISBN per work: its most-rated edition, most popular works first.
best = cat.sort_values("n_ratings", ascending=False).drop_duplicates("work_id").head(N_WORKS)

have = {r["isbn"] for r in read_jsonl(EDITIONS)}
todo = [i for i in best["isbn"] if i not in have]
run(fetch_editions, [todo[i:i + 100] for i in range(0, len(todo), 100)], 2, "editions (batches of 100)")

have = {r["work_key"] for r in read_jsonl(WORKS)}
keys = list(dict.fromkeys(r["work_key"] for r in read_jsonl(EDITIONS) if r["work_key"] and r["work_key"] not in have))
run(fetch_work, keys, 2, "works")
print("finished")
