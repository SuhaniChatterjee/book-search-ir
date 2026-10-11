"""Build the test collection (queries + relevance judgements).

Book-Crossing ships without queries or qrels, so we simulate four query types:

  title    known-item search: the book's title, or title keywords + author surname
  typo     the same known-item titles with spelling mistakes
  author   an author's name; every catalog entry by that author is relevant
  topic    an Open Library subject heading such as "missing persons"; books
           tagged with it are relevant. Subjects are never indexed, so the
           models must find these books from titles and descriptions alone.
  cbf      content-based filtering: three books a user liked form the query,
           the other books that user rated are the (graded) relevant set

Run:  python build_testset.py
"""
import json
import random
from collections import Counter

from src.data import PROCESSED, load_catalog, load_raw
from src.preprocess import STOPWORDS, clean, tokenize

SEED = 42
N_TITLE, N_TYPO, N_AUTHOR, N_CBF = 300, 150, 150, 300

rng = random.Random(SEED)
cat = load_catalog()
docs_of_work = cat.groupby("work_id").indices  # work_id -> array of doc ids
queries = []


def add(qtype, text, qrels, exclude=()):
    queries.append({
        "qid": f"{qtype}-{sum(q['type'] == qtype for q in queries) + 1:03d}",
        "type": qtype, "query": text,
        "qrels": {int(d): int(g) for d, g in qrels.items()},
        "exclude": [int(d) for d in exclude],
    })


def content_words(text):
    return [t for t in tokenize(clean(text)) if t not in STOPWORDS and len(t) >= 3]


def misspell(word):
    """One random edit that keeps the first letter: delete, swap or substitute."""
    i = rng.randrange(1, len(word) - 1)
    op = rng.choice(["delete", "swap", "substitute"])
    if op == "delete":
        return word[:i] + word[i + 1:]
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + rng.choice([c for c in "abcdefghijklmnopqrstuvwxyz" if c != word[i]]) + word[i + 1:]


# ---- known-item pool: works that readers actually rated, with a usable title
work = cat.groupby("work_id").agg(
    title_key=("title_key", "first"), author=("author", "first"),
    author_key=("author_key", "first"), n_ratings=("n_ratings", "sum"),
)
n_content = work["title_key"].map(lambda t: len(content_words(t)))
n_words = work["title_key"].str.split().str.len()
pool = work[(work["n_ratings"] >= 5) & (n_content >= 2) & (n_words <= 8) & (work["author_key"] != "")]
sample = pool.sample(N_TITLE + N_TYPO, random_state=SEED)

for i, (wid, w) in enumerate(sample.iloc[:N_TITLE].iterrows()):
    qrels = {d: 1 for d in docs_of_work[wid]}
    if i % 2 == 0:  # full title
        add("title", w.title_key, qrels)
    else:  # up to two title keywords + author surname
        words = content_words(w.title_key)
        picked = sorted(rng.sample(range(len(words)), min(2, len(words))))
        surname = tokenize(clean(w.author))[-1]
        add("title", " ".join([words[j] for j in picked] + [surname]), qrels)

for wid, w in sample.iloc[N_TITLE:].iterrows():
    words = w.title_key.split()
    eligible = [j for j, t in enumerate(words) if len(t) >= 5 and t.isalpha()]
    if not eligible:
        continue
    for j in rng.sample(eligible, min(len(eligible), 1 if len(words) <= 3 else 2)):
        words[j] = misspell(words[j])
    add("typo", " ".join(words), {d: 1 for d in docs_of_work[wid]})

# ---- author queries: authors with 5-100 entries so Recall@100 can reach 1.0
by_author = cat[cat["author_key"] != ""].groupby("author_key").agg(
    n=("isbn", "size"), n_ratings=("n_ratings", "sum"),
    name=("author", lambda s: Counter(s.str.lower()).most_common(1)[0][0]),
)
authors = by_author[by_author["n"].between(5, 100) & (by_author["n_ratings"] >= 20)]
docs_of_author = cat.groupby("author_key").indices
for akey, a in authors.sample(N_AUTHOR, random_state=SEED).iterrows():
    add("author", " ".join(tokenize(clean(a["name"]))), {d: 1 for d in docs_of_author[akey]})

# ---- topic queries: hand-picked topical subject headings attached to 25-300 works
TOPICS = """vampires, time travel, wizards, witches, dragons, serial murders, missing persons, kidnapping,
orphans, cats, dogs, horses, fairy tales, ghost stories, haunted houses, werewolves, pirates, robots,
artificial intelligence, genetic engineering, space colonies, interplanetary voyages, extraterrestrial beings,
organized crime, mafia, terrorism, spies, assassins, conspiracies, political corruption, lawyers, physicians,
journalists, teachers, artists, actors, rock musicians, scientists, archaeologists, psychiatrists, twins,
mothers and daughters, fathers and sons, brothers and sisters, female friendship, male friendship, first loves,
marriage, divorce, adultery, weddings, pregnancy, adoption, child abuse, alcoholism, drug abuse, suicide, grief,
cancer, mental illness, depression, amnesia, racism, slavery, immigrants, holocaust, concentration camps,
war stories, soldiers, veterans, shipwrecks, wilderness survival, survival, revenge, good and evil,
coming of age, boarding schools, high schools, college students, christmas stories, cooking, gardening,
weight loss, personal finance, parenting, buddhism, christian life, spirituality, mythology, arthurian romances,
knights and knighthood, kings and rulers, princesses, middle ages, horse racing, ranch life, farm life, cowboys,
dystopias, reincarnation, telepathy, psychics, angels, cults, prisoners, death row inmates, epidemics, evolution,
physics, computers, internet, feminism, homeless persons, treasure troves"""
tagged = cat[cat["subjects"].map(len) > 0]
for topic in [t.strip() for t in TOPICS.replace("\n", " ").split(",")]:
    # also accept catalogue variants such as "vampires, fiction" and "vampires in fiction"
    match = lambda subs: any(s == topic or s.startswith(topic + ",") or s == topic + " in fiction" for s in subs)
    add("topic", topic, {d: 1 for d in tagged.index[tagged["subjects"].map(match)]})

# ---- content-based filtering from explicit ratings
_, ratings, _ = load_raw()
ratings["isbn"] = ratings["isbn"].str.strip().str.upper()
r = ratings[ratings["rating"] > 0].merge(cat[["isbn", "work_id"]], on="isbn")
r = r.groupby(["user", "work_id"], as_index=False)["rating"].max()
liked = r[r["rating"] >= 8].groupby("user").size()
total = r.groupby("user").size()
users = sorted(u for u in liked.index if liked[u] >= 8 and 15 <= total[u] <= 200)
grade = lambda x: 3 if x == 10 else 2 if x >= 8 else 1 if x >= 6 else 0

for user in rng.sample(users, N_CBF):
    ur = r[r["user"] == user]
    profile = rng.sample(sorted(ur.loc[ur["rating"] >= 8, "work_id"]), 3)
    text = " ".join(f"{work.title_key[w]} {' '.join(tokenize(clean(work.author[w])))}" for w in profile)
    qrels = {}
    for wid, rating in zip(ur["work_id"], ur["rating"]):
        if wid not in profile and grade(rating):
            qrels.update({d: grade(rating) for d in docs_of_work[wid]})
    add("cbf", text, qrels, exclude=[d for w in profile for d in docs_of_work[w]])

out = PROCESSED / "testset.json"
out.write_text(json.dumps(queries))
print(f"saved {len(queries)} queries -> {out}")
for t in ("title", "typo", "author", "topic", "cbf"):
    qs = [q for q in queries if q["type"] == t]
    n_rel = [sum(g > 0 for g in q["qrels"].values()) for q in qs]
    print(f"\n{t}: {len(qs)} queries, mean relevant docs {sum(n_rel) / len(n_rel):.1f}")
    for q in qs[:3]:
        d = next(iter(q["qrels"]))
        print(f"   {q['query'][:70]!r}  ->  {cat.title[d][:45]} / {cat.author[d]}")
