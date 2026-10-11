"""Text preprocessing: cleaning, tokenization, stop-word removal, stemming."""
import re
import unicodedata
from functools import lru_cache

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STOPWORDS = frozenset(stopwords.words("english"))
_stemmer = PorterStemmer()
_token_re = re.compile(r"[a-z0-9]+")


def clean(text: str) -> str:
    """Lowercase, strip accents and HTML entities (BX titles contain '&amp;')."""
    text = str(text).replace("&amp;", "&").lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    return _token_re.findall(text)


@lru_cache(maxsize=None)
def stem(token: str) -> str:
    # Cached: a 270k-book catalog repeats the same words constantly.
    return _stemmer.stem(token)


def preprocess(text: str, remove_stopwords: bool = True, do_stem: bool = True) -> list[str]:
    """Full pipeline: raw string -> list of index terms."""
    tokens = tokenize(clean(text))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if do_stem:
        tokens = [stem(t) for t in tokens]
    return tokens
