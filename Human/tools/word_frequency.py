# file: scripts/wordfreq_from_csv.py
"""Fast word-frequency analysis from an already aggregated CSV.

Input CSV must contain the text columns to analyze (default: keyFactors, strategy).
This script is optimized for speed using scikit-learn's CountVectorizer. It falls back
to a lightweight regex counter if scikit-learn is unavailable.
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

# ---- Optional fast path dependencies ----
try:
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    _HAS_SKLEARN = True
except Exception:
    CountVectorizer = None  # type: ignore
    ENGLISH_STOP_WORDS = frozenset()
    _HAS_SKLEARN = False

try:
    from nltk.stem import SnowballStemmer
    _HAS_NLTK = True
    _STEMMER = SnowballStemmer("english")
except Exception:
    _HAS_NLTK = False
    _STEMMER = None

# ---- Tokenization helpers ----
_WORD_RE = re.compile(r"[A-Za-z]{2,}")  # skip 1-char & digits for quality/speed


def _regex_tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _custom_analyzer_factory(use_stem: bool, extra_stop: Iterable[str]):
    stop = set(ENGLISH_STOP_WORDS) | {s.lower() for s in extra_stop}
    stemmer = _STEMMER if (use_stem and _HAS_NLTK) else None

    def analyze(doc: str) -> List[str]:
        toks = _regex_tokenize(doc or "")
        if stemmer is not None:
            # Why: stemming is slower; only enabled when asked
            toks = [stemmer.stem(t) for t in toks]
        return [t for t in toks if t not in stop]

    return analyze


# ---- Counting ----

def fast_count_sklearn(texts: List[str], use_stem: bool, extra_stop: Iterable[str]) -> List[Tuple[str, int]]:
    if use_stem:
        analyzer = _custom_analyzer_factory(True, extra_stop)
        vectorizer = CountVectorizer(analyzer=analyzer)
    else:
        vectorizer = CountVectorizer(
            lowercase=True,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
        )
    X = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    counts = X.sum(axis=0).A1
    items = list(zip(vocab, counts))
    items.sort(key=lambda x: x[1], reverse=True)
    return items


def fast_count_fallback(texts: List[str], use_stem: bool, extra_stop: Iterable[str]) -> List[Tuple[str, int]]:
    from collections import Counter

    stop = set(ENGLISH_STOP_WORDS) | {s.lower() for s in extra_stop}
    stemmer = _STEMMER if (use_stem and _HAS_NLTK) else None

    counter: Counter[str] = Counter()
    for doc in texts:
        toks = _regex_tokenize(doc or "")
        if stemmer is not None:
            toks = [stemmer.stem(t) for t in toks]
        toks = [t for t in toks if t not in stop]
        counter.update(toks)
    return counter.most_common()


# ---- I/O ----

def read_aggregated_csv(path: str, columns: List[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")
    try:
        df = pd.read_csv(path, usecols=columns, dtype=str, engine="pyarrow")
    except Exception:
        df = pd.read_csv(path, usecols=columns, dtype=str)
    # Use pandas StringDtype for memory efficiency
    return df.astype({col: "string" for col in columns})


def compute_and_write(
    df: pd.DataFrame,
    columns: List[str],
    out_prefix: str,
    use_stem: bool,
    extra_stop: Iterable[str],
    topk: Optional[int],
    min_count: int,
) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for col in columns:
        texts = df[col].fillna("").astype(str).tolist()
        if _HAS_SKLEARN:
            counts = fast_count_sklearn(texts, use_stem=use_stem, extra_stop=extra_stop)
        else:
            counts = fast_count_fallback(texts, use_stem=use_stem, extra_stop=extra_stop)

        if min_count > 1:
            counts = [(w, c) for (w, c) in counts if c >= min_count]
        if topk is not None and topk > 0:
            counts = counts[:topk]

        out_path = f"{out_prefix}_{col}.csv"
        pd.DataFrame(counts, columns=["word", "frequency"]).to_csv(out_path, index=False)
        summary[col] = sum(c for _, c in counts)
    return summary


# ---- CLI ----

def parse_columns(value: str) -> List[str]:
    # Accept comma-separated or repeated flags handled by nargs="+"
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--columns must specify at least one column name")
    return parts


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Fast word frequency from an aggregated CSV.")
    parser.add_argument("input_csv", help="Aggregated CSV path (already merged).")
    parser.add_argument(
        "--columns",
        type=parse_columns,
        default=parse_columns("keyFactors,strategy"),
        help="Comma-separated column names to analyze (default: keyFactors,strategy).",
    )
    parser.add_argument("--out-prefix", default="word_freq", help="Prefix for frequency CSV outputs.")
    parser.add_argument("--stem", action="store_true", help="Enable stemming (slower, optional).")
    parser.add_argument("--stop", nargs="*", default=[], help="Extra stopwords to exclude.")
    parser.add_argument("--topk", type=int, default=None, help="Keep only top-K tokens per column.")
    parser.add_argument("--min-count", type=int, default=1, help="Drop tokens with freq < min-count.")

    args = parser.parse_args(argv)

    # Validate columns exist
    # Read only once to keep it fast
    try:
        df = read_aggregated_csv(args.input_csv, args.columns)
    except ValueError as e:
        # If any column missing, re-read without usecols to print choices
        full = pd.read_csv(args.input_csv, nrows=5)
        have = ", ".join(full.columns.astype(str).tolist())
        raise SystemExit(f"Column missing. Available columns: {have}\nOriginal error: {e}")

    summary = compute_and_write(
        df,
        columns=args.columns,
        out_prefix=args.out_prefix,
        use_stem=bool(args.stem),
        extra_stop=args.stop,
        topk=args.topk,
        min_count=args.min_count,
    )

    print("Done.")
    for col, total in summary.items():
        print(f"  {col}: {total} kept tokens (after filters)")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])