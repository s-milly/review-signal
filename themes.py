"""Read every review: overall sentiment, plus which themes it mentions and
whether each mention is good (+1) or bad (-1).

    python3 themes.py --raw data/raw     ->  data/work/reviews.parquet
"""
import argparse
import gzip
import json
import os
import re
from multiprocessing import Pool

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

THEMES = {
    "wait": r"\b(wait|waited|waiting|slow|quick|quickly|fast|prompt|promptly|forever|took (?:so )?long|long line|minutes|an hour|seated)\b",
    "service": r"\b(server|servers|waiter|waitress|staff|service|rude|friendly|host|hostess|manager|attentive|ignored)\b",
    "price": r"\b(price|prices|priced|expensive|overpriced|pricey|value|cheap|worth|cost|money)\b",
    "food": r"\b(food|taste|tasted|tasty|delicious|bland|flavor|flavorful|fresh|stale|cold|dish|dishes|meal)\b",
    "clean": r"\b(clean|dirty|filthy|gross|sticky|bathroom|restroom|smell|smelled|bugs?|roach(?:es)?|spotless)\b",
}
PATTERNS = {k: re.compile(v, re.I) for k, v in THEMES.items()}
GENERIC_FOOD = re.compile(r"\b(food|meal|dish|dishes)\b", re.I)
HEDGE = re.compile(r"\b(could be|could use|could have been|needs? (?:to be|some)|wish|should be|not very|not that|not so)\b", re.I)
SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|,\s*|;\s*|\s+but\s+|\s+although\s+|\s+though\s+", re.I)
# restaurant words VADER scores as neutral
EXTRA_WORDS = {"slow": -1.5, "forever": -1.2, "overpriced": -2.0, "pricey": -0.8, "bland": -1.6,
               "stale": -1.6, "soggy": -1.4, "greasy": -1.0, "sticky": -1.0, "lukewarm": -1.0,
               "undercooked": -1.8, "overcooked": -1.4, "inattentive": -1.6, "spotless": 2.0,
               "attentive": 1.6, "fresh": 1.4, "quick": 1.2, "prompt": 1.2, "reasonable": 1.0}
_vader = None


def score(row):
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
        _vader.lexicon.update(EXTRA_WORDS)
    text, stars = row
    out = {"compound": _vader.polarity_scores(text)["compound"]}
    # star rating decides neutral sentences ("we waited 45 minutes" scores neutral)
    lean = -1 if stars <= 2 else (1 if stars >= 4 else 0)
    votes = {k: [] for k in THEMES}
    for sent in SPLIT.split(text):
        hits = [k for k, p in PATTERNS.items() if p.search(sent)]
        # "we waited an hour for our food" is about the wait, not the food
        if "food" in hits and len(hits) > 1 and not PATTERNS["food"].search(GENERIC_FOOD.sub("", sent)):
            hits.remove("food")
        if not hits:
            continue
        c = _vader.polarity_scores(sent)["compound"]
        tone = 1 if c >= 0.05 else (-1 if c <= -0.05 else lean)
        if HEDGE.search(sent) and tone > 0:  # "could be cleaner" is a complaint
            tone = -1
        if tone:
            for k in hits:
                votes[k].append(tone)
    for k, v in votes.items():
        total = sum(v)  # clauses vote; on a tie the last one wins ("pricey but worth it")
        out[k] = (1 if total > 0 else -1) if total else (v[-1] if v else 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/work")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with gzip.open(os.path.join(a.raw, "review.jsonl.gz"), "rt", encoding="utf-8") as f:
        df = pd.DataFrame(json.loads(line) for line in f)
    with Pool() as pool:
        scored = pool.map(score, zip(df["text"], df["stars"]), chunksize=2000)
    df = pd.concat([df.drop(columns=["text"]), pd.DataFrame(scored)], axis=1)
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(os.path.join(a.out, "reviews.parquet"), index=False)
    print(f"{len(df)} reviews scored")
    for k in THEMES:
        m = df[k] != 0
        print(f"  {k:8s} mentioned in {m.mean():5.1%} of reviews, bad in {(df[k] < 0).sum() / max(m.sum(), 1):5.1%} of those")


if __name__ == "__main__":
    main()
