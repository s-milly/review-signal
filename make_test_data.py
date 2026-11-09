"""Make FAKE data in the same format shrink_yelp.py writes.

This is only for building and testing the pipeline. Never report results
from it: the patterns are planted on purpose, so the model will look better
on it than on real data.

    python3 make_test_data.py   ->  data/test/*.jsonl.gz
"""
import gzip
import json
import os
import sys

import numpy as np

rng = np.random.default_rng(7)
LEAD = int(os.environ.get("LEAD", "2"))  # months between a quality change and traffic reacting
OUT = os.environ.get("OUT", "data/test")
THEMES = ["wait", "service", "price", "food", "clean"]
GOOD = {
    "wait": ["we were seated right away", "food came out fast"],
    "service": ["our server was so friendly", "the staff were great"],
    "price": ["great value for the money", "fair prices"],
    "food": ["the food was delicious", "best tacos I have had", "everything was fresh"],
    "clean": ["the place was spotless", "very clean dining room"],
}
BAD = {
    "wait": ["we waited 45 minutes for a table", "the food took forever", "slow, long wait"],
    "service": ["our server was rude", "the staff ignored us", "terrible service"],
    "price": ["way overpriced", "too expensive for what you get"],
    "food": ["the food was cold and bland", "my order was wrong", "tasted stale"],
    "clean": ["the tables were dirty", "the bathroom was gross"],
}
NAMES = ["Corner", "Golden", "Little", "Blue", "Urban", "Old Town", "Sunny", "Red Door", "Harbor", "Maple"]
KINDS = ["Grill", "Kitchen", "Diner", "Taqueria", "Pizza", "Noodle Bar", "Cafe", "Bistro", "BBQ", "Deli"]
months = [f"{y}-{m:02d}" for y in range(2014, 2022) for m in range(1, 13)]


def main():
    os.makedirs(OUT, exist_ok=True)
    biz, revs, chk = [], [], []
    for i in range(400):
        bid = f"test{i:04d}"
        base = rng.normal(0, 1, len(THEMES))          # each theme's normal quality
        size = rng.lognormal(3.0, 0.5)                 # typical monthly check-ins
        start = int(rng.integers(0, 30))
        shock = {}                                     # a theme that goes bad at some point
        if rng.random() < 0.5:
            shock = {"theme": int(rng.integers(len(THEMES))), "at": int(rng.integers(start + 12, len(months) - 6)),
                     "size": rng.uniform(1.5, 3.5)}
        traffic, closed_at, q_hist = size, None, []
        months_kept = {}
        for t in range(start, len(months)):
            q = base.copy()
            if shock and t >= shock["at"]:
                q[shock["theme"]] -= shock["size"] * min(1, (t - shock["at"] + 1) / 4)
            q += rng.normal(0, 0.3, len(THEMES))
            q_hist.append(q.mean())
            # traffic follows quality a few months later, plus seasonality and noise
            lagged = np.mean(q_hist[-(LEAD + 4):-LEAD]) if len(q_hist) > LEAD + 1 else q.mean()
            season = 1 + 0.15 * np.sin(2 * np.pi * (t % 12) / 12)
            traffic = 0.7 * traffic + 0.3 * size * np.exp(0.8 * lagged) * season
            n_chk = int(rng.poisson(max(traffic, 0.1)))
            if n_chk:
                months_kept[months[t]] = n_chk
            for _ in range(int(rng.poisson(max(traffic, 0.5) * 0.12))):
                k = int(rng.integers(len(THEMES)))
                good = rng.random() < 1 / (1 + np.exp(-1.2 * q[k]))
                stars = int(np.clip(round(rng.normal(4.4 if good else 1.8, 0.6)), 1, 5))
                text = rng.choice(GOOD[THEMES[k]] if good else BAD[THEMES[k]])
                extra = THEMES[int(rng.integers(len(THEMES)))]
                text = f"{text}. {rng.choice(GOOD[extra] if rng.random() < 0.6 else BAD[extra])}."
                revs.append({"business_id": bid, "stars": stars,
                             "date": f"{months[t]}-{int(rng.integers(1, 29)):02d}", "useful": 0, "text": text.capitalize()})
            normal = size * np.exp(0.8 * base.mean())
            if t > start + 18 and traffic < normal * 0.7 and rng.random() < 0.15:
                closed_at = t
                break
        n_rev = sum(1 for r in revs if r["business_id"] == bid)
        biz.append({"business_id": bid, "name": f"{rng.choice(NAMES)} {rng.choice(KINDS)} (test)",
                    "address": "", "city": "Testville", "state": "PA", "latitude": 39.95, "longitude": -75.16,
                    "stars": 3.5, "review_count": n_rev, "is_open": 0 if closed_at else 1,
                    "categories": "Restaurants"})
        chk.append({"business_id": bid, "months": months_kept})
    for name, rows in (("business", biz), ("review", revs), ("checkin_monthly", chk)):
        with gzip.open(f"{OUT}/{name}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"fake data: {len(biz)} businesses, {len(revs)} reviews, {sum(b['is_open'] == 0 for b in biz)} closed")


if __name__ == "__main__":
    main()
