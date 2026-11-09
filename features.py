"""One row per business per month (the "forecast date"), holding only what
was known by the end of that month, plus what happened next.

    python3 features.py --raw data/raw --work data/work  ->  data/work/panel.parquet

Feature groups (train.py compares models with and without the review group):
  hist_*  past traffic, season and age
  rev_*   review volume, stars, sentiment and complaint themes
Targets:
  y_traffic  log(1 + traffic over the next 3 months)
  y_close    1 if the business closed within the next 12 months
"""
import argparse
import gzip
import json
import os

import numpy as np
import pandas as pd

from themes import THEMES

SMOOTH = 5  # reviews of "prior" mixed into small samples so 1 bad review is not 100%


def load_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def roll(g, col, n, shift=0):
    return g[col].transform(lambda s: s.shift(shift).rolling(n, min_periods=1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--work", default="data/work")
    ap.add_argument("--traffic", choices=["checkins", "reviews"], default="checkins",
                    help="what stands in for customer traffic")
    a = ap.parse_args()

    biz = pd.DataFrame(load_jsonl(os.path.join(a.raw, "business.jsonl.gz")))
    rev = pd.read_parquet(os.path.join(a.work, "reviews.parquet"))
    rev["month"] = rev["date"].dt.to_period("M")

    chk_rows = [(c["business_id"], m, n) for c in load_jsonl(os.path.join(a.raw, "checkin_monthly.jsonl.gz"))
                for m, n in c["months"].items()]
    chk = pd.DataFrame(chk_rows, columns=["business_id", "month", "checkins"])
    chk["month"] = pd.PeriodIndex(chk["month"], freq="M")

    # monthly review totals
    agg = {"n": ("stars", "size"), "stars": ("stars", "sum"), "compound": ("compound", "sum")}
    for k in THEMES:
        rev[f"{k}_bad"] = (rev[k] < 0).astype(int)
        rev[f"{k}_good"] = (rev[k] > 0).astype(int)
        agg[f"{k}_bad"] = (f"{k}_bad", "sum")
        agg[f"{k}_good"] = (f"{k}_good", "sum")
    monthly = rev.groupby(["business_id", "month"]).agg(**agg).reset_index()

    # full month grid from each business's first activity to the end of the data
    end = max(monthly["month"].max(), chk["month"].max())
    first = pd.concat([monthly[["business_id", "month"]], chk[["business_id", "month"]]]).groupby("business_id")["month"].min()
    last = pd.concat([monthly[["business_id", "month"]], chk[["business_id", "month"]]]).groupby("business_id")["month"].max()
    grid = pd.concat([pd.DataFrame({"business_id": b, "month": pd.period_range(m, end, freq="M")})
                      for b, m in first.items()], ignore_index=True)
    p = grid.merge(monthly, on=["business_id", "month"], how="left").merge(chk, on=["business_id", "month"], how="left")
    p = p.fillna(0).sort_values(["business_id", "month"]).reset_index(drop=True)
    p["traffic"] = p["checkins"] if a.traffic == "checkins" else p["n"]
    g = p.groupby("business_id")

    # history group
    p["hist_t3"] = np.log1p(roll(g, "traffic", 3))
    p["hist_t3_prev"] = np.log1p(roll(g, "traffic", 3, shift=3))
    p["hist_t12"] = np.log1p(roll(g, "traffic", 12) / 4)  # per 3 months, comparable to hist_t3
    p["hist_t3_lastyear"] = np.log1p(roll(g, "traffic", 3, shift=9))  # same season a year ago, next 3 months
    p["hist_trend"] = p["hist_t3"] - p["hist_t12"]
    p["hist_month"] = p["month"].dt.month
    p["hist_age"] = g.cumcount()

    # review group: last 3 months vs the 12 before that
    n3, n12 = roll(g, "n", 3), roll(g, "n", 12, shift=3)
    p["rev_n3"] = np.log1p(n3)
    p["rev_n_trend"] = np.log1p(n3) - np.log1p(n12 / 4)
    star_prior = rev["stars"].mean()
    s3 = (roll(g, "stars", 3) + SMOOTH * star_prior) / (n3 + SMOOTH)
    s12 = (roll(g, "stars", 12, shift=3) + SMOOTH * star_prior) / (n12 + SMOOTH)
    p["rev_stars3"], p["rev_stars_delta"] = s3, s3 - s12
    cs = g["stars"].cumsum(); cn = g["n"].cumsum()
    p["rev_stars_all"] = (cs + SMOOTH * star_prior) / (cn + SMOOTH)
    c3 = roll(g, "compound", 3) / (n3 + SMOOTH)
    p["rev_sent3"], p["rev_sent_delta"] = c3, c3 - roll(g, "compound", 12, shift=3) / (n12 + SMOOTH)
    for k in THEMES:
        prior = rev[f"{k}_bad"].mean()
        b3 = (roll(g, f"{k}_bad", 3) + SMOOTH * prior) / (n3 + SMOOTH)
        b12 = (roll(g, f"{k}_bad", 12, shift=3) + SMOOTH * prior) / (n12 + SMOOTH)
        p[f"rev_{k}_bad3"], p[f"rev_{k}_bad_delta"] = b3, b3 - b12

    # targets
    fut = g["traffic"].transform(lambda s: s[::-1].rolling(3, min_periods=3).sum()[::-1].shift(-1))
    p["y_traffic"] = np.log1p(fut)
    p["traffic_next3"] = fut
    closed = biz.set_index("business_id")["is_open"].eq(0)
    p["last_active"] = p["business_id"].map(last)
    p["closed"] = p["business_id"].map(closed).fillna(False)
    months_left = (p["last_active"] - p["month"]).apply(lambda d: d.n)
    p["y_close"] = (p["closed"] & (months_left <= 12)).astype(int)
    p["closure_known"] = (end - p["month"]).apply(lambda d: d.n) >= 12
    # a closed business is only "at risk" while it is still active
    p = p[~p["closed"] | (months_left >= 1)]
    p = p[p["hist_age"] >= 6]  # need a little history first

    p["month"] = p["month"].dt.to_timestamp()
    keep = [c for c in p.columns if c.startswith(("hist_", "rev_", "y_"))] + \
           ["business_id", "month", "traffic", "traffic_next3", "closure_known", "n", "stars"] + \
           [f"{k}_bad" for k in THEMES]
    p[keep].to_parquet(os.path.join(a.work, "panel.parquet"), index=False)
    print(f"panel: {len(p)} rows, {p['business_id'].nunique()} businesses, months {p['month'].min():%Y-%m} to {p['month'].max():%Y-%m}")
    print(f"closures within 12 months among labeled rows: {p.loc[p.closure_known, 'y_close'].mean():.1%}")


if __name__ == "__main__":
    main()
