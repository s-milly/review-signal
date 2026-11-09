"""Train and test the forecasts, then export the dashboard data.

    python3 train.py --work data/work --out site

The test is a backtest: models learn from earlier months and are scored on
later months they never saw. Traffic forecasts compared:
  naive     next 3 months = last 3 months
  history   LightGBM on past traffic, season and age only
  reviews   the same plus every review feature
also at 6 and 12 months ahead, and closure risk with and without reviews in
two separate test years. On the Philadelphia data, reviews did not improve the
traffic forecast, so the dashboard forecasts from history and uses reviews for
closure risk and for explaining what customers complain about.
"""
import argparse
import gzip
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from themes import THEMES

P = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=50,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=0)
LABELS = {"wait": "Wait times", "service": "Service", "price": "Prices", "food": "Food quality", "clean": "Cleanliness"}


def fit(X, y, **kw):
    return lgb.LGBMRegressor(**P, **kw).fit(X, y)


def typical_error(y, yhat):
    """Median percent miss on the next-3-month traffic."""
    a, b = np.expm1(y), np.expm1(yhat)
    ok = a > 0
    return float(np.median(np.abs(b[ok] - a[ok]) / a[ok]))


def group_of(col):
    if col.startswith("hist_"):
        return {"hist_month": "Season"}.get(col, "Recent traffic")
    for k in THEMES:
        if col.startswith(f"rev_{k}_"):
            return LABELS[k]
    if col.startswith("rev_stars"):
        return "Star rating"
    if col.startswith("rev_sent"):
        return "Review tone"
    return "Review volume"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--work", default="data/work")
    ap.add_argument("--out", default="site")
    ap.add_argument("--test-start", default="2018-07")
    ap.add_argument("--test-end", default="2019-09", help="last forecast date; also the dashboard's as-of date")
    ap.add_argument("--label", default="", help="shown on the page, e.g. 'Philadelphia restaurants'")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    p = pd.read_parquet(os.path.join(a.work, "panel.parquet")).sort_values(["business_id", "month"])
    H = [c for c in p.columns if c.startswith("hist_")]
    R = [c for c in p.columns if c.startswith("rev_")]
    ts, te = pd.Timestamp(a.test_start), pd.Timestamp(a.test_end)
    gap = pd.DateOffset(months=3)
    tr = p[(p.month < ts - gap) & p.y_traffic.notna()]
    te_ = p[(p.month >= ts) & (p.month <= te) & p.y_traffic.notna()]

    # ---- traffic backtest, 3 months ahead (the dashboard's forecast)
    m_hist, m_full = fit(tr[H], tr.y_traffic), fit(tr[H + R], tr.y_traffic)
    preds = {"naive": te_.hist_t3.values, "history": m_hist.predict(te_[H]), "reviews": m_full.predict(te_[H + R])}
    lo = fit(tr[H], tr.y_traffic, objective="quantile", alpha=0.1)
    hi = fit(tr[H], tr.y_traffic, objective="quantile", alpha=0.9)
    plo, phi = lo.predict(te_[H]), hi.predict(te_[H])
    y = te_.y_traffic.values
    traffic = {
        "rows": int(len(te_)), "businesses": int(te_.business_id.nunique()), "train_rows": int(len(tr)),
        "typical_error": {k: typical_error(y, v) for k, v in preds.items()},
        "range_coverage": float(np.mean((y >= plo) & (y <= phi))),
    }

    # ---- the main question: do reviews improve the forecast further ahead?
    g = p.groupby("business_id")
    horizons = []
    for h in (3, 6, 12):
        fut = g["traffic"].transform(lambda s: s[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1))
        d = p.assign(y=np.log1p(fut)).dropna(subset=["y"])
        off = pd.DateOffset(months=h)
        dtr = d[d.month < ts - off]
        dte = d[(d.month >= ts) & (d.month <= pd.Timestamp("2019-12") - off)]
        row = {"months": h, "rows": int(len(dte))}
        for name, cols in (("history", H), ("reviews", H + R)):
            row[name] = typical_error(dte.y.values, fit(dtr[cols], dtr.y).predict(dte[cols]))
        horizons.append(row)

    # ---- closure backtest, two separate test years
    known = p[p.closure_known]
    cp = dict(P, n_estimators=300)
    closure = []
    for year in (2017, 2018):
        cs = pd.Timestamp(f"{year}-01")
        ctr = known[known.month < cs - pd.DateOffset(months=12)]
        cte = known[(known.month >= cs) & (known.month <= cs + pd.DateOffset(months=11))]
        row = {"year": year, "rows": int(len(cte)), "rate": float(cte.y_close.mean())}
        for name, cols in (("history", H), ("reviews", H + R)):
            s = lgb.LGBMClassifier(**cp).fit(ctr[cols], ctr.y_close).predict_proba(cte[cols])[:, 1]
            top = s >= np.quantile(s, 0.9)
            row[name] = {"auc": float(roc_auc_score(cte.y_close, s)),
                         "avg_precision": float(average_precision_score(cte.y_close, s)),
                         "top10_share": float(cte.y_close[top].sum() / max(cte.y_close.sum(), 1))}
        closure.append(row)

    metrics = {"label": a.label, "test": [a.test_start, a.test_end], "traffic": traffic,
               "horizons": horizons, "closure": closure}
    json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    print(json.dumps(metrics, indent=1))

    # ---- dashboard export: forecasts made on the as-of date, by models that
    # only saw outcomes already known on that date, next to what really happened.
    # Traffic uses history only (it tested best); closure risk uses reviews too.
    asof = te
    seen = p[(p.month <= asof - gap) & p.y_traffic.notna()]
    f_mid = fit(seen[H], seen.y_traffic)
    f_lo = fit(seen[H], seen.y_traffic, objective="quantile", alpha=0.1)
    f_hi = fit(seen[H], seen.y_traffic, objective="quantile", alpha=0.9)
    cseen = known[known.month <= asof - pd.DateOffset(months=12)]
    f_close = lgb.LGBMClassifier(**cp).fit(cseen[H + R], cseen.y_close)
    base_close = float(cseen.y_close.mean())

    now = p[p.month == asof].copy()
    now = now[now.hist_t12 > np.log1p(10)]  # skip businesses with very little traffic
    now["mid"], now["lo"], now["hi"] = f_mid.predict(now[H]), f_lo.predict(now[H]), f_hi.predict(now[H])
    X = now[H + R]
    now["risk"] = f_close.predict_proba(X)[:, 1]
    contrib = f_close.predict(X, pred_contrib=True)[:, :-1]  # log-odds of closing
    groups = sorted({group_of(c) for c in H + R})
    gidx = {gr: [i for i, c in enumerate(H + R) if group_of(c) == gr] for gr in groups}
    typical = {k: float(seen[f"rev_{k}_bad3"].median()) for k in THEMES}

    with gzip.open(os.path.join(a.raw, "business.jsonl.gz"), "rt", encoding="utf-8") as fh:
        names = {b["business_id"]: b for b in map(json.loads, fh)}

    hist_from = asof - pd.DateOffset(months=23)
    series = p[(p.month >= hist_from) & (p.month <= asof + gap)].set_index(["business_id", "month"])
    have = set(series.index.get_level_values(0))
    out = []
    for i, (_, r) in enumerate(now.iterrows()):
        bid = r.business_id
        # how many times more (or less) likely to close each factor makes it
        drivers = sorted(({"name": gr, "times": float(np.exp(contrib[i, gidx[gr]].sum()))} for gr in groups),
                         key=lambda d: -abs(np.log(d["times"])))
        hist = []
        if bid in have:
            for m, row in series.loc[bid].iterrows():
                hist.append({"m": f"{m:%Y-%m}", "t": int(row.traffic), "n": int(row.n),
                             "stars": round(row.stars / row.n, 2) if row.n else None})
        b = names.get(bid, {})
        out.append({
            "id": bid, "name": b.get("name", bid), "address": b.get("address", ""),
            "categories": b.get("categories", ""), "closed_later": bool(b.get("is_open", 1) == 0),
            "last3": float(np.expm1(r.hist_t3)), "forecast": max(0.0, float(np.expm1(r["mid"]))),
            "low": max(0.0, float(np.expm1(r["lo"]))), "high": max(0.0, float(np.expm1(r["hi"]))),
            "actual": None if pd.isna(r.traffic_next3) else float(r.traffic_next3),
            "risk": float(r["risk"]), "drivers": drivers[:6],
            "themes": {k: {"now": float(r[f"rev_{k}_bad3"]), "change": float(r[f"rev_{k}_bad_delta"]),
                           "typical": typical[k]} for k in THEMES},
            "history": hist,
        })
    out.sort(key=lambda d: -d["risk"])
    import pyarrow.parquet as pq
    study = {"reviews": int(pq.ParquetFile(os.path.join(a.work, "reviews.parquet")).metadata.num_rows),
             "restaurants": len(names)}
    data = {"asof": f"{asof:%Y-%m}", "label": a.label, "base_close_rate": base_close, "study": study,
            "themes": LABELS, "metrics": metrics, "businesses": out}
    with open(os.path.join(a.out, "data.json"), "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"exported {len(out)} businesses, {os.path.getsize(os.path.join(a.out, 'data.json')) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
