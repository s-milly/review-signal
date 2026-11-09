# Review Signal

**A product case study: we set out to forecast restaurant sales from customer
reviews, tested the idea on 552,940 real reviews, found it did not hold up,
and changed the product to match the evidence.**

Built on the Yelp Open Dataset: 1,616 Philadelphia restaurants with 100 or
more reviews, 2005 to early 2022.

## The problem

A small restaurant owner gets reviews every week but has no analyst. The
common belief is that reviews are an early warning: complaints pile up before
sales fall. If that is true, a tool that reads reviews could warn owners months
ahead and tell them what to fix.

## The hypothesis

> A forecast that reads the reviews beats a forecast that only looks at past
> traffic.

Success metric: lower forecast error on months the model has never seen.
Yelp check-ins stand in for foot traffic.

## How it was tested

1. **Read every review.** Sentiment for each review, plus five complaint
   themes (wait times, service, prices, food quality, cleanliness), each
   marked good or bad clause by clause.
2. **Build a monthly table.** For every restaurant and month, only what was
   known by the end of that month: past traffic, season, review volume, stars,
   tone and complaint share by theme.
3. **Backtest.** Models (LightGBM) learned from earlier months, then forecast
   2018 and 2019 months they never saw. Same model, with and without the
   review features. Also tested 6 and 12 months ahead, and closure risk in two
   separate test years.

## What we found

| Traffic forecast, typical miss | History only | History plus reviews |
|---|---|---|
| 3 months ahead | 30.2% | 30.6% |
| 6 months ahead | 26.2% | 26.9% |
| 12 months ahead | 24.9% | 27.4% |

| Closure risk ranking (AUC, higher is better) | History only | History plus reviews |
|---|---|---|
| Test year 2017 | 0.67 | 0.73 |
| Test year 2018 | 0.69 | 0.69 |

- **The hypothesis failed.** Reviews did not improve the traffic forecast at
  any range. Past traffic already holds what the reviews would have said.
- **Reviews did help flag closures**, clearly in one test year, slightly in
  the other.
- **Reality check:** of the 20 restaurants ranked riskiest in September 2019,
  9 had closed by the time the data ends in early 2022, against 62 of 547
  overall (11%).
- **Reviews are the only source that says *why*** customers are unhappy.

## What we changed

| Before | After | Why |
|---|---|---|
| Forecast from history plus reviews | Forecast from history only | It tested more accurate |
| "Fix this complaint, gain this much" slider | Removed | The data did not support that promise |
| Reviews drive the forecast | Reviews drive closure risk and the complaint breakdown | That is where they added value |
| Accuracy hidden at the bottom | A "What we tested" panel | Users should know what to trust |

The new positioning: **reviews tell you why; traffic history tells you how much.**

## What I would do next

- **Test with real sales data** from a few restaurants instead of check-ins,
  which are a weak stand-in for traffic.
- **Interview owners** about whether a closure-risk score would change any
  decision they make, before building more on it.
- **Improve theme tagging** with a small hand-labeled set to measure it (for
  example, "wrong order" should count against food, not wait time).

## Run it

1. Download the Yelp Open Dataset (free; you accept Yelp's terms). Put
   `shrink_yelp.py` next to it and run it (or double-click it). It finds the
   files, even inside the `.tar`, and writes three small files to `yelp_upload/`.
   Copy them to `data/raw/`.
2. Build:
   ```
   pip install pandas numpy lightgbm scikit-learn pyarrow vaderSentiment
   python3 themes.py   --raw data/raw --out data/work
   python3 features.py --raw data/raw --work data/work
   python3 train.py    --raw data/raw --work data/work --out site --label "Philadelphia restaurants, Yelp Open Dataset"
   ```
3. `python3 -m http.server`, then open `http://localhost:8000/site/`.

`make_test_data.py` makes fake data in the same format for checking the code;
the page shows a yellow banner when it is loaded.

| File | What it does |
|---|---|
| `shrink_yelp.py` | Cuts the Yelp download to one city |
| `themes.py` | Sentiment and complaint themes per review |
| `features.py` | One row per restaurant per month |
| `train.py` | Backtests, forecasts, closure risk, dashboard data |
| `site/index.html` | The dashboard (one page, no build step) |

Data: Yelp Open Dataset, used under Yelp's dataset terms. The raw data and the
generated `site/data.json` are not in this repository.
