"""Shrink the Yelp Open Dataset to one city so it is small enough to upload.

Easiest: put this file in the same folder as the Yelp download (for example
Downloads) and double-click it. It searches the folders around it for the
Yelp files, reading them straight out of the .tar if it was never unpacked.
It needs only Python 3, no extra packages. Or run it with options:

    python shrink_yelp.py --src path/to/yelp_dataset --city Philadelphia

It writes three small files to a folder named yelp_upload next to this file.
Upload those three files.
"""
import argparse
import gzip
import json
import os
import sys
import tarfile
import traceback
from collections import Counter


HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ("business", "review", "checkin")


def find_source(start):
    """Return ('dir', folder) or ('tar', tarpath) for the Yelp files under start."""
    tars = []
    for root, dirs, files in os.walk(start):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("AppData", "node_modules")]
        if "yelp_academic_dataset_business.json" in files:
            return "dir", root
        tars += [os.path.join(root, f) for f in files if f.lower().endswith(".tar")]
    for t in tars:
        try:
            with tarfile.open(t) as tf:
                if any(m.name.endswith("yelp_academic_dataset_business.json") for m in tf):
                    return "tar", t
        except tarfile.TarError:
            pass
    return None, None


def lines(src, name):
    kind, where = src
    target = f"yelp_academic_dataset_{name}.json"
    if kind == "dir":
        f = open(os.path.join(where, target), encoding="utf-8")
    else:
        tf = tarfile.open(where)
        member = next(m for m in tf if m.name.endswith(target))
        import io
        f = io.TextIOWrapper(tf.extractfile(member), encoding="utf-8")
    with f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="folder with the yelp_academic_dataset_*.json files, or the .tar (found automatically if left out)")
    ap.add_argument("--city", default="Philadelphia")
    ap.add_argument("--category", default="Restaurants")
    ap.add_argument("--min-reviews", type=int, default=100)
    ap.add_argument("--max-text", type=int, default=600, help="keep this many characters of each review")
    ap.add_argument("--out", default=os.path.join(HERE, "yelp_upload"))
    a = ap.parse_args()
    if a.src and a.src.lower().endswith(".tar"):
        src = ("tar", a.src)
    elif a.src:
        src = ("dir", a.src)
    else:
        print(f"Looking for the Yelp files in {HERE} ...")
        src = find_source(HERE)
        if not src[0]:
            print("Could not find the Yelp files. Put this script in the folder that holds the")
            print("Yelp download (for example Downloads) and run it again.")
            return
    print(f"Using: {src[1]}")
    print(f"Keeping {a.category} in {a.city} with at least {a.min_reviews} reviews. This takes several minutes.")
    os.makedirs(a.out, exist_ok=True)

    keep = {}
    with gzip.open(os.path.join(a.out, "business.jsonl.gz"), "wt", encoding="utf-8") as out:
        for b in lines(src, "business"):
            cats = b.get("categories") or ""
            if b.get("city", "").lower() != a.city.lower() or a.category not in cats:
                continue
            if b.get("review_count", 0) < a.min_reviews:
                continue
            keep[b["business_id"]] = True
            row = {k: b.get(k) for k in ("business_id", "name", "address", "city", "state",
                                         "latitude", "longitude", "stars", "review_count",
                                         "is_open", "categories")}
            out.write(json.dumps(row) + "\n")
    print(f"businesses kept: {len(keep)}")

    n = 0
    with gzip.open(os.path.join(a.out, "review.jsonl.gz"), "wt", encoding="utf-8") as out:
        for r in lines(src, "review"):
            if r["business_id"] not in keep:
                continue
            row = {"business_id": r["business_id"], "stars": r["stars"], "date": r["date"][:10],
                   "useful": r.get("useful", 0), "text": (r.get("text") or "")[: a.max_text]}
            out.write(json.dumps(row) + "\n")
            n += 1
            if n % 100000 == 0:
                print(f"  {n} reviews kept so far ...")
    print(f"reviews kept: {n}")

    m = 0
    with gzip.open(os.path.join(a.out, "checkin_monthly.jsonl.gz"), "wt", encoding="utf-8") as out:
        for c in lines(src, "checkin"):
            if c["business_id"] not in keep:
                continue
            months = Counter(d.strip()[:7] for d in c["date"].split(",") if d.strip())
            out.write(json.dumps({"business_id": c["business_id"], "months": dict(months)}) + "\n")
            m += 1
    print(f"businesses with check-ins: {m}")

    for f in sorted(os.listdir(a.out)):
        p = os.path.join(a.out, f)
        print(f"  {p}: {os.path.getsize(p) / 1e6:.1f} MB")
    print(f"\nDone. Upload the 3 files in {a.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("\nSomething went wrong. Copy the text above and send it to Claude.")
    if len(sys.argv) == 1:  # opened by double-click: keep the window open
        input("\nPress Enter to close this window.")
