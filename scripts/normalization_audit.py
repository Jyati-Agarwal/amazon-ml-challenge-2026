#!/usr/bin/env python3
"""Normalization audit for the production candidate generator.

Imports normalize_text / norm_series from scripts/generate_candidates.py (the exact
production code) and checks behavior on targeted synthetic cases + real sampled rows
from S1/S2/S3. Read-only: raw TSVs are never modified.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_candidates import normalize_text, norm_series, read_tsv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "dataset" / "train"

print("== 1. Synthetic edge cases (production normalize_text) ==")
cases = [
    ("case", "McDonald's CORP.", "mcdonald s corp"),
    ("punct", "A.B.C. & Co., Ltd.!!", "a b c co ltd"),
    ("whitespace", "  Foo\t Bar  Baz  ", "foo bar baz"),
    ("latin-accent", "Café Crème SARL", "cafe creme sarl"),
    # Default production norm strips Indic combining marks (Python re \\w excludes
    # category Mn). Measured recall-NEUTRAL on the 20k validation (0.9794 legacy vs
    # 0.9792 with --matra-norm); kept: exactly matches the validated benchmark.
    ("devanagari", "राम मार्केटिंग प्राइवेट लिमिटेड", "र म म र क ट ग प र इव ट ल म ट ड"),
    ("tamil", "சென்னை உணவகம்", "ச ன ன உணவகம"),
    ("mixed-script", "Sharma & Sons (शर्मा)", "sharma sons शर म"),
    ("empty", "", ""),
    ("punct-only", "***!!!", ""),
    ("digits-street", "1795 Westchester Drive, High Point, NC", "1795 westchester drive high point nc"),
    ("house-no", "H.No-130/H, Sector 21", "h no 130 h sector 21"),
    ("legal-suffix", "Acme Pvt. Ltd.", "acme pvt ltd"),
    ("hyphen-name", "Coca-Cola", "coca cola"),
    ("underscore", "foo_bar", "foo_bar"),  # \w keeps underscore — known, consistent
]
fails = 0
for label, raw, expect in cases:
    got = normalize_text(raw)
    ok = got == expect
    idem = normalize_text(got) == got
    if not (ok and idem):
        fails += 1
    print(f"  [{'OK ' if ok and idem else 'FAIL'}] {label}: {raw!r} -> {got!r}"
          + ("" if ok else f" (expected {expect!r})")
          + ("" if idem else " NOT IDEMPOTENT"))

print("\n== 2. Scalar vs vectorized consistency ==")
raws = [c[1] for c in cases]
vec = norm_series(pd.Series(raws)).tolist()
scal = [normalize_text(r) for r in raws]
same = vec == scal
print(f"  norm_series == normalize_text on all cases: {same}")
if not same:
    fails += 1

print("\n== 3. Real data sample (50k rows/source, seed 7) ==")
for name, fn in (("S1", "train_source1.tsv"), ("S2", "train_source2.tsv"),
                 ("S3", "train_source3.tsv")):
    df = read_tsv(TRAIN / fn).sample(50000, random_state=7)
    nn = norm_series(df["business_name"])
    na = norm_series(df["business_address"])
    raw_empty_a = (df["business_address"].str.strip() == "").mean()
    print(f"  {name}: empty nname {(nn == '').mean():.4%} | "
          f"empty naddr {(na == '').mean():.4%} (raw empty addr {raw_empty_a:.4%}) | "
          f"names collapsed to '' from non-empty raw: "
          f"{((nn == '') & (df['business_name'].str.strip() != '')).sum()}")
    # long-address behavior
    long_a = df.loc[df["business_address"].str.len().idxmax(), "business_address"]
    print(f"    longest addr ({len(long_a)} ch) normalizes to "
          f"{len(normalize_text(long_a))} ch without error")
    # idempotency on real data
    idem = (norm_series(nn.head(5000)) == nn.head(5000)).all()
    print(f"    idempotent on 5k real names: {idem}")
    if not idem:
        fails += 1

print("\n== 4. S1 vs S2/S3 consistency ==")
print("  Same norm_series function object is applied to every source in "
      "generate_candidates.py (single code path) — consistent by construction.")

print(f"\nAUDIT {'PASS' if fails == 0 else f'FAIL ({fails} problems)'}")
sys.exit(1 if fails else 0)
