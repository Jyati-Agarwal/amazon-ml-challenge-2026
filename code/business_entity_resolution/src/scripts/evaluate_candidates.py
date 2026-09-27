#!/usr/bin/env python3
"""Evaluate a generate_candidates.py output directory against TRAIN ground truth.

Reports pair recall (overall / per-country / per-source), entity all-match recall,
candidate-count distribution (including zero-candidate S1s), and correctness checks
(dupes, cross-country leakage, ID validity, channel consistency).

Usage:
  .venv/bin/python scripts/evaluate_candidates.py --dir experiments/candidates/val20k
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def read_tsv(p):
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--gt", default=str(ROOT / "dataset/train/train_ground_truth.tsv"))
    ap.add_argument("--s1", default=str(ROOT / "dataset/train/train_source1.tsv"))
    ap.add_argument("--report", default=None, help="optional JSON report path")
    args = ap.parse_args()
    d = Path(args.dir)

    idx = pd.concat([pd.read_parquet(f) for f in
                     sorted(glob.glob(str(d / "s1_index" / "*.parquet")))],
                    ignore_index=True)
    part_files = sorted(glob.glob(str(d / "parts" / "*.parquet")))
    cand = pd.concat([pd.read_parquet(f) for f in part_files], ignore_index=True)
    print(f"loaded {len(cand):,} candidate rows from {len(part_files)} parts; "
          f"{len(idx):,} S1 in index")

    checks = {}
    checks["dup_pairs"] = int(cand.duplicated(["s1_id", "cand_id"]).sum())
    checks["s1_prefix_ok"] = bool(cand["s1_id"].str.startswith("S1-").all())
    src_pref = cand["cand_id"].str[:2]
    checks["cand_from_s2s3"] = bool(src_pref.isin(["S2", "S3"]).all())
    checks["source_col_matches_prefix"] = bool(
        (cand["source"].astype(int).astype(str) == src_pref.str[1]).all())
    checks["channels_nonzero"] = int((cand["channels"] == 0).sum())
    checks["word_rank_consistent"] = bool(
        (((cand["channels"] & 1) > 0) == (cand["word_rank"] >= 0)).all())
    checks["char_rank_consistent"] = bool(
        (((cand["channels"] & 2) > 0) == (cand["char_rank"] >= 0)).all())
    checks["dup_s1_in_index"] = int(idx["s1_id"].duplicated().sum())

    s1 = read_tsv(args.s1)
    s1_country = dict(zip(s1["entity_id"], s1["country"]))
    checks["s1_country_correct"] = bool(
        (cand["country"] == cand["s1_id"].map(s1_country)).all())

    gt = read_tsv(args.gt)
    in_scope = set(idx["s1_id"])
    gt = gt[gt["source1_entity_id"].isin(in_scope)].copy()
    gt["matched"] = gt["matched_entity_ids"].apply(
        lambda x: x.split(",") if x else [])
    pairs = gt[["source1_entity_id", "matched"]].explode("matched")
    pairs = pairs[pairs["matched"].notna() & (pairs["matched"] != "")].rename(
        columns={"source1_entity_id": "s1_id", "matched": "m_id"})
    pairs["country"] = pairs["s1_id"].map(s1_country)
    pairs["src"] = pairs["m_id"].str[:2]
    print(f"GT in scope: {len(gt):,} S1 rows, {len(pairs):,} true pairs")

    key = cand["s1_id"] + "|" + cand["cand_id"]
    have = set(key)
    hit = np.fromiter(((a + "|" + b) in have
                       for a, b in zip(pairs["s1_id"], pairs["m_id"])),
                      dtype=bool, count=len(pairs))

    # cross-country leakage: candidate country (from corpus side) must equal S1's.
    # cheap proxy: any hit pair whose sources disagree is impossible by construction;
    # direct check on the candidate table:
    metrics = {"pair_recall": round(float(hit.mean()), 4),
               "total_candidates": int(len(cand)),
               "total_true_pairs": int(len(pairs))}
    hs = pd.Series(hit)
    for c in sorted(pairs["country"].unique()):
        metrics[f"recall_{c}"] = round(float(hs[(pairs["country"] == c).to_numpy()].mean()), 4)
    for sname in ("S2", "S3"):
        m = (pairs["src"] == sname).to_numpy()
        if m.any():
            metrics[f"recall_{sname}"] = round(float(hs[m].mean()), 4)
    ent = hs.groupby(pairs["s1_id"].to_numpy()).all()
    metrics["entity_all_match_recall"] = round(float(ent.mean()), 4)

    counts = idx["n_cands"].to_numpy()
    metrics.update({
        "cand_mean": round(float(counts.mean()), 1),
        "cand_median": int(np.median(counts)),
        "cand_p95": int(np.percentile(counts, 95)),
        "cand_p99": int(np.percentile(counts, 99)),
        "cand_max": int(counts.max()),
        "zero_cand_s1": int((counts == 0).sum()),
    })

    print("\n== METRICS ==")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\n== CHECKS ==")
    for k, v in checks.items():
        print(f"  {k}: {v}")

    if args.report:
        Path(args.report).write_text(json.dumps(
            {"metrics": metrics, "checks": checks}, indent=1))
        print("\nreport ->", args.report)


if __name__ == "__main__":
    main()
