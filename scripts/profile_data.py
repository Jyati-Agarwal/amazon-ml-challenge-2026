"""One-off data reconnaissance for the ML Challenge 2026 dataset.

Processes one file at a time to keep memory bounded. Writes findings to
docs/data_profile.json (raw numbers; DATA_PROFILE.md is written from these).
Read-only with respect to dataset/.
"""
import gc
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data_profile.json"
OUT.parent.mkdir(exist_ok=True)

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
NON_ASCII = re.compile(r"[^\x00-\x7F]")

results = {}


def pct(x, n):
    return round(100.0 * x / n, 3) if n else 0.0


def len_stats(s: pd.Series):
    lens = s.str.len()
    return {
        "mean": round(float(lens.mean()), 1),
        "median": int(lens.median()),
        "p5": int(lens.quantile(0.05)),
        "p95": int(lens.quantile(0.95)),
        "max": int(lens.max()),
        "empty_pct": pct(int((lens == 0).sum()), len(s)),
    }


def profile_source(path: Path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                     quoting=3, engine="c")
    n = len(df)
    r = {
        "rows": n,
        "columns": list(df.columns),
        "entity_id_unique": bool(df["entity_id"].is_unique),
        "country_counts": df["country"].value_counts().to_dict(),
        "missing_pct": {c: pct(int((df[c] == "").sum()), n) for c in df.columns},
        "name_len": len_stats(df["business_name"]),
        "addr_len": len_stats(df["business_address"]),
    }
    # duplicates
    r["dup_name_addr_country_rows"] = int(df.duplicated(
        subset=["business_name", "business_address", "country"]).sum())
    r["dup_business_name_rows"] = int(df.duplicated(subset=["business_name"]).sum())
    nonempty_addr = df[df["business_address"] != ""]
    r["dup_address_rows_nonempty"] = int(
        nonempty_addr.duplicated(subset=["business_address"]).sum())
    # script characteristics on a sample (full regex over 5M rows is fine but sample is enough)
    samp = df.sample(min(200_000, n), random_state=0)
    r["name_devanagari_pct"] = pct(
        int(samp["business_name"].str.contains(DEVANAGARI).sum()), len(samp))
    r["name_nonascii_pct"] = pct(
        int(samp["business_name"].str.contains(NON_ASCII).sum()), len(samp))
    by_country = {}
    for ctry, g in samp.groupby("country"):
        by_country[ctry] = {
            "name_devanagari_pct": pct(int(g["business_name"].str.contains(DEVANAGARI).sum()), len(g)),
            "name_len_mean": round(float(g["business_name"].str.len().mean()), 1),
            "addr_len_mean": round(float(g["business_address"].str.len().mean()), 1),
            "addr_empty_pct": pct(int((g["business_address"] == "").sum()), len(g)),
        }
    r["by_country_sample"] = by_country
    r["examples"] = df.head(3).to_dict(orient="records")
    del df, samp, nonempty_addr
    gc.collect()
    return r


for split in ("train", "test"):
    for i in (1, 2, 3):
        p = ROOT / "dataset" / split / f"{split}_source{i}.tsv"
        key = f"{split}_source{i}"
        print("profiling", key, flush=True)
        results[key] = profile_source(p)
        OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))

# ---- ground truth ----
print("profiling ground truth", flush=True)
gt = pd.read_csv(ROOT / "dataset/train/train_ground_truth.tsv", sep="\t",
                 dtype=str, keep_default_na=False, quoting=3)
n = len(gt)
match_lists = gt["matched_entity_ids"].str.split(",")
counts = match_lists.apply(lambda L: 0 if L == [""] else len(L))
s2_counts = gt["matched_entity_ids"].str.count("S2-")
s3_counts = gt["matched_entity_ids"].str.count("S3-")

dist = counts.value_counts().sort_index()
g = {
    "s1_entities": n,
    "s1_ids_unique": bool(gt["source1_entity_id"].is_unique),
    "zero_match": int((counts == 0).sum()),
    "one_match": int((counts == 1).sum()),
    "multi_match": int((counts >= 2).sum()),
    "avg_matches_per_s1": round(float(counts.mean()), 3),
    "max_matches": int(counts.max()),
    "match_count_distribution": {str(k): int(v) for k, v in dist.head(20).items()},
    "total_matched_ids": int(counts.sum()),
    "total_s2_ids": int(s2_counts.sum()),
    "total_s3_ids": int(s3_counts.sum()),
    "s1_with_any_s2": int((s2_counts > 0).sum()),
    "s1_with_any_s3": int((s3_counts > 0).sum()),
    "s1_with_both": int(((s2_counts > 0) & (s3_counts > 0)).sum()),
    "s1_only_s2": int(((s2_counts > 0) & (s3_counts == 0)).sum()),
    "s1_only_s3": int(((s3_counts > 0) & (s2_counts == 0)).sum()),
    "s2_matches_dist": {str(k): int(v) for k, v in s2_counts.value_counts().sort_index().head(10).items()},
    "s3_matches_dist": {str(k): int(v) for k, v in s3_counts.value_counts().sort_index().head(10).items()},
}

# do matched IDs get reused across S1 entities (i.e. is it a clean partition)?
all_ids = [x for L in match_lists for x in L if x]
g["matched_id_mentions"] = len(all_ids)
g["matched_id_unique"] = len(set(all_ids))

# what fraction of S2/S3 records are matched to anything
src2_n = results["train_source2"]["rows"]
src3_n = results["train_source3"]["rows"]
uniq = set(all_ids)
u2 = sum(1 for x in uniq if x.startswith("S2-"))
u3 = sum(1 for x in uniq if x.startswith("S3-"))
g["s2_records_matched"] = u2
g["s3_records_matched"] = u3
g["s2_matched_pct_of_source"] = pct(u2, src2_n)
g["s3_matched_pct_of_source"] = pct(u3, src3_n)

results["ground_truth"] = g
OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))
print("done ->", OUT)
