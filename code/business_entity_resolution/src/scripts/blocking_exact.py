"""Blocking study, part A: exact-key methods on the FULL training set.

Methods: exact normalized name / address, each with and without country.
Also: GT country-consistency check, S2/S3/script-mismatch recall breakdowns,
candidate-volume stats, and hard-negative (giant block) diagnostics.

Reads dataset/ read-only. Appends metric rows to
experiments/blocking/blocking_results.csv and writes diagnostics under
experiments/blocking/.
"""
import gc
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "experiments" / "blocking"
OUTDIR.mkdir(parents=True, exist_ok=True)
RESULTS = OUTDIR / "blocking_results.csv"

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
WS = re.compile(r"\s+")


def strip_latin_accents(s: str) -> str:
    # remove combining marks only for Latin-range base chars; leaves Devanagari matras alone
    out = []
    for ch in unicodedata.normalize("NFD", s):
        if unicodedata.category(ch) == "Mn" and ord(ch) < 0x0900:
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def norm_series(s: pd.Series) -> pd.Series:
    s = s.map(lambda x: strip_latin_accents(x).lower())
    s = s.str.replace(PUNCT, " ", regex=True)
    s = s.str.replace(WS, " ", regex=True).str.strip()
    return s


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def append_result(row: dict):
    df = pd.DataFrame([row])
    header = not RESULTS.exists()
    df.to_csv(RESULTS, mode="a", header=header, index=False)


def cand_stats(counts: np.ndarray) -> dict:
    return {
        "cand_mean": round(float(counts.mean()), 1),
        "cand_median": int(np.median(counts)),
        "cand_p95": int(np.percentile(counts, 95)),
        "cand_p99": int(np.percentile(counts, 99)),
        "cand_max": int(counts.max()),
    }


t0 = time.time()
print("loading data", flush=True)
s1 = read_tsv(ROOT / "dataset/train/train_source1.tsv")
s2 = read_tsv(ROOT / "dataset/train/train_source2.tsv")
s3 = read_tsv(ROOT / "dataset/train/train_source3.tsv")
gt = read_tsv(ROOT / "dataset/train/train_ground_truth.tsv")
corpus = pd.concat([s2, s3], ignore_index=True)
del s2, s3
gc.collect()

print("normalizing", flush=True)
for df in (s1, corpus):
    df["nname"] = norm_series(df["business_name"])
    df["naddr"] = norm_series(df["business_address"])

# expanded GT pairs
gt_nonempty = gt[gt["matched_entity_ids"] != ""].copy()
pairs = gt_nonempty.assign(
    matched=gt_nonempty["matched_entity_ids"].str.split(",")
).explode("matched")[["source1_entity_id", "matched"]]
print(f"pairs: {len(pairs)}", flush=True)

# attach attributes
s1_idx = s1.set_index("entity_id")
c_idx = corpus.set_index("entity_id")
pairs = pairs.join(s1_idx[["nname", "naddr", "country"]], on="source1_entity_id")
pairs = pairs.join(
    c_idx[["nname", "naddr", "country", "business_name"]],
    on="matched", rsuffix="_m",
)
pairs["m_source"] = pairs["matched"].str[:2]
pairs["m_deva"] = pairs["business_name"].str.contains(DEVANAGARI)

# ---- GT country consistency ----
same_country = (pairs["country"] == pairs["country_m"]).mean()
print(f"GT same-country rate: {same_country:.6f}", flush=True)
(OUTDIR / "gt_country_check.json").write_text(json.dumps({
    "same_country_rate": round(float(same_country), 6),
    "n_pairs": len(pairs),
    "deva_pair_count": int(pairs["m_deva"].sum()),
    "deva_pair_pct": round(float(pairs["m_deva"].mean() * 100), 3),
}, indent=1))

total_pairs = len(pairs)
n_s1 = len(s1)


def evaluate(method_name, s1_key, pair_key_s1, pair_key_m, runtime_start):
    """s1_key: Series of key per S1 row (aligned with s1). pair_key_*: keys per GT pair."""
    hit = (pair_key_s1 == pair_key_m) & (pair_key_s1 != "")
    rec = float(hit.mean())
    by_country = pairs.groupby("country").apply(
        lambda g: float(hit.loc[g.index].mean()), include_groups=False)
    by_source = pairs.groupby("m_source").apply(
        lambda g: float(hit.loc[g.index].mean()), include_groups=False)
    deva_rec = float(hit[pairs["m_deva"]].mean())
    latin_rec = float(hit[~pairs["m_deva"]].mean())
    # entity-level: all matches hit (entities with >=1 match); singletons excluded here
    ent = hit.groupby(pairs["source1_entity_id"]).all()
    ent_rec = float(ent.mean())
    # candidate volume: block size in corpus for each S1 key
    vc = corpus_key_counts
    counts = s1_key.map(vc).fillna(0)
    counts[s1_key == ""] = 0
    counts = counts.to_numpy()
    row = {
        "method": method_name, "scope": "full_train",
        "recall": round(rec, 4),
        "recall_US": round(float(by_country.get("US", np.nan)), 4),
        "recall_India": round(float(by_country.get("India", np.nan)), 4),
        "recall_France": "NA (no France in train)",
        "recall_S2": round(float(by_source.get("S2", np.nan)), 4),
        "recall_S3": round(float(by_source.get("S3", np.nan)), 4),
        "recall_script_mismatch": round(deva_rec, 4),
        "recall_latin": round(latin_rec, 4),
        "entity_recall_all_matches": round(ent_rec, 4),
        **cand_stats(counts),
        "runtime_s": round(time.time() - runtime_start, 1),
        "notes": "",
    }
    append_result(row)
    print(method_name, row["recall"], row["cand_mean"], flush=True)
    return counts


# ---- Method 1: exact normalized name ----
t = time.time()
corpus_key_counts = corpus["nname"].value_counts()
evaluate("M1_exact_name", s1["nname"], pairs["nname"], pairs["nname_m"], t)

# hard negatives: biggest name blocks
top_names = corpus_key_counts.head(25)
top_names.to_csv(OUTDIR / "hardneg_top_name_blocks.csv", header=["corpus_count"])

# ---- Method 3: name + country ----
t = time.time()
corpus_key = corpus["nname"] + "||" + corpus["country"]
corpus_key_counts = corpus_key.value_counts()
evaluate("M3_exact_name_country",
         s1["nname"] + "||" + s1["country"],
         pairs["nname"] + "||" + pairs["country"],
         pairs["nname_m"] + "||" + pairs["country_m"], t)

# ---- Method 2: exact normalized address ----
t = time.time()
corpus_key_counts = corpus.loc[corpus["naddr"] != "", "naddr"].value_counts()
evaluate("M2_exact_addr", s1["naddr"], pairs["naddr"], pairs["naddr_m"], t)
corpus_key_counts.head(25).to_csv(OUTDIR / "hardneg_top_addr_blocks.csv", header=["corpus_count"])

# ---- Method 4: address + country ----
t = time.time()
mask = corpus["naddr"] != ""
corpus_key = corpus.loc[mask, "naddr"] + "||" + corpus.loc[mask, "country"]
corpus_key_counts = corpus_key.value_counts()
pk1 = pairs["naddr"] + "||" + pairs["country"]
pkm = pairs["naddr_m"].where(pairs["naddr_m"] != "", "<EMPTY>") + "||" + pairs["country_m"]
evaluate("M4_exact_addr_country", s1["naddr"] + "||" + s1["country"], pk1, pkm, t)

# ---- Union of M1+M2 style exact keys (name+country OR addr+country) ----
t = time.time()
name_key_counts = (corpus["nname"] + "||" + corpus["country"]).value_counts()
hit_name = ((pairs["nname"] == pairs["nname_m"]) & (pairs["nname"] != "")
            & (pairs["country"] == pairs["country_m"]))
hit_addr = ((pairs["naddr"] == pairs["naddr_m"]) & (pairs["naddr"] != "")
            & (pairs["country"] == pairs["country_m"]))
hit = hit_name | hit_addr
ent = hit.groupby(pairs["source1_entity_id"]).all()
addr_counts = (s1["naddr"] + "||" + s1["country"]).map(corpus_key_counts).fillna(0)
name_counts = (s1["nname"] + "||" + s1["country"]).map(name_key_counts).fillna(0)
counts = (name_counts + addr_counts).to_numpy()  # upper bound (double-counts overlap)
append_result({
    "method": "M_union_exactname_exactaddr_country", "scope": "full_train",
    "recall": round(float(hit.mean()), 4),
    "recall_US": round(float(hit[pairs["country"] == "US"].mean()), 4),
    "recall_India": round(float(hit[pairs["country"] == "India"].mean()), 4),
    "recall_France": "NA (no France in train)",
    "recall_S2": round(float(hit[pairs["m_source"] == "S2"].mean()), 4),
    "recall_S3": round(float(hit[pairs["m_source"] == "S3"].mean()), 4),
    "recall_script_mismatch": round(float(hit[pairs["m_deva"]].mean()), 4),
    "recall_latin": round(float(hit[~pairs["m_deva"]].mean()), 4),
    "entity_recall_all_matches": round(float(ent.mean()), 4),
    **cand_stats(counts),
    "runtime_s": round(time.time() - t, 1),
    "notes": "cand counts are sum of the two blocks (upper bound)",
})
print("union exact done", flush=True)

# ---- missed-match examples for the exact union (diagnostics) ----
missed = pairs[~hit].sample(min(2000, int((~hit).sum())), random_state=0)
missed_out = missed[["source1_entity_id", "matched", "country", "m_source", "m_deva"]].copy()
missed_out["s1_name"] = missed["nname"]
missed_out["m_name"] = missed["nname_m"]
missed_out["s1_addr"] = missed["naddr"]
missed_out["m_addr"] = missed["naddr_m"]
missed_out.head(500).to_csv(OUTDIR / "missed_by_exact_union_sample.csv", index=False)

print(f"total {time.time() - t0:.0f}s", flush=True)
