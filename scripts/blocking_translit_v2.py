"""Follow-up: does a cleaned-up transliteration (unidecode + collapse repeated
letters) improve script-mismatch recall for India, on top of the name+addr
representations that already carry the Latin-address signal?

India only, same 8k sample (seed 42), full India train S2+S3 pool.
Configs: char34 name+addr and word name+addr, corpus deva rows transliterated.
"""
import gc
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from unidecode import unidecode

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "experiments" / "blocking"
RESULTS = OUTDIR / "blocking_results.csv"
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
WS = re.compile(r"\s+")
REPEAT = re.compile(r"(.)\1+")
SEED = 42
KS = [25, 50, 100, 200]


def strip_latin_accents(s):
    out = [ch for ch in unicodedata.normalize("NFD", s)
           if not (unicodedata.category(ch) == "Mn" and ord(ch) < 0x0900)]
    return unicodedata.normalize("NFC", "".join(out))


def norm_series(s):
    s = s.map(lambda x: strip_latin_accents(x).lower())
    s = s.str.replace(PUNCT, " ", regex=True)
    return s.str.replace(WS, " ", regex=True).str.strip()


def translit2(x):
    x = unidecode(x).lower()
    x = PUNCT.sub(" ", x)
    x = REPEAT.sub(r"\1", x)          # collapse doubled letters
    return WS.sub(" ", x).strip()


def read_tsv(p):
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)


t0 = time.time()
s1 = read_tsv(ROOT / "dataset/train/train_source1.tsv")
s2 = read_tsv(ROOT / "dataset/train/train_source2.tsv")
s3 = read_tsv(ROOT / "dataset/train/train_source3.tsv")
gt = read_tsv(ROOT / "dataset/train/train_ground_truth.tsv")
corp = pd.concat([s2, s3], ignore_index=True)
del s2, s3
gc.collect()
corp = corp[corp["country"] == "India"].reset_index(drop=True)
s1 = s1[s1["country"] == "India"]

corp["deva"] = corp["business_name"].str.contains(DEVANAGARI)
corp["nname"] = norm_series(corp["business_name"])
corp["naddr"] = norm_series(corp["business_address"])
q = s1.sample(8000, random_state=SEED).reset_index(drop=True)  # same as part B
q["nname"] = norm_series(q["business_name"])
q["naddr"] = norm_series(q["business_address"])
gt_map = gt.set_index("source1_entity_id")["matched_entity_ids"]
q["matched_list"] = q["entity_id"].map(gt_map).fillna("").apply(
    lambda x: x.split(",") if x else [])
pairs = q[["entity_id", "matched_list"]].explode("matched_list")
pairs = pairs[pairs["matched_list"].notna() & (pairs["matched_list"] != "")]
pairs = pairs.rename(columns={"matched_list": "m_id"})
deva_map = corp.set_index("entity_id")["deva"]
pairs["m_deva"] = pairs["m_id"].map(deva_map).fillna(False)
print(f"pairs {len(pairs)} deva {int(pairs.m_deva.sum())}", flush=True)

# joint field; transliterate name for deva rows only (address is already Latin)
corp["tjoint"] = corp["nname"] + " " + corp["naddr"]
mask = corp["deva"].to_numpy()
corp.loc[mask, "tjoint"] = (corp.loc[mask, "business_name"].map(translit2)
                            + " " + corp.loc[mask, "naddr"])
q["tjoint"] = q["nname"] + " " + q["naddr"]
corp_ids = corp["entity_id"].to_numpy()

for cfg, analyzer, ngr in [("M9v2_char34_nameaddr_translit2", "char_wb", (3, 4)),
                           ("M9v2_word_nameaddr_translit2", "word", (1, 1))]:
    t = time.time()
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngr, min_df=3,
                          max_df=0.4, dtype=np.float32)
    X = vec.fit_transform(corp["tjoint"])
    Q = vec.transform(q["tjoint"])
    B = X.T.tocsr()
    del X
    gc.collect()
    C = sp_matmul_topn(Q, B, top_n=200, threshold=0.05, sort=True, n_threads=12)
    del B
    gc.collect()
    indptr, idx = C.indptr, C.indices
    top = {eid: corp_ids[idx[indptr[i]:indptr[i + 1]]]
           for i, eid in enumerate(q["entity_id"])}
    for k in KS:
        sets = {e: set(v[:k]) for e, v in top.items()}
        hit = pairs.apply(lambda r: r["m_id"] in sets.get(r["entity_id"], ()), axis=1)
        ent = hit.groupby(pairs["entity_id"]).all()
        row = {
            "method": f"{cfg}@k{k}", "scope": "sample8k_India_fullpool",
            "recall": round(float(hit.mean()), 4),
            "recall_US": "", "recall_India": round(float(hit.mean()), 4),
            "recall_France": "NA", "recall_S2": "", "recall_S3": "",
            "recall_script_mismatch": round(float(hit[pairs["m_deva"]].mean()), 4),
            "recall_latin": round(float(hit[~pairs["m_deva"]].mean()), 4),
            "entity_recall_all_matches": round(float(ent.mean()), 4),
            "cand_mean": k, "cand_median": k, "cand_p95": k, "cand_p99": k,
            "cand_max": k, "runtime_s": round(time.time() - t, 1),
            "notes": "deva corpus names transliterated (unidecode+collapse)",
        }
        pd.DataFrame([row]).to_csv(RESULTS, mode="a", header=False, index=False)
        print(row["method"], row["recall"], "deva", row["recall_script_mismatch"], flush=True)
    del C
    gc.collect()

print("done", round((time.time() - t0) / 60, 1), "min", flush=True)
