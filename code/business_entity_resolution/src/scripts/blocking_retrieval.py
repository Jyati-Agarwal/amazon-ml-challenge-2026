"""Blocking study, part B: retrieval methods on a stratified S1 sample vs FULL corpora.

Sample: 20,000 train S1 entities (12k US, 8k India; seed 42), preserving their real
ground-truth links. Retrieval corpus is the FULL same-country train S2+S3 pool
(US: ~6.19M, India: ~4.13M records) so recall is not artificially inflated.

Methods:
  M5  rare-token blocking (name tokens, address tokens; df thresholds)
  M6  char n-gram TF-IDF top-k (name; name+addr)
  M7  word TF-IDF top-k (name+addr)
  M9  transliteration-aware char TF-IDF top-k (unidecode on non-ASCII)
  M10 unions incl. exact name+country

Appends rows to experiments/blocking/blocking_results.csv.
"""
import gc
import re
import resource
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from unidecode import unidecode

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "experiments" / "blocking"
OUTDIR.mkdir(parents=True, exist_ok=True)
RESULTS = OUTDIR / "blocking_results.csv"

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
WS = re.compile(r"\s+")
NON_ASCII = re.compile(r"[^\x00-\x7F]")
SEED = 42
SAMPLE = {"US": 12000, "India": 8000}
KS = [10, 25, 50, 100, 200]
TOPN = 200
NTHREADS = 12


def log(*a):
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    print(f"[{time.strftime('%H:%M:%S')}] (peak {mem:.1f}GB)", *a, flush=True)


def strip_latin_accents(s: str) -> str:
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


def translit_series(s: pd.Series) -> pd.Series:
    mask = s.str.contains(NON_ASCII)
    out = s.copy()
    out.loc[mask] = out.loc[mask].map(
        lambda x: WS.sub(" ", PUNCT.sub(" ", unidecode(x).lower())).strip())
    return out


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def append_result(row):
    df = pd.DataFrame([row])
    df.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)


t0 = time.time()
log("loading")
s1 = read_tsv(ROOT / "dataset/train/train_source1.tsv")
s2 = read_tsv(ROOT / "dataset/train/train_source2.tsv")
s3 = read_tsv(ROOT / "dataset/train/train_source3.tsv")
gt = read_tsv(ROOT / "dataset/train/train_ground_truth.tsv")
corpus_all = pd.concat([s2, s3], ignore_index=True)
del s2, s3
gc.collect()

log("normalizing corpus")
corpus_all["nname"] = norm_series(corpus_all["business_name"])
corpus_all["naddr"] = norm_series(corpus_all["business_address"])
corpus_all["deva"] = corpus_all["business_name"].str.contains(DEVANAGARI)
s1["nname"] = norm_series(s1["business_name"])
s1["naddr"] = norm_series(s1["business_address"])

# ---- sample S1, keep real GT links ----
rng = np.random.RandomState(SEED)
samples = []
for ctry, n in SAMPLE.items():
    sub = s1[s1["country"] == ctry]
    samples.append(sub.sample(n, random_state=SEED))
qs = pd.concat(samples, ignore_index=True)
gt_map = gt.set_index("source1_entity_id")["matched_entity_ids"]
qs["matched"] = qs["entity_id"].map(gt_map).fillna("")
qs["matched_list"] = qs["matched"].apply(lambda x: x.split(",") if x else [])
n_true_total = int(qs["matched_list"].str.len().sum())
log(f"sample: {len(qs)} S1, {n_true_total} true matches")

# per-pair frame for breakdowns
pairs = qs[["entity_id", "country", "matched_list"]].explode("matched_list")
pairs = pairs[pairs["matched_list"].notna() & (pairs["matched_list"] != "")]
pairs = pairs.rename(columns={"matched_list": "m_id"})
c_deva = corpus_all.set_index("entity_id")["deva"]
pairs["m_deva"] = pairs["m_id"].map(c_deva).fillna(False)
pairs["m_source"] = pairs["m_id"].str[:2]
log(f"pairs: {len(pairs)}, deva pairs: {int(pairs.m_deva.sum())}")


def summarize(method, hitmask, cand_counts, runtime_s, notes="", k=None):
    """hitmask: bool per pairs row. cand_counts: per sampled-S1 candidate counts."""
    ent = hitmask.groupby(pairs["entity_id"]).all()
    counts = np.asarray(cand_counts)
    row = {
        "method": method + (f"@k{k}" if k else ""), "scope": "sample20k_fullpool",
        "recall": round(float(hitmask.mean()), 4),
        "recall_US": round(float(hitmask[pairs["country"] == "US"].mean()), 4),
        "recall_India": round(float(hitmask[pairs["country"] == "India"].mean()), 4),
        "recall_France": "NA (no France in train)",
        "recall_S2": round(float(hitmask[pairs["m_source"] == "S2"].mean()), 4),
        "recall_S3": round(float(hitmask[pairs["m_source"] == "S3"].mean()), 4),
        "recall_script_mismatch": round(float(hitmask[pairs["m_deva"]].mean()), 4),
        "recall_latin": round(float(hitmask[~pairs["m_deva"]].mean()), 4),
        "entity_recall_all_matches": round(float(ent.mean()), 4),
        "cand_mean": round(float(counts.mean()), 1),
        "cand_median": int(np.median(counts)),
        "cand_p95": int(np.percentile(counts, 95)),
        "cand_p99": int(np.percentile(counts, 99)),
        "cand_max": int(counts.max()),
        "runtime_s": round(runtime_s, 1),
        "notes": notes,
    }
    append_result(row)
    log(row["method"], "recall", row["recall"], "cand_mean", row["cand_mean"])


# storage of candidate sets for union analysis: {method: {s1_id: set(m_ids)}}
cand_store = {}


def eval_from_store(method, store, runtime_s, notes="", k=None):
    hit = pairs.apply(lambda r: r["m_id"] in store.get(r["entity_id"], EMPTY), axis=1)
    counts = qs["entity_id"].map(lambda e: len(store.get(e, EMPTY))).to_numpy()
    summarize(method, hit, counts, runtime_s, notes, k)
    return hit


EMPTY = frozenset()

# =====================================================================
# per-country processing
# =====================================================================
retrieval_configs = [
    ("M6_char34_name", "char_wb", (3, 4), "nname", False),
    ("M6_char34_nameaddr", "char_wb", (3, 4), "namejoint", False),
    ("M7_word_nameaddr", "word", (1, 1), "namejoint", False),
    ("M9_char34_name_translit", "char_wb", (3, 4), "nname", True),
]
for cfg, *_ in retrieval_configs:
    cand_store[cfg] = {}
cand_store["M_exact_name_country"] = {}

for ctry in ("US", "India"):
    log(f"===== country {ctry} =====")
    corp = corpus_all[corpus_all["country"] == ctry].reset_index(drop=True)
    corp_ids = corp["entity_id"].to_numpy()
    q = qs[qs["country"] == ctry].reset_index(drop=True)
    log(f"corpus {len(corp)}, queries {len(q)}")

    corp["namejoint"] = corp["nname"] + " " + corp["naddr"]
    q = q.assign(namejoint=q["nname"] + " " + q["naddr"])

    # ---- exact name+country candidates (for union + reference) ----
    t = time.time()
    qnames = set(q["nname"])
    m = corp["nname"].isin(qnames)
    grp = corp.loc[m].groupby("nname")["entity_id"].apply(set)
    for eid, nm in zip(q["entity_id"], q["nname"]):
        cand_store["M_exact_name_country"][eid] = grp.get(nm, set())
    log("exact name+country candidates built", round(time.time() - t, 1), "s")

    # ---- translit column (used by M9) ----
    t = time.time()
    corp["tname"] = translit_series(corp["nname"])
    q = q.assign(tname=translit_series(q["nname"]))
    log("translit done", round(time.time() - t, 1), "s")

    # ---- TF-IDF retrieval configs ----
    for cfg, analyzer, ngr, col, use_translit in retrieval_configs:
        t = time.time()
        ccol = "tname" if use_translit else col
        qcol = "tname" if use_translit else col
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngr, min_df=3,
                              max_df=0.4, dtype=np.float32)
        X = vec.fit_transform(corp[ccol])
        Q = vec.transform(q[qcol])
        log(cfg, ctry, "tfidf built", X.shape, f"nnz={X.nnz/1e6:.0f}M",
            round(time.time() - t, 1), "s")
        t2 = time.time()
        B = X.T.tocsr()
        del X
        gc.collect()
        C = sp_matmul_topn(Q, B, top_n=TOPN, threshold=0.05, sort=True,
                           n_threads=NTHREADS)
        del B
        gc.collect()
        log(cfg, ctry, "matmul done", round(time.time() - t2, 1), "s")
        # store top-200 ids per query
        store = cand_store[cfg]
        indptr, idx = C.indptr, C.indices
        for i, eid in enumerate(q["entity_id"]):
            ids = corp_ids[idx[indptr[i]:indptr[i + 1]]]
            store[eid] = ids  # sorted by score desc
        del C
        gc.collect()

    # ---- M5 rare-token blocking ----
    t = time.time()
    name_tokens = corp["nname"].str.split()
    df_counter = Counter()
    for toks in name_tokens:
        df_counter.update(set(toks))
    addr_tokens = corp["naddr"].str.split()
    adf_counter = Counter()
    for toks in addr_tokens:
        adf_counter.update(set(toks))
    log("token dfs built", round(time.time() - t, 1), "s")

    for T, key, counter, tokcol in [
        (10, "M5_rare_name_token_df10", df_counter, "nname"),
        (100, "M5_rare_name_token_df100", df_counter, "nname"),
        (10, "M5_rare_addr_token_df10", adf_counter, "naddr"),
    ]:
        t = time.time()
        cand_store.setdefault(key, {})
        # rare tokens used by queries
        q_rare = {}
        wanted = set()
        for eid, text in zip(q["entity_id"], q[tokcol]):
            rt = {tk for tk in text.split() if counter.get(tk, 0) <= T and len(tk) > 1}
            q_rare[eid] = rt
            wanted |= rt
        # scan corpus once, build postings for wanted tokens
        postings = {}
        src_toks = name_tokens if tokcol == "nname" else addr_tokens
        for i, toks in enumerate(src_toks):
            for tk in set(toks):
                if tk in wanted:
                    postings.setdefault(tk, []).append(i)
        for eid in q["entity_id"]:
            idxs = set()
            for tk in q_rare[eid]:
                idxs.update(postings.get(tk, ()))
            cand_store[key][eid] = set(corp_ids[list(idxs)]) if idxs else set()
        log(key, ctry, "done", round(time.time() - t, 1), "s")

    del corp, name_tokens, addr_tokens, df_counter, adf_counter, postings
    gc.collect()

# =====================================================================
# evaluation
# =====================================================================
log("evaluating stores")

# exact name+country
eval_from_store("M_exact_name_country_sample", cand_store["M_exact_name_country"], 0,
                notes="sample replica of M3 for union math")

# token methods
for key in ("M5_rare_name_token_df10", "M5_rare_name_token_df100", "M5_rare_addr_token_df10"):
    eval_from_store(key, cand_store[key], 0)

# retrieval methods at budgets
memb = {}  # method -> k -> {eid: set}
for cfg, *_ in retrieval_configs:
    memb[cfg] = {}
    for k in KS:
        store_k = {eid: set(ids[:k]) for eid, ids in cand_store[cfg].items()}
        memb[cfg][k] = store_k
        eval_from_store(cfg, store_k, 0, k=k)

# =====================================================================
# M8 / M10 unions (at moderate budgets)
# =====================================================================
def union_stores(stores):
    out = {}
    for st in stores:
        for eid, ids in st.items():
            if eid in out:
                out[eid] = out[eid] | set(ids)
            else:
                out[eid] = set(ids)
    return out


unions = [
    ("M8_char100_word50", [memb["M6_char34_name"][100], memb["M7_word_nameaddr"][50]]),
    ("M10_exact+char100", [cand_store["M_exact_name_country"], memb["M6_char34_name"][100]]),
    ("M10_exact+char100+word50",
     [cand_store["M_exact_name_country"], memb["M6_char34_name"][100],
      memb["M7_word_nameaddr"][50]]),
    ("M10_exact+char100+word50+translit50",
     [cand_store["M_exact_name_country"], memb["M6_char34_name"][100],
      memb["M7_word_nameaddr"][50], memb["M9_char34_name_translit"][50]]),
    ("M10_exact+char200+word100+translit100+chNA100",
     [cand_store["M_exact_name_country"], memb["M6_char34_name"][200],
      memb["M7_word_nameaddr"][100], memb["M9_char34_name_translit"][100],
      memb["M6_char34_nameaddr"][100]]),
]
union_hits = {}
for name, stores in unions:
    t = time.time()
    u = union_stores(stores)
    union_hits[name] = eval_from_store(name, u, time.time() - t)

# ---- diagnostics: misses of the biggest union ----
best = "M10_exact+char200+word100+translit100+chNA100"
miss = pairs[~union_hits[best]]
if len(miss):
    c_idx = corpus_all.set_index("entity_id")
    s1_idx = qs.set_index("entity_id")
    rows = []
    for _, r in miss.head(300).iterrows():
        rows.append({
            "s1_id": r["entity_id"],
            "m_id": r["m_id"],
            "country": r["country"],
            "m_deva": r["m_deva"],
            "s1_name": s1_idx.loc[r["entity_id"], "business_name"],
            "m_name": c_idx.loc[r["m_id"], "business_name"],
            "s1_addr": s1_idx.loc[r["entity_id"], "business_address"],
            "m_addr": c_idx.loc[r["m_id"], "business_address"],
        })
    pd.DataFrame(rows).to_csv(OUTDIR / "missed_by_best_union_sample.csv", index=False)

log("ALL DONE", round((time.time() - t0) / 60, 1), "min")
