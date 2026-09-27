"""Blocking follow-up: k-budget comparison, asymmetric country budgets, rescue
channels, and miss-tail categorization.

Same validation design as blocking_retrieval.py: 20k stratified S1 sample (seed 42),
FULL same-country train S2+S3 pools. Recomputes only the two configs that matter
(word and char34 TF-IDF over name+addr), persists top-200 candidate lists to parquet
(experiments/blocking/cand_sample/) so downstream analysis never recomputes retrieval.

Rescue channels tested (marginal gain over base union word50+char50+exact_name):
  R1 exact normalized address (+country)
  R2 rare name token (corpus df<=100)

Outputs: experiments/blocking/blocking_budget_comparison.csv,
         experiments/blocking/miss_tail_categories.json,
         experiments/blocking/missed_by_base_union.csv
"""
import gc
import json
import re
import resource
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "experiments" / "blocking"
CANDDIR = OUTDIR / "cand_sample"
CANDDIR.mkdir(parents=True, exist_ok=True)
BUDGET_CSV = OUTDIR / "blocking_budget_comparison.csv"

INDIC = re.compile(r"[ऀ-৿਀-૿஀-౿ಀ-ൿ]")
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
WS = re.compile(r"\s+")
SEED = 42
SAMPLE = {"US": 12000, "India": 8000}
TOPN = 200
NTHREADS = 12


def log(*a):
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    print(f"[{time.strftime('%H:%M:%S')}] (peak {mem:.1f}GB)", *a, flush=True)


def strip_latin_accents(s):
    out = [ch for ch in unicodedata.normalize("NFD", s)
           if not (unicodedata.category(ch) == "Mn" and ord(ch) < 0x0900)]
    return unicodedata.normalize("NFC", "".join(out))


def norm_series(s):
    s = s.map(lambda x: strip_latin_accents(x).lower())
    s = s.str.replace(PUNCT, " ", regex=True)
    return s.str.replace(WS, " ", regex=True).str.strip()


def read_tsv(p):
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)


t0 = time.time()
log("loading")
s1 = read_tsv(ROOT / "dataset/train/train_source1.tsv")
s2 = read_tsv(ROOT / "dataset/train/train_source2.tsv")
s3 = read_tsv(ROOT / "dataset/train/train_source3.tsv")
gt = read_tsv(ROOT / "dataset/train/train_ground_truth.tsv")
corpus_all = pd.concat([s2, s3], ignore_index=True)
del s2, s3
gc.collect()

log("normalizing")
corpus_all["nname"] = norm_series(corpus_all["business_name"])
corpus_all["naddr"] = norm_series(corpus_all["business_address"])
s1["nname"] = norm_series(s1["business_name"])
s1["naddr"] = norm_series(s1["business_address"])

# ---- sample (identical to blocking_retrieval.py) ----
samples = []
for ctry, n in SAMPLE.items():
    samples.append(s1[s1["country"] == ctry].sample(n, random_state=SEED))
qs = pd.concat(samples, ignore_index=True)
gt_map = gt.set_index("source1_entity_id")["matched_entity_ids"]
qs["matched_list"] = qs["entity_id"].map(gt_map).fillna("").apply(
    lambda x: x.split(",") if x else [])
pairs = qs[["entity_id", "country", "matched_list"]].explode("matched_list")
pairs = pairs[pairs["matched_list"].notna() & (pairs["matched_list"] != "")]
pairs = pairs.rename(columns={"matched_list": "m_id"}).reset_index(drop=True)
c_idx = corpus_all.set_index("entity_id")
pairs["m_name"] = pairs["m_id"].map(c_idx["business_name"])
pairs["m_naddr"] = pairs["m_id"].map(c_idx["naddr"])
pairs["m_indic"] = pairs["m_name"].str.contains(INDIC)
pairs["m_source"] = pairs["m_id"].str[:2]
log(f"sample {len(qs)} S1, {len(pairs)} pairs, indic {int(pairs.m_indic.sum())}")

stores = {}          # name -> {eid: np.array of ids (ranked)} or {eid: set}
runtimes = {}

for ctry in ("US", "India"):
    log(f"===== {ctry} =====")
    corp = corpus_all[corpus_all["country"] == ctry].reset_index(drop=True)
    corp_ids = corp["entity_id"].to_numpy()
    q = qs[qs["country"] == ctry].reset_index(drop=True)
    corp["joint"] = corp["nname"] + " " + corp["naddr"]
    q = q.assign(joint=q["nname"] + " " + q["naddr"])

    # exact name
    t = time.time()
    qnames = set(q["nname"])
    grp = corp.loc[corp["nname"].isin(qnames)].groupby("nname")["entity_id"].apply(set)
    st = stores.setdefault("exact_name", {})
    for eid, nm in zip(q["entity_id"], q["nname"]):
        st[eid] = grp.get(nm, set())
    runtimes[f"exact_name_{ctry}"] = time.time() - t

    # R1: exact normalized address (non-empty)
    t = time.time()
    qaddrs = set(q["naddr"]) - {""}
    grpa = corp.loc[corp["naddr"].isin(qaddrs)].groupby("naddr")["entity_id"].apply(set)
    st = stores.setdefault("exact_addr", {})
    for eid, ad in zip(q["entity_id"], q["naddr"]):
        st[eid] = grpa.get(ad, set()) if ad else set()
    runtimes[f"exact_addr_{ctry}"] = time.time() - t

    # R2: rare name token df<=100
    t = time.time()
    dfc = Counter()
    toks_col = corp["nname"].str.split()
    for tk in toks_col:
        dfc.update(set(tk))
    wanted = set()
    q_rare = {}
    for eid, text in zip(q["entity_id"], q["nname"]):
        rt = {tk for tk in text.split() if dfc.get(tk, 0) <= 100 and len(tk) > 1}
        q_rare[eid] = rt
        wanted |= rt
    postings = {}
    for i, tk_set in enumerate(toks_col):
        for tk in set(tk_set):
            if tk in wanted:
                postings.setdefault(tk, []).append(i)
    st = stores.setdefault("rare_token", {})
    for eid in q["entity_id"]:
        idxs = set()
        for tk in q_rare[eid]:
            idxs.update(postings.get(tk, ()))
        st[eid] = set(corp_ids[list(idxs)]) if idxs else set()
    runtimes[f"rare_token_{ctry}"] = time.time() - t
    del postings, toks_col, dfc
    gc.collect()

    # retrieval configs
    for cfg, analyzer, ngr in [("word", "word", (1, 1)), ("char", "char_wb", (3, 4))]:
        t = time.time()
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngr, min_df=3,
                              max_df=0.4, dtype=np.float32)
        X = vec.fit_transform(corp["joint"])
        Q = vec.transform(q["joint"])
        B = X.T.tocsr()
        del X
        gc.collect()
        t_mat = time.time()
        C = sp_matmul_topn(Q, B, top_n=TOPN, threshold=0.05, sort=True,
                           n_threads=NTHREADS)
        runtimes[f"{cfg}_matmul_{ctry}"] = time.time() - t_mat
        runtimes[f"{cfg}_total_{ctry}"] = time.time() - t
        del B
        gc.collect()
        st = stores.setdefault(cfg, {})
        indptr, idx = C.indptr, C.indices
        rows = []
        for i, eid in enumerate(q["entity_id"]):
            ids = corp_ids[idx[indptr[i]:indptr[i + 1]]]
            st[eid] = ids
            rows.append(pd.DataFrame({"s1_id": eid, "rank": np.arange(len(ids)),
                                      "cand_id": ids}))
        pd.concat(rows, ignore_index=True).to_parquet(
            CANDDIR / f"{cfg}_{ctry}.parquet", index=False)
        del C, rows
        gc.collect()
        log(cfg, ctry, "done", round(runtimes[f"{cfg}_total_{ctry}"], 1), "s",
            f"(matmul {runtimes[f'{cfg}_matmul_{ctry}']:.0f}s)")
    del corp
    gc.collect()

(OUTDIR / "budget_runtimes.json").write_text(json.dumps(
    {k: round(v, 1) for k, v in runtimes.items()}, indent=1))

# =====================================================================
# budget evaluation
# =====================================================================
EMPTY = frozenset()


def build_union(spec):
    """spec: list of (store_name, k or None). Returns {eid: set}."""
    out = {}
    for eid in qs["entity_id"]:
        s = set()
        for name, k in spec:
            v = stores[name].get(eid, EMPTY)
            if k is not None:
                v = v[:k]
            s.update(v)
        out[eid] = s
    return out


def evaluate(label, union, notes=""):
    hit = np.fromiter((m in union.get(e, EMPTY)
                       for e, m in zip(pairs["entity_id"], pairs["m_id"])),
                      dtype=bool, count=len(pairs))
    hs = pd.Series(hit)
    ent = hs.groupby(pairs["entity_id"]).all()
    counts = qs["entity_id"].map(lambda e: len(union.get(e, EMPTY))).to_numpy()
    row = {
        "strategy": label,
        "pair_recall": round(float(hit.mean()), 4),
        "recall_US": round(float(hs[pairs["country"] == "US"].mean()), 4),
        "recall_India": round(float(hs[pairs["country"] == "India"].mean()), 4),
        "recall_script_mismatch": round(float(hs[pairs["m_indic"]].mean()), 4),
        "entity_all_match_recall": round(float(ent.mean()), 4),
        "cand_mean": round(float(counts.mean()), 1),
        "cand_median": int(np.median(counts)),
        "cand_p95": int(np.percentile(counts, 95)),
        "cand_p99": int(np.percentile(counts, 99)),
        "cand_max": int(counts.max()),
        "notes": notes,
    }
    hdr = not BUDGET_CSV.exists()
    pd.DataFrame([row]).to_csv(BUDGET_CSV, mode="a", header=hdr, index=False)
    log(label, "recall", row["pair_recall"], "ent", row["entity_all_match_recall"],
        "cand", row["cand_mean"])
    return hit, counts


# singles
for cfg in ("word", "char"):
    for k in (25, 50, 100, 200):
        evaluate(f"{cfg}@{k}", build_union([(cfg, k)]))
evaluate("exact_name", build_union([("exact_name", None)]))
evaluate("exact_addr(R1 alone)", build_union([("exact_addr", None)]))
evaluate("rare_token(R2 alone)", build_union([("rare_token", None)]))

# symmetric unions
base_specs = {
    "w25+c25+e": [("word", 25), ("char", 25), ("exact_name", None)],
    "w50+c50+e": [("word", 50), ("char", 50), ("exact_name", None)],
    "w100+c50+e": [("word", 100), ("char", 50), ("exact_name", None)],
    "w50+c100+e": [("word", 50), ("char", 100), ("exact_name", None)],
    "w100+c100+e": [("word", 100), ("char", 100), ("exact_name", None)],
    "w200+c200+e": [("word", 200), ("char", 200), ("exact_name", None)],
}
unions = {}
for label, spec in base_specs.items():
    unions[label] = build_union(spec)
    evaluate(label, unions[label])

# asymmetric country budgets: US small, India big
asym = {}
small = base_specs["w50+c50+e"]
big = base_specs["w100+c100+e"]
for eid, ctry in zip(qs["entity_id"], qs["country"]):
    spec = small if ctry == "US" else big
    s = set()
    for name, k in spec:
        v = stores[name].get(eid, EMPTY)
        if k is not None:
            v = v[:k]
        s.update(v)
    asym[eid] = s
evaluate("asym US(w50c50e)/India(w100c100e)", asym,
         notes="country-conditional budget, open-set: default=big budget")

# rescue channels on top of base union w50+c50+e
base = unions["w50+c50+e"]
base_hit, base_counts = None, None
base_hit = np.fromiter((m in base.get(e, EMPTY)
                        for e, m in zip(pairs["entity_id"], pairs["m_id"])),
                       dtype=bool, count=len(pairs))
for rname, store_name in [("R1 exact_addr", "exact_addr"),
                          ("R2 rare_token", "rare_token")]:
    aug = {eid: base[eid] | set(stores[store_name].get(eid, EMPTY))
           for eid in base}
    hit, counts = evaluate(f"w50+c50+e + {rname}", aug)
    extra_pairs = int((hit & ~base_hit).sum())
    hs = pd.Series(hit)
    ent_aug = hs.groupby(pairs["entity_id"]).all()
    bs = pd.Series(base_hit)
    ent_base = bs.groupby(pairs["entity_id"]).all()
    extra_ents = int((ent_aug & ~ent_base).sum())
    base_cand = qs["entity_id"].map(lambda e: len(base.get(e, EMPTY))).to_numpy()
    log(f"  {rname}: +{extra_pairs} pairs (+{100*extra_pairs/len(pairs):.3f}pp), "
        f"+{extra_ents} fully-recovered entities, "
        f"+{float((counts-base_cand).mean()):.1f} cands/S1 mean")

# both rescues
aug2 = {eid: base[eid] | set(stores["exact_addr"].get(eid, EMPTY))
        | set(stores["rare_token"].get(eid, EMPTY)) for eid in base}
evaluate("w50+c50+e + R1 + R2", aug2)

# =====================================================================
# miss-tail categorization for base union
# =====================================================================
miss = pairs[~base_hit].copy()
s1_map = qs.set_index("entity_id")
miss["s1_name"] = miss["entity_id"].map(s1_map["business_name"])
miss["s1_naddr"] = miss["entity_id"].map(s1_map["naddr"])
miss["s1_nname"] = miss["entity_id"].map(s1_map["nname"])
miss["m_nname"] = miss["m_id"].map(c_idx["nname"])

def tokset(x):
    return set(str(x).split())

name_ov = [len(tokset(a) & tokset(b)) for a, b in zip(miss["s1_nname"], miss["m_nname"])]
miss["zero_name_overlap"] = np.array(name_ov) == 0
miss["empty_m_addr"] = miss["m_naddr"] == ""
miss["script_mismatch"] = miss["m_indic"]
miss["domain_name"] = miss["m_name"].str.contains(r"\.(?:com|net|org|in)\b", case=False)
# addr token overlap
addr_ov = [len(tokset(a) & tokset(b)) for a, b in zip(miss["s1_naddr"], miss["m_naddr"])]
miss["low_addr_overlap"] = (np.array(addr_ov) <= 1) & ~miss["empty_m_addr"]
# severe typo: no exact tokens shared but high char similarity
import difflib
miss["severe_typo"] = [
    (z and difflib.SequenceMatcher(None, a, b).ratio() > 0.6)
    for z, a, b in zip(miss["zero_name_overlap"], miss["s1_nname"], miss["m_nname"])]

n_miss = len(miss)
cats = {
    "total_misses": n_miss,
    "miss_rate_pairs": round(float(n_miss / len(pairs)), 4),
    "zero_name_token_overlap": int(miss["zero_name_overlap"].sum()),
    "severe_typo_subset": int(miss["severe_typo"].sum()),
    "script_mismatch_indic": int(miss["script_mismatch"].sum()),
    "empty_match_address": int(miss["empty_m_addr"].sum()),
    "low_addr_overlap(<=1 tok, nonempty)": int(miss["low_addr_overlap"].sum()),
    "domain_style_name": int(miss["domain_name"].sum()),
    "by_country": miss["country"].value_counts().to_dict(),
    "by_source": miss["m_source"].value_counts().to_dict(),
    "note": "categories overlap; base union = w50+c50+exact_name",
}
(OUTDIR / "miss_tail_categories.json").write_text(json.dumps(cats, indent=1))
cols = ["entity_id", "m_id", "country", "s1_name", "m_name", "s1_naddr", "m_naddr",
        "zero_name_overlap", "empty_m_addr", "script_mismatch", "domain_name",
        "severe_typo", "low_addr_overlap"]
miss[cols].head(500).to_csv(OUTDIR / "missed_by_base_union.csv", index=False)
log("miss categories:", json.dumps(cats))
log("ALL DONE", round((time.time() - t0) / 60, 1), "min")
