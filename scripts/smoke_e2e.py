"""Session 3 integration smoke test — tiny end-to-end pipeline in .venv312.

Proves the components work together on a few hundred S1 / few thousand S2+S3 rows.
NOT the production pipeline: features are provisional (subset of MODEL_FEATURE_SPEC),
the model is a throwaway, and nothing here tunes anything.

Parts:
  A. pandas-3 compatibility checks for idioms used in scripts/blocking_*.py etc.
  B. tiny e2e: normalize -> country partition -> word+char TF-IDF -> sp_matmul_topn
     top-k -> exact-name channel -> union/dedupe -> provisional features -> LightGBM.
  C. output schema: writes matching_results.tsv / candidate_pairs.tsv to a temp dir
     and runs utils/validate_submission.py against a fabricated mini test dir.
  D. prints per-stage runtime and peak RSS.

Temp outputs go to experiments/smoke_tmp/ (git-ignored) and are removed on success.
Run:  .venv312/bin/python scripts/smoke_e2e.py
"""
import re
import resource
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "experiments" / "smoke_tmp"
SEED = 42
N_S1 = {"US": 200, "India": 100}
N_DISTRACT = 5000
TOPK = 20

PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
WS = re.compile(r"\s+")
NON_ASCII = re.compile(r"[^\x00-\x7F]")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

FAILURES = []
TIMINGS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def timed(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    TIMINGS.append((label, dt))
    print(f"  ({label}: {dt:.2f}s)")
    return out


# same normalization as scripts/blocking_retrieval.py
def strip_latin_accents(s):
    out = []
    for ch in unicodedata.normalize("NFD", s):
        if unicodedata.category(ch) == "Mn" and ord(ch) < 0x0900:
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def norm_series(s):
    s = s.map(lambda x: strip_latin_accents(x).lower())
    s = s.str.replace(PUNCT, " ", regex=True)
    s = s.str.replace(WS, " ", regex=True).str.strip()
    return s


def read_tsv(path, **kw):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=3, **kw)


# ---------------------------------------------------------------- Part A
def part_a():
    print("\n=== Part A: pandas-3 idiom checks (patterns from project scripts) ===")
    print(f"  pandas {pd.__version__}, numpy {np.__version__}")
    df = read_tsv(ROOT / "dataset/train/train_source2.tsv", nrows=5000)
    check("read_tsv keep_default_na/quoting=3", len(df) == 5000 and df["business_address"].dtype is not None)
    check("empty addresses read as ''", (df["business_address"] == "").any(),
          f"{int((df['business_address'] == '').sum())} empty in 5k rows")

    # blocking_retrieval.py:110-120 idioms
    m = pd.Series(["A,B", None, "C"]).fillna("")
    lst = m.apply(lambda x: x.split(",") if x else [])
    try:
        n = int(lst.str.len().sum())
        check(".str.len() on list column", n == 3, f"got {n}")
    except Exception as e:
        check(".str.len() on list column", False, f"{type(e).__name__}: {e}")
    try:
        flags = pd.Series(["x", "y"]).map({"x": True}.get).fillna(False)
        check(".map().fillna(False)", bool(flags.tolist() == [True, False]))
    except Exception as e:
        check(".map().fillna(False)", False, f"{type(e).__name__}: {e}")

    ex = pd.DataFrame({"id": ["a"], "lst": [["1", "2"]]}).explode("lst")
    check("explode()", len(ex) == 2)
    grp = df.head(100).groupby("country")["entity_id"].apply(set)
    check("groupby.apply(set)", isinstance(grp.iloc[0], set))
    # pandas 3: string-col .values is ArrowStringArray, no longer np.ndarray.
    # Project scripts only rely on numeric .values (ndarray, still true), string
    # .to_numpy() (ndarray), and fancy-indexing/set() over string .values —
    # all verified here (real_blocking_validation.py:373-374 pattern).
    check("numeric .values is ndarray", isinstance(pd.Series([1.0]).values, np.ndarray))
    check("string .to_numpy() is ndarray",
          isinstance(df["entity_id"].to_numpy(), np.ndarray))
    ids = df["entity_id"].values[np.argsort(df["entity_id"].to_numpy())][:3]
    check("string .values fancy-index + set()", len(set(ids)) == 3,
          f".values type is {type(df['entity_id'].values).__name__}")
    norm = norm_series(df["business_name"].head(500))
    check("norm_series (regex str.replace chain)", norm.notna().all())
    sampled = df.sample(50, random_state=SEED)
    check("sample(random_state)", len(sampled) == 50)
    counts = df["country"].map(df["country"].value_counts()).fillna(0)
    check(".map(value_counts).fillna(0)", counts.notna().all())


# ---------------------------------------------------------------- Part B
def load_sample():
    s1 = read_tsv(ROOT / "dataset/train/train_source1.tsv")
    gt = read_tsv(ROOT / "dataset/train/train_ground_truth.tsv")
    gt_map = gt.set_index("source1_entity_id")["matched_entity_ids"]

    rng = np.random.RandomState(SEED)
    picks = []
    for ctry, n in N_S1.items():
        sub = s1[s1["country"] == ctry]
        picks.append(sub.sample(n, random_state=SEED))
    qs = pd.concat(picks, ignore_index=True)
    qs["matched"] = qs["entity_id"].map(gt_map).fillna("")
    qs["matched_list"] = qs["matched"].apply(lambda x: x.split(",") if x else [])
    true_ids = {m for lst in qs["matched_list"] for m in lst}

    import pyarrow.csv as pv
    frames = []
    for src in ("train_source2.tsv", "train_source3.tsv"):
        tbl = pv.read_csv(ROOT / "dataset/train" / src,
                          parse_options=pv.ParseOptions(delimiter="\t"),
                          convert_options=pv.ConvertOptions(strings_can_be_null=False,
                                                            column_types=None))
        frames.append(tbl.to_pandas())
    corpus_full = pd.concat(frames, ignore_index=True)
    corpus_full["business_address"] = corpus_full["business_address"].fillna("")
    is_true = corpus_full["entity_id"].isin(true_ids)
    same_ctry = corpus_full["country"].isin(N_S1.keys())
    distract = corpus_full[~is_true & same_ctry].sample(N_DISTRACT, random_state=SEED)
    corpus = pd.concat([corpus_full[is_true], distract], ignore_index=True)
    del corpus_full, frames
    return qs, corpus, true_ids


def part_b():
    print("\n=== Part B: tiny end-to-end pipeline ===")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import sp_matmul_topn
    from rapidfuzz.distance import JaroWinkler, Levenshtein
    from unidecode import unidecode
    import lightgbm as lgb

    qs, corpus, true_ids = timed("load + sample", load_sample)
    print(f"  sample: {len(qs)} S1, corpus {len(corpus)} rows, "
          f"{len(true_ids)} true-match ids ({int(corpus['entity_id'].isin(true_ids).sum())} present)")
    check("Devanagari rows present in corpus",
          corpus["business_name"].str.contains(DEVANAGARI).any())
    check("empty addresses present in corpus",
          (corpus["business_address"].str.strip() == "").any())

    def normalize(df):
        df = df.copy()
        df["nname"] = norm_series(df["business_name"])
        df["naddr"] = norm_series(df["business_address"])
        df["joint"] = df["nname"] + " " + df["naddr"]
        return df

    qs = timed("normalization", lambda: normalize(qs))
    corpus = normalize(corpus)
    check("normalization handles Unicode without NaN",
          qs["joint"].notna().all() and corpus["joint"].notna().all())

    # country partition + retrieval
    cand = {}  # s1_id -> set of corpus entity_ids
    t_tfidf = t_topk = 0.0
    for ctry in N_S1:
        q = qs[qs["country"] == ctry].reset_index(drop=True)
        c = corpus[corpus["country"] == ctry].reset_index(drop=True)
        c_ids = c["entity_id"].to_numpy()
        for analyzer, ngr in (("word", (1, 1)), ("char_wb", (3, 4))):
            t0 = time.perf_counter()
            vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngr, min_df=1,
                                  dtype=np.float32)
            X = vec.fit_transform(c["joint"])
            Q = vec.transform(q["joint"])
            t_tfidf += time.perf_counter() - t0
            check(f"TF-IDF dims {ctry}/{analyzer}",
                  X.shape == (len(c), len(vec.vocabulary_)) and Q.shape[0] == len(q),
                  f"X{X.shape} Q{Q.shape}")
            t0 = time.perf_counter()
            C = sp_matmul_topn(Q, X.T.tocsr(), top_n=TOPK, threshold=0.05,
                               sort=True, n_threads=14)
            t_topk += time.perf_counter() - t0
            for i, eid in enumerate(q["entity_id"]):
                ids = c_ids[C.indices[C.indptr[i]:C.indptr[i + 1]]]
                cand.setdefault(eid, set()).update(ids)
        # exact normalized-name channel
        grp = c.groupby("nname")["entity_id"].apply(set)
        for eid, nm in zip(q["entity_id"], q["nname"]):
            cand.setdefault(eid, set()).update(grp.get(nm, set()))
    TIMINGS.append(("TF-IDF fit/transform (4x)", t_tfidf))
    TIMINGS.append(("sparse top-k retrieval (4x)", t_topk))
    print(f"  (TF-IDF: {t_tfidf:.2f}s, top-k: {t_topk:.2f}s)")

    # union + dedupe -> pair list
    pair_rows = [(s1_id, m) for s1_id, ids in cand.items() for m in ids]
    pairs = pd.DataFrame(pair_rows, columns=["s1_id", "cand_id"])
    n_before = len(pairs)
    pairs = pairs.drop_duplicates()
    check("no duplicate pairs after union+dedupe",
          len(pairs) == n_before == len(set(map(tuple, pair_rows))),
          f"{len(pairs)} pairs")

    # candidate IDs map back to real corpus rows; countries never cross
    c_idx = corpus.set_index("entity_id")
    check("all candidate ids exist in corpus", pairs["cand_id"].isin(c_idx.index).all())
    s1_ctry = qs.set_index("entity_id")["country"]
    cand_ctry = pairs["cand_id"].map(c_idx["country"])
    check("country partition respected",
          (pairs["s1_id"].map(s1_ctry) == cand_ctry).all())

    # blocking recall on the tiny sample (informational only)
    gt_sets = dict(zip(qs["entity_id"], qs["matched_list"].apply(set)))
    hits = sum(len(cand.get(e, set()) & g) for e, g in gt_sets.items())
    total = sum(len(g) for g in gt_sets.values())
    print(f"  tiny-sample blocking recall: {hits}/{total} = {hits/total:.3f} (not representative)")

    # provisional features (NOT the final 16-feature spec)
    def translit(s):
        return unidecode(s) if NON_ASCII.search(s) else s

    def feats(r1, r2):
        n1, n2 = translit(r1["nname"]), translit(r2["nname"])
        a1, a2 = r1["naddr"], r2["naddr"]
        g1, g2 = ({n[i:i+3] for i in range(len(n)-2)} for n in (n1, n2))
        t1, t2 = set(a1.split()), set(a2.split())
        d1 = {t for t in a1.split() if t.isdigit()}
        d2 = {t for t in a2.split() if t.isdigit()}
        return (
            len(g1 & g2) / max(len(g1 | g2), 1),
            JaroWinkler.similarity(n1, n2),
            len(t1 & t2) / max(min(len(t1), len(t2)), 1),
            Levenshtein.normalized_similarity(a1, a2),
            len(d1 & d2) / max(len(d1 | d2), 1),
            float(a1 == "" or a2 == ""),
        )

    q_idx = qs.set_index("entity_id")

    def build_features():
        return np.array([feats(q_idx.loc[a], c_idx.loc[b])
                         for a, b in zip(pairs["s1_id"], pairs["cand_id"])],
                        dtype=np.float32)

    F = timed(f"feature calc ({len(pairs)} pairs, single-core)", build_features)
    check("feature matrix finite (no NaN/inf)", np.isfinite(F).all(),
          f"shape {F.shape}")
    check("feature values in sane range", F.min() >= 0 and F[:, :5].max() <= 1.0 + 1e-6)

    y = np.array([int(b in gt_sets.get(a, set()))
                  for a, b in zip(pairs["s1_id"], pairs["cand_id"])])
    print(f"  labels: {y.sum()} pos / {len(y)} pairs")

    def train_predict():
        model = lgb.train({"objective": "binary", "verbose": -1, "num_threads": 14},
                          lgb.Dataset(F, y), num_boost_round=50)
        return model.predict(F)

    prob = timed("LightGBM train+predict", train_predict)
    check("prediction shape matches pairs", prob.shape == (len(pairs),))
    check("predictions are probabilities", 0 <= prob.min() and prob.max() <= 1)
    pairs["prob"] = prob
    return qs, corpus, pairs, cand


# ---------------------------------------------------------------- Part C
def part_c(qs, corpus, pairs, cand):
    print("\n=== Part C: output schema + validator ===")
    TMP.mkdir(parents=True, exist_ok=True)

    pred = pairs[pairs["prob"] >= 0.7].groupby("s1_id")["cand_id"].apply(
        lambda s: ",".join(sorted(s)))
    matching = pd.DataFrame({"source1_entity_id": qs["entity_id"]})
    matching["matched_entity_ids"] = matching["source1_entity_id"].map(pred).fillna("")
    matching.to_csv(TMP / "matching_results.tsv", sep="\t", index=False)

    cand_out = pd.DataFrame({"source1_entity_id": qs["entity_id"]})
    cand_out["candidate_entity_ids"] = cand_out["source1_entity_id"].map(
        lambda e: ",".join(sorted(cand.get(e, set())))).fillna("")
    cand_out.to_csv(TMP / "candidate_pairs.tsv", sep="\t", index=False)

    with open(TMP / "matching_results.tsv") as f:
        hdr = f.readline().rstrip("\n").split("\t")
    check("matching_results.tsv header/delimiter",
          hdr == ["source1_entity_id", "matched_entity_ids"])

    # every predicted match must be inside the candidate set
    ok = all(set(m.split(",")) <= cand.get(e, set())
             for e, m in zip(matching["source1_entity_id"],
                             matching["matched_entity_ids"]) if m)
    check("predictions subset of candidates", ok)

    # fabricate a mini test dir so utils/validate_submission.py fully runs
    mini = TMP / "mini_test"
    mini.mkdir(exist_ok=True)
    qs[["entity_id", "business_name", "business_address", "country"]].to_csv(
        mini / "test_source1.tsv", sep="\t", index=False)
    for src in ("S2", "S3"):
        sub = corpus[corpus["entity_id"].str.startswith(src)]
        sub[["entity_id", "business_name", "business_address", "country"]].to_csv(
            mini / f"test_source{src[1]}.tsv", sep="\t", index=False)

    r = subprocess.run(
        [sys.executable, str(ROOT / "utils/validate_submission.py"),
         "--matching", str(TMP / "matching_results.tsv"),
         "--candidate", str(TMP / "candidate_pairs.tsv"),
         "--test-dir", str(mini), "--check-ids"],
        capture_output=True, text=True)
    tail = "\n".join(r.stdout.strip().splitlines()[-6:])
    print("  validator output (tail):\n    " + tail.replace("\n", "\n    "))
    check("utils/validate_submission.py exit 0", r.returncode == 0,
          f"rc={r.returncode}")


# ---------------------------------------------------------------- main
def main():
    part_a()
    out = part_b()
    part_c(*out)

    print("\n=== Part D: timings ===")
    for label, dt in TIMINGS:
        print(f"  {label:44s} {dt:7.2f}s")
    print(f"  peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB")

    if FAILURES:
        print(f"\nSMOKE TEST: {len(FAILURES)} FAILURE(S)")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    shutil.rmtree(TMP, ignore_errors=True)
    print("\nSMOKE TEST: ALL PASS (temp outputs removed)")


if __name__ == "__main__":
    main()
