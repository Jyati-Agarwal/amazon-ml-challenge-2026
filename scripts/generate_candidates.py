#!/usr/bin/env python3
"""Production candidate generation for Business Entity Resolution.

Implements the locked blocking architecture (docs/BLOCKING_IMPLEMENTATION_PLAN.md):
per-country partition -> normalize -> word TF-IDF top-k + char_wb(3,4) TF-IDF top-k
+ exact normalized-name (capped) -> union -> dedupe -> parquet parts.

Open-set country budgets: BUDGETS dict lookup with a conservative default for any
label not present (France / unseen countries automatically get the default).

Ground truth is NEVER read here. Evaluation lives in scripts/evaluate_candidates.py.

Resumable at (country, chunk) granularity via <out>/manifest.json: completed chunk
part-files are skipped on rerun (vectorizers are refit deterministically).

S1 sharding (--shard-id/--num-shards): deterministic MD5(entity_id)%num_shards
assignment so N workers can split S1; every shard still searches the FULL
same-country S2/S3 corpus (TF-IDF models are fit on the corpus only, so shard
outputs are exactly the unsharded rows for those S1s). See docs/PARALLEL_SHARDING.md.

Examples (production env = .venv312; validated 20k benchmark ran under .venv 3.9):
  20k validation sample:
    .venv312/bin/python scripts/generate_candidates.py \
        --out experiments/candidates/val20k --sample "US:12000,India:8000" --seed 42
  full train, worker for shard 0 of 3:
    .venv312/bin/python scripts/generate_candidates.py \
        --out experiments/candidates/train_sharded/shard_0 --shard-id 0 --num-shards 3
"""
import argparse
import gc
import hashlib
import json
import os
import re
import resource
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# v1 (production DEFAULT — the validated benchmark normalization): Python re's \w
# does not match combining marks (category Mn), so `[^\w\s] -> " "` strips Indic
# matras/viramas ("प्राइवेट" -> "प र इव ट"). Measured on the 20k validation this is
# recall-NEUTRAL vs preserving them (0.9794 vs 0.9792 pair recall) because Latin
# addresses carry the retrieval signal for Indic-name records; we keep v1 because
# it exactly matches the validated benchmark and yields slightly fewer candidates.
# v2 (--matra-norm): keeps the full Indic block ऀ-ൿ (matras survive; danda -> space).
DANDA = re.compile(r"[।॥]")
PUNCT_V2 = re.compile(r"[^\w\sऀ-ൿ]", re.UNICODE)
PUNCT_V1 = re.compile(r"[^\w\s]", re.UNICODE)
PUNCT = PUNCT_V1
WS = re.compile(r"\s+")

CH_WORD, CH_CHAR, CH_EXACT = 1, 2, 4


def log(*a):
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    print("[%s] (peak %.1fGB)" % (time.strftime("%H:%M:%S"), mem), *a, flush=True)


# ---------------------------------------------------------------- normalization
def strip_latin_accents(s):
    out = [ch for ch in unicodedata.normalize("NFD", s)
           if not (unicodedata.category(ch) == "Mn" and ord(ch) < 0x0900)]
    return unicodedata.normalize("NFC", "".join(out))


def normalize_text(s):
    """Scalar normalization (audit/tests). Must match norm_series exactly."""
    s = strip_latin_accents(s).lower()
    if PUNCT is PUNCT_V2:
        s = DANDA.sub(" ", s)
    s = PUNCT.sub(" ", s)
    return WS.sub(" ", s).strip()


def norm_series(s):
    s = s.map(lambda x: strip_latin_accents(x).lower())
    if PUNCT is PUNCT_V2:
        s = s.str.replace(DANDA, " ", regex=True)
    s = s.str.replace(PUNCT, " ", regex=True)
    return s.str.replace(WS, " ", regex=True).str.strip()


def read_tsv(p):
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)


# ------------------------------------------------------------------- sharding
def shard_of(entity_id, num_shards):
    """Deterministic, machine-independent S1 shard assignment (D11).

    MD5 of the UTF-8 entity_id -> 128-bit int -> modulo num_shards.
    Never Python's built-in hash() (salted per process). Only S1 is sharded;
    every shard searches the FULL same-country S2/S3 corpus.
    """
    return int(hashlib.md5(entity_id.encode("utf-8")).hexdigest(), 16) % num_shards


def shard_mask(ids, num_shards, shard_id):
    return np.fromiter((shard_of(e, num_shards) == shard_id for e in ids),
                       dtype=bool, count=len(ids))


# ------------------------------------------------------------------- manifest
def load_manifest(out_dir):
    p = out_dir / "manifest.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"config": None, "countries": {}, "state": "new"}


def save_manifest(out_dir, man):
    tmp = out_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(man, indent=1))
    os.replace(tmp, out_dir / "manifest.json")


# ------------------------------------------------------------------ retrieval
def topk_to_arrays(C):
    """CSR top-k result -> (query_row, corpus_col, rank) int arrays."""
    counts = np.diff(C.indptr)
    qi = np.repeat(np.arange(C.shape[0], dtype=np.int64), counts)
    rank = (np.arange(len(C.indices), dtype=np.int64)
            - np.repeat(C.indptr[:-1].astype(np.int64), counts))
    return qi, C.indices.astype(np.int64), rank


def process_country(country, q, corp, args, man, out_parts, out_index):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import sp_matmul_topn

    kw, kc = args.budgets.get(country, args.default_budget)
    n_q = len(q)
    n_chunks = (n_q + args.chunk_size - 1) // args.chunk_size
    cman = man["countries"].setdefault(country, {"chunks": {}, "status": "running"})
    cman.update({"n_s1": n_q, "n_corpus": len(corp), "k_word": kw, "k_char": kc,
                 "n_chunks": n_chunks})
    if cman.get("status") == "done" and all(
            str(i) in cman["chunks"] for i in range(n_chunks)):
        log(country, "already done — skipping")
        return
    log("=====", country, "S1=%d corpus=%d budget=(%d,%d) chunks=%d"
        % (n_q, len(corp), kw, kc, n_chunks))

    corp_ids = corp["entity_id"].to_numpy()
    corp_src = corp["source"].to_numpy()
    corp_joint = (corp["nname"] + " " + corp["naddr"])
    q_ids = q["entity_id"].to_numpy()
    q_joint = (q["nname"] + " " + q["naddr"]).to_numpy()
    q_names = q["nname"].to_numpy()

    # exact normalized-name blocks (only for names present in this S1 partition;
    # empty normalized names excluded; block capped at first exact_cap corpus rows
    # in file order — matches the validated Session-2 reconstruction)
    t = time.time()
    qname_set = set(q_names) - {""}
    mask = corp["nname"].isin(qname_set).to_numpy()
    exact_blocks = {}
    if mask.any():
        sub = pd.DataFrame({"nname": corp["nname"].to_numpy()[mask],
                            "pos": np.nonzero(mask)[0]})
        for nm, g in sub.groupby("nname")["pos"]:
            exact_blocks[nm] = g.to_numpy()[: args.exact_cap]
    log(country, "exact blocks: %d names, %.1fs" % (len(exact_blocks), time.time() - t))

    # vectorizers (fit on the FULL country corpus; deterministic across resumes)
    mats = {}
    for cfg, analyzer, ngr in (("word", "word", (1, 1)), ("char", "char_wb", (3, 4))):
        t = time.time()
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngr, min_df=3,
                              max_df=0.4, dtype=np.float32)
        X = vec.fit_transform(corp_joint)
        B = X.T.tocsr()
        del X
        gc.collect()
        mats[cfg] = (vec, B)
        log(country, cfg, "matrix built %.0fs nnz=%d" % (time.time() - t, B.nnz))
    del corp, corp_joint
    gc.collect()

    for ci in range(n_chunks):
        if str(ci) in cman["chunks"]:
            continue
        t0 = time.time()
        lo, hi = ci * args.chunk_size, min((ci + 1) * args.chunk_size, n_q)
        texts = q_joint[lo:hi]

        frames = {}
        for cfg, k in (("word", kw), ("char", kc)):
            vec, B = mats[cfg]
            Q = vec.transform(texts)
            C = sp_matmul_topn(Q, B, top_n=k, threshold=args.threshold,
                               sort=True, n_threads=args.n_threads)
            qi, col, rank = topk_to_arrays(C)
            frames[cfg] = pd.DataFrame(
                {"qi": qi, "ci": col, cfg + "_rank": rank.astype(np.int16)})
            del Q, C

        # exact channel for this chunk
        eq, ec = [], []
        for i in range(lo, hi):
            blk = exact_blocks.get(q_names[i])
            if blk is not None and len(blk):
                eq.append(np.full(len(blk), i - lo, dtype=np.int64))
                ec.append(blk)
        if eq:
            dfe = pd.DataFrame({"qi": np.concatenate(eq), "ci": np.concatenate(ec)})
            dfe["exact"] = np.uint8(1)
        else:
            dfe = pd.DataFrame({"qi": pd.Series(dtype=np.int64),
                                "ci": pd.Series(dtype=np.int64),
                                "exact": pd.Series(dtype=np.uint8)})

        u = frames["word"].merge(frames["char"], on=["qi", "ci"], how="outer")
        u = u.merge(dfe, on=["qi", "ci"], how="outer")
        u["word_rank"] = u["word_rank"].fillna(-1).astype(np.int16)
        u["char_rank"] = u["char_rank"].fillna(-1).astype(np.int16)
        u["exact"] = u["exact"].fillna(0).astype(np.uint8)
        u["channels"] = ((u["word_rank"] >= 0).astype(np.uint8) * CH_WORD
                         | (u["char_rank"] >= 0).astype(np.uint8) * CH_CHAR
                         | u["exact"] * CH_EXACT)

        part = pd.DataFrame({
            "s1_id": q_ids[lo:hi][u["qi"].to_numpy()],
            "cand_id": corp_ids[u["ci"].to_numpy()],
            "source": corp_src[u["ci"].to_numpy()],
            "country": country,
            "channels": u["channels"].to_numpy(),
            "word_rank": u["word_rank"].to_numpy(),
            "char_rank": u["char_rank"].to_numpy(),
        })
        fn = out_parts / ("cand_%s_%05d.parquet" % (country.replace("/", "_"), ci))
        part.to_parquet(fn, index=False)

        # per-chunk S1 index (covers zero-candidate S1s)
        counts = np.zeros(hi - lo, dtype=np.int32)
        got = u["qi"].value_counts()
        counts[got.index.to_numpy()] = got.to_numpy()
        pd.DataFrame({"s1_id": q_ids[lo:hi], "country": country,
                      "n_cands": counts}).to_parquet(
            out_index / ("s1_%s_%05d.parquet" % (country.replace("/", "_"), ci)),
            index=False)

        secs = time.time() - t0
        cman["chunks"][str(ci)] = {"rows": int(hi - lo), "cands": int(len(part)),
                                   "secs": round(secs, 1)}
        save_manifest(args.out, man)
        log(country, "chunk %d/%d rows=%d cands=%d %.0fs"
            % (ci + 1, n_chunks, hi - lo, len(part), secs))
        del frames, dfe, u, part
        gc.collect()

    cman["status"] = "done"
    save_manifest(args.out, man)
    del mats, exact_blocks
    gc.collect()


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s1", default=str(ROOT / "dataset/train/train_source1.tsv"))
    ap.add_argument("--s2", default=str(ROOT / "dataset/train/train_source2.tsv"))
    ap.add_argument("--s3", default=str(ROOT / "dataset/train/train_source3.tsv"))
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--budgets", default='{"US": [50, 50]}',
                    help='JSON {country: [k_word, k_char]}')
    ap.add_argument("--default-budget", type=int, nargs=2, default=[100, 100],
                    help="k_word k_char for countries not in --budgets")
    ap.add_argument("--sample", default=None,
                    help='per-country S1 sample, e.g. "US:12000,India:8000" '
                         "(replicates the benchmark sampler; unlisted countries "
                         "are excluded)")
    ap.add_argument("--s1-ids", default=None,
                    help="optional file with one S1 entity_id per line (subset mode)")
    ap.add_argument("--shard-id", type=int, default=None,
                    help="0-based S1 shard to process (requires --num-shards); "
                         "assignment = MD5(entity_id) %% num_shards. S2/S3 stay FULL.")
    ap.add_argument("--num-shards", type=int, default=None,
                    help="total number of S1 shards (requires --shard-id)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-size", type=int, default=100000)
    ap.add_argument("--sources", default="s2,s3", help="comma subset of s2,s3")
    ap.add_argument("--countries", default=None,
                    help="optional comma list restricting processed partitions")
    ap.add_argument("--exact-cap", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--n-threads", type=int, default=12)
    ap.add_argument("--matra-norm", action="store_true",
                    help="preserve Indic combining marks in normalization "
                         "(measured recall-neutral; default = benchmark norm)")
    args = ap.parse_args()
    if args.matra_norm:
        global PUNCT
        PUNCT = PUNCT_V2
    if (args.shard_id is None) != (args.num_shards is None):
        ap.error("--shard-id and --num-shards must be given together")
    if args.shard_id is not None and not (0 <= args.shard_id < args.num_shards):
        ap.error("--shard-id must be in [0, num_shards)")

    args.out = Path(args.out)
    args.budgets = {k: tuple(v) for k, v in json.loads(args.budgets).items()}
    args.default_budget = tuple(args.default_budget)
    out_parts = args.out / "parts"
    out_index = args.out / "s1_index"
    out_parts.mkdir(parents=True, exist_ok=True)
    out_index.mkdir(parents=True, exist_ok=True)

    man = load_manifest(args.out)
    config = {"s1": args.s1, "s2": args.s2, "s3": args.s3,
              "budgets": {k: list(v) for k, v in args.budgets.items()},
              "default_budget": list(args.default_budget),
              "sample": args.sample, "s1_ids": bool(args.s1_ids),
              "shard_id": args.shard_id, "num_shards": args.num_shards,
              "seed": args.seed, "chunk_size": args.chunk_size,
              "sources": args.sources, "exact_cap": args.exact_cap,
              "threshold": args.threshold, "matra_norm": args.matra_norm}
    if man["config"] not in (None, config):
        sys.exit("manifest config mismatch — use a fresh --out dir or delete "
                 + str(args.out / "manifest.json"))
    man["config"] = config

    t_start = time.time()
    log("loading inputs")
    s1 = read_tsv(args.s1)
    srcs = [s.strip() for s in args.sources.split(",")]
    parts = []
    for name, path in (("s2", args.s2), ("s3", args.s3)):
        if name in srcs:
            df = read_tsv(path)
            df["source"] = np.uint8(int(name[1]))
            parts.append(df)
    corpus = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    log("S1 %d, corpus %d" % (len(s1), len(corpus)))

    log("normalizing")
    for df in (s1, corpus):
        df["nname"] = norm_series(df["business_name"])
        df["naddr"] = norm_series(df["business_address"])
        df.drop(columns=["business_name", "business_address"], inplace=True)
    gc.collect()

    # S1 subset selection (validation modes)
    if args.sample:
        picks = []
        for spec in args.sample.split(","):
            ctry, n = spec.split(":")
            picks.append(s1[s1["country"] == ctry].sample(int(n),
                                                          random_state=args.seed))
        s1 = pd.concat(picks, ignore_index=True)
        log("sampled S1: %d rows" % len(s1))
    if args.s1_ids:
        wanted = set(Path(args.s1_ids).read_text().split())
        s1 = s1[s1["entity_id"].isin(wanted)].reset_index(drop=True)
        log("s1-ids subset: %d rows" % len(s1))
    if args.shard_id is not None:
        s1 = s1[shard_mask(s1["entity_id"].to_numpy(), args.num_shards,
                           args.shard_id)].reset_index(drop=True)
        log("shard %d/%d: %d S1 rows (MD5(entity_id) %% %d == %d)"
            % (args.shard_id, args.num_shards, len(s1), args.num_shards,
               args.shard_id))

    countries = sorted(s1["country"].unique())
    if args.countries:
        keep = {c.strip() for c in args.countries.split(",")}
        countries = [c for c in countries if c in keep]
    man.setdefault("country_order", countries)
    save_manifest(args.out, man)

    for country in countries:
        q = s1[s1["country"] == country].reset_index(drop=True)
        corp = corpus[corpus["country"] == country].reset_index(drop=True)
        if len(corp) == 0:
            log(country, "EMPTY corpus partition — %d S1 get zero candidates" % len(q))
            pd.DataFrame({"s1_id": q["entity_id"], "country": country,
                          "n_cands": np.int32(0)}).to_parquet(
                out_index / ("s1_%s_00000.parquet" % country.replace("/", "_")),
                index=False)
            man["countries"][country] = {"chunks": {}, "status": "done",
                                         "n_s1": len(q), "n_corpus": 0}
            save_manifest(args.out, man)
            continue
        process_country(country, q, corp, args, man, out_parts, out_index)

    man["state"] = "done"
    man["total_secs"] = round(time.time() - t_start, 1)
    save_manifest(args.out, man)
    total = sum(ch["cands"] for c in man["countries"].values()
                for ch in c["chunks"].values())
    log("ALL DONE: %d candidate pairs, %.1f min"
        % (total, (time.time() - t_start) / 60))


if __name__ == "__main__":
    main()
