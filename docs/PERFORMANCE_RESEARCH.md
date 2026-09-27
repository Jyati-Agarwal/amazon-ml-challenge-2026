# PERFORMANCE RESEARCH — Amazon ML Challenge 2026 (Entity Resolution)

Measured 2026-09-25. Read-only analysis; dataset untouched. Data-content details live in
`docs/DATA_PROFILE.md` — this doc covers compute, memory, and pipeline architecture.

## 1. Machine and environment

- **Hardware:** Apple M4 Pro, 14 cores, 48 GB RAM. Plenty for this dataset if we stay sparse.
- **Python:** system Python **3.9.6** (`/usr/bin/python3`); `.venv` is built on it.
- **Installed in `.venv`:** numpy 2.0.2, pandas 2.3.3, pyarrow 21.0.0 — **and nothing else**.
  No scikit-learn, no scipy, no polars, no rapidfuzz, no LightGBM/XGBoost, no transliteration libs.
  No conda, no uv, no Homebrew Python found.

**Environment gap is the #1 infrastructure risk.** The modeling stack must be installed before
any real work starts. Python 3.9 is old enough that the *latest* wheels of some libraries no
longer support it (recent polars and scikit-learn ≥1.7 require 3.10+). Two options:

1. Stay on 3.9 and pin slightly older versions (scikit-learn ≤1.6, polars ~1.8, rapidfuzz,
   lightgbm — all have 3.9 wheels). Fastest path.
2. Install Python 3.11/3.12 (Homebrew) and rebuild the venv. Cleaner, ~30 min cost.

Recommended installs (nothing installed automatically): `scikit-learn scipy rapidfuzz
lightgbm polars unidecode` (+ optionally `sparse_dot_topn` for fast sparse top-k cosine).

## 2. Data scale (from DATA_PROFILE.md + fresh benchmark)

- 7 TSVs, 2.3 GB total, ~24.2M rows, 4 short string columns each.
- Load benchmark on train_source1 (200 MB, 2.2M rows): **pandas C engine 1.8 s / 0.7 GB;
  pyarrow 0.4 s / 0.24 GB**. Extrapolated: the largest file (~5.3M rows) loads in ~5 s and
  ~1.7 GB in pandas. Whole dataset fits in RAM simultaneously (~6–8 GB in pandas,
  ~2.5 GB in Arrow). **No chunked reading needed.**

## 3. pandas vs Polars

- **pandas is sufficient** for I/O and per-file transforms at this scale (seconds per file).
- **Polars helps in three specific places**, all string-heavy and multi-core:
  group-bys/joins over 10M rows (candidate-pair assembly), bulk string normalization
  (lowercase/strip/regex over 24M strings — Polars is parallel, pandas `.str` is single-core),
  and streaming aggregation of large intermediate pair tables. Worth installing, not mandatory.
- Convert TSVs to **Parquet once** (pyarrow, already installed): ~5–10× faster reload,
  ~4× smaller in RAM via Arrow strings. Do this first; every later iteration benefits.

## 4. Sparse vs dense, TF-IDF, char n-grams

- **Dense is impossible.** Char-trigram vocab over multi-script names is ~50k–200k dims;
  dense 10M × 50k float32 ≈ 2 PB. Everything vectorized must stay CSR sparse.
- **Sparse TF-IDF is cheap.** Names average ~25 chars → ~23 trigrams/row. Name-only matrix
  over all ~10M test S2+S3 rows: ~2.3×10⁸ non-zeros ≈ **1.9 GB** as float32 CSR.
  Name+address (~75 chars → ~70 nnz/row): ~7×10⁸ nnz ≈ **5.6 GB**. Both fit in 48 GB,
  and per-country blocking cuts each block to a fraction of that.
  Use `TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), dtype=np.float32)` —
  the float32 dtype halves memory vs the float64 default.
- **The danger is not the TF-IDF matrix — it's the similarity product.** A naive
  `X_s1 @ X_s2s3.T` materializes a sparse matrix whose nnz is unbounded (common trigrams
  connect millions of pairs). Never compute it whole. Options, best first:
  1. `sparse_dot_topn` — top-k cosine per row with a hard k, multithreaded, bounded memory.
  2. Manual chunking: 5–20k S1 rows at a time × block index, take top-k per row, discard chunk.
  3. `sklearn.neighbors.NearestNeighbors(metric='cosine')` on the sparse matrix — simpler
     but slower and memory-hungrier at this scale.

## 5. Candidate-pair volume (the make-or-break number)

- All-pairs: 1.73M × 9.97M ≈ **1.7×10¹³** — impossible.
- Country blocking alone: India block 0.81M × 4.7M ≈ **3.8×10¹²** — still impossible.
- **Target: 20–50 candidates per S1 entity → 3.5×10⁷–8.6×10⁷ pairs total.** That is the
  budget every blocking idea must be measured against. Ground truth averages 3.46 true
  matches/entity (max 11), so k=50 gives huge headroom if blocking recall is good.
- At 5×10⁷ pairs: a float32 feature matrix with 20 features is **4 GB** — fine.
  At 5×10⁸ pairs it's 40 GB — the machine dies. **Cap k per entity, always.**

## 6. Expensive vs cheap features

| Cost | Features |
|---|---|
| Cheap (vectorized) | TF-IDF cosine (falls out of candidate gen for free — keep it), token overlap/Jaccard via set ops, length diffs, exact-token flags, country match |
| Medium | rapidfuzz `token_sort_ratio` / `partial_ratio` (C++-fast, ~1–5 µs/pair → 5×10⁷ pairs ≈ 1–4 core-minutes ×14 cores), digit-sequence (house/khasra number) comparison |
| Expensive — use sparingly | Edit distance on long addresses in pure Python (never), transliteration per pair (precompute per record instead), any embedding model over 24M strings (hours on CPU; skip or restrict to a hard subset) |

Rule: anything per-*record* (normalization, transliteration, token sets, phonetic keys)
is O(24M) — precompute once and cache. Anything per-*pair* is O(5×10⁷) — must be
rapidfuzz/numpy-vectorized, never a Python-level loop (a 10 µs/pair Python function
costs 8 minutes at 5×10⁷; a 1 ms/pair function costs 14 hours).

## 7. Parallelization

- 14 real cores. LightGBM, Polars, and `sparse_dot_topn` parallelize internally — free wins.
- For per-pair featurization: `multiprocessing.Pool` (or `concurrent.futures`) over
  **pair-chunks of ~1M**. macOS uses *spawn* start method — pass file paths / shared inputs,
  not giant DataFrames, to workers (pickling a 5 GB frame per worker kills you).
  rapidfuzz also has `rapidfuzz.process.cdist(..., workers=-1)` for batch scoring.
- Blocking naturally parallelizes by country × source block (5–6 independent blocks).

## 8. Batching, caching, precomputation

**Precompute once, cache to Parquet (pyarrow already installed), keyed by a version string:**

1. TSV → Parquet conversion of all 7 files (`data/parquet/`).
2. Normalized text per record: lowercase, strip punctuation/junk prefixes, legal-suffix
   canonicalization, whitespace collapse. One column each for name/address.
3. Devanagari→Latin transliteration per record (13% of S2-India names) — per-record, never per-pair.
4. Token sets / digit sequences per record.
5. Fitted TF-IDF vectorizers (pickle) per country block.
6. **Candidate pairs** (S1 id, S2/S3 id, cosine score) — the single most valuable cache;
   regenerating candidates is the most expensive step (~10–30 min), and every downstream
   experiment reuses the same pairs.
7. Pair feature matrix (Parquet) — retrain models in seconds without recomputing features.

With 6+7 cached, a model iteration (retrain + rescore + threshold sweep) is **minutes**,
which is what makes 48 hours enough.

## 9. Models practical in 48 hours

- **LightGBM binary classifier on pair features — the recommended core.** Trains on
  10–50M pairs in minutes on 14 cores, handles nulls (3% empty addresses) natively,
  and threshold tuning directly optimizes F₀.₅ (precision-weighted — tune the threshold
  high, and give singleton prediction real attention: 5.6% of entities, scored 1.0/0.0).
- Logistic regression on the same features — good day-1 baseline, seconds to train.
- **Not practical:** transformer embeddings or cross-encoders over 24M strings on CPU
  (many hours per pass, no GPU stack installed); training deep models; clustering the
  full graph. Sentence embeddings *only* as a rescoring pass on <1M hard pairs, if time remains.
- Post-processing: enforce the one-to-at-most-one constraint (each S2/S3 id matches ≤1 S1
  entity) via greedy assignment on scores — cheap (sort 5×10⁷ scores) and boosts precision.

## 10. Experiments that can explode (guardrails)

| Trap | Blast radius | Safeguard |
|---|---|---|
| Un-capped sparse similarity product | 100s of GB, OOM/swap-death | Always top-k per row, chunked; never materialize full product |
| Blocking key too loose (e.g. first-token) | 10¹⁰⁺ pairs, disk/RAM blowup | Print candidate count *before* materializing; abort if > 2×10⁸ |
| Python-loop per-pair feature | 14+ hours silently | Benchmark on 10k pairs first; extrapolate before full run |
| `.apply()` string ops on 24M rows | 10–60 min each, ×N experiments | Vectorized `.str`/Polars/precomputed columns only |
| Pandas cross-join for candidates | RAM explosion | Build pairs from inverted index / top-k lists, never `merge(how='cross')` |
| float64 TF-IDF over name+address | 2× memory (11 GB+) per matrix | `dtype=np.float32` in every vectorizer |
| Multiprocessing with big pickled args (macOS spawn) | RAM × n_workers | Workers read cached Parquet themselves |
| Accidental dataset-derived CSV in repo | Data-leak / repo bloat | Keep all outputs in gitignored dirs; never commit dataset derivatives |

Universal safeguards: run every long job with a **10k-row dry-run flag first**; log
row/pair counts at each stage; wrap experiments with a wall-clock budget and RSS check
(`resource.getrusage`) that aborts past ~35 GB.

## 11. Recommendations (summary)

- **Data stack:** pandas + pyarrow (installed) for I/O; convert to Parquet immediately;
  add Polars for the string-normalization and big-join steps. Install scikit-learn, scipy,
  rapidfuzz, lightgbm (+ sparse_dot_topn, unidecode) before anything else — the venv is
  currently empty of ML tooling, and Python 3.9 pins you to slightly older wheels
  (or rebuild the venv on 3.11+ first, ~30 min).
- **Vectorization strategy:** char 2–4-gram TF-IDF, `float32`, CSR sparse, fit per
  country block; name-first (1.9 GB), name+address if recall needs it (5.6 GB).
- **Candidate-generation architecture:** block by country → sparse TF-IDF top-k cosine
  (sparse_dot_topn or chunked matmul, k≈30–50) per block → optional cheap union with an
  exact-normalized-name inverted index for high-precision easy matches → cache pairs to
  Parquet. Budget: ≤10⁸ pairs total.
- **Likely bottlenecks (in order):** environment setup, candidate generation over the
  10M-row India block, per-pair featurization at 5×10⁷ scale, and repeated recomputation
  if caching is skipped.
- **Runaway protection:** count-before-materialize, top-k caps, dry-run flags, RSS/time
  budgets, no Python-level per-pair loops.
