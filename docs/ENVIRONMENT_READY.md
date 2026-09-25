# ENVIRONMENT READY — Session 3 report

Date: 2026-09-25. Machine: Apple M4 Pro, 14 cores, 48 GB RAM, macOS (Darwin 25.6).
Benchmarks reproducible via `scripts/benchmark_env.py` (sampled/synthetic data only).
Integration smoke test: `scripts/smoke_e2e.py` (see §9). Pins: `requirements.txt`.

## READY / NOT READY

**READY.** All required packages install, import, and pass functional benchmarks in a new
Python 3.12 project venv. One external dependency (`libomp`) was needed for LightGBM and is
installed. `faiss-cpu` was deliberately **not** installed (see below). The full integration
smoke test (`scripts/smoke_e2e.py`) passes end-to-end, including pandas-3 compatibility of
all existing project code patterns, real project-module imports, Session-2 parquet caches,
and the submission validator. **Nothing is NOT READY.** No blockers remain for the full pipeline.

## 1. Python versions found

| Interpreter | Version | Notes |
|---|---|---|
| `/usr/bin/python3` (system) | 3.9.6 | Old; untouched |
| `/opt/homebrew/bin/python3.12` | **3.12.14** | Newly installed via `brew install python@3.12` |
| 3.11 / 3.13 | not present | Not needed |

No pyenv, no uv, no conda. Homebrew was already present; installing `python@3.12` adds a
user-local interpreter and does **not** modify the system Python or the default `python3`.

## 2. Environments

- **`.venv312` (new, recommended)** — Python 3.12.14, at project root, used for everything below.
  Activate: `source .venv312/bin/activate` or call `.venv312/bin/python` directly.
- **`.venv` (old)** — Python 3.9.6 with only numpy/pandas/pyarrow. **Left untouched**; keep it
  until the pipeline runs end-to-end on `.venv312`, then it can be deleted.

## 3. Installed packages (`.venv312`)

| Package | Version | Justification |
|---|---|---|
| numpy | 2.5.3 | everything |
| pandas | 3.0.6 | I/O, tabular ops — **note: pandas 3.0 is a new major version** (copy-on-write and string dtype by default); avoid pre-3.0 idioms like chained assignment |
| scipy | 1.18.1 | sparse matrices (CSR) |
| scikit-learn | 1.9.1 | TfidfVectorizer, metrics, LogisticRegression baseline |
| rapidfuzz | 3.14.6 | fast per-pair string similarity |
| lightgbm | 4.7.0 | pair classifier (needed `brew install libomp` — done) |
| pyarrow | 25.0.1 | Parquet caching |
| polars | 1.44.2 | parallel string normalization / big joins over 10M+ rows |
| sparse-dot-topn | 1.2.0 | bounded-memory top-k sparse cosine — core of candidate generation |
| Unidecode | 1.4.0 | Devanagari/accents → ASCII transliteration (13% of S2-India names) |

**Not installed — `faiss-cpu`:** the pipeline is sparse-TF-IDF-based; `sparse_dot_topn`
covers top-k retrieval with better memory behavior for sparse vectors. FAISS only becomes
useful if dense embeddings enter the pipeline (unlikely on CPU in 48 h). Install later if needed.

## 4. Compatibility issues encountered (none hidden)

1. **LightGBM `OSError: libomp.dylib not found`** on first import — the macOS arm64 wheel
   links OpenMP dynamically. Fixed with `brew install libomp`; import and training verified.
2. **pandas 3.0** is very new. Nothing failed in benchmarks, but existing scripts written
   against pandas 2.x semantics (e.g. `scripts/profile_data.py`) should be smoke-tested
   before reuse. If anything breaks, `pip install 'pandas<3'` in `.venv312` is a safe fallback.
3. Python 3.9 (old venv) cannot run current wheels of scikit-learn/polars — this is why
   `.venv312` exists rather than upgrading `.venv` in place.

## 5. Benchmark results (sampled/synthetic only — no full-dataset work)

200k real names sampled from train_source2; synthetic data for LightGBM. 14 threads where supported.

| Benchmark | Runtime | Memory note |
|---|---:|---|
| Load 200k names (pandas) | 0.22 s | +0.11 GB |
| Char TF-IDF 2–4gram fit_transform, 200k names | 3.0 s | 105 MB CSR, 175k vocab, 13M nnz |
| Word TF-IDF, 200k names | 0.33 s | 657k nnz — much sparser/cheaper |
| **Naive full sparse cosine, 10k × 200k** | **26 s** | **1.48B nnz ≈ 12 GB** — proof this must never be done at scale |
| **sparse_dot_topn top-30, 10k × 200k** | **1.7 s** | bounded: 276k nnz kept |
| rapidfuzz `token_sort_ratio` cdist, 4M pairs | 0.09 s | ~45M pairs/s with 14 workers |
| LightGBM train, 100k × 20, 100 trees | 0.86 s | negligible |
| LightGBM predict, 100k rows | 0.07 s | negligible |

Peak RSS for the whole run: 7.1 GB — dominated entirely by the deliberate naive-product demo.

## 6. Extrapolations to full scale (planning numbers)

- **Char TF-IDF over all ~10M test S2+S3 rows:** ~2.5 min fit_transform, ~5 GB CSR. Fine.
- **Candidate generation, biggest block (India: 0.81M × 4.7M) with sparse_dot_topn:**
  scaling the 1.7 s benchmark by rows×cols ⇒ roughly **30–60 min** single pass. Chunk the
  query side (~50k rows/chunk) and write each chunk's pairs to Parquet.
- **rapidfuzz over 5×10⁷ candidate pairs:** seconds-to-minutes per scorer — not a bottleneck.
- **LightGBM on ~5×10⁷ training pairs × 20 features:** ~4 GB float32 features, minutes per
  100 trees on 14 threads. Fine.

## 7. Remaining bottleneck risks for full experiments

1. **Naive sparse products** — the 26 s / 12 GB demo at 10k×200k becomes instant OOM at
   block scale. Only `sp_matmul_topn` (or equivalent chunked top-k), never `A @ B.T`.
2. **Candidate-count blowout** — always log pair counts per chunk; abort if a block trends
   past ~2×10⁸ total pairs.
3. **Char TF-IDF vocab growth** — 175k features at 200k rows; at 10M multi-script rows the
   vocab may reach ~1M. Memory stays fine (nnz-bound, not vocab-bound), but consider
   `min_df=2` to trim hapax n-grams and shrink the model.
4. **pandas 3.0 unknowns** — prefer Polars or pyarrow for the heavy string/join steps anyway.
5. **macOS `spawn` multiprocessing** — pass file paths, not DataFrames, to worker processes.
   rapidfuzz's built-in `workers=` avoids the issue entirely for pair scoring.

## 8. Recommendation summary

- **Python:** 3.12.14 (Homebrew) via project venv **`.venv312`**.
- **Run everything with** `.venv312/bin/python`; leave `.venv` (3.9) untouched until migration is confirmed.
- **Stack:** pandas/pyarrow for I/O + Parquet caching, polars for heavy string work,
  sklearn TF-IDF (float32) + sparse_dot_topn for candidates, rapidfuzz for pair features,
  LightGBM for the classifier, Unidecode for transliteration.
- **Optional later:** faiss-cpu (only if embeddings), `pandas<3` fallback if 3.0 misbehaves.

## 9. Integration smoke test (Session 3 final stage, 2026-09-25)

`scripts/smoke_e2e.py` — ALL PASS under `.venv312`. Coverage:

**A. pandas-3 compatibility of existing-script idioms** (patterns lifted from
`blocking_*.py`, `feature_signal_research.py`, `model_baseline.py`,
`real_blocking_validation.py`): `read_csv(keep_default_na=False, quoting=3)`,
`.fillna("")`/`.fillna(False)`/`.fillna(0)`, `.str.len()` on list columns, `explode`,
`groupby.apply(set)`, regex `str.replace` chains, `sample(random_state)` — all pass.

**One behavioral change found (no code breaks):** in pandas 3, `.values` on a *string*
column returns `ArrowStringArray`, not `np.ndarray`. Audit of every `.values` use in the
project: all are on dicts or numeric columns (still ndarray) except
`scripts/real_blocking_validation.py:373-374` (`g.cand_id.values` fancy-indexed then
`set()`-ed) — tested explicitly, works correctly with ArrowStringArray. Recommendation:
prefer `.to_numpy()` for string columns in new code. **`pandas<3` pin not needed.**

**Real project code under pandas 3:** `model_baseline.py` and `feature_signal_research.py`
import cleanly in `.venv312`; `norm`, `core_name`, `expected_f05_select` produce correct
outputs; both Session-2 caches (`experiments/cache/pairs_features.parquet` 357,108×44,
`pairs_raw.parquet` 357,108×11) read cleanly with pyarrow 25 / pandas 3, zero numeric NaNs.
(`model_baseline.py` was not *executed* end-to-end — it overwrites Session-2 artifacts;
first full-scale run should be done deliberately, not as a smoke test.)

**B. Tiny end-to-end pipeline** (300 S1 = 200 US + 100 India, 6,045-row corpus =
1,045 true matches + 5,000 distractors, incl. Devanagari names and empty addresses):
normalize → country partition → word + char_wb TF-IDF (name+address) →
`sp_matmul_topn` top-20 → exact-normalized-name channel → union + dedupe (8,214 pairs,
verified duplicate-free) → 6 provisional features (finite, in-range, no NaN/inf) →
LightGBM train + predict (correct shape, valid probabilities). Verified: candidate IDs
map back to corpus rows; countries never cross; Unicode and missing addresses don't crash;
tiny-sample blocking recall 1044/1045 (not representative).

**C. Output schema:** `matching_results.tsv` + `candidate_pairs.tsv` written with correct
headers/tab delimiter; predictions verified ⊆ candidates; `utils/validate_submission.py
--check-ids` run against a fabricated mini test dir → exit 0, "Safe to submit".
Temp outputs auto-deleted.

**D. Timings (tiny sample; identify-bottlenecks only, no aggressive extrapolation):**
load+sample 4.2 s (dominated by full S2/S3 read for honest sampling), normalization ~0 s,
TF-IDF ×4 0.21 s, top-k ×4 0.01 s, features 0.54 s for 8,214 pairs **single-core Python
loop** (~15k pairs/s → the one stage that must be vectorized/parallelized at 10⁷⁺ scale),
LightGBM 0.34 s. Peak RSS 6.4 GB (the corpus load).

**Files needing changes later (not now):** `scripts/real_blocking_validation.py:373-374`
(cosmetic: switch string `.values` → `.to_numpy()`); production feature computation must
use rapidfuzz batch APIs / multiprocessing, not the per-pair Python loop from the smoke test.
