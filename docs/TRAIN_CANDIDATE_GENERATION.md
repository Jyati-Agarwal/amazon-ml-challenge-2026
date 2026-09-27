# TRAIN CANDIDATE GENERATION — Production run report

Session 1 (blocking implementation), 2026-09-25. Status: **VALIDATED; full unsharded
run intentionally STOPPED — pivoting to 3-way S1 sharding** (team decision). This file
is the run report + resume manual for the full-train candidate generation.

## Implementation

- Generator: `scripts/generate_candidates.py` (production; **never reads ground truth**)
- Evaluator: `scripts/evaluate_candidates.py` (GT lives only here)
- Audit: `scripts/normalization_audit.py`
- Interpreter: `.venv/bin/python` (Python 3.9, pandas 2.3.3, sklearn 1.6.1,
  sparse_dot_topn 1.2.0 — the exact stack that produced the 20k benchmark)

Architecture (locked, docs/BLOCKING_IMPLEMENTATION_PLAN.md): per-country partition
(open-set) → normalize name/address → word TF-IDF top-k + char_wb(3,4) TF-IDF top-k
over `nname + " " + naddr` (min_df=3, max_df=0.4, float32, sp_matmul_topn
threshold=0.05) → exact normalized-name channel (block capped at first 200 corpus
rows, empty names excluded) → union → dedupe. Budgets: `{"US": (50,50)}`, default
`(100,100)` for every other/unseen country (France ⇒ default automatically).
No rescue channels (R1/R2 rejected by experiment).

## Normalization audit (PASS, after one fix)

`scripts/normalization_audit.py` audits the exact production functions. Findings:

1. **BUG FOUND in the benchmark normalization**: Python `re`'s `\w` does not match
   combining marks (category Mn), so the study regex `[^\w\s] -> " "` stripped Indic
   matras/viramas: `प्राइवेट लिमिटेड` → `प र इव ट ल म ट ड`. The accent-stripper had been
   written to preserve them (`ord >= 0x0900`) but the punctuation pass undid it.
   The 0.9794 benchmark itself ran with matras stripped — D5's "preserves matras"
   claim was false in practice.
2. **Fix (production default)**: punctuation regex keeps the full Indic block
   `ऀ-෿` (`[^\w\sऀ-ൿ]`), with Devanagari danda `।॥` mapped to space first.
   The old behavior is preserved behind `--legacy-norm` for benchmark reproduction.
3. All other checks pass: case/punct/whitespace/Latin-accent handling, digits and
   house numbers preserved, empty and punctuation-only strings → "", idempotent,
   scalar==vectorized, no non-empty name collapses to empty on 150k sampled real
   rows, S1 and S2/S3 share one code path. Empty normalized addresses: 0% (S1),
   ~3.35% (S2), ~3.38% (S3) — all from raw-empty addresses, as profiled.
4. Known/accepted quirks: `_` is kept by `\w` (consistent everywhere); Latin accents
   stripped but Indic vowel signs kept (by design).

## Output schema

`<out>/parts/cand_<country>_<chunk>.parquet`:
`s1_id (str), cand_id (str), source (uint8 2|3), country (str),
channels (uint8 bitmask 1=word|2=char|4=exact), word_rank (int16, -1 absent),
char_rank (int16, -1 absent)`. One row per unique (s1_id, cand_id).
`<out>/s1_index/s1_<country>_<chunk>.parquet`: `s1_id, country, n_cands` for every
processed S1 (zero-candidate entities included).
`<out>/manifest.json`: config + per-(country, chunk) progress (rows, cands, secs).

## Smoke test (800 S1: 500 US / 300 India, chunk 400)

pair recall 0.9799 (US 0.993 / India 0.959), entity 0.9397, cand mean 116.6 /
p99 255; 0 dup pairs; all ID/country/source/channel checks pass; resume-skip
verified; 8.7 min, peak 15.8 GB.

## 20k validation (12k US / 8k India, seed 42, FULL same-country S2+S3 pools) — PASS

| Config | Pair recall | US | India | Entity | Cand mean | median | p95 | p99 | max | Pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Benchmark target (B04) | 0.9794 | 0.9943 | 0.9571 | 0.9400 | 116.3 | 92 | 182 | 254 | 577 | — |
| **Benchmark norm (ADOPTED)** | **0.9794** | 0.9943 | 0.9571 | 0.9400 | 115.8 | 92 | 182 | 254 | 291 | 2,315,556 |
| Matra-fix norm (`--matra-norm`) | 0.9792 | 0.9943 | 0.9565 | 0.9395 | 117.6 | 92 | 185 | 254 | 291 | 2,352,468 |

Digit-for-digit benchmark reproduction (S2 0.9804 / S3 0.9784 also match M02).
Runtime ≈ 18 min/config, peak RSS 14.4 GB. The matra fix is recall-neutral-to-negative
(Latin addresses already carry Indic-name retrieval) → **production default = benchmark
norm**; matra variant retained behind `--matra-norm`. Correctness checks all pass:
0 duplicate pairs, all cands from S2/S3, source col == ID prefix, country partition
correct (no cross-country pairs), channel bitmask ⇔ rank consistency, 0 zero-cand S1
(tracked in s1_index regardless), GT untouched by the generator. Reports:
`experiments/candidates/val20k_legacy_report.json`, `val20k_report.json`.
(cand_max 291 vs 577 in B04: the study's max included uncapped exact blocks; the
production exact cap of 200 trims the single outlier entity. Recall unaffected.)

## Full TRAIN run — STOPPED BY DESIGN (not a failure)

Launched 18:03 (`--out experiments/candidates/train_full`, chunk 100k, 9 India +
14 US chunks planned), then **intentionally stopped at ~3.5 min** (during the India
char-matrix build, before any chunk completed) because the team switched the execution
plan to **3-way S1 sharding** so three members can generate candidates in parallel.
Partial state: only `train_full/manifest.json` (0 completed chunks, no part files) —
safe to delete or reuse. The generator already supports sharding via `--s1-ids <file>`
(one S1 entity_id per line) or `--sample`; each shard should use a distinct `--out`.

Single-machine command (if ever run unsharded):
```
.venv/bin/python scripts/generate_candidates.py \
    --out experiments/candidates/train_full --chunk-size 100000
```
Expected: ~257M pairs, 2–4 GB parquet, ~17 h (char channel dominates; US char is the
single biggest cost — dropping it costs −0.4 pp US recall, saves ~11 h).

## Resume instructions

The run is resumable at (country, chunk) granularity. If it stops for any reason,
re-run the exact same command with the same `--out`; completed chunks are read from
`manifest.json` and skipped (vectorizers are refit deterministically, ~3 min/country
fixed cost). Do not change any flag between resumes — the manifest rejects config
mismatches. Progress: check `manifest.json` or the log file.
