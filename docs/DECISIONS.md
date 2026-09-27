# DECISIONS

Running log of locked-in project decisions. Newest last.

## D1 — Git safety model (2026-09-25)
Repo-local git identity only: `Jyati-Agarwal / Jyati-Agarwal@users.noreply.github.com`.
Remote fixed to `https://github.com/Jyati-Agarwal/amazon-ml-challenge-2026.git`.
Global git config never touched. No commits or pushes yet (both require explicit user OK).
`.gitignore` blocks dataset/, all TSV/parquet/models/outputs/env files.

## D2 — Environment (2026-09-25)
`.venv` (Python 3, pandas 2.3.3, scikit-learn 1.6.1, scipy, sparse_dot_topn 1.2.0,
unidecode, pyarrow). Machine: 48 GB RAM, 14 cores. Run scripts with `.venv/bin/python`.

## D3 — Validation design for blocking (2026-09-25)
20,000 stratified train S1 (12k US / 8k India, `random_state=42` via
`df[df.country==c].sample(n, random_state=42)`), real GT links kept, retrieval corpus =
FULL same-country train S2+S3 (never subsampled). Justified because GT match rate across
countries is exactly 0 (measured 1.000000 same-country over all 7.64M pairs).

## D4 — Per-country partitioning is lossless (2026-09-25)
GT never crosses country → partition S1/S2/S3 by the raw `country` string (open set,
no hardcoded country list). France at test time is just another partition value.

## D5 — Blocking backbone (2026-09-25)
Word TF-IDF (1-gram) and char_wb (3,4)-gram TF-IDF, both over
`normalized_name + " " + normalized_address`, cosine top-k via sparse_dot_topn,
plus exact-normalized-name union. Name-only representations rejected (72% vs 96%+
recall @k100). Naive and cleaned transliteration rejected for blocking (≤+2.5 pp on the
Devanagari subset, ~0 overall; Latin addresses already carry the signal).
Normalization: lowercase, strip Latin accents (NFD, drop Mn below U+0900),
punctuation→space, collapse whitespace.

## D6 — TF-IDF hyperparameters (2026-09-25)
`min_df=3, max_df=0.4, dtype=float32, threshold=0.05, sorted top-200 retained`.
Not tuned further by design (small informative grid only).

## D7 — Pairwise feature schema locked (Session 2, 2026-09-25)
16 KEEP features per docs/MODEL_FEATURE_SPEC.md (address containment/3gram/lev, digit_jac,
translit-aware name 3gram+JW, name_tok_jac, squash containment, missingness, indic flag,
source flag, log1p chain-ness, length diffs, locality). TF-IDF cosine features, raw
name_lev/jw, exact-equality flags, pincode features DROPPED (redundant or dead — pincodes
present in ~12% US / ~0% India addresses). country used as candidate FILTER, not feature.

## D9 — Production blocking normalization (2026-09-25)
Keep the benchmark normalization (Python `re` `[^\w\s]`→space, which strips Indic
combining marks) as the production default in scripts/generate_candidates.py.
The matra-stripping behavior was discovered by the normalization audit, then A/B
measured on the 20k validation: benchmark 0.9794/0.9400 vs matra-preserving
0.9792/0.9395 — neutral, and benchmark norm is the exactly-validated config with
fewer candidates. Matra-preserving variant available via `--matra-norm`.
Docs claiming "matras preserved" (D5 wording) were wrong in practice — corrected here.

## D10 — Full TRAIN generation execution plan: 3-way S1 sharding (2026-09-25)
The single-machine unsharded run (~17 h) was launched and intentionally stopped after
~3.5 min (0 chunks written) — the team will split S1 three ways and run
generate_candidates.py in parallel on three machines. Implementation and 20k
validation remain authoritative. Shard mechanism defined in D11.

## D11 — Shard assignment + sharded execution mechanics (2026-09-25)
`shard = int(MD5(entity_id utf-8).hexdigest, 16) % num_shards` (never built-in
`hash()` — salted per process). Only S1 is sharded; S2/S3 stay FULL per shard, and
TF-IDF models are fit on the full country corpus only, so merged shard output ==
unsharded output on the same S1 rows (row order aside). CLI: `--shard-id I
--num-shards N` in generate_candidates.py; per-shard `--out`
(experiments/candidates/train_sharded/shard_I). Merge via
scripts/merge_candidate_shards.py; GT sharded for eval via
scripts/shard_ground_truth.py. Full spec + verified guarantees:
docs/PARALLEL_SHARDING.md. All NEW runs use `.venv312/bin/python` (per Session-3
D10-environment decision); 20k validation re-run under .venv312 before any
production shard run.

## D8 — Decision stage (Session 2, 2026-09-25)
Global probability threshold, tuned on entity-level validation macro-F0.5; current best
t=0.7 (NOT 0.5). Per-S1 expected-F0.5 rule deferred until isotonic calibration is added
(model overconfident in 0.5–0.9 band; rule underperformed 0.9326 vs 0.9341). One-to-one
post-processing supported by GT structure (0 violations / 7.64M links) but unmeasurable at
research-sample scale — adopt only if full-scale validation confirms no F0.5 loss.

## D9 — End-to-end validation config frozen for scale-up (Session 2 stage 3, 2026-09-25)
On REAL blocking candidates (locked spec, pair recall 0.9794, 115.8 cand/S1):
19-feature LightGBM (spec + digit_exact_conflict), **raw threshold 0.70**, end-to-end
val macro-F0.5 0.9246 (misses counted as FN). Isotonic calibration: keep the machinery,
not required (raw ≈ calibrated on real negatives). Per-S1 expected-F0.5 rule: rejected
(ties global threshold, more complexity). One-to-one: include as cheap final safety step
(1 conflict / 18,220 pairs at sample scale; structurally safe; resolves by model score).
Threshold is a plateau 0.60–0.75 — re-verify at full-train scale before final inference.

## D10 — Production environment locked (Session 3, 2026-09-25)
All pipeline runs use **`.venv312`** (Python 3.12.14, Homebrew python@3.12; lightgbm
requires brew libomp — installed). Exact pins in `requirements.txt`. **pandas 3.0.6
kept (no `<3` pin)**: every project idiom + real module imports + Session-2 parquet
caches verified compatible by `scripts/smoke_e2e.py` (ALL PASS). Sole behavioral note:
string-column `.values` → ArrowStringArray (all existing uses still work; new code
should use `.to_numpy()`). faiss-cpu rejected (sparse pipeline; sparse_dot_topn covers
top-k). Old `.venv` (3.9) retained untouched until first full-scale run succeeds.

## D12 — Production feature pipeline: global-CSR vectorization (Session 2 stage 4, 2026-09-25)
scripts/feature_pipeline.py replaces the per-pair Python loop for scale-up. Feature
DEFINITIONS unchanged (exact copies of stage-3 functions); restructured as (a) one-time
per-RECORD prep parquet, (b) one-time cached global binary CSR set matrices per set
field (rows = all S1 + all candidates; 3-gram ids via vectorized UTF-32 codepoint
packing), (c) per-chunk pair features = row gather + sparse multiply + rapidfuzz
process.cpdist. Verified byte-identical to rb_features.parquet (max abs diff 0.0 on
all 19 features, 2.32M pairs, 0 NaN/inf) at 119.7k pairs/s end-to-end (162k steady;
old loop 28.8k) — measured while candidate-gen shard 0 held ~6 cores. Full docs:
docs/FEATURE_PIPELINE.md.

## D13 — cand_addr_ntok computed but EXCLUDED from the model (2026-09-25)
The stage-3 "try next" candidate-address token-count feature measurably HURTS on the
20k smoke set: macro-F0.5 0.8983 with vs 0.9113 without (PR-AUC 0.8798 vs 0.9192) on
identical rows/split. It remains in the feature parquet (FEATURES, 20 cols) for
analysis; the trained model uses MODEL_FEATURES = 19 (train_final_model.py).

## D15 — FINAL model + decision config (2026-09-26, full-train validation)
Trained on the full 255.7M-pair train candidates (230.1M fit / 25.6M val,
D14 MD5 split val_pct 10, positives 2.9261%): models/lgbm_final.txt
(19 MODEL_FEATURES, 833 s train). Validation on 220,429 val S1 with blocking
misses counted as FN: **threshold 0.625 -> macro-F0.5 0.9316; one-to-one
post-processing (drops 739/688,480 pairs) -> 0.9322 — ADOPTED.**
Plateau 0.600–0.700 (all >=0.9312). US 0.9487 / India 0.9062; singleton
correct-empty 89.57%; P 0.9707 / R 0.8770 at 0.625. Inference config:
lgbm_final.txt + t=0.625 + one-to-one by model score.
Artifacts: experiments/final_validation_metrics.csv, validation_summary.json.

## D14 — Final-model validation split + metric-noise caveat (2026-09-25)
Entity-level split `MD5(s1_id)//1000 % 100 < val_pct` (default 10): deterministic,
machine-independent, decorrelated from the D11 %3 shard hash by the //1000. Control
experiment: the validated stage-3 features under THIS split score macro-F0.5
0.9138–0.9144 vs the 0.9246 headline on the old rng-seed-21 split — same features,
same config. Conclusion: at 6k-entity validation scale, macro-F0.5 comparisons across
DIFFERENT entity splits carry ~±1.5 pp composition noise; only same-split comparisons
are meaningful. Full-train validation (~220k val S1) will shrink this.

## D16 — V0 locked; error-analysis stage completed without changes (2026-09-26)
The D15 config is frozen as **V0** (models/lgbm_final.txt, t=0.625, one-to-one,
19 features, D14 split). A deep error analysis was performed READ-ONLY
(scripts/error_analysis_v0.py; report experiments/error_analysis/ERROR_ANALYSIS.md):
model, threshold, blocking, candidates, features all unchanged; TEST untouched.
Key measured facts locking future choices: threshold grid re-confirmed 0.625 optimal
(plateau 0.60–0.675, <=0.04pp); one-to-one keeps +0.06pp (36 FN) — both closed as
tuning avenues. Blocking misses cost <=0.74pp and stay out of scope. V1 candidates
ranked by evidence: H1 soft digit/house-number similarity (digit-conflict positives
detected 62.0% vs 92.0% — largest single error mass), H2 clean-transliteration name
sims for Indic candidates (25% of India FPs are doubled-letter translit artifacts).
Any V1 test must retrain a CHALLENGER model on the SAME D14 split and compare
same-split macro-F0.5 (per D14); V0 artifacts are never overwritten.

## D17 — H1 challenger result; V0 still locked (2026-09-26)
H1 (V0 + digit_lev_best + digit_prefix_cont from cached digit-token sets; identical
data/split/config/threshold/o2o) scored **macro-F0.5 0.9348 vs V0 0.9322 (+0.26pp)**
on the locked D14 validation: recall +0.66pp at flat precision (+0.01pp); recovered
6,179 of 22,610 digit-conflict FNs; net +5,088 TP / +100 FP. US +0.37pp, India
+0.11pp. Challenger artifacts isolated: models/lgbm_v1_h1.txt +
experiments/v1_h1/ (scripts/v1_h1_challenger.py). **V0 (models/lgbm_final.txt,
t=0.625, o2o) remains the production config regardless of this result** — adoption
of H1 requires an explicit user decision. No threshold re-tuning, no H2/M1, no
further optimization performed after the measurement.

## D18 — H1 ADOPTED as production model; full TEST pipeline executed (2026-09-27)
User directive (2026-09-26 22:40, autonomous overnight run): H1 is the selected
production model for final inference; V0 stays untouched as fallback. Executed:
TEST shards 0/1/2 generated locally (locked D9/D11 config; S1 577,575/577,462/
577,507; pairs 73,312,490/73,280,775/73,211,657), verified disjoint + covering
all 1,732,544 TEST S1, merged (219,804,922 pairs). Isolated TEST caches
(experiments/cache_test), base features (219,804,922 rows, verified counts/
dup-free/finite), H1 columns merged in (experiments/test_features_h1, schema ==
booster header). Inference: models/lgbm_v1_h1.txt, threshold 0.625, one-to-one
(exact D15 semantics, deterministic tie-break) -> 5,561,174 matched pairs,
1,628,143 S1 matched, 104,401 empty. Official validator PASS (exit 0), including
the full --check-ids pass. Submission package built: team_submission.zip (1.1 GB;
output TSVs + code/business_entity_resolution/{src,README,requirements} + filled
Documentation_template.md; no datasets/parquets/logs/models). Final config of
record: **H1 (models/lgbm_v1_h1.txt, 21 features) + t=0.625 + o2o; validation
macro-F0.5 0.9348**. V0 fallback unchanged (models/lgbm_final.txt, 0.9322).
