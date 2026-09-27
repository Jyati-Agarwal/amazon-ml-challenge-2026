# FINAL SUBMISSION READINESS AUDIT (2026-09-26)

> **FINAL STATUS (2026-09-27 10:21): PIPELINE EXECUTED END-TO-END, VALIDATOR
> PASS (incl. --check-ids), PACKAGE BUILT.** H1 adopted (D18):
> models/lgbm_v1_h1.txt + t=0.625 + o2o. Outputs: output/matching_results.tsv
> (1,732,544 rows; 104,401 empty) + output/candidate_pairs.tsv (219,804,922
> pairs). Package: team_submission.zip (1.1 GB — rename to
> <team_name>_submission.zip before upload). Details in docs/PROGRESS.md
> and D18.

> **UPDATE (same day): BOTH BLOCKERS FIXED — code-only, NO TEST data
> processed, V0 unchanged.**
> - Fix 1: `--cache-dir` added to `feature_pipeline.py` prep/features
>   (`set_cache_dir()` redirects prep parquets + setmats together; default =
>   TRAIN cache, backward compatible).
> - Fix 2: `scripts/predict_test.py` created (verifies booster feature order
>   vs imported MODEL_FEATURES; t=0.625 default; validated o2o semantics with
>   stable (s1_id, cand_id) tie-break; every S1 emitted; candidate_pairs.tsv
>   = COMPLETE scored candidate set; sorted, dup-rejecting, deterministic).
> - Verified by `scripts/test_predict_synthetic.py` — **23/23 checks PASS**
>   on purely synthetic data, including official-validator exit 0 and a
>   "TRAIN cache untouched" mtime check.
> - Consistency vs real artifacts (read-only): booster header == meta.json ==
>   MODEL_FEATURES; validation_summary best_threshold == 0.625 == script
>   default. models/lgbm_final.txt untouched.
> The command sequence below is now runnable as written once all three TEST
> candidate shards are local. **No remaining code blockers.**

Code/readiness audit only — no TEST data read, nothing run, nothing modified.
Locked V0: `models/lgbm_final.txt` (19 features), threshold **0.625**,
one-to-one by model score, validated macro-F0.5 **0.9322**.

## Verdict summary

| # | Item | Verdict |
|---|---|---|
| 1 | TEST inference script | **C — BLOCKER: does not exist** |
| 2 | Feature caches for TEST (prep/setmat paths) | **C — BLOCKER: hardcoded to train; silent-corruption risk** |
| 3 | Candidate shard merge | **A — READY** (`merge_candidate_shards.py`) |
| 4 | Feature computation on TEST candidates | A — READY (`feature_pipeline.py features`), after #2 fixed |
| 5 | Model / feature order / threshold artifacts | A — READY |
| 6 | Official validator | A — READY (`utils/validate_submission.py`) |
| 7 | o2o determinism | B — small fix, fold into the new inference script |

## Q&A (the 17 audit questions)

1. **Which script performs TEST inference?** NONE. `train_final_model.py` has
   only featurize/train/validate; `featurize` requires GT labels and
   `validate` reads labeled `val_features.parquet`. No script loads the model,
   scores unlabeled pairs, applies a threshold/o2o, or writes
   `matching_results.tsv` / `candidate_pairs.tsv`. **BLOCKER — a new
   `scripts/predict_test.py` (~100 lines) is required.**
2. **Can it consume merged shards?** The featurizer can:
   `feature_pipeline.py features --pairs <dir>` reads `<dir>/parts/*.parquet`
   + `<dir>/s1_index/` (for global n_cands). Works on the merged dir.
3. **Multiple shard dirs directly?** Not in one invocation
   (`--pairs` is single). Running it 3× into ONE `--out` is UNSAFE: the
   manifest and output names key on the bare part filename
   (`cand_India_00000.parquet` exists in every shard → shard_1/2 parts would
   be skipped as "done" and outputs would collide). Either merge first
   (recommended — merged parts are `shardN_`-prefixed, collision-free) or use
   three separate `--out` dirs.
4. **Exact inputs/args** — see "Command sequence" below.
5. **Outputs produced**: merged candidates (`parts/`, `s1_index/`,
   `merge_manifest.json`); feature parquets `feat_shardN_cand_*.parquet` +
   `manifest.json`; then (new script) `output/matching_results.tsv`,
   `output/candidate_pairs.tsv`.
6. **candidate_pairs.tsv = final candidate set fed to model?** Yes by
   construction IF the new script writes both files from the same merged
   `parts/` (that is the requirement placed on `predict_test.py`). No current
   script writes it.
7. **Predictions ⊆ candidate_pairs.tsv?** Guaranteed structurally: inference
   only scores candidate pairs. The validator also cross-checks (warning).
8. **o2o after scoring, before writing?** Will be — validated semantics
   (`train_final_model.py` stage_validate lines 287–291): select prob≥t, sort
   by prob desc, drop duplicate cand_id keep-first. The new script must copy
   this exactly.
9. **Threshold 0.625 configured safely?** Recorded machine-readably in
   `experiments/train_final/validation_summary.json` (`best_threshold: 0.625`)
   and D15. The new script should read it from there (or a `--threshold`
   defaulting to 0.625) — never retune.
10. **Feature order matches lgbm_final.txt?** YES, triple-verified:
    `FEATURES` (feature_pipeline.py, fixed order) → `MODEL_FEATURES`
    (train_final_model.py, drops cand_addr_ntok) == `meta.json features` ==
    booster header `feature_names`. Inference must select
    `df[MODEL_FEATURES]` (import it — do not hand-copy the list).
11. **S1/S2/S3 IDs preserved?** Yes — `s1_id`/`cand_id` are carried verbatim
    (strings, `S1-`/`S2-`/`S3-` prefixes) through candidates → features;
    prep/featurize raise KeyError on any unknown ID.
12. **Empty-match S1 rows?** Not automatic — pairs above threshold cover only
    matched S1. The new script MUST emit one row per test S1 (empty second
    column when no match), sourcing the full S1 list from the merged
    `s1_index` (covers zero-candidate S1s; union of shards = all 1,732,544
    test S1). Validator enforces this.
13. **Duplicate predicted pairs impossible?** Yes: generate_candidates
    dedupes per (country, chunk), each S1 lives in exactly one chunk of
    exactly one shard, merge verifies disjointness + per-part dedupe. o2o
    additionally makes cand_ids unique across S1.
14. **Deterministic ordering?** Feature/scoring path is deterministic. Two
    requirements on the new script: sort output rows by s1_id, and use a
    stable sort for the o2o prob ordering (`kind='stable'`, tie-break by
    (s1_id, cand_id)) — `sort_values` default quicksort is deterministic for
    a fixed input but tie order is arbitrary; make it explicit.
15. **Validator works with planned paths?** Yes:
    `python3 utils/validate_submission.py --matching output/matching_results.tsv
    --candidate output/candidate_pairs.tsv --test-dir dataset/test`
    (exit 0 = safe). Optionally `--check-ids` (a few GB RAM, diagnostic only).
16. **Hidden merged-vs-sharded assumptions?**
    (a) The Q3 filename-collision trap — merge first.
    (b) `merge_candidate_shards.py` requires each teammate dir to contain
    `manifest.json` (state=done) + `parts/` + `s1_index/`, and configs must
    be identical except shard_id — teammates must copy the WHOLE shard dir.
    (c) n_cands comes from summed s1_index — correct because S1 sets are
    disjoint.
17. **Disk/RAM estimate** (scaled from train actuals: 255.7M pairs → features
    5.1 GB, featurize peak 14.2 GB; test ≈232M pairs):
    merged candidate copy ~2.8 GB + TEST prep/setmat caches ~3 GB + feature
    parquets ~4.7 GB + output TSVs ~3 GB ≈ **~14 GB disk** (95 GB free — OK).
    RAM peaks: prep ~6 GB, setmat build ~10 GB, featurize ~14 GB, inference
    (streamed per feat file, ≥t rows only kept: ~2.7% ≈ 6M pairs) <8 GB —
    all fine on 48 GB.

## BLOCKER 2 detail — cache-path collision (must fix before TEST featurize)

`feature_pipeline.py` hardcodes `PREP_S1/PREP_CAND` = `experiments/cache/
prep_{s1,cand}.parquet` and setmats = `experiments/cache/setmat_*.npz`:
- Running `prep --s1 dataset/test/...` would **overwrite the train prep
  parquets** (destroys artifacts).
- Worse: `FeatureComputer._load_setmats` loads any existing `setmat_*.npz`
  without checking what corpus built them → TEST rows against TRAIN matrices
  = silently garbage features, no error raised.

**Minimal fix (B):** add `--cache-dir` (default `experiments/cache`) to
`prep`/`features` and thread it through `stage_prep`/`FeatureComputer`
(~10 lines; replaces module-global CACHE/PREP_* at arg-parse time). TEST run
then uses `--cache-dir experiments/cache_test`. Train caches untouched.

## Command sequence once all 3 TEST shards are local (AFTER the 2 fixes)

```bash
# 0. teammates deliver whole dirs -> experiments/candidates/test_sharded/shard_{1,2}
#    (shard_0 generated here later, same layout). Verify on receipt:
#    manifest.json state=done, config identical except shard_id.

# 1. merge (verifies disjointness/dupes/config)                      [~5 min]
.venv312/bin/python scripts/merge_candidate_shards.py \
    --shards experiments/candidates/test_sharded/shard_0 \
             experiments/candidates/test_sharded/shard_1 \
             experiments/candidates/test_sharded/shard_2 \
    --out experiments/candidates/test_sharded/merged

# 2. TEST prep (isolated cache dir — needs the --cache-dir fix)      [~5 min]
.venv312/bin/python scripts/feature_pipeline.py prep \
    --s1 dataset/test/test_source1.tsv \
    --s2 dataset/test/test_source2.tsv \
    --s3 dataset/test/test_source3.tsv \
    --workers 4 --cache-dir experiments/cache_test

# 3. featurize merged TEST candidates (resumable; setmats built once) [~25 min]
.venv312/bin/python scripts/feature_pipeline.py features \
    --pairs experiments/candidates/test_sharded/merged \
    --out experiments/test_features \
    --workers 6 --cache-dir experiments/cache_test

# 4. inference + submission files (NEW SCRIPT — blocker 1)           [~10 min]
.venv312/bin/python scripts/predict_test.py \
    --features experiments/test_features \
    --cands experiments/candidates/test_sharded/merged \
    --model models/lgbm_final.txt --threshold 0.625 \
    --out output/
#    -> output/matching_results.tsv + output/candidate_pairs.tsv

# 5. official validator (must PASS, exit 0)
.venv312/bin/python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Merging: **REQUIRED** (Q3 collision trap + free integrity verification).

## H1 INTEGRATION AUDIT (2026-09-26 evening, read-only + 2 minimal fixes)

H1 = models/lgbm_v1_h1.txt (V0's 19 MODEL_FEATURES + digit_lev_best +
digit_prefix_cont; identical D14 split/params; val macro-F0.5 **0.9348** vs
V0 0.9322). V0 stays the LOCKED fallback; H1 adoption is a user decision (D17).

Findings and fixes (both verified synthetically, no TEST data, no heavy runs):

1. **predict_test.py was V0-hardwired** (hard equality assert on 19 features;
   read/scored only MODEL_FEATURES) → FIXED: it now takes the feature list
   from the booster's own header, REQUIRING it to start with the exact
   MODEL_FEATURES order and only extend it. V0 path byte-identical (23/23
   synthetic checks re-passed); real H1 header verified ==
   MODEL_FEATURES + [digit_lev_best, digit_prefix_cont]; wrong-order boosters
   still rejected. Threshold 0.625 default, o2o, sorting, candidate-pair
   handling all UNCHANGED.
2. **H1 features are NOT produced by feature_pipeline.py** — they come from
   v1_h1_challenger.py `featurize`, which was hardwired to TRAIN paths →
   FIXED: added --features-dir/--cache-dir/--out (defaults = TRAIN paths,
   backward compatible) and --merge (writes FULL parquets = base cols + 2 H1
   cols). Definitions identical to training by construction: same
   h1_features() code, driven by the prep 'digits' column that TEST prep
   (--cache-dir experiments/cache_test) produces with the same stage_prep
   code. Synthetic check: columns row-aligned, values correct (lev 0.75 case),
   base columns preserved.

**H1 runbook delta** (only if user adopts H1; V0 = skip both deltas):
after step 3 (featurize) insert
```bash
# 3b. H1 columns + merge into full parquets            [+~5 GB disk, ~20 min]
.venv312/bin/python scripts/v1_h1_challenger.py featurize \
    --features-dir experiments/test_features \
    --cache-dir experiments/cache_test \
    --out experiments/test_features_h1 --merge
```
and in step 4 use
`--features experiments/test_features_h1 --model models/lgbm_v1_h1.txt`
(threshold stays default 0.625). Steps 1, 2, 3, 5 unchanged.

## Final zip package audit (docs/PROBLEM_STATEMENT.md, lines 149–199)

Required structure `<team_name>_submission.zip`:
- `output/matching_results.tsv` + `output/candidate_pairs.tsv` (exactly what
  predict_test.py + validator produce/check).
- `code/business_entity_resolution/{src/, README.md, requirements.txt}` —
  self-contained runnable pipeline. Currently `src/` holds only a .gitkeep:
  **scripts/*.py + utils/validate_submission.py must be copied in**, plus a
  README with the exact runbook and the pinned requirements.txt (already
  pinned, incl. lightgbm 4.7.0 / pandas 3.0.6).
- `Documentation_template.md` filled in (exists at repo root, still template).

MUST NOT ship: dataset/, experiments/ (candidates, caches, features),
*.log, .venv*, .git. Model binaries are NOT required (pipeline must
regenerate them); lgbm .txt files are 3.5 MB each — include at most the one
production model if desired, never mandatory. Build the zip from an explicit
allowlist, not by zipping the repo. Model-license rule OK: LightGBM
(MIT-licensed library, ~thousands of tree parameters ≪ 8B).

Validator (unchanged): `python3 utils/validate_submission.py --matching
output/matching_results.tsv --candidate output/candidate_pairs.tsv
--test-dir dataset/test` — needs dataset/test/test_source1.tsv for the
required-S1 set; exit 0 + PASS = safe; optional --check-ids is memory-heavy.

## Required fixes before TEST inference (both small, neither touches V0 model)

1. **BLOCKER** — write `scripts/predict_test.py`: stream feat parquets,
   `booster.predict` in 5M-row chunks on `df[MODEL_FEATURES]`, keep rows
   ≥ threshold, o2o (stable sort by prob desc, tie-break (s1_id, cand_id),
   drop duplicate cand_id keep-first), emit `matching_results.tsv` with one
   row per test S1 (empty allowed, from merged s1_index) and
   `candidate_pairs.tsv` from merged parts; both sorted by s1_id, UTF-8,
   tab-separated, headers `source1_entity_id\tmatched_entity_ids` /
   `source1_entity_id\tcandidate_entity_ids`.
2. **BLOCKER** — `--cache-dir` parameter in `feature_pipeline.py` (see above).

Everything else in the chain is READY.
