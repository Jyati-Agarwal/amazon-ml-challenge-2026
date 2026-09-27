# PROGRESS

State file for session continuity. Update at every milestone; read after any
compaction together with DECISIONS.md, DATA_PROFILE.md, BLOCKING_FINDINGS.md,
BLOCKING_IMPLEMENTATION_PLAN.md, experiments/EXPERIMENT_LOG.csv.

## Phase status

| Phase | Status |
|---|---|
| Reconnaissance / data profile | DONE (docs/DATA_PROFILE.md) |
| Blocking recall study (exact, token, TF-IDF, translit, unions) | DONE (docs/BLOCKING_FINDINGS.md) |
| K-budget + rescue channels + implementation spec | DONE (docs/BLOCKING_IMPLEMENTATION_PLAN.md) |
| Matching model (classifier) | RESEARCH BASELINE DONE (Session 2 — see below) |
| Production generator (generate_candidates.py) | DONE + VALIDATED (20k reproduces benchmark exactly) |
| Full candidate generation on train | NOT STARTED — 3-way sharding implemented + validated, awaiting go-ahead |
| Full candidate generation on test | NOT STARTED (forbidden this stage) |

## Session 2 — feature signal + LightGBM baseline (2026-09-25, DONE)

- Feature-signal study on 357K sampled pairs (4,000 S1, 13,815 pos, hard negatives):
  docs/FEATURE_SIGNAL_RESEARCH.md. Final schema: docs/MODEL_FEATURE_SPEC.md (16 KEEP).
- One-to-one verified on full GT: 0 of 7,638,365 S2/S3 IDs map to >1 S1.
- Chain-ness (S1 core-name freq, log1p): decisive conditional signal
  (P(match|same name): 0.905 @freq1 → 0.020 @freq>100) but model recovers it without —
  keep, not critical. Computed leakage-free from S1 inputs only.
- LightGBM baseline (entity-level split 2,784/1,200 S1): ROC-AUC 0.9987, PR-AUC 0.976.
  **Best global threshold t=0.7 → macro-F0.5 0.9341** (t=0.5 gives 0.9270).
  US 0.9458 / India 0.9174. Singletons correct-empty 85.9%.
- Per-S1 expected-F0.5 rule: 0.9326 — loses to tuned threshold due to overconfidence in
  the 0.5–0.9 prob band → NEXT: isotonic calibration, then re-test rule.
- One-to-one post-processing: 0 pairs affected at sample scale (untestable here);
  validate on full-pipeline validation before adopting.
- Scripts: scripts/feature_signal_research.py, scripts/model_baseline.py.
  Caches: experiments/cache/*.parquet (git-ignored). Metrics:
  experiments/model_baseline_metrics.csv, model_error_samples.csv.
- Exact next step: full-scale candidate generation on a train validation slice →
  retrain on real blocking negatives → isotonic calibration → re-run threshold +
  decision-rule + one-to-one comparison at scale. Add digit_exact_conflict feature.

## Session 2 stage 3 — REAL-blocking validation experiment (2026-09-25, DONE)

Script: scripts/real_blocking_validation.py (stages: candidates | features | train).
Reuses Session-1 blocker output experiments/blocking/cand_sample/*.parquet
(20k stratified S1: 12k US / 8k India, top-200/channel vs FULL corpus), assembles the
locked union spec (US w50+c50+exact_name cap200; India w100+c100+exact_name cap200).

STATUS:
- [DONE] stage `candidates`: pair recall 0.9794 (US 0.9943 / India 0.9571;
  S2 0.9804 / S3 0.9784), entity all-match recall 0.9400, cand/S1 mean 115.8
  median 92 p95 182 max 291. 2,315,563 pairs, 1,424 blocking misses.
  Caches: experiments/cache/rb_candidates.parquet, rb_blocking_misses.parquet,
  rb_gt_counts.parquet. Log: experiments/cache/rb_candidates.log.
- [DONE] stage `features`: 19 features for 2.32M pairs → rb_features.parquet.
- [DONE] stage `train` + docs. RESULTS (full: docs/REAL_BLOCKING_MODEL_RESULTS.md):
  fit 11,200 / cal 2,800 / val 6,000 S1. Raw ROC 0.9982 PR 0.9691.
  **Best: raw threshold 0.70 → end-to-end macro-F0.5 0.9246** (US 0.9447 / India 0.8951;
  plateau 0.60–0.75). Isotonic ≈ no-op (overconfidence gone on real negatives;
  cal best 0.9240). Per-S1 expected-F0.5 rule ties (0.9245). One-to-one: 1 conflict in
  18,220 selected pairs, F0.5 unchanged — harmless safety step only.
  Singletons 91.1% correct-empty. Indic slice weakest (P .936 / R .798).
  Errors (3,487): A blocking 12%, B feature 43%, C calib/threshold 39%, D ambiguous 6%.
  Outputs written: real_blocking_validation_metrics.csv, threshold_results.csv,
  calibration_results.csv, model_error_samples.csv, REAL_BLOCKING_MODEL_RESULTS.md.
- NOTE: macro-F0.5 uses n_true from rb_gt_counts (blocking misses count as FN —
  honest end-to-end numbers, expect lower than research-sample 0.9341).
- digit_exact_conflict := 1 if both addresses contain ≥1 purely-numeric token and
  the sets don't intersect, else 0.
- Exact next step toward test inference: implement full-scale blocking runner
  (BLOCKING_IMPLEMENTATION_PLAN.md) on full train (~10 h) → multiprocess feature
  computation (~200M pairs) → retrain LightGBM on full train candidates → rerun
  threshold sweep at scale → freeze config → then (with user go-ahead) test inference.
  Also try: candidate-address token-count feature (address-over-trust FPs, India).

## Session 2 stage 4 — production feature+training pipeline (2026-09-25, DONE)

Full design + runbook: docs/FEATURE_PIPELINE.md. Decisions: D12/D13/D14.
Everything below ran CONCURRENT with candidate-gen shard 0 (~6 of 14 cores) —
throughput numbers are conservative. The running shard jobs were never touched.

- [DONE] scripts/feature_pipeline.py — production feature computation.
  `prep` (per-RECORD one-time normalization -> experiments/cache/prep_s1.parquet
  + prep_cand.parquet; 12.5M records, 230s, peak 5.9GB) + one-time cached global
  binary CSR set matrices (setmat_*.npz, 512s build, peak 9.7GB) + per-chunk
  features = CSR row-gather intersections + rapidfuzz process.cpdist + numpy.
  Feature definitions are EXACT stage-3 copies. 20 float32 cols = validated 19
  + cand_addr_ntok. NaN/inf guard. Chunked, manifest-resumable `features` stage.
- [DONE] Benchmark (experiments/feature_pipeline_benchmark.json, 2,315,563 pairs):
  old loop 28,756 pairs/s (200k sample) vs new **119,665 pairs/s** end-to-end
  (162,004 steady-state) = x4.2-5.6; peak RSS 12.5GB; **max abs diff 0.0 on all
  19 features vs rb_features.parquet; 0 NaN/inf**. Full-train ~200M pairs
  projected ~30-45 min.
- [DONE] scripts/train_final_model.py (featurize | train | validate), resumable.
  Entity split MD5(s1_id)//1000 % 100 < val_pct (D14). Labels from TRAIN GT only.
  s1_meta.parquet covers zero-candidate S1. Model uses MODEL_FEATURES = 19
  (**cand_addr_ntok EXCLUDED — hurts: 0.8983 vs 0.9113 macro-F0.5, D13**).
  Validate: grid 0.50-0.85 (0.60-0.75 focus), macro-F0.5 w/ blocking misses as
  FN, singletons, one-to-one, country slices.
- [DONE] Integration smoke on val20k_py312 THROUGH the scripts: featurize 2.32M
  pairs @66k pairs/s (incl. I/O+labels) -> train 52s -> validate: best
  **macro-F0.5 0.9120 @ t=0.70** (plateau 0.60-0.75), o2o dropped 0, US 0.9385 /
  India 0.8737, singletons 86.1% (1,949 val S1 @ val-pct 10). Reference control
  (stage-3 rb_features, same split scheme): 0.9138-0.9144 -> pipeline within
  split noise; the 0.9246 headline was a different rng split (D14 caveat).
- [READY] When shard_{0,1,2} finish, run the 3 commands in
  docs/FEATURE_PIPELINE.md "Runbook". Full training NOT run (user gate).
  Do NOT run TEST inference.

## What was learned (compressed)

- Dataset: train S1 2.207M / S2 5.035M / S3 5.286M; test S1 1.733M / S2 4.887M / S3 5.082M.
  GT: 5.6% singletons, avg 3.461 matches per S1 (max 11); each S2/S3 record matches ≤1 S1.
  ~26% of S2/S3 are distractors. France only in test (15% of test S1).
- GT matches NEVER cross country (1.000000) → per-country blocking lossless.
- Exact keys weak (name 25.8%, addr 8.3%, union 32.5%).
- Word TF-IDF name+addr top-100 per country: 97.4% pair recall / 92.4% entity recall.
  Char(3,4) name+addr close behind; name-only representations are 25 pp worse — the
  address is load-bearing. S2≈S3 under every method.
- India lags US 4–5 pp everywhere (9 Indic scripts = 16.9% of India corpus names; only
  61% of those Devanagari). Transliteration (naive AND cleaned) rejected for blocking:
  Latin addresses already retrieve most script-mismatch records (86% @word k100).
- Miss tail ~2.5% at practical budgets: zero name-token overlap (DBA renames, domain
  names, leet typos) 63%, empty S2/S3 address 32%, script-mismatch + weak address ≥24%.

## K-budget / rescue study (2026-09-25, DONE — do NOT rerun)

`scripts/blocking_budget.py` finished (18 min, peak 13.7 GB). Outputs:
experiments/blocking/blocking_budget_comparison.csv, budget_runtimes.json,
miss_tail_categories.json, missed_by_base_union.csv,
cand_sample/{word,char}_{US,India}.parquet (top-200 per sampled S1 — REUSE these).

Results (unions include exact normalized name):
- w50+c50+e: pair recall 0.9750 (US 0.9943 / India 0.9461), entity 0.9296, 85.8 cands/S1.
- w100+c100+e: 0.9806 / entity 0.9441 @ 164.3 cands.
- **Asym US(w50c50e)/India+default(w100c100e): 0.9794 (US 0.9943 / India 0.9571),
  entity 0.9400 @ 116.3 cands (p99 254). ADOPTED** — matches big-budget India recall at
  29% less volume. Open-set config: only US gets the small budget; default = large.
- Rescue channels REJECTED: R1 exact-addr +1 pair; R2 rare-token +32 pairs (+0.046 pp)
  for +8.5% candidates.
- Miss tail (2.50% of pairs; overlapping cats): zero name-token overlap 60%,
  Indic script (9 scripts, widened flag) 48%, empty address 27%, domain names 6.5%;
  86% of misses are India. Accepted as ceiling; no cheap channel reaches them.
- Per-query retrieval cost: word 2.2–3.1 ms, char 23–26 ms (char = bottleneck).
  Projection: TRAIN ~257M pairs / 2–4 GB parquet / ~17 h; TEST ~232M / 2–4 GB / ~11 h.
  Lever: drop US char channel (−0.4 pp US) saves ~11 h across both splits.
- Spec: docs/BLOCKING_IMPLEMENTATION_PLAN.md (normalization, budgets dict, block cap 200,
  chunked sp_matmul_topn, parquet bitmask output).

## Production candidate generation (2026-09-25, Session 1 — VALIDATED, do NOT redo)

Full report: docs/TRAIN_CANDIDATE_GENERATION.md. Summary:

- `scripts/generate_candidates.py` — production generator (GT-free), locked asym spec,
  open-set budgets `{"US": (50,50)}` default `(100,100)`, exact-name cap 200 (empty
  names excluded), chunked sp_matmul_topn, parquet parts with channels bitmask
  (1=word/2=char/4=exact) + word/char ranks, (country, chunk) resume via manifest.json.
  `scripts/evaluate_candidates.py` (GT only here), `scripts/normalization_audit.py`.
  Run everything with `.venv/bin/python` (3.9 — the stack that made the benchmark).
- **Normalization finding:** the benchmark norm strips Indic matras (Python `re` `\w`
  excludes Mn) — `प्राइवेट` → `प र इव ट`. Measured A/B at 20k: benchmark norm 0.9794/0.9400
  vs matra-preserving 0.9792/0.9395 → recall-neutral; **benchmark norm kept as default**,
  matra variant behind `--matra-norm`. Audit otherwise clean (idempotent, S1==S2/S3 path,
  0 non-empty names collapse to empty, ~3.35% empty corpus naddr as profiled).
- **20k validation: digit-for-digit reproduction** — pair 0.9794 (US 0.9943 / India
  0.9571), entity 0.9400, mean 115.8 / p99 254, 2,315,556 pairs, 18 min, peak 14.4 GB.
  All correctness checks pass (0 dups, no cross-country, source/ID/channel consistent).
- **Full unsharded TRAIN run (~17 h) was launched then INTENTIONALLY STOPPED after
  ~3.5 min** (0 chunks completed; only train_full/manifest.json exists) — team decision
  to move to **3-way S1 sharding** so three members run generation in parallel.
  Not a failure; implementation unchanged and validated. Sharding hooks already exist:
  `--s1-ids <file>` (one entity_id/line) or `--sample`, distinct `--out` per shard.
  Outputs live under experiments/candidates/ (git-ignored): smoke/, val20k_legacy/,
  val20k/ (+ *_report.json) — keep these.

## 3-way S1 sharding implementation (2026-09-25, Session 1 — spec: docs/PARALLEL_SHARDING.md)

- Full TRAIN unsharded run was intentionally stopped after ~3.5 min (reason: move
  to 3-way sharding). **No usable full TRAIN candidate set exists**; the partial
  `experiments/candidates/train_full/` (manifest only, 0 chunks) is NOT final and
  must not be treated as data.
- Environment: all NEW runs use **`.venv312/bin/python`** (3.12.14). Old `.venv`
  (3.9) is fallback only.
- Sharding: `MD5(entity_id)%num_shards` (D11) via `--shard-id/--num-shards` in
  generate_candidates.py. Only S1 sharded; S2/S3 full; TF-IDF fit on full corpus →
  merged shards ≡ unsharded. Train shard sizes 735,761/736,288/734,772; test
  577,575/577,462/577,507; GT sharded + verified
  (experiments/candidates/train_sharded/gt/).
- New scripts: scripts/merge_candidate_shards.py (streaming merge + disjointness/
  dup checks), scripts/shard_ground_truth.py.
- Controlled test (2,000 S1: US:1200/India:800 seed 42, chunk 1000, .venv312) —
  ALL PASS: shards 656/660/684 (disjoint, union complete), merged output
  row-for-row IDENTICAL to unsharded (234,838 rows, all columns incl. channels +
  ranks), recall identical (pair 0.9785, entity 0.9380 — in line with 20k bench
  at 2k noise), 0 dup pairs, full corpora searched per shard, correct budgets.
  Artifacts: experiments/candidates/shardtest/ (+ reports, log).
- **20k re-validation under .venv312 — PASS, byte-identical to the 3.9 benchmark**
  (pair 0.9794 US 0.9943/India 0.9571, entity 0.9400, 2,315,556 pairs, 16.5 min;
  val20k_py312/ rows == val20k_legacy rows in every column). .venv312 is safe for
  production; the "re-validate before switching env" prerequisite is satisfied.
- Merge test: merge_candidate_shards.py PASS (disjointness, config match, no dup
  (s1_id, source, cand_id), counts preserved).

## Runtime / memory observations

- Full exact-key study (2.2M×10.3M hash joins): 4.3 min total.
- TF-IDF retrieval: build 6.19M-doc char nameaddr matrix ~168 s; matmul 12k queries
  vs 6.19M docs, top-200, 12 threads ~105–190 s. Whole 20k-sample study: 30 min,
  peak RSS 16.2 GB. Transliteration follow-up (India only): 8.1 min.
- Extrapolation: full-train retrieval (2.2M queries) ≈ 3–6 h per config per country pair;
  both configs both splits ≈ 10 h train + ~8 h test. Feasible in 48 h budget.

## Failed / rejected approaches (do not retry blindly)

- Naive unidecode transliteration for retrieval: no effect (doubled-letter artifacts).
- Cleaned transliteration (collapse repeats): +2.5 pp deva subset on char, 0 on word —
  rejected for blocking (kept as candidate matcher FEATURE later).
- Rare-token blocking as primary: ≤18.4% recall standalone.
- Name-only retrieval: 72% @k100 — rejected.
- exact addr / addr+country as primary: 8.3%.

## Unresolved questions

1. France recall — unmeasurable (no train labels); mitigated by default-large budget.
2. Isotonic calibration of the LightGBM baseline, then re-test per-S1 decision rule
   (Session 2 finding).
3. One-to-one assignment post-processing — validate at full-pipeline scale.
4. Whether US char channel is worth ~11 h of retrieval for +0.4 pp US recall (decide
   when scheduling the full runs).

## Session 3 — environment + integration smoke test (2026-09-25, DONE)

- New env **`.venv312`** (Python 3.12.14, Homebrew) — use it for all future runs;
  old `.venv` (3.9) kept untouched as fallback. Pins in `requirements.txt`.
  Packages: numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, sklearn 1.9.1, rapidfuzz 3.14.6,
  lightgbm 4.7.0 (needed `brew install libomp` — done), pyarrow 25.0.1, polars 1.44.2,
  sparse-dot-topn 1.2.0, unidecode 1.4.0. faiss-cpu deliberately skipped.
- Perf micro-benchmarks: scripts/benchmark_env.py (docs/ENVIRONMENT_READY.md §5–6).
- **Integration smoke test: scripts/smoke_e2e.py — ALL PASS** (ENVIRONMENT_READY.md §9):
  pandas-3 idiom checks, real module imports, Session-2 parquet caches read OK,
  tiny e2e (300 S1 / 6k corpus: normalize → partition → word+char TF-IDF →
  sp_matmul_topn top-k → exact-name → union/dedupe → provisional features → LightGBM),
  output schema + utils/validate_submission.py exit 0. Temp outputs deleted.
- **pandas 3 verdict: compatible, no `<3` pin needed.** One behavioral note: string-col
  `.values` is now ArrowStringArray (works in all project uses; prefer `.to_numpy()`).
  Cosmetic future fix: real_blocking_validation.py:373-374.
- Known bottleneck for full runs: per-pair Python-loop featurization (~15k pairs/s
  single-core) — production must use rapidfuzz batch APIs / multiprocessing.
- model_baseline.py NOT executed end-to-end (would overwrite Session-2 artifacts).

## Code shared with team (2026-09-25)

Pushed commit f94766c ("Add parallel entity-resolution pipeline") to
github.com/Jyati-Agarwal/amazon-ml-challenge-2026 main (rebased onto the team's
existing scaffolding; problem-statement README moved to docs/PROBLEM_STATEMENT.md;
.gitignore merged to strict union, experiments/ fully ignored). Code + docs only —
verified no dataset/TSV/CSV/parquet/model/log files in the diff.

## FULL TRAIN SHARD RUNS (user green-lit; all on this machine, .venv312)

- **Shard 0 — DONE 2026-09-26 01:48** (launched 09-25 20:05, 343.3 min, peak
  17.5 GB, exit 0). 735,761 S1 → **85,296,843 pairs** (India 295,264 S1 / 3 chunks
  / 45,736,145 cands; US 440,497 S1 / 5 chunks / 39,560,698 cands). Manifest
  state=done, 8/8 chunks. Output: experiments/candidates/train_sharded/shard_0
  (parts/ + s1_index/), log shard_0.log. Merge-ready.
- **Shard 1 — DONE 2026-09-26 07:16** (326.9 min, peak 17.2 GB, exit 0):
  736,288 S1 → **85,264,716 pairs** (India 45,545,075 / US 39,719,641).
- **Shard 2 — DONE 2026-09-26 12:43** (326.9 min, peak 17.3 GB, exit 0):
  734,772 S1 → **85,142,372 pairs** (India 45,568,345 / US 39,574,027).
- **ALL THREE SHARDS COMPLETE: 2,206,821 S1 (full train), 255,703,931 candidate
  pairs total** (vs ~257M projection). All manifests state=done, all chunks
  complete. ~952 MB parquet per shard (~2.8 GB total), disk has 107 GB free.
  Outputs: experiments/candidates/train_sharded/shard_{0,1,2}/ — merge-ready.
- NOT merged yet (user gate). NOT evaluated vs GT yet.

## FINAL SPRINT — full-train featurize -> train -> validate (2026-09-26, USER GREEN-LIT)

User authorized the full pipeline (<24h to deadline). Locked config, NO new
features, NO test candidates, NO git changes. train_final_model.py consumes the
3 shard dirs directly (no merge step needed — featurize reads parts/ + s1_index/
per shard). Caches verified present + reused (prep_s1/prep_cand + 6 setmat npz).

- [DONE 13:22] featurize: 255,703,931 pairs in 969 s (263,918 pairs/s), peak
  RSS 14.2 GB, exit 0. GT 2,206,821 S1 / 7,638,365 pos links; 0 zero-candidate
  S1. Output: experiments/train_final/{features/, s1_meta.parquet,
  featurize_manifest.json}. Log: experiments/train_final_featurize.log.
- [DONE 13:38] train: 957 s, peak RSS 21.4 GB, exit 0. Split (D14 MD5 val_pct
  10): 230,138,888 train pairs / 25,565,043 val pairs; positives 6,734,146
  (2.9261%). Model: models/lgbm_final.txt (+ .meta.json), 19 MODEL_FEATURES.
  Top importances: name_jw_translit, name_3gram_jac_translit, name_tok_jac,
  addr_3gram_jac, addr_tok_cont, addr_len_diff. Val rows ->
  experiments/train_final/val_features.parquet. Log: train_final_train.log.
- [DONE 13:39] validate: 51 s, peak 7.1 GB, exit 0. 25,565,043 val pairs,
  220,429 val S1 (12,413 singletons), 762,069 true links (blocking misses as
  FN). **BEST t=0.625: macro-F0.5 0.9316; +one-to-one (739 of 688,480 pairs
  dropped) -> 0.9322.** Plateau 0.600-0.700 all >=0.9312. P=0.9707 R=0.8770
  at t=0.625. US 0.9487 (131,816 S1) / India 0.9062 (88,613 S1). Singletons
  correct-empty 89.57%. Outputs: experiments/final_validation_metrics.csv,
  validation_summary.json. Log: train_final_validate.log. No warnings.
- FINAL MODEL: models/lgbm_final.txt + threshold 0.625 + one-to-one (D15).

## Final submission readiness audit (2026-09-26, code-only — no TEST data read)

Full audit: docs/FINAL_SUBMISSION_READINESS.md. Verdict: chain is READY
EXCEPT two blockers, both small and neither touching the locked V0 model:
(1) no TEST inference/submission script exists (need scripts/predict_test.py:
stream features -> predict -> t=0.625 -> o2o -> matching_results.tsv +
candidate_pairs.tsv); (2) feature_pipeline.py cache paths hardcoded to
experiments/cache -> TEST prep would overwrite TRAIN caches and stale TRAIN
setmats would silently corrupt TEST features — need a --cache-dir arg
(~10 lines). Merge of the 3 TEST shards is REQUIRED before featurizing
(bare part filenames collide across shards in feature_pipeline's manifest).
Exact command sequence is in the readiness doc. Disk ~14 GB / RAM <=14 GB —
fits. Feature order verified: FEATURES == meta.json == booster header.

## Readiness blockers FIXED (2026-09-26, code-only — NO TEST data processed)

Both audit blockers implemented + synthetically tested. V0 unchanged
(models/lgbm_final.txt untouched, mtime 13:38; threshold 0.625 + o2o intact).
- FIX 1: `--cache-dir` on feature_pipeline.py prep/features (default =
  TRAIN cache experiments/cache, fully backward compatible). set_cache_dir()
  redirects prep_s1/prep_cand/setmat_*.npz together, so stale TRAIN setmats
  can never be loaded against TEST prep rows. TEST must use
  `--cache-dir experiments/cache_test`.
- FIX 2: NEW scripts/predict_test.py — streams feat_*.parquet, verifies
  booster feature order == MODEL_FEATURES (imported, not copied), predicts
  in 5M chunks, t=0.625 (D15 default, warns on override), o2o with stable
  sort + (s1_id, cand_id) tie-break, emits output/matching_results.tsv
  (every S1, empty allowed, from merged s1_index) + candidate_pairs.tsv
  (COMPLETE scored candidate set), both sorted by s1_id. Rejects duplicate
  input pairs and cross-file S1 straddling.
- NEW scripts/test_predict_synthetic.py: 23/23 PASS on purely synthetic data
  (byte-identical reruns, threshold, o2o + tie-break, subset property,
  zero-cand S1, complete candidate set, official validator exit 0, cache
  isolation incl. "TRAIN cache untouched" mtime check).
- Consistency verified vs real artifacts (read-only): booster header ==
  meta.json == MODEL_FEATURES == FEATURES minus cand_addr_ntok;
  validation_summary best_threshold == 0.625 == predict_test default.
- Runbook once all 3 TEST shards local: docs/FINAL_SUBMISSION_READINESS.md
  (merge -> prep --cache-dir experiments/cache_test -> features -> 
  predict_test -> validate_submission). No remaining code blockers.

## Session 2 stage 5 — V0 error analysis (2026-09-26, DONE — analysis only)

**V0 LOCKED**: models/lgbm_final.txt + t=0.625 + one-to-one, macro-F0.5 0.9322
(D15/D16). Nothing modified. Full report:
experiments/error_analysis/ERROR_ANALYSIS.md (+ 9 CSVs, examples.txt,
val_probs.npy). Script: scripts/error_analysis_v0.py (score | analyze,
read-only, 45s + 29s).

- Error budget (oracle): remove all FP -> 0.9542 (+2.20pp); recover all
  candidate FN -> 0.9709 (+3.87pp); blocking ceiling 0.9926.
  FP 19,438 / FN-lowscore 77,760 / FN-o2o 36 / blocking miss 15,970 (2.10%).
- **Sharpest finding: digit_exact_conflict positives detected 62.0% vs 92.0%
  without** (8% of positives; house-number typos in true matches). 29.1% of
  FNs + 13.6% of FPs are digit-conflict cases; 49.7% of those FNs at p>=0.25.
- FP core: same/similar address wrong business (66.9% addr_high; 19.5%
  addr_high+name_low, 85% India); chains 31.4%; Indic translit doubled-letter
  artifacts 14.4% (100% India).
- Blocking misses: 82% India, 63% zero name overlap, 51% Indic cand name,
  25% empty cand addr; 95.2% of missed pairs' S1s still have another TP
  candidate -> cost bounded <=0.74pp. Blocking stays locked.
- Threshold grid 0.50-0.75: 0.625 exactly optimal (plateau 0.60-0.675
  within 0.04pp). o2o: +0.06pp, 36 FN. Both exhausted.
- India 0.9073 vs US 0.9489: worse on all 3 error classes (FP rate 4.18% vs
  1.95%, miss 4.29% vs 0.62%).
- V1 ranking: H1 soft digit/house-number similarity (FIRST — recall,
  low risk), H2 clean-translit name sims for Indic (P+R, India), M1
  addr_missing x name-strength, M2 within-S1 rank features. LOW: threshold,
  o2o, blocking, DBA, cand_addr_ntok, isotonic. NOT implemented (user gate).

## TEST CANDIDATE GENERATION (user green-lit, 2026-09-26)

Assignment change: THIS machine runs **shard 0**; teammates run shard 1 and
shard 2 on their machines. The brief local shard-1 attempt (started 15:42,
killed ~15:45 by session teardown, 0 chunks completed) is quarantined at
experiments/candidates/test_sharded/shard_1_partial_local_DO_NOT_USE/ —
never merge it; the teammate's shard_1 dir replaces it.

- [**DONE** 19:38, exit 0] TEST **shard 0** on this machine (15:58–19:38,
  220.2 min, caffeinate). Locked config confirmed in manifest (US 50/50,
  default 100/100, exact cap 200, threshold 0.05, benchmark norm, no
  sampling, seed 42, sources s2,s3, shard 0/3).
  - S1 rows: **577,575** (matches expected exactly): France 86,380 /
    India 270,330 / US 220,865. Corpus 9,969,589.
  - Candidate pairs: **73,312,490** (France 13,076,848 / India 41,720,458 /
    US 18,515,184).
  - Peak memory 17.3 GB; wall 3 h 40 m (faster than the ~5.5 h estimate).
  - Manifest: experiments/candidates/test_sharded/shard_0/manifest.json,
    state=done, all 7 (country,chunk) entries done.
  - Output: experiments/candidates/test_sharded/shard_0/ (parts/ 7 parquets,
    s1_index/, manifest.json; 835 MB total).
    Log: experiments/candidates/test_sharded/shard_0.log.
- ASSIGNMENT CHANGE (user, 21:18): teammate delivery too slow — THIS machine
  now runs shards 1 and 2 SEQUENTIALLY (shard 1 first), then auto-merges all
  three. Shard 0 untouched. NO featurize/inference after merge (user gates).
- [**DONE** 00:57 (2026-09-27), exit 0] TEST **shard 1** (21:18–00:57,
  218.8 min): S1 **577,462** (exact expected) = France 86,707 / India
  270,270 / US 220,485; pairs **73,280,775** (Fr 13,115,100 / In 41,700,109
  / US 18,465,566); peak 15.2 GB; manifest state=done, parts/ + s1_index/
  present, config locked (US 50/50, default 100/100, cap 200, thr 0.05,
  seed 42). Quarantined partial NOT used.
- [**DONE** 04:34 (2026-09-27), exit 0] TEST **shard 2** (00:58–04:34,
  216.7 min): S1 **577,507** (exact expected); pairs **73,211,657**;
  peak 14.2 GB; manifest state=done, locked config identical.
- [**VERIFIED** ~09:20] All 3 shards: state=done, configs identical except
  shard_id, S1 disjoint, union = 1,732,544 = full test_source1 ID set.
- [**MERGE DONE**, exit 0] experiments/candidates/test_sharded/merged/:
  **219,804,922 pairs** (= exact sum 73,312,490+73,280,775+73,211,657),
  1,732,544 S1, 21 parts + s1_index + merge_manifest.json. Shard dirs
  untouched. Log: experiments/candidates/test_sharded/merge.log.
- [**PREP DONE** 09:23, exit 0] TEST prep -> experiments/cache_test/
  (prep_s1 173M / prep_cand 1.0G; 1,732,544 S1 + 9,969,589 cand records,
  64 s). TRAIN cache experiments/cache/ untouched (mtimes Sep 25).
- [**FEATURES DONE** 09:36, exit 0] base TEST featurize ->
  experiments/test_features: 219,804,922 rows in 445 s (21 files, 4.5 GB);
  verified: row total == merged manifest, no dup pairs, all finite.
- [**H1 FEATURES DONE** 09:44, exit 0] v1_h1_challenger featurize --merge ->
  experiments/test_features_h1 (348 s, 4.6 GB); verified: schema contains
  all 21 booster features (2 H1 cols appended), row-aligned to base,
  H1 values finite in [0,1].
- [**H1 INFERENCE DONE** 09:56, exit 0] predict_test with
  models/lgbm_v1_h1.txt (21 features verified vs booster header), t=0.625,
  o2o (dropped 482,848 dup-cand claims): **5,561,174 matched pairs;
  1,628,143 S1 with >=1 match; 104,401 empty (6.0%)**; both output TSVs
  1,732,544 rows. Log: experiments/predict_test_h1.log.
- [**VALIDATOR PASS**, exit 0] utils/validate_submission.py PASS, and a
  second run with --check-ids (full ID-existence) also PASS.
- [**PACKAGE BUILT** 10:21] team_submission.zip (1.1 GB) at repo root from
  allowlist staging dir submission_build/: output/{matching_results,
  candidate_pairs}.tsv + code/business_entity_resolution/{src/scripts (21
  .py), src/utils/validate_submission.py, README.md, requirements.txt} +
  filled Documentation_template.md. Audited: no dataset/experiments/logs/
  parquets/models inside. **RENAME to <team_name>_submission.zip before
  upload** (team name unknown to this machine); leaderboard file =
  output/matching_results.tsv.
- FINAL CONFIG (D18): H1 models/lgbm_v1_h1.txt + t=0.625 + o2o
  (val macro-F0.5 0.9348). V0 fallback untouched. PIPELINE COMPLETE.
- AUTONOMOUS OVERNIGHT DIRECTIVE (user, 22:40, asleep — do not wait between
  phases): shard 1 → shard 2 → verify 3 shards → merge → prep
  (--cache-dir experiments/cache_test) → features (experiments/test_features)
  → H1 featurize --merge (experiments/test_features_h1) → predict_test with
  **models/lgbm_v1_h1.txt (H1 ADOPTED as production model, D17)**, threshold
  0.625 + o2o → official validator (must PASS) → build final submission
  package → final audit → update docs. V0 = untouched fallback. STOP only on
  genuine unrecoverable error (record command+error+next action here). Never:
  retrain, H2/M1, change threshold/blocking, delete artifacts, touch
  train_sharded/ or the quarantined partial.
- Sequence after shard 1: shard 2 (--shard-id 2 --out .../shard_2, expected
  S1 577,507), verify all 3 (state=done, disjoint S1, total 1,732,544),
  merge_candidate_shards.py --out .../merged, then CONTINUE through
  inference/validation/packaging per the directive above.

## Session 2 stage 6 — H1 challenger experiment (2026-09-26, DONE — V0 UNCHANGED)

Controlled challenger per D16-H1: V0 methodology + 2 soft digit features
(digit_lev_best = max cross-token Levenshtein sim of cached digit tokens;
digit_prefix_cont = proper-prefix containment). Same 255.7M pairs, same D14
split (verified: 230,138,888/25,565,043/6,734,146 pos identical to V0), same
D9 config, t=0.625 + o2o. Script scripts/v1_h1_challenger.py; model
models/lgbm_v1_h1.txt; report experiments/v1_h1/RESULTS.md.

- **H1 macro-F0.5 0.9348 vs V0 0.9322 (+0.26pp)**; P +0.0001 / R +0.0066 —
  gain is pure precision-safe recall. US +0.37pp (0.9526), India +0.11pp
  (0.9084). Singletons 0.8961 (-0.15pp).
- Digit-conflict FNs recovered 6,179/22,610 (27.3%); net +5,088 TP vs net
  +100 FP (3,095 new / 2,995 fixed).
- Decision rule: improvement found -> reported and STOPPED. **V0 stays
  LOCKED**; adoption is a user decision (D17). No further optimization done.

## H1 integration readiness audit (2026-09-26 evening, while waiting on shards 1+2)

Read-only audit + 2 minimal code fixes; NO heavy computation, NO TEST
inference/featurize/merge, shard dirs untouched, V0 model/threshold untouched.

- A) H1 model compatibility: **PASS (after fix)** — predict_test.py now takes
  feature order from the booster header (must start with exact MODEL_FEATURES,
  may only extend); real H1 header verified = 19 V0 features +
  [digit_lev_best, digit_prefix_cont]. V0 path byte-identical: full 23/23
  synthetic suite re-passed; wrong-order boosters still rejected.
- B) TEST availability of H1 features: **PASS (after fix)** —
  feature_pipeline.py does NOT compute them; v1_h1_challenger.py featurize
  now accepts --features-dir/--cache-dir/--out/--merge (defaults = TRAIN,
  backward compatible) and writes merged full parquets for TEST. Same
  h1_features() code + prep 'digits' column ⇒ identical definitions.
- C) Inference readiness: **PASS** — threshold 0.625, o2o, deterministic
  sort, candidate_pairs handling unchanged for both V0 and H1.
- Package audit: zip needs output/ TSVs + code/business_entity_resolution/
  {src/ (currently only .gitkeep — copy scripts/ + utils/ in), README,
  pinned requirements.txt} + filled Documentation_template.md; exclude
  dataset/, experiments/, logs, venvs; models not required (3.5 MB each if
  wanted). Details + H1 runbook delta in docs/FINAL_SUBMISSION_READINESS.md.
- V0 remains the fallback; H1 adoption still a user decision (D17). No H2/M1.

## Exact next step

Final sprint pipeline DONE (featurize -> train -> validate, all exit 0).
V0 error analysis DONE (stage 5) and H1 challenger DONE (stage 6:
+0.26pp, awaiting user adoption decision).
STOPPED per instruction before TEST work. Awaiting user go-ahead for:
TEST candidate generation (sharded, other machines per plan), TEST prep/
featurize/inference, submission build. Do NOT push to GitHub yet. Partial
experiments/candidates/train_full/ (manifest only, 0 chunks) is NOT data.

---

## Appendix — teammate shard-1 report (merged from origin/main)

A teammate independently generated TEST shard 1 on their machine before the
assignment moved back here; their counts match this machine's shard-1 run
EXACTLY (73,280,775 pairs; per-country identical), confirming cross-machine
determinism of the locked config. Their report preserved below.

# TEST Candidate Generation — Shard 1/3

## Status
DONE

## Environment
- Branch: main
- Python: .venv312 (Python 3.12.14)
- Production architecture: LOCKED

## Input
- S1: dataset/test/test_source1.tsv
- S2: dataset/test/test_source2.tsv
- S3: dataset/test/test_source3.tsv

## Sharding
- shard_id: 1
- num_shards: 3
- S1 rows processed: 577,462
- Assignment: MD5(entity_id) % 3
- S2/S3: full, unsharded

## Locked configuration
- US: word=50, char=50
- All other countries: word=100, char=100
- Exact normalized-name cap: 200
- Benchmark normalization
- Word TF-IDF + char TF-IDF + exact normalized-name
- No sampling
- No train generation
- No merge
- No TEST inference

## Results
- Total candidate pairs: 73,280,775
- France candidates: 13,115,100
- India candidates: 41,700,109
- US candidates: 18,465,566
- Runtime: 42,555.7 seconds (~11h 49m)
- Manifest state: done

## Verification
- Candidate parquet files: 7
- S1 index files: 7
- S1 count check: PASS
- Candidate count check: PASS

## Output
experiments/candidates/test_sharded/shard_1/

## Resume
If needed, rerun the exact production command; completed chunks are resumable.
