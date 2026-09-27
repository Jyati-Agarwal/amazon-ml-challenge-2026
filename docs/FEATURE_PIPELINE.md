# FEATURE_PIPELINE — production feature computation + training (Session 2 stage 4)

Scripts: `scripts/feature_pipeline.py` (features) and `scripts/train_final_model.py`
(featurize | train | validate). Environment: **`.venv312/bin/python`** only.
Consumes `scripts/generate_candidates.py` shard output directly
(`<shard>/parts/*.parquet` + `<shard>/s1_index/*.parquet`); never loads the full
200M+ pair set into RAM.

## Design

Feature semantics are EXACT copies of the validated stage-3 implementation
(`scripts/real_blocking_validation.py`) — verified byte-identical (see Benchmark).
The speed comes from restructuring, not from changing any definition:

1. **`prep` stage (one-time, per-RECORD not per-pair).** Every S1 + corpus record
   is normalized once (norm / core tokens / squash / digit tokens / locality
   tokens / Indic-transliterated name view / lengths / flags) →
   `experiments/cache/prep_s1.parquet` (232 MB) + `prep_cand.parquet` (1.14 GB).
   S1 chain-ness freq (`s1_name_freq_log`) built here from S1 inputs only.
   12.5M records, 230 s, peak 5.9 GB (ProcessPoolExecutor, spawn, 4 workers).

2. **Global per-record set matrices (one-time, cached).** For each of the 6 set
   fields (naddr tokens, naddr 3-grams, digit tokens, core-name tokens,
   translit-name 3-grams, locality tokens) a single binary CSR matrix over ALL
   records (rows = S1 then candidates at offset `n_s1`) is built and cached to
   `experiments/cache/setmat_*.npz`. 3-gram ids use a UTF-32 codepoint trick
   (vectorized, collision-free incl. len-1/2 fallback). Build: 512 s, peak
   9.7 GB, once ever.

3. **Per-chunk pair features = row gathers + sparse ops.**
   - 9 set features: `M[a_rows].multiply(M[b_rows]).getnnz(axis=1)` →
     jaccard/containment/conflict, in 500k-pair slices.
   - `addr_lev`, `name_jw_translit`: `rapidfuzz.process.cpdist(..., workers=N)`.
   - Length/flag/freq features: pure numpy on cached column arrays.
   - `name_squash_cont`: small Python loop (substring test, len>6 guard) —
     negligible cost.
   - `n_cands` from the shard `s1_index` (exact, covers multi-part S1s);
     `src_s3`/`country_india` from the pair columns.

4. **Output schema** (`FEATURES`, 20 float32 columns): the 19 validated model
   features + `cand_addr_ntok`. **The model uses only the 19**
   (`MODEL_FEATURES` in train_final_model.py): cand_addr_ntok measurably hurts
   (smoke macro-F0.5 0.8983 with vs 0.9113 without; PR-AUC 0.8798 vs 0.9192).
   It stays in the parquet for future analysis.

Missing addresses, Indic scripts, empty strings: handled identically to stage 3
(addr_missing flag, translit view only for Indic candidate names, empty-set
features = 0, digit_exact_conflict requires both sides non-empty). A NaN/inf
guard raises on any non-finite feature value (0 hits on 2.32M pairs).

## Benchmark (experiments/feature_pipeline_benchmark.json)

20k validation artifacts (2,315,563 pairs), run CONCURRENT with candidate-gen
shard 0 (~6 of 14 cores busy) — numbers are conservative:

| metric | old per-pair loop | new pipeline |
|---|---|---|
| throughput | 28,756 pairs/s (200k sample) | **119,665 pairs/s** end-to-end; 162,004 steady-state |
| peak RSS | — | 12.5 GB (incl. one-time setmat build; 13.3 GB in featurize with labels) |
| correctness | reference | **max abs diff 0.0 on all 19 features** vs rb_features.parquet |
| NaN/inf | — | 0 |

Full-train projection (~200M pairs): **~30–45 min** featurization on this
machine, vs ~2 h+ for the old loop.

## Runbook — full TRAIN shards (when all 3 finish)

```bash
# 0. one-time (already done on this machine; cached — reruns are no-ops)
.venv312/bin/python scripts/feature_pipeline.py prep --workers 4

# 1. features + labels (resumable per part file; ~30-45 min)
.venv312/bin/python scripts/train_final_model.py featurize \
    --cands experiments/candidates/train_sharded/shard_0 \
            experiments/candidates/train_sharded/shard_1 \
            experiments/candidates/train_sharded/shard_2 \
    --out experiments/train_final --workers 6

# 2. train (entity-level split MD5(s1_id)//1000%100 < 10 => validation)
.venv312/bin/python scripts/train_final_model.py train \
    --out experiments/train_final          # -> models/lgbm_final.txt (+ meta)

# 3. validate (threshold grid, macro-F0.5 w/ misses as FN, singletons, o2o)
.venv312/bin/python scripts/train_final_model.py validate \
    --out experiments/train_final
```

Notes:
- `featurize` accepts 1–3 shard dirs; run it on whatever has finished, rerun
  with all three later — the manifest skips completed parts. `s1_meta.parquet`
  is rebuilt each invocation from all listed shards.
- Shards generated on other machines: copy the whole shard dir (`parts/` +
  `s1_index/` + `manifest.json`).
- Train matrix estimate at ~200M pairs × 19 feats × 4 B ≈ 15 GB float32 —
  fits in 48 GB. If tight, raise `--val-pct` or subsample negatives (not
  currently needed).
- GT (`--gt`) is used ONLY for labels/metrics. TEST data is never touched.

## Integration smoke test (val20k_py312, full path through the scripts)

featurize 2,315,556 pairs at 66k pairs/s (incl. I/O + labels, concurrent with
shard 0) → train 52 s → validate. 19-feature model, MD5 val split (6,008 S1):
macro-F0.5 **0.9113** (plateau 0.60–0.75). Reference control (validated
rb_features + same MD5 split): 0.9138–0.9144 → pipeline within noise of
reference. The stage-3 headline 0.9246 used a different (rng seed-21) split —
macro-F0.5 carries ~±1.5 pp split-composition noise at 6k-entity scale; the
full-train validation (~220k val S1 at val-pct 10) will be far tighter.

## Caches (experiments/cache/, git-ignored)

prep_s1.parquet, prep_cand.parquet, setmat_{naddr_tok,naddr_gram,digits_tok,
core_tok,nname_gram,loc_tok}.npz. Delete only to force a rebuild (prep 230 s +
setmats 512 s). They are TRAIN-input derived; TEST inference will need its own
prep run (same commands, different --s1/--s2/--s3 — not yet wired, deliberate).
