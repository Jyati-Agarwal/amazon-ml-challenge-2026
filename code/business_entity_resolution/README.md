# Business Entity Resolution — reproduction guide

Pipeline: per-country TF-IDF blocking → LightGBM pair classifier →
threshold 0.625 → one-to-one assignment → submission TSVs.

Everything below runs from `src/` with the challenge data unpacked at
`dataset/train/` and `dataset/test/` (same layout as the challenge bundle).
Environment: Python 3.12, `pip install -r ../requirements.txt`
(macOS additionally needs `brew install libomp` for LightGBM).
Hardware used: Apple M4 Pro, 14 cores, 48 GB RAM; peak RSS of any stage
≤ 22 GB; total disk for intermediates ~25 GB.

All long stages write a `manifest.json` and are resumable by rerunning the
identical command.

## 1. Candidate generation (blocking) — train and test

S1 is split into 3 disjoint shards by `MD5(entity_id) % 3` so shards can run
on separate machines; S2/S3 are always the full corpus, so results are
identical to an unsharded run. Locked configuration (all defaults): word
TF-IDF (1-gram) + char_wb(3,4) over normalized `name + " " + address`,
top-k per channel with score threshold 0.05, budgets US (50, 50) and
default (100, 100) — the default also covers countries never seen in
training (test adds France) — plus exact normalized-name blocks capped at
200 candidates.

```bash
for S in 0 1 2; do
python scripts/generate_candidates.py \
    --s1 dataset/test/test_source1.tsv \
    --s2 dataset/test/test_source2.tsv \
    --s3 dataset/test/test_source3.tsv \
    --shard-id $S --num-shards 3 \
    --out experiments/candidates/test_sharded/shard_$S
done   # ~3.7 h and ~17 GB RAM per shard (sequential shown; shards may run on 3 machines)

python scripts/merge_candidate_shards.py \
    --shards experiments/candidates/test_sharded/shard_0 \
             experiments/candidates/test_sharded/shard_1 \
             experiments/candidates/test_sharded/shard_2 \
    --out experiments/candidates/test_sharded/merged
```

For training, run the same three commands with the train TSVs
(`--s1 dataset/train/train_source1.tsv` etc.) and
`--out experiments/candidates/train_sharded/shard_$S`.
Train: 255,703,931 pairs; test: 219,804,922 pairs.

## 2. Features

Prep caches are corpus-specific — always give train and test separate
`--cache-dir`s:

```bash
# TEST
python scripts/feature_pipeline.py prep \
    --s1 dataset/test/test_source1.tsv --s2 dataset/test/test_source2.tsv \
    --s3 dataset/test/test_source3.tsv --workers 4 \
    --cache-dir experiments/cache_test
python scripts/feature_pipeline.py features \
    --pairs experiments/candidates/test_sharded/merged \
    --out experiments/test_features --workers 6 \
    --cache-dir experiments/cache_test          # ~8 min after cache build

# add the 2 digit-similarity features and write full model-ready parquets
python scripts/v1_h1_challenger.py featurize \
    --features-dir experiments/test_features \
    --cache-dir experiments/cache_test \
    --out experiments/test_features_h1 --merge  # ~6 min
```

## 3. Model training (reproduces models/lgbm_v1_h1.txt deterministically)

```bash
# TRAIN candidates + features first (steps 1–2 with train data,
# --cache-dir experiments/cache), then:
python scripts/train_final_model.py featurize \
    --cands experiments/candidates/train_sharded/shard_0 \
            experiments/candidates/train_sharded/shard_1 \
            experiments/candidates/train_sharded/shard_2 \
    --out experiments/train_final --workers 6
python scripts/train_final_model.py train      # V0 baseline (19 features)
python scripts/train_final_model.py validate
python scripts/v1_h1_challenger.py featurize   # H1 columns for train pairs
python scripts/v1_h1_challenger.py train       # final model (21 features)
python scripts/v1_h1_challenger.py validate
```

LightGBM 500 trees / 63 leaves / lr 0.07 / `random_state=0`; validation is a
fixed 10% split by `MD5(s1_id)`. Selected decision rule: threshold 0.625 +
one-to-one by model score (validation macro-F0.5 0.9348).

## 4. Inference → submission files

```bash
python scripts/predict_test.py \
    --features experiments/test_features_h1 \
    --cands experiments/candidates/test_sharded/merged \
    --model models/lgbm_v1_h1.txt \
    --out output                                 # ~11 min
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test                      # must print PASS
```

`output/matching_results.tsv` = final matches (one row per test S1, empty
allowed); `output/candidate_pairs.tsv` = the complete blocking candidate set
scored by the model. Threshold defaults to the locked 0.625.

`scripts/test_predict_synthetic.py` is a self-contained synthetic test of the
inference/packaging path (23 checks, no challenge data needed).

## Source map

- `scripts/generate_candidates.py`, `merge_candidate_shards.py` — blocking
- `scripts/feature_pipeline.py` — normalization, prep caches, 20 base features
- `scripts/train_final_model.py` — V0 training/validation (19 features)
- `scripts/v1_h1_challenger.py` — 2 digit features + final 21-feature model
- `scripts/predict_test.py` — inference, threshold, one-to-one, TSV writers
- `utils/validate_submission.py` — official format validator
- remaining `scripts/*.py` — data profiling, blocking research, error
  analysis, and environment benchmarks that motivated the locked
  configuration (not needed for reproduction)

No external data, APIs, or lookups are used anywhere; only the provided
challenge TSVs. LightGBM is MIT-licensed and the model has far fewer than
8B parameters (3.5 MB text dump).
