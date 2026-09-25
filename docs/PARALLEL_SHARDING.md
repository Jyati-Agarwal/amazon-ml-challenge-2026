# PARALLEL SHARDING — 3-way S1 split for candidate generation (D11)

Execution plan for the full TRAIN candidate run: three team members each run
`scripts/generate_candidates.py` on one third of S1, in parallel, on their own
machines. Replaces the single-machine ~17 h unsharded run (intentionally stopped,
see D10). Status: implemented + validated on a 2,000-S1 controlled test
(see §Validation). **The full TRAIN run has NOT been started.**

## Design

- **Only S1 is sharded.** S2/S3 are never split: every shard searches the FULL
  same-country S2+S3 corpus, so recall per S1 is identical to the unsharded run.
- **TF-IDF consistency:** the word and char vectorizers are fit per country on the
  full country corpus only (queries are merely `transform`ed), so all shards learn
  the *same* vocabulary/IDF model. Sharding changes only which S1 rows are queried.
  Therefore `shard_0 ∪ shard_1 ∪ shard_2` = the unsharded output on the same S1
  rows, up to row order and file layout.
- **Candidate logic unchanged:** budgets US=(word 50, char 50), all other
  countries=(100, 100), plus exact normalized-name capped at 200; no pincode /
  rare-token / exact-address rescue channels. Same normalization (D9 default).

## Shard assignment (deterministic, machine-independent)

```python
shard = int(hashlib.md5(entity_id.encode("utf-8")).hexdigest(), 16) % num_shards
```

Implemented as `shard_of()` in `scripts/generate_candidates.py`. MD5 of the UTF-8
entity_id string → 128-bit integer → modulo `num_shards`. Python's built-in
`hash()` is NOT used (it is salted per process). The same entity_id maps to the
same shard on every run and every machine.

Measured shard sizes (num_shards=3):

| Split | shard 0 | shard 1 | shard 2 | total |
|---|---:|---:|---:|---:|
| train S1 (+ GT, same ids) | 735,761 | 736,288 | 734,772 | 2,206,821 |
| test S1 | 577,575 | 577,462 | 577,507 | 1,732,544 |

Verified: every id in exactly one shard, zero overlap, union == original set,
re-running the assignment is bit-identical. Spot value: `S1-965667` → md5
`f9176d73…` → shard 0.

## CLI

```
--shard-id I --num-shards N     # both or neither; 0 <= I < N
```

The shard filter is applied after `--sample` / `--s1-ids` (all subset modes
compose). `shard_id`/`num_shards` are recorded in the output `manifest.json`
config, so resuming with different shard flags in the same `--out` is rejected.

## Environment

All NEW runs use **`.venv312/bin/python`** (Python 3.12.14, Session-3 env; pins in
`requirements.txt`). Do not use the old `.venv` (3.9) for production runs. The 20k
validation was re-run under `.venv312` to confirm recall vs the 3.9 benchmark —
results in §Validation.

## Three-member commands (full TRAIN — do not start without go-ahead)

```bash
# Member 1
.venv312/bin/python scripts/generate_candidates.py \
    --shard-id 0 --num-shards 3 \
    --out experiments/candidates/train_sharded/shard_0

# Member 2
.venv312/bin/python scripts/generate_candidates.py \
    --shard-id 1 --num-shards 3 \
    --out experiments/candidates/train_sharded/shard_1

# Member 3
.venv312/bin/python scripts/generate_candidates.py \
    --shard-id 2 --num-shards 3 \
    --out experiments/candidates/train_sharded/shard_2
```

All three use the same repo state, the same default budgets/flags, and the same
full S2/S3 TSVs; only `--shard-id` and `--out` differ. Never point two workers at
the same `--out`. Expected per-shard wall clock ≈ 1/3 of the unsharded ~17 h
projection ≈ **~6 h** (TF-IDF fits are repeated per machine, so slightly more than
a perfect third). Each run is resumable at (country, chunk) granularity: rerun the
identical command after an interruption and completed chunks are skipped.

For TEST later (do NOT run yet): identical commands with
`--s1 dataset/test/test_source1.tsv --s2 dataset/test/test_source2.tsv
--s3 dataset/test/test_source3.tsv` and a fresh `--out`.

## Output layout

```
experiments/candidates/train_sharded/
  shard_0/ {manifest.json, parts/cand_<country>_<chunk>.parquet,
            s1_index/s1_<country>_<chunk>.parquet}
  shard_1/ ...
  shard_2/ ...
  gt/      train_ground_truth.shard_{0,1,2}.tsv   (eval convenience only)
  merged/  (created by the merge utility)
```

Parquet schema is unchanged: `s1_id, cand_id, source (uint8 2|3), country,
channels (uint8 bitmask 1=word|2=char|4=exact), word_rank, char_rank (int16, -1
absent)`. `s1_index` covers zero-candidate S1s. Everything under
`experiments/candidates/` is git-ignored — never commit candidate data.

## Merging

```bash
.venv312/bin/python scripts/merge_candidate_shards.py \
    --shards experiments/candidates/train_sharded/shard_0 \
             experiments/candidates/train_sharded/shard_1 \
             experiments/candidates/train_sharded/shard_2 \
    --out experiments/candidates/train_sharded/merged
```

Streams one part file at a time (never loads the full dataset). Checks: all shard
manifests `state=done` with identical configs (except `shard_id`); shard S1 sets
pairwise disjoint (from the small `s1_index` files); no duplicate
`(s1_id, source, cand_id)` inside any part. Disjoint S1 sets + per-chunk dedupe
make cross-shard duplicate pairs impossible. Parts are copied verbatim
(schema preserved) with a `shardN_` prefix; `merge_manifest.json` records totals.

## GT sharding (evaluation convenience)

```bash
.venv312/bin/python scripts/shard_ground_truth.py \
    --input dataset/train/train_ground_truth.tsv \
    --id-col source1_entity_id --num-shards 3 \
    --out-dir experiments/candidates/train_sharded/gt
```

Uses the same `shard_of()`; verified GT_shard0 ∪ GT_shard1 ∪ GT_shard2 == original
GT (2,206,821 rows, no missing/duplicated S1). The candidate generator itself
remains GT-free.

## Validation (controlled test, 2026-09-25, `.venv312`) — ALL PASS

1. **Sharded ≡ unsharded.** Same 2,000-S1 sample (`US:1200,India:800`, seed 42,
   chunk 1000) generated once unsharded and once as shard_0(656)+shard_1(660)+
   shard_2(684). Merged via merge_candidate_shards.py (S1 disjointness OK, no dup
   pairs). Row-for-row comparison sorted on (s1_id, source, cand_id): **all
   234,838 rows identical in every column** (channels bitmask + word/char ranks
   included). Every shard run searched the FULL corpora (India 4,133,346 /
   US 6,186,873) with correct budgets (US 50/50, others 100/100).
2. **Recall identical.** Unsharded and merged both: pair recall 0.9785
   (US 0.9940 / India 0.9551), entity all-match 0.9380, cand mean 117.4 / p99 256,
   0 dup pairs, all correctness checks pass — in line with the 20k benchmark
   (0.9794/0.9400) given 2k-sample noise. Reports:
   `experiments/candidates/shardtest/{unsharded,merged}_report.json`.
3. **`.venv312` ≡ `.venv` (3.9).** The 20k validation re-run under `.venv312`
   (`experiments/candidates/val20k_py312/`, 16.5 min) reproduces the benchmark
   digit-for-digit — pair 0.9794 (US 0.9943 / India 0.9571), entity 0.9400,
   mean 115.8, p99 254, 2,315,556 pairs — and is **byte-identical row-for-row**
   to the 3.9 run (val20k_legacy) in all columns. Environment switch is safe.

Runtimes on this machine (M4 Pro): unsharded 2k-sample 10.0 min; per-shard runs
6.7–9.7 min (TF-IDF fits dominate at small query counts); peak RSS ≤ 17.2 GB.
