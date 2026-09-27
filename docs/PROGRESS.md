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
