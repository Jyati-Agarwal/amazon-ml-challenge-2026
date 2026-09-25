# BLOCKING IMPLEMENTATION PLAN

Final specification for the candidate-generation stage. Evidence:
`docs/BLOCKING_FINDINGS.md`, `experiments/blocking/blocking_budget_comparison.csv`.
Status: spec locked 2026-09-25; **IMPLEMENTED as scripts/generate_candidates.py and
20k-validated (digit-for-digit: 0.9794 pair / 0.9400 entity)** — see
docs/TRAIN_CANDIDATE_GENERATION.md. Note (D9): the normalization below nominally
"keeps all Unicode scripts", but Python `re` `\w` excludes combining marks, so Indic
matras are stripped in practice; measured recall-neutral and kept as the validated
default (`--matra-norm` preserves them). Execution plan: 3-way S1 sharding
(D10/D11) — `--shard-id/--num-shards`, MD5(entity_id)%N, S2/S3 never split,
TF-IDF fit on full corpus per country so shards ≡ unsharded output; full spec
docs/PARALLEL_SHARDING.md. All new runs use `.venv312/bin/python`.

## Input / output contract

**Input:** one split (`train` or `test`): `sourceN.tsv` files read with
`pd.read_csv(sep="\t", dtype=str, keep_default_na=False, quoting=3)`.

**Output:** `candidates_{split}.parquet` with columns
`s1_id (str), cand_id (str), channels (uint8 bitmask: 1=word, 2=char, 4=exact_name)`,
one row per unique (s1_id, cand_id) pair, plus per-channel rank columns
`word_rank, char_rank (int16, -1 if absent)` for downstream features.
Also derivable: `candidate_pairs.tsv` (submission artifact) by grouping cand_id per s1_id.

## Pipeline

```
S1, S2, S3
  └─ 1. normalize name, address            (vectorized pandas)
  └─ 2. partition by raw country string    (open set — no country list in code)
       per partition:
         3. word TF-IDF retrieval  top-k_word   (name+" "+addr)
         4. char TF-IDF retrieval  top-k_char   (name+" "+addr)
         5. exact normalized-name block (cap 200 per block)
         6. union + dedupe → bitmask + ranks
  └─ 7. concat partitions → parquet
```

## Step definitions

### 1. Normalization (identical for S1 queries and S2/S3 corpus)

```python
def normalize(s):           # applied to business_name and business_address separately
    s = strip_latin_accents(s)   # NFD, drop category-Mn chars with ord < 0x0900, NFC
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)   # punctuation -> space (keeps all Unicode scripts)
    return re.sub(r"\s+", " ", s).strip()
joint = nname + " " + naddr      # retrieval representation; naddr may be ""
```

- **Missing addresses:** empty string; `joint` degrades gracefully to the name (the
  trailing space is harmless). No imputation. Empty-address records are the main recall
  tail — accepted (see FINDINGS §11).
- No transliteration, no suffix stripping, no token reordering (all evaluated, rejected).

### 2. Country partitioning

`groupby(country)` on the raw string. Queries = S1 partition; corpus = concat(S2, S3)
partition. GT never crosses country (measured 1.000000) so this is lossless. A partition
whose country was never seen in train (e.g. France) is processed identically — the
TF-IDF models are **fit per partition on that partition's corpus**, so nothing depends
on training-time vocabulary.

### 3–4. TF-IDF retrieval channels

| Channel | Vectorizer | k (top-k per S1) |
|---|---|---|
| word | `TfidfVectorizer(analyzer="word", ngram_range=(1,1), min_df=3, max_df=0.4, dtype=float32)` on `joint` | from budget config |
| char | `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,4), min_df=3, max_df=0.4, dtype=float32)` on `joint` | from budget config |

- Fit on corpus, transform queries. Cosine top-k via
  `sparse_dot_topn.sp_matmul_topn(Q_chunk, X.T.tocsr(), top_n=k, threshold=0.05,
  sort=True, n_threads=14)`.
- **Budget config** (open-set dict lookup, default = large):

```python
BUDGETS = {"US": (50, 50)}                 # (k_word, k_char) — measured cheap & sufficient
DEFAULT_BUDGET = (100, 100)                # India, France, any unseen country
k_word, k_char = BUDGETS.get(country, DEFAULT_BUDGET)
```

Only `US` earns a reduced budget from measurement; every other label falls through to
the conservative default. This is configuration, not architecture — no country logic.

### 5. Exact normalized-name channel

Hash join `S1.nname == corpus.nname` within the partition. **Cap each block at 200
candidates** (drop blocks larger than 200 — generic names like `physical therapy`
(1,071 records) flood candidates with near-zero precision; top-k channels still cover
those entities). Adds ~11 cands/S1 mean, recall +~0.4 pp over TF-IDF-only.

### 6. Union + dedupe

Concatenate the three channels' (s1_idx, cand_idx) arrays per partition, dedupe with
`np.unique` on the packed int64 `(s1_idx << 32) | cand_idx` (or pandas drop_duplicates),
OR-ing channel bitmasks and keeping min rank per channel. Never materialize per-entity
Python sets at full scale.

### Rejected components (do not add back without new evidence)

Rescue channels R1 exact-address (+1 pair) and R2 rare-token (+0.05 pp for +8.5%
volume); transliteration retrieval; name-only representations; pincode keys (untested
as retrieval but tail analysis shows empty/weak addresses dominate misses).

## Memory & runtime management

- Process one country partition at a time; `del` + `gc.collect()` between partitions.
- Chunk S1 queries at **100k rows** per `sp_matmul_topn` call; append results to the
  parquet file incrementally (pyarrow `ParquetWriter`). Peak RSS stays ≤ ~18 GB
  (largest corpus matrix ~530M nnz float32 + its transpose during conversion).
- Build the corpus TF-IDF matrix once per (partition, channel); build `X.T.tocsr()`
  once and reuse across query chunks.
- char channel is 8–10× word cost (23–26 ms/query vs 2.2–3.1). If wall-clock is
  critical, US char can be dropped for −0.4 pp US recall (saves ~11 h across
  train+test).

## Expected performance (from measurement)

| Metric | Value |
|---|---|
| Pair recall (train, asym budgets) | 0.979 |
| Entity all-match recall | 0.940 |
| Candidates/S1 | mean 116, median 92, p99 254, max ~600 |
| TRAIN volume / storage / runtime | ~257M pairs, 2–4 GB parquet, ~17 h (word 1.6 h + char 15 h) |
| TEST volume / storage / runtime | ~232M pairs, 2–4 GB parquet, ~11 h |
| Known ceiling loss | ~2.1% of true pairs (India-heavy tail: Indic-script names + weak addresses) |

## Validation hooks

- On train, join candidates against GT and report pair/entity recall per country —
  must reproduce ≈0.979/0.940 before the matcher is trained on these candidates.
- `matching_results.tsv` ⊆ `candidate_pairs.tsv` is enforced by construction.
- Sanity: every S1 entity present exactly once in outputs, even with zero candidates.
