# BLOCKING FINDINGS — Candidate Generation Recall Study

Measured 2026-09-25 on train data. Methods and full tables:
`experiments/blocking/blocking_summary.md` and `blocking_results.csv`.
Retrieval numbers come from a 20k stratified S1 sample (seed 42) queried against the
FULL same-country S2+S3 pools (6.19M US / 4.13M India); exact-key numbers are full-train.

## 1. Best exact strategy

Exact keys are weak everywhere: normalized name = **25.8%** recall, normalized address =
8.3%, their union = **32.5%** (entity-level only 5.6%). Adding country changes nothing
(names rarely collide across countries). Exact matching is useful only as a free,
zero-cost component inside a union — it can never be the backbone.

## 2. Best fuzzy/retrieval strategy

**Word-level TF-IDF over `name + address`, cosine top-k per country** is the single best
method: 97.4% recall @ k=100 (92.4% entity-level), 95.5% @ k=25. Char (3,4)-gram TF-IDF
on name+address is close (96.2% @ k=100) and is more typo-robust; char n-grams on name
alone are far worse (72.2% @ k=100) — **the address is the load-bearing field**.
Retrieval cost is low: ~105 s per 12k queries against 6.19M docs (12 threads,
sparse_dot_topn), extrapolating to ~10 h for all 2.2M train S1 across both main configs —
feasible on this machine.

## 3. Transliteration impact

- Naive unidecode on names only: **no effect** (deva-subset recall 0.8→1.6% across k;
  unidecode renders Hindi with doubled letters — `प्राइवेट लिमिटेड` → `praaivett limittedd`
  vs true `private limited` — so n-grams barely overlap).
- Cleaned transliteration (unidecode + collapse repeats) on the name+address field:
  +2.5 pp on the Devanagari subset for char n-grams (80.1→82.5% @ k=100), **≈0** for word
  retrieval (86.2→85.9%).
- Reason: script-mismatch records are already retrievable through their **Latin-script
  addresses** — word name+addr alone gets 86% on Devanagari pairs.
- Script census: 16.9% of India corpus names (~700k records) are non-Latin across **9
  scripts** (Devanagari 10.3%, Telugu/Kannada/Tamil/Gujarati/Bengali ~1.2–1.5% each).

**Verdict: transliteration is NOT worth the complexity for blocking.** Revisit it later
only as a *feature* for the matching model (name similarity on transliterated text),
where the collapsed-unidecode trick is cheap and generic across all 9 scripts.

## 4. Best multi-block combination

- `exact(name+country) ∪ char34(name+addr)@100 ∪ word(name+addr)@50`: **97.5%** recall,
  92.9% entity-level, ~145 candidates/S1 (p99 = 241).
- Budget-doubled union (@200/@100 + extra configs): **98.4%**, 95.4% entity-level,
  ~339 candidates/S1. The last +0.9 pp costs 2.3× the volume.
- Word and char retrieval are highly redundant (union of the two ≈ word alone +0.14 pp at
  matched budget); keep both mainly for typo robustness, at modest k.

## 5. Candidate recall vs candidate volume (word name+addr, per S1)

| k | Recall | Entity recall | Pairs to score on test (1.73M S1) |
|---:|---:|---:|---:|
| 10 | 0.930 | 0.812 | 17M |
| 25 | 0.955 | 0.875 | 43M |
| 50 | 0.966 | 0.902 | 87M |
| 100 | 0.974 | 0.924 | 173M |
| 200 | 0.980 | 0.940 | 346M |
| union @145 | 0.975 | 0.929 | ~250M |

Diminishing returns start at k≈50; k=100 is the knee given F_0.5 is precision-weighted
(recall above ~97% ceiling buys little if the matcher must then reject ~97 of 100
candidates anyway).

## 6. Recommended candidate budget

**k=50 word + k=50 char (name+addr) + exact-name union, per country: ~100–120 candidates
per S1, ≈97.2% pair recall / ≈92% entity recall.** On test that is ~180–210M pairs — heavy
but tractable with vectorized features on 14 cores. If pairwise scoring proves cheap,
raise to k=100+k=100 (~0.5 pp more recall); if slow, k=25 word-only still holds 95.5%.

## 7. Biggest blocking failure modes (from 300 sampled misses of the best union)

1. **Zero name-token overlap** (63% of misses): DBA/renames (`Cinder Bakery LLC` ↔
   `Calomira`), domain-style names (`Sonal Law Chambers` ↔ `lcsonal.com`, 9% of misses),
   severe typo/leet corruption (`At1antic Grlaf`).
2. **Non-Latin name + weak address** (≥24% of misses; Devanagari, Tamil, Telugu, …):
   when the address is short/empty the Latin address can't rescue the record.
3. **Empty S2/S3 address** (32% of misses): match must survive on a corrupted name alone.
4. **Different locality strings** for the same entity (`Islip` vs `Ronkonkkoma`,
   `Boston` vs `Jamaica Plain`) — address helps names but sometimes betrays them.
5. **Giant generic-name blocks** are a precision (not recall) hazard: top corpus name
   blocks are `physical therapy` (1,071), `primary care` (1,070), `pediatric dental`
   (1,004) — exact-name blocking on these floods 1k candidates; top-k retrieval caps this
   automatically. India addresses like `no 1 chennai tn` collide too but only ~35×.

These four classes together account for essentially all of the ~2% unrecoverable pairs.

## Structural facts confirmed

- GT matches **never cross country** (rate 1.000000 over 7.64M pairs) → per-country
  blocking is lossless and cuts the search space ~2.5×.
- S2 vs S3 behave almost identically under every method (Δrecall < 0.5 pp) → one unified
  pipeline for both.
- India recall lags US by ~4–5 pp under every fuzzy method (more scripts, noisier
  addresses) → country-specific k budgets (larger for India) are justified; separate
  *methods* per country are not.

## Recommended stack for the matching stage

Superseded by the k-budget follow-up below — see §8–§11 and
`docs/BLOCKING_IMPLEMENTATION_PLAN.md` for the final spec.

---

# Follow-up study: k-budgets, asymmetric country budgets, rescue channels

Measured 2026-09-25, `scripts/blocking_budget.py` (18 min, peak 13.7 GB). Same 20k
sample / full-pool design; script-mismatch flag widened from Devanagari-only to **all 9
Indic scripts** (4,864 of 69,023 GT pairs = 7.0%). Full table:
`experiments/blocking/blocking_budget_comparison.csv`. Top-200 candidate lists persisted
to `experiments/blocking/cand_sample/{word,char}_{US,India}.parquet` for reuse.

## 8. K-budget comparison (unions include exact normalized name)

| Strategy | Pair recall | US | India | Entity recall | Cand mean | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| w25+c25+e | 0.9682 | 0.9916 | 0.9331 | 0.9113 | 46.7 | 100 | 218 | 537 |
| **w50+c50+e** | 0.9750 | 0.9943 | 0.9461 | 0.9296 | 85.8 | 138 | 254 | 577 |
| w100+c50+e | 0.9794 | 0.9956 | 0.9553 | 0.9409 | 127.7 | 179 | 296 | 611 |
| w50+c100+e | 0.9770 | 0.9955 | 0.9494 | 0.9352 | 130.1 | 179 | 299 | 621 |
| w100+c100+e | 0.9806 | 0.9963 | 0.9571 | 0.9441 | 164.3 | 214 | 329 | 651 |
| w200+c200+e | 0.9848 | 0.9974 | 0.9661 | 0.9560 | 320.1 | 381 | 479 | 811 |
| **asym: US w50c50e / India w100c100e** | **0.9794** | 0.9943 | **0.9571** | 0.9400 | **116.3** | 182 | 254 | 577 |

Word k is the better spend: w100+c50 beats w50+c100 at equal volume. Individual channels:
word@50 = 0.9655, char@50 = 0.9530, exact = 0.2592 (11 cands, nearly free).

## 9. Asymmetric country budgets — justified

The asymmetric stack matches symmetric w100+c100+e India recall exactly (0.9571) and
loses only 0.2 pp US (0.9943 vs 0.9963) while cutting mean candidates 29% (116 vs 164).
Implementation stays open-set: budgets come from a `{country: (k_word, k_char)}` config
with a **default = large budget** for any unseen label (France therefore automatically
gets the conservative large budget). No country names are hardcoded in logic.

## 10. Rescue channels — both rejected

Marginal gain over base union w50+c50+e (67,295 hit / 1,728 missed pairs):

| Channel | Extra true pairs | Extra fully-recovered entities | Extra cands/S1 |
|---|---:|---:|---:|
| R1 exact normalized address | **+1** (+0.001 pp) | +1 | +0.0 |
| R2 rare name token (df≤100) | +32 (+0.046 pp) | +26 | +7.2 (+8.5%) |

R1 is subsumed by the TF-IDF channels (identical addresses score high anyway). R2 buys
0.05 pp for 8.5% more volume — not worth it. **No rescue channel recommended.**

## 11. The recall tail (1,728 missed pairs = 2.50%, categories overlap)

| Category | Share of misses |
|---|---:|
| Zero name-token overlap (renames, DBA, domains, leet typos) | 60% (1,036) |
| Indic-script name (any of 9 scripts) | 48% (837) |
| Empty S2/S3 address | 27% (458) |
| Domain-style name (`lcsonal.com`) | 6.5% (112) |
| Severe typo (0 shared tokens but char-sim > 0.6) | 5% (85) |
| Low address overlap (≤1 shared token, non-empty) | 3% (50) |
| — India share of all misses | **86%** (1,493) |

The archetypal unrecoverable pair is an India record with a native-script or renamed
business name AND a weak/empty address — no cheap textual channel reaches it (R1/R2
confirmed this empirically). Raising India k from 100→200 recovers ~0.9 pp; beyond that
the tail requires per-script name matching, which §3 showed is poor value. **Accept a
~2.1% pair-recall ceiling loss (asym stack) and spend effort on matcher precision.**

## 12. Volume and runtime projection (asym stack, measured per-query costs)

Per-query retrieval cost (12 threads): word ≈ 2.2–3.1 ms, char ≈ 23–26 ms (char is 8–10×
the cost and the clear bottleneck; TF-IDF builds are minutes).

| Scope | Est. pairs | Est. parquet size | Est. retrieval runtime |
|---|---:|---:|---:|
| Full TRAIN (2.207M S1) | ~257M | ~2–4 GB | word ~1.6 h + char ~15 h ≈ **17 h** |
| Full TEST (1.733M S1) | ~232M | ~2–4 GB | word ~0.8 h + char ~10 h ≈ **11 h** |

Cost levers if 28 h of retrieval is too much alongside model work: (a) drop the char
channel for US only (US word+exact ≈ 0.991; char adds +0.4 pp US) — saves ~11 h total;
(b) raise char sp_matmul threshold 0.05→0.15; (c) run train retrieval overnight once and
cache candidates to parquet (they are reused by every downstream experiment).

---

# Production implementation (2026-09-25) — VALIDATED

`scripts/generate_candidates.py` implements this spec and reproduces the asym benchmark
**digit-for-digit** on the 20k validation (pair 0.9794, US 0.9943 / India 0.9571,
entity 0.9400, mean 115.8, p99 254). Normalization audit finding (D9): the study
normalization strips Indic matras (Python `re` `\w` excludes category-Mn marks);
an A/B at 20k showed preserving them is recall-neutral (0.9792/0.9395), so the
validated benchmark norm remains the default. Full report + run/resume manual:
`docs/TRAIN_CANDIDATE_GENERATION.md`. Execution moved to 3-way S1 sharding (D10).
