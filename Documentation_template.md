# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We resolve Source-1 businesses against the 10M-record Source-2/3 corpus with a
two-stage pipeline: per-country sparse TF-IDF blocking (word + character
channels plus exact-name blocks) that keeps ~98% of true pairs while scoring
only ~127 candidates per entity, followed by a 21-feature LightGBM pair
classifier thresholded at 0.625 with a one-to-one assignment step. The final
model reaches **macro-F0.5 0.9348** on a held-out 10% validation split; a key
innovation is a pair of "soft digit" features (cross-token Levenshtein and
prefix containment over house/street numbers) derived from our own error
analysis, which recovered 27% of digit-conflict false negatives at unchanged
precision.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA findings that shaped the design (scripts/profile_data.py,
docs/DATA_PROFILE.md):

- Heavy name/address noise: abbreviations (Rd/Road), token reordering,
  dropped components (no PIN/state), landmark-style Indian addresses, and
  transliterated/Indic-script name variants — so exact keys alone cannot
  block, and string-similarity features must be script-aware.
- Numbers are decisive: house/street/PIN digits are often the only signal
  separating two same-name branches; conversely 1-digit typos and
  truncations ("4804" vs "2804", "3859" vs "385") cause missed matches.
- The metric is precision-heavy (F0.5) and macro-averaged per S1 entity, and
  singletons (no true match) score 1.0 only for an empty prediction — so the
  operating point must strongly prefer precision, and every S1 entity must be
  emitted, empty allowed.
- `country` is an open set: France appears only in the test data, so nothing
  is hard-coded to {US, India}; unseen countries take the default blocking
  budget and the model's country indicator is simply 0.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier
**Core Innovation:** error-analysis-driven soft digit-similarity features
(digit_lev_best, digit_prefix_cont) added to a strong 19-feature baseline,
adopted only after a controlled challenger experiment on an identical split
(+0.26 pp macro-F0.5, precision unchanged); plus fully deterministic,
manifest-resumable sharding that lets 3 machines generate candidates with
results identical to a single-machine run.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** per-country partition; two sparse retrieval
  channels over normalized `name + " " + address` — word TF-IDF (1-gram) and
  char_wb TF-IDF (3-4 grams), each taking top-k neighbours by cosine score
  (sparse_dot_topn, score threshold 0.05) with budgets k_word = k_char = 50
  for US and 100 for all other countries (the default budget also covers
  countries absent from training, e.g. France); union with exact
  normalized-name blocks (capped at 200 candidates per name).
- **Candidate pairs generated:** test 219,804,922 for 1,732,544 S1 entities
  (≈127/entity); train 255,703,931.
- **How you ensured true matches were not lost:** measured pair recall
  against the training ground truth = **0.9799** (US 0.993 / India 0.959)
  at these budgets; budgets, thresholds, and the exact-name channel were
  selected by sweeping configurations against GT recall vs. pair-count cost
  (docs/TRAIN_CANDIDATE_GENERATION.md). S1 is sharded 3 ways by
  MD5(entity_id) with the full S2/S3 corpus and corpus-wide TF-IDF
  statistics in every shard, so sharding changes nothing about the result;
  a merge step verifies shard disjointness, config equality, and duplicate
  freedom before features are computed.

---

## 4. Matching Model

**Features used (21):**
- Name features: token Jaccard, transliterated 3-gram Jaccard,
  transliterated Jaro-Winkler, squashed-name containment, token-count
  difference, S1 name frequency (log), Indic-script indicator.
- Address features: token containment, 3-gram Jaccard, Levenshtein,
  digit-set Jaccard, exact-digit conflict flag, locality overlap,
  street-number match, length difference, address-missing flag.
- Digit-similarity (our error-analysis additions): digit_lev_best (max
  cross-token normalized Levenshtein over house/street digit tokens),
  digit_prefix_cont (proper-prefix containment, min length 2).
- Other: source indicator (S2/S3), country-India indicator, candidate count.

**Model type:** LightGBM binary classifier (500 trees, 63 leaves,
lr 0.07, min_child_samples 60, subsample/colsample 0.9, random_state 0),
trained on 230.1M pairs (2.93% positive) from the 90% train side of a fixed
MD5(s1_id) split.
**Threshold selection method:** F0.5 grid search on the held-out 10%
validation split (plateau 0.600-0.700, chosen 0.625), followed by a
one-to-one constraint: candidates ranked by model score, each S2/S3 entity
awarded to its highest-scoring S1 (deterministic tie-break), which raised
precision further at trivial recall cost.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9348** (final 21-feature model, t=0.625 +
  one-to-one; 220,429-entity validation split). Baseline 19-feature model:
  0.9322. Precision 0.9718, recall 0.8836; US 0.9526, India 0.9084;
  singleton correct-empty rate 89.6%.
- **Common false positives (wrong merges):** same-brand different branches
  where addresses are sparse or landmark-only, and high name-similarity
  pairs whose only conflicting evidence is a weak address signal (the
  one-to-one step removes the largest such cluster: duplicate claims of one
  S2/S3 entity by several S1 records).
- **Common false negatives (missed matches):** true matches with exactly
  conflicting digit tokens (22,610 in validation — typos like 4804/2804 and
  truncations like 3859/385). The two digit-similarity features recovered
  6,179 (27.3%) of these while adding only ~100 net false positives; the
  rest are heavy transliteration/renaming cases below threshold.

---

## 6. Conclusion

A precision-first blocking + LightGBM pipeline with careful, measured
normalization reached 0.9322 macro-F0.5; a single error-analysis-driven
feature addition, validated as a controlled challenger on an identical
split, lifted it to 0.9348 without hurting precision. Key lessons:
budget-swept sparse blocking gets 98% pair recall at ~127
candidates/entity; the F0.5 metric rewards a high threshold plus a
one-to-one constraint; and disciplined experiment hygiene (locked configs,
deterministic splits, resumable manifests) is what made a 3-machine,
overnight final run safe.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains all source under `src/`
(`scripts/` + `utils/`), a `README.md` with the exact end-to-end commands,
and a pinned `requirements.txt` (Python 3.12, LightGBM 4.7.0 — MIT license,
model ≪ 8B parameters; no external data or APIs anywhere). Entry points:
`scripts/generate_candidates.py` + `scripts/merge_candidate_shards.py`
(blocking), `scripts/feature_pipeline.py` and
`scripts/v1_h1_challenger.py featurize --merge` (features),
`scripts/train_final_model.py` + `scripts/v1_h1_challenger.py train`
(model), `scripts/predict_test.py` (inference → `output/matching_results.tsv`
+ `output/candidate_pairs.tsv`), `utils/validate_submission.py` (format
check; the shipped outputs PASS including the full ID-existence check).

### B. Additional Results

- Blocking sweep, normalization audit, and per-country budget analysis:
  docs/TRAIN_CANDIDATE_GENERATION.md, docs/REAL_BLOCKING_MODEL_RESULTS.md.
- Baseline-model threshold grid and validation methodology:
  experiments/train_final (validation summary; best_threshold 0.625).
- Challenger comparison (baseline vs final, identical split):
  +0.26 pp macro-F0.5, ΔP +0.0001, ΔR +0.0066, digit-conflict FN recovery
  27.3%, net new FPs ≈ 100.
- Test-set predictions: 5,561,174 matched pairs across 1,628,143 S1
  entities; 104,401 predicted singletons (6.0%).

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
