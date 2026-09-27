# STRATEGY RESEARCH — Business Entity Resolution (Amazon ML Challenge 2026)

Independent research analysis, 2026-09-25. Read-only session; no code, data, or Git state was
modified. Empirical numbers below come from two sampled probes run against the training data
(5,518 sampled S1 entities → 19,085 true pairs; token DF computed over all 10.3M S2+S3 rows),
plus the existing `docs/DATA_PROFILE.md`.

---

## 0. TL;DR

Pipeline shape is forced by the data: **per-country TF-IDF/char-n-gram candidate generation
(top-k cosine) → LightGBM pairwise classifier on cheap string features → calibrated
per-entity expected-F0.5 decision rule with one-to-one enforcement from the S2/S3 side.**
The three things that will actually move the leaderboard: (1) candidate-gen recall,
(2) Indic-script transliteration for India (≈25% of India true pairs have zero Latin token
overlap; India is 47% of test), (3) the precision-side decision policy (F0.5 macro with
singletons worth 1.0 rewards abstention far more than teams usually expect).

---

## 1. Problem formulation

- Bipartite, directed matching: each of 1.73M test S1 entities gets a (possibly empty) set of
  matches from S2∪S3 (~10M records). Avg 3.46 true matches, max 11; 5.6% singletons.
- **Clean partition (measured on all 7.64M GT pairs): every S2/S3 record matches at most one
  S1 entity.** This converts the problem into an assignment problem from the S2/S3 side and is
  directly exploitable at inference (see §10).
- ~26% of S2/S3 rows match nothing — deliberate distractors; precision traps.
- Metric: macro-averaged F0.5 per S1 entity. A correct empty prediction scores 1.0; any
  prediction on a true singleton scores 0.0. Precision is weighted 2× recall.
- Formulate as: candidate generation (recall stage) + pairwise binary classification
  (precision stage) + per-entity set-selection (decision stage). Do NOT formulate as
  clustering — the bipartite + clean-partition structure makes clustering machinery
  (transitive closure, correlation clustering) unnecessary.

## 2. Name normalization

Measured match tiers on true pairs (cumulative):

| Tier | Coverage of true pairs |
|---|---:|
| Exact string equality | 4.8% |
| Lowercase + strip accents/punct | 26.2% |
| + drop legal suffixes (Inc/Corp/Pvt/Ltd/LLC/SARL/…), sorted-token equality | 47.8% |
| Core-token Jaccard ≥ 0.5 | 74.6% |
| Zero token overlap (the hard tail) | 14.6% (US 8.2%, India 24.6%) |

Normalization pipeline that pays for itself: NFKD + casefold + accent strip; punctuation → space;
`&`→`and`; strip junk prefixes (`--`, `<<`); **legal-suffix dictionary across all three
countries** (Inc, Corp, LLC, Ltd, Pvt, Private, Limited, LLP, SARL, S.A.S, SCI, SA, plus
Devanagari प्राइवेट लिमिटेड and equivalents) — strip as tokens but keep as a categorical feature
(suffix mismatch is weak negative evidence, not disqualifying).
Special forms found in the zero-overlap tail, all recoverable:
- **Domain names**: `maurewilliamscolombier.com` ↔ `Maure Williams Colombier Inc`. Fix: strip
  TLD, compare against space-squashed S1 name (exact/substring). Cheap and high precision.
- **Indic scripts**: not only Devanagari — Kannada observed (`ಸ್ಟಾರ್ ಟೆಕ್ನಾಲಜೀಸ್` ↔ `Star
  Technologies`). S2 India ~13% non-Latin names, S3 ~7.6%; S1 is 100% ASCII. Requires
  offline transliteration (see §16 experiment E2).
- **Acronyms** (`CM` ↔ `Casuarina Mart`): initial-letter feature; low volume, do last.

## 3. Address normalization

- Token Jaccard ≥ 0.3 on lightly normalized addresses already holds for **87.4%** of true
  pairs — address is the main rescue signal when names diverge. 4.2% of true pairs have an
  empty S2/S3 address (name must carry those alone).
- Worthwhile normalizations (dictionary-based, hours not days): St/Street, Rd/Road, Ave/Avenue;
  US state abbrev ↔ full name (`TX`↔`Texas` — confirmed cross-source inconsistency); India
  state abbrevs (`HR`↔Haryana, `DL`↔Delhi); lowercase+accent-strip (France).
- Do NOT attempt full address parsing (libpostal-style component extraction). Component
  reordering (`IA, Iowa City, 1064 Newton Rd`) is handled for free by bag-of-tokens /
  char-n-gram similarity. Geocoding is explicitly banned.
- Numeric tokens (house numbers, PIN codes, khasra numbers) are the most discriminative
  address atoms — extract digit-token overlap as a dedicated feature.

## 4–5. Blocking & candidate generation (the recall ceiling)

All-pairs is 1.7×10¹³; even country-blocked it's 3.8×10¹². Need ~20–100 candidates/S1.

Measured recall of single **rare-token blocking** (share ≥1 token with DF ≤ cap, name or
address, + domain-squash key), on true pairs:

| DF cap | name-tok | addr-tok | either | +domain key | median cand/S1 (bound) | p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 27.0% | 40.6% | 56.4% | 84.0% | 31 | 254 |
| 1000 | 41.8% | 65.8% | 79.8% | 92.6% | 420 | 1,556 |
| 5000 | 57.0% | 85.7% | 94.1% | 97.6% | 3,662 | 9,814 |

**Critical finding:** the misses are not noisy pairs — they are *identical or near-identical
records whose shared tokens are all individually common* (e.g. `Gurgaon Services Private
Limited` at the exact same address, missed because gurgaon/services/vipul/sohna all have
DF > 1000). Single-token keys are structurally wrong; co-occurrence of two common tokens is
rare even when each token is common.

**Recommended candidate generation** (union, per country block):
1. **TF-IDF cosine top-k** (k≈30–50) via `sklearn TfidfVectorizer` + `sparse_dot_topn`, on
   word 1-grams + char 3–5-grams of `name + ' ' + address`. Char n-grams simultaneously handle
   typos, word order, abbreviations, and squashed domain names. This is the workhorse.
2. **Domain-squash exact key** (space-removed, TLD-stripped name equality/containment).
3. **Transliterated-name TF-IDF** for non-Latin S2/S3 records (after §16 E2).
4. Optionally keep rare-token (DF ≤ ~200) inverted index as a cheap booster — it's nearly free.

Run S2 and S3 separately (same code). Target: **≥97% pair recall at ≤60 candidates/S1**,
measured on held-out GT. Blocking recall directly caps the recall term of F0.5; at β=0.5 a
97% ceiling costs little, a 90% ceiling costs a lot of entities their 4th match.
`candidate_pairs.tsv` requirement means this stage's output must be persisted anyway.

## 6. Pairwise features (aim for ~15–25, all cheap)

Name: exact/normalized/core-token equality flags; token Jaccard & containment (handles
subset names); Jaro-Winkler; Levenshtein ratio; char-3-gram cosine; squashed-name
containment; same-legal-suffix flag; acronym match flag; token count difference.
Address: token Jaccard (raw + abbrev-normalized); char-n-gram cosine; **digit-token overlap**
(house/PIN); city/state token overlap (last-k tokens heuristic); either-address-empty flag.
Context: country (categorical, open-set — hash or leave-as-string in LightGBM);
source flag (S2 vs S3); S1-name-frequency (is this a chain name? — count of identical
normalized names within S1); candidate rank & cosine score from blocking; number of
candidates for this S1 (crowdedness).
Use `rapidfuzz` (C++-backed) — pure-Python edit distance at 10⁸ pairs will not finish.

## 7. Model choice

**LightGBM binary classifier** on candidate pairs. Rationale: 7.6M positives + blocking
negatives ≈ 50–80M available labeled pairs (sample to ~10–20M); GBMs dominate tabular
string-feature ER, train in minutes, and produce scores that calibrate well (isotonic/Platt on
a held-out fold — calibration matters because the decision rule in §8 consumes probabilities).
Logistic regression is the 2-hour fallback and a useful sanity baseline; expect GBM to beat it
on interaction effects (e.g. "name weak but address digit-match strong"). XGBoost/CatBoost:
no advantage here worth the switching cost. Train on hard negatives (blocking candidates that
are not GT matches) — never on random negatives.

## 8. Threshold optimization / decision stage

Do not use a single global threshold. With calibrated per-candidate probabilities p₁≥p₂≥…
for an entity, expected macro-F0.5 is maximized by evaluating expected F0.5 of each prefix
set {top-1, top-2, …, ∅} under independence and picking the best — O(k) per entity, closed
form. This natively handles:
- **Singletons**: predicting ∅ is optimal unless top p is high — with F0.5, roughly p > ~0.4–0.5
  needed before a single guess beats abstention on expectation (tune empirically, don't trust
  the closed form blindly — validate on held-out macro-F0.5).
- **Multiple matches**: add the 4th candidate only if its p clears the marginal-benefit bar.
Tune any residual free parameter (calibration temperature, per-country offset) directly
against held-out macro-F0.5 via grid search — it's cheap.

## 9. Singleton handling

5.6% of entities, each worth a full 1.0. The expected-F0.5 rule (§8) handles them if
probabilities are honest. Additional guard: distractor rate is ~26% of S2/S3, so the model
sees realistic negatives in training — do not artificially balance classes; keep natural
candidate-set class ratio so probabilities stay meaningful.

## 10. Multiple-match & one-to-one handling

- Per-entity independent selection first (§8), then enforce the **clean-partition constraint**:
  if one S2/S3 record is selected by multiple S1 entities, keep it for the highest-probability
  S1 and drop elsewhere (greedy by score; full Hungarian is overkill at this sparsity).
  This is a pure precision gain — measured on GT, the constraint holds exactly.
- Exact duplicate rows in S2/S3 (distinct IDs, identical content, ~20–26K per file) can BOTH
  match the same S1 — do not dedupe candidates by content.

## 11–12. Country behavior & S2 vs S3

- **Train US 60/India 40; test US 38/India 47/France 15. France has zero training rows.**
  Keep everything country-generic: per-country TF-IDF vocabularies (fit on test-country data —
  TF-IDF is unsupervised, so France gets its own vocabulary for free), no country one-hots
  that can't absorb a new label, suffix dictionaries that already include SARL/SAS/SCI.
- Proxy test for France: train the classifier on US-only, evaluate on India (and vice versa).
  If the drop is small, features are country-robust and France risk is low; if large, remove
  country-anchored features. French names are Latin-script with accents — closer to US
  difficulty than India (no transliteration problem).
- S2 vs S3 differences are moderate (S2: 13.4% vs 7.6% Devanagari; S3 truncates addresses
  more; S3 allows 6 matches vs S2's 5). One model with a source flag; no separate pipelines.

## 13. Validation design

- Split **by S1 entity** (never by pair — leaks the entity), e.g. 80/20, stratified by
  country and match-count bucket. Freeze the 20% until final threshold tuning.
- Implement the exact macro-F0.5 scorer (singletons included) on day one; every experiment
  reports it. Track separately: blocking pair-recall, classifier PR-AUC, end-to-end F0.5.
- Two extra eval slices: US-only↔India-only cross-training (France proxy, §12), and the
  public leaderboard used sparingly as a distribution-shift check, not a tuning signal
  (private LB decides).

## 14. Error analysis strategy

Bucket every held-out error as: (a) blocking miss (true match not in candidates) — fix in
candidate gen, not the model; (b) scored-but-rejected (threshold/calibration) ; (c) false
positive. Slice all three by country × script × name-sim tier × match-count. Manually read
50 FPs and 50 FNs after each major change — with ~30% duplicate business names in S1
(chains/franchises), expect FPs to concentrate on same-name-different-address; verify the
digit-token address features are firing there.

## 15. Computational constraints (48 GB RAM, 14-core Apple Silicon, no GPU assumed)

- TF-IDF over 10M strings: sparse, fits easily. `sparse_dot_topn` top-k over
  (1.7M × ~5M) per country block: chunked, ~tens of minutes per block, embarrassingly parallel.
- Feature computation at ~10⁸ candidate pairs × 20 features with rapidfuzz +
  multiprocessing: the single biggest wall-clock item, budget 1–3 hours per full run —
  cache candidate pairs to disk (parquet) so re-featurization doesn't repeat blocking.
- LightGBM on 10–20M pairs: minutes. Everything fits the 48-hour budget with ~3–5 full
  end-to-end iterations, which is exactly why heavier models are excluded (§C).
- venv currently has only pandas/numpy — install scikit-learn, sparse_dot_topn (or scipy
  fallback), rapidfuzz, lightgbm, and an offline Indic transliteration package early; these
  are code libraries, not external data lookup (README bans data/APIs; the ≤8B
  MIT/Apache model clause explicitly contemplates pretrained models — still, flag
  transliteration libs in the methodology doc for transparency).

---

## A. Three strongest baselines (build in this order)

1. **Deterministic high-precision matcher** (~2h): normalized core-token name equality AND
  address token Jaccard ≥ 0.3 (plus domain-squash key), same country. Covers ~48% of true
  pairs at near-perfect precision; establishes the scoring harness and a real LB number.
2. **TF-IDF cosine end-to-end** (~half day): candidate gen as in §4 + accept candidates above
  a tuned cosine threshold (name and address cosines combined linearly). No learning yet;
  gets word-order, typo, and abbreviation cases.
3. **LightGBM pairwise classifier** (day 1–2): features §6 on candidates §4, decision rule §8,
  one-to-one enforcement §10. This is the expected final system skeleton.

## B. Three high-value experiments

1. **Expected-F0.5 decision rule vs global threshold** (§8) — cheap to run, historically worth
   several points on macro precision-heavy metrics, directly exploits singleton scoring.
2. **Indic transliteration for S2/S3 names** — transliterate non-Latin names to Latin, re-run
   blocking + features. Targets the 24.6% zero-overlap tail in India (47% of test S1). Even a
   crude phoneme-map transliteration should recover most, since GT transliterations are literal.
3. **Blocking recall ablation** — measure held-out pair recall and candidate counts for each
   generator in the §4 union; tune k and n-gram ranges. Recall ceiling is the hard bound on
   the final score, and `candidate_pairs.tsv` is audited.

## C. Three approaches to NOT spend time on

1. **Transformer cross-encoders / fine-tuned BERT pair scorers** — 10⁷–10⁸ pair inference on
   CPU will not finish; marginal gain over GBM on short noisy strings is small.
2. **Graph/collective ER (clustering, transitive closure, GNNs)** — the clean-partition
   bipartite structure means greedy assignment captures all the global signal that exists.
3. **Deep address parsing / trained sequence taggers for address components** (and anything
   geocoding-shaped, which is banned) — bag-of-tokens + digit-overlap gets ~all of the value.
   Also skip generic LLM embeddings for all 12M records: embedding + ANN infra cost eats the
   48h for an unproven gain over char-n-gram TF-IDF on this data.

## D. Biggest likely source of leaderboard performance

**Candidate-generation recall combined with a calibrated precision-side decision policy.**
Blocking recall caps the score; after that, because F0.5 double-weights precision and
singletons pay 1.0 for abstention, the win comes from *knowing when not to predict* —
calibration + expected-F0.5 selection + one-to-one enforcement. India transliteration is the
single biggest recall unlock (≈25% of India pairs are otherwise unreachable by any Latin
string similarity).

## E. Biggest likely source of false positives

**Same-name, different-branch businesses**: ~30% of S1 names are non-unique
(chains/franchises), and ~26% of S2/S3 records are deliberate no-match distractors. A
name-dominated model will merge branches. Mitigations: digit-token address features,
S1-name-frequency ("chain-ness") feature, one-to-one assignment, and the abstention-friendly
decision rule. Secondary FP source: near-identical names that are genuinely different entities
(`Fiify` vs `FRYF`-style noise makes low-edit-distance ≠ same entity) — let the model learn
from hard blocking negatives rather than hand-tuning edit-distance cutoffs.

---
*Probes were sampled (every 400th GT row); percentages carry ~±1% sampling error. Raw probe
code was run inline and not saved, per research-session constraints.*
