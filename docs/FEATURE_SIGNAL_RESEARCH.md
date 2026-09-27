# FEATURE SIGNAL RESEARCH — Pairwise Features for Entity Matching

Session 2 (feature-signal research), 2026-09-25. Reproducible via
`scripts/feature_signal_research.py` (stages: `sample`, `features`, `analyze`);
per-feature stats in `experiments/feature_signal_results.csv`. Dataset read-only;
intermediates cached in `experiments/cache/*.parquet` (git-ignored). No external
data or internet enrichment used; libraries (rapidfuzz, scikit-learn, unidecode,
pyarrow) installed into the project `.venv` only.

---

## 0. TL;DR

- **Address similarity is the strongest single signal family** (AUC ≈ 0.94), beating every
  name feature (best ≈ 0.86). Address rescues **99%** of positives whose names don't overlap.
- **Name × address interaction is the whole game**: P(match) = 83% when both are high,
  ~2% when only name is high, ~9% when only address is high, ~0.06% when neither.
- **Name equality alone is dangerous** (9–15% precision vs hard negatives — chains and
  engineered near-twins). **Pincode is a dud** (present in ~12% of US and ~0% of India
  addresses). **House-number digits are the tiebreaker** for the hardest cases.
- **Transliteration (unidecode) on Indic-script names: AUC 0.53 → 0.89–0.96** on that slice.
  Must-have for India.
- **One-to-one verified exactly**: 0 of 7,638,365 matched S2/S3 IDs map to more than one S1.
  A hard one-to-one assignment constraint at inference is fully supported.

---

## 1. Sample construction

- **4,000 randomly sampled S1 entities** (seed 7) from the 2.2M train ground-truth rows:
  13,815 true matches, 244 singletons.
- **Positives**: every GT match of a sampled S1 (all included by ID, so positive feature
  stats are not biased toward "easy to find" pairs).
- **Hard negatives**: candidates from a full scan of all 10.3M S2+S3 rows via inverted
  indexes on the sampled S1s' name tokens (DF ≤ 3,000), address tokens (DF ≤ 2,000), and
  squashed-name keys; a pair was kept when it shared ≥ 2 indexed tokens or ≥ 1 rare token
  (DF ≤ 300), capped at 150 candidates per S1 by overlap count. This naturally yields
  exactly the requested hard-negative types: same/similar name at a different address
  (chains), shared address tokens with a different business, and high lexical similarity
  non-matches. Labels checked against the S1's full GT set, so multi-matches are never
  mislabeled as negatives.
- **Result: 357,108 pairs — 13,815 positives (base rate 3.9%), ~343K hard negatives.**
  The candidate generator alone recovers 79.8% of positives (a blocking-recall data point:
  token blocking at these DF caps is not sufficient on its own — consistent with Session 1).
- Caveat: precision numbers below are *versus hard negatives*, i.e. pessimistic relative to
  the full candidate population; AUC comparisons between features are the reliable currency.

## 2. Feature definitions (34 features)

Normalization: NFKD + casefold + accent-strip + punctuation→space; `core tokens` also drop
legal suffixes (Inc/Corp/LLC/Ltd/Pvt/Private/Limited/LLP/SARL/SAS/…); `squash` = name with
all non-alphanumerics removed (domain-name trick); transliterated variants apply `unidecode`
to Indic-script candidate names before comparing. Address digit tokens = purely numeric
tokens; pincode = last 6-digit (India) / 5-digit (US) token; locality = last 3 alphabetic
address tokens. Full definitions in `scripts/feature_signal_research.py`.

## 3. True-match vs hard-negative separation (single-feature AUC)

| Feature | AUC | pos mean | neg mean |
|---|---:|---:|---:|
| **name_addr_mean** (avg of name & addr Jaccard) | **0.958** | 0.65 | 0.15 |
| addr_tfidf_cos | 0.941 | 0.75 | 0.17 |
| addr_tok_cont | 0.941 | 0.77 | 0.19 |
| addr_tok_jac | 0.940 | 0.61 | 0.11 |
| addr_3gram_jac | 0.933 | 0.61 | 0.11 |
| name_addr_prod | 0.892 | 0.40 | 0.01 |
| **name_3gram_jac_translit** | **0.863** | 0.68 | 0.19 |
| name_jw_translit | 0.853 | 0.91 | 0.66 |
| addr_lev | 0.847 | 0.64 | 0.29 |
| name_3gram_jac | 0.833 | 0.67 | 0.19 |
| digit_jac | 0.822 | 0.67 | 0.10 |
| name_tok_jac | 0.815 | 0.68 | 0.20 |
| locality_overlap | 0.809 | 0.42 | 0.11 |
| street_num_match | 0.801 | 0.70 | 0.11 |
| name_tfidf_cos | 0.790 | 0.73 | 0.27 |
| name_squash_cont | 0.721 | 0.56 | 0.12 |
| name_core_eq | 0.693 | 0.50 | 0.11 |
| name_exact_norm | 0.579 | 0.26 | 0.10 |
| addr_exact_norm | 0.540 | 0.08 | 0.00 |
| pin_eq / pin_prefix3 / pin_present_both | 0.51 | — | — |
| country_eq / addr_missing / indic_script / name_acronym | ≤ 0.52 | — | — |
| name_ntok_diff / addr_len_diff / name_len_diff | 0.30–0.38 (inverse) | — | — |

(Full table with precision/coverage at high cutoffs: `experiments/feature_signal_results.csv`.)

**Q1 — strongest individual separators:** address token containment/Jaccard/TF-IDF
(interchangeable, AUC 0.94), transliteration-aware name char-3-gram Jaccard (0.86), address
char/edit similarity (0.85), digit-token overlap (0.82), name token Jaccard (0.82).

## 4. Redundant features (Q2)

Spearman ρ ≥ 0.85 groups — keep one per group:
- **{addr_tok_cont, addr_tok_jac, addr_tfidf_cos, addr_3gram_jac}** (ρ 0.90–0.99) → keep
  addr_tok_cont + addr_3gram_jac (the char version adds typo robustness; TF-IDF adds nothing
  once containment is in).
- **{name_tok_jac, name_tok_cont, name_tfidf_cos}** (ρ 0.95) → keep name_tok_jac.
- **{name_3gram_jac, name_lev, name_jw}** (ρ 0.86–0.89) → keep the *translit* variants of
  3-gram + JW; drop plain name_lev.
- **{digit_jac, street_num_match}** (ρ 0.90) → keep digit_jac, optionally both (cheap).

## 5. Dangerous-alone features (Q3)

Precision when the feature fires, measured against hard negatives:

| Rule | n | Precision |
|---|---:|---:|
| name_exact_norm == 1 | 38,825 | **0.093** |
| name_core_eq == 1 | 44,866 | **0.153** |
| pin_eq == 1 | 8,944 | **0.070** |
| addr_exact_norm == 1 | 1,282 | 0.868 |
| name_core_eq AND addr_cont ≥ 0.5 | 6,660 | **0.932** |
| name_core_eq AND addr_cont < 0.2 | 35,116 | **0.010** |

- **Identical name is a trap**: chains/franchises and deliberate distractors mean an exact
  name match with a dissimilar address is a match only 1% of the time.
- **Same pincode is far too weak** even when present.
- **Even identical addresses are wrong 13% of the time** (multi-tenant buildings /
  distractors) — never auto-accept on address alone either.

## 6. Interactions (Q8) — the core structure

P(match) by quadrant (name_tok_jac vs addr_tok_cont at 0.5):

| | addr < 0.5 | addr ≥ 0.5 |
|---|---:|---:|
| **name < 0.5** | 0.0006 (n=262K) | 0.089 (n=39K) |
| **name ≥ 0.5** | 0.021 (n=46K) | **0.831** (n=11K) |

Conjunction is worth ~40× over either margin. A 2-feature logistic model
(name_3gram_translit + addr_tok_cont) already reaches **AUC 0.983**; all 33 features →
**0.996**. Within the hard both-high quadrant, the discriminators are **digit_jac (AUC
0.852)** and **street_num_match (0.820)** — the negatives there are engineered near-twins:
same street with the house number nudged (`130` vs `134 Screaming Eagle`, `TC 4/914` vs
`TC 4/927`) or one name word swapped (`…Markets Pvt Ltd` vs `…Pharmaceuticals Pvt Ltd`).
Name edit similarity (0.70) and name_3gram (0.77) also retain power there; locality and
addr_exact do not.

## 7. Address rescue of name failures (Q4)

16.0% of positives have name_tok_jac < 0.3 (garbled names, scripts, domain forms).
Of those, **96.7% have addr_tok_cont ≥ 0.5 and 99.2% ≥ 0.3**. Address similarity almost
completely covers the name-failure tail. The residual 1.2% of positives weak on *both*
(162/13,815) are mostly empty-address candidates with heavily garbled names — char-level
name features (JW-translit ≥ 0.9, squash containment for domain forms) still catch most.

## 8. Pincode findings (Q5)

Near-useless. Presence: **US addresses ~11–13% contain a ZIP; India ~0% contain a PIN**
(addresses in this dataset simply end at city/state). pin_eq AUC 0.51, precision-alone 7%.
Verdict: keep at most `pin_eq` as a minor US-only feature; do not invest further. (digit_jac
already subsumes pincode digits when they exist.)

## 9. Street/building number and locality (Q6)

- **digit_jac / street_num_match: AUC 0.82/0.80 overall, and the best discriminators
  (0.85/0.82) inside the ambiguous both-high quadrant** — this is the single most
  cost-effective address enrichment. Stronger in India (0.87) than US (0.79), because Indian
  addresses carry more numeric atoms (plot/khasra/flat numbers).
- **locality_overlap (last-3 alpha tokens): AUC 0.81 US / 0.75 India** — moderate, partially
  redundant with addr_tok_cont; keep as a cheap extra, low priority.

## 10. Transliteration findings (Q7)

On the 20,304 Indic-script pairs (5.7% of sample; 993 positives):

| Feature | raw AUC | transliterated AUC |
|---|---:|---:|
| name_tok_jac | 0.529 | 0.560 |
| name_3gram_jac | 0.525 | **0.887** |
| name_jw | 0.665 | **0.957** |

`unidecode` transliteration is phonetically approximate (word-level tokens rarely match
exactly — token Jaccard barely moves) but **char-n-gram and Jaro-Winkler on the
transliterated string recover nearly all the signal**. Without it, Indic-script pairs are
blind on the name side (AUC ≈ 0.53 = coin flip). S2 India is ~13% Indic-script names, S3
~7.6% → this is a must-have. Note: Indic script also appears in *addresses* (state names in
Bengali/Kannada/Malayalam observed) — worth transliterating addresses too (untested here,
likely small win since the rest of the address is Latin).

## 11. S2 vs S3 (Q9)

AUCs are close (addr_tok_cont 0.948 vs 0.940; name_tok_jac 0.822 vs 0.786). S3 name signal
is slightly weaker across the board; S2 has ~2× the Indic-script share. No structural
difference that justifies separate models — one model with a `source` categorical flag.
(The sample's per-source positive rates differ, 2.8% vs 6.2%, but that is a candidate-
generation artifact, not a population property.)

## 12. US vs India (Q10)

- Name features: US stronger (tok_jac 0.822 vs 0.803); the India gap is entirely the script
  problem — **Latin-only India pairs have the *strongest* name signal of any slice**
  (3gram 0.919, name_addr_mean 0.980).
- Address features: India slightly stronger on token containment (0.952 vs 0.943) and much
  stronger on digit features (0.87 vs 0.79); US stronger on locality (0.86 vs 0.75).
- Practical: one model, country as a feature, transliteration for India; expect France to
  behave like "US with accents" (Latin script; accent-stripping already handled).

## 13. Difficult and ambiguous cases (Q11)

- **Hardest negatives** (both-high quadrant, 17% negative): near-twin records differing only
  in house number or one name token — resolved by digit_jac + name edit distance (§6).
- **Hardest positives** (1.2% low-low): empty candidate address + garbled/leet name
  (`Waterman Wor1d Inc`, `M0ore, Mcmath & Pinon`), domain-concatenations, and a few
  genuinely broken names (`Stubbe Holding LLC` ↔ `Umbrakelo` — likely irreducible noise).
- **Typo'd digits exist in true matches** (`15306` vs `5306 Peachmeadow`) — so digit
  features must be graded (containment/overlap), never a hard filter.

## 14. One-to-one structural check

Full ground-truth scan (all 2,206,821 rows, 7,638,365 matched IDs):
- **S2 IDs mapped to > 1 S1: 0**
- **S3 IDs mapped to > 1 S1: 0**
- Duplicate IDs within a single row's list: 0. All 7.64M matched IDs unique.

Zero violations → the evidence fully supports a **hard one-to-one assignment constraint**
from the S2/S3 side at inference (greedy: keep each S2/S3 record only for its
highest-scoring S1). Pure precision gain under the F0.5 metric.

## 15. Recommended initial feature set for the GBT/LightGBM model

Ordered by value; 16 core + 4 context features:

1. `addr_tok_cont` — top address signal (pick over jac/tfidf twins)
2. `addr_3gram_jac` — char-level address (typo robustness)
3. `addr_lev` — address edit similarity (order-sensitive complement)
4. `digit_jac` (+ optionally `street_num_match`) — house/plot number overlap; quadrant tiebreaker
5. `name_tok_jac` — word-level name
6. `name_3gram_jac_translit` — char-level name, transliteration-aware
7. `name_jw_translit` — edit-style name, transliteration-aware
8. `name_core_eq` — suffix-stripped equality flag (only in combination)
9. `name_squash_cont` — domain-name/concatenation catcher
10. `locality_overlap` — city/state tail overlap
11. `addr_missing` — missingness indicator (lets the model rely on name-only when needed)
12. `name_ntok_diff`, `addr_len_diff` — weak but cheap length signals
13. `indic_script` — script-mismatch indicator (interaction handle for the model)
14. `pin_eq` — marginal, US-only value; keep only if free
15. `source` (S2/S3), `country` (categorical, open-set safe)
16. **Add (not built here): S1 name-frequency ("chain-ness") within S1**, and candidate-set
    size per S1 — both target the same-name trap identified in §5.

Drop: `name_tfidf_cos`, `addr_tfidf_cos`, `name_tok_cont`, `name_lev`, `name_jw` (raw),
`addr_exact_norm`, `name_exact_norm`, `pin_prefix3`, `pin_present_both`, `name_acronym`
(fires 0.01% of pairs), `country_eq` (use as candidate filter instead), `name_len_diff`.

Sanity baseline: 2 features (name_3gram_translit + addr_tok_cont) give AUC 0.983 and all 33
give 0.996 with plain logistic regression — the GBT's job is the last mile: quadrant
interactions, digit tiebreakers, and calibrated probabilities for the F0.5 decision rule.

---

> **Stage 3 (real blocking candidates) supersedes the numbers below for pipeline
> decisions — see `docs/REAL_BLOCKING_MODEL_RESULTS.md`** (end-to-end macro-F0.5 0.9246
> at raw t=0.70; calibration no-op; per-S1 rule ties; o2o 1 conflict/18,220).

# STAGE 2 ADDENDUM — Chain-ness, LightGBM baseline, decision rules (2026-09-25)

Reproducible via `scripts/model_baseline.py` (stages `chainness`, `train`).
Metrics: `experiments/model_baseline_metrics.csv`; errors: `experiments/model_error_samples.csv`;
final schema: `docs/MODEL_FEATURE_SPEC.md`.

## A. Chain-ness (S1 name frequency)

Definition: count of the suffix-stripped normalized name within full train S1 (2.2M rows;
1,345,737 unique core names). Input-data only — no GT. Distribution: median 1, p90 = 2,
p99 = 8, p99.9 = 66, max 536; 14.9% of names are non-unique, covering 48.1% of S1 rows.
Use as **log1p(freq)** (heavy right tail).

- Alone: AUC 0.499 — pure interaction feature, useless as a ranker.
- Conditioned on identical core name, decisive: P(match) = **0.905** at freq 1 → 0.483 at
  freq 2–3 → 0.242 at 4–10 → 0.049 at 11–100 → **0.020** at freq > 100.
- In-model ablation: PR-AUC 0.976 with vs 0.978 without — the GBM recovers the signal from
  other features, so chain-ness is **helpful-but-not-critical**. Verdict: KEEP (cheap, kills
  the residual same-name FP tail, interpretable), but it is not a make-or-break feature.

## B. LightGBM baseline (entity-level split: 2,784 train / 1,200 val S1 entities)

19 features (spec §KEEP + country/n_cands), 400 trees. **ROC-AUC 0.9987, PR-AUC 0.9763.**
Top gain: name_3gram_translit, name_jw_translit, addr_len_diff, n_cands, addr_3gram_jac.

Threshold sweep (validation):

| t | pair-P | pair-R | macro-F0.5 |
|---:|---:|---:|---:|
| 0.5 | 0.941 | 0.926 | 0.9270 |
| 0.6 | 0.953 | 0.914 | 0.9319 |
| **0.7** | 0.961 | 0.902 | **0.9341** |
| 0.8 | 0.969 | 0.885 | 0.9303 |

Slices at t=0.7: US P/R 0.974/0.913 (macro-F0.5 0.9458), India 0.942/0.887 (0.9174),
S2 0.959/0.904, S3 0.963/0.900 (equivalent), Indic-script 0.919/0.818 (weakest slice),
ambiguous high-name/high-addr quadrant 0.976/0.956 (the GBM largely solved it — digit
features work). Singletons predicted correctly empty: 85.9% (61/71).

## C. Decision rules

- **Global threshold 0.7 is the current best** (macro-F0.5 0.9341). 0.5 is NOT optimal —
  the precision-heavy metric pushes the operating point up.
- **Per-S1 expected-F0.5 prefix rule: 0.9326** — slightly *below* the tuned global
  threshold. Root cause: model overconfidence in the 0.5–0.9 band (predicted 0.59–0.81 vs
  actual 0.53–0.71); the rule trusts inflated probabilities and over-selects. Isotonic
  calibration first, then re-test — the rule should win after calibration in theory, but
  the current evidence favors the simple threshold.
- **One-to-one post-processing: no measurable effect on this sample** (0 pairs dropped —
  with only 4k of 2.2M S1 entities sampled, two sampled S1s essentially never compete for
  the same S2/S3 record). The full-GT structural fact (0 violations in 7.64M links) still
  supports it; it can only act at full scale. Decision: keep as a **candidate** post-step,
  validate on the full-pipeline validation split before adopting — do not hard-wire yet.

## D. Error analysis (val, t=0.7: 155 FP / 412 FN)

FP causes (overlapping): 46% high-address-similarity with different names (shared
building/locality), 34% house-number near-twins that digit features didn't fully separate,
21% same-name chain pairs (only 8/32 of those have freq > 10 — chain-ness helps but names
with freq 2–10 remain risky). 48% of FPs have prob > 0.9 — confidently wrong, mostly
engineered near-twins; a `digit_exact_conflict` feature (house numbers present on both
sides but unequal) is the most promising fix.
FN causes: 43% sit at prob 0.3–0.7 (recoverable by calibration + decision rule, not new
features); 37% weak-name; 14% Indic-script; 11% empty-address. Only 2.2% are weak on both
name and address — the irreducible tail (DBA renames like `Superior Nuclear LLC` ↔
`1820237449 Quolum -`), which is a **candidate-generation problem, not a feature problem**.

Addressability estimate: ~45% of remaining errors addressable by calibration/threshold
work, ~25% by 1–2 new features (digit-conflict, freq-2–10 chain handling), ~20% by better
candidate generation/transliteration coverage, ~10% irreducible.
