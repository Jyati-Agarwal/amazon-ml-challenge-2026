# MODEL FEATURE SPEC — Pairwise Matching Model (LightGBM)

Session 2, 2026-09-25. Feature schema for the final pairwise classifier, finalized from
`docs/FEATURE_SIGNAL_RESEARCH.md` (single-feature AUCs, redundancy analysis) and the
baseline run in `scripts/model_baseline.py` (metrics: `experiments/model_baseline_metrics.csv`).

Shared preprocessing: `norm(s)` = NFKD + casefold + accent-strip + punctuation→space +
whitespace collapse (Indic chars preserved). `core tokens` = norm tokens minus legal-suffix
stopwords (inc, corp, llc, ltd, pvt, private, limited, llp, sarl, sas, sci, …).
`translit(s)` = `unidecode(s)` applied when the string contains Indic-script characters.
All similarity features are in [0, 1] unless noted. No feature uses ground truth — the only
corpus-level statistic (chain-ness) is computed from S1 input rows only (leakage-safe).

## KEEP — core features (16)

| # | Feature | Definition | Why it matters | Type/Range | Missing behavior |
|---|---|---|---|---|---|
| 1 | `addr_tok_cont` | \|A∩B\| / min(\|A\|,\|B\|) over norm address tokens | Strongest single signal (AUC 0.941); containment tolerates the truncated S2/S3 addresses | float [0,1] | 0 when either address empty (see #13) |
| 2 | `addr_3gram_jac` | Jaccard of char 3-gram sets of norm address (spaces removed) | Typo-robust address complement (AUC 0.933) | float [0,1] | 0 when empty |
| 3 | `addr_lev` | rapidfuzz normalized Levenshtein similarity of norm addresses | Order-sensitive complement; separates reordered vs different (AUC 0.847) | float [0,1] | 0 when empty |
| 4 | `digit_jac` | Jaccard of purely-numeric token sets of both addresses | House/plot/khasra number agreement; best tiebreaker inside high-name/high-addr near-twins (in-quadrant AUC 0.852) | float [0,1] | 0 when either has no digits |
| 5 | `name_tok_jac` | Jaccard over core name tokens | Word-level name signal (AUC 0.815) | float [0,1] | 0 if a side has no core tokens |
| 6 | `name_3gram_jac_translit` | Jaccard of char 3-grams of norm names, candidate transliterated first | Char-level name signal that works across scripts (AUC 0.863 overall, 0.887 on Indic) | float [0,1] | 0 when empty |
| 7 | `name_jw_translit` | Jaro-Winkler similarity of norm names, candidate transliterated | Prefix-weighted edit view; 0.957 AUC on Indic slice | float [0,1] | 0 when empty |
| 8 | `name_squash_cont` | 1 if alnum-squashed name of one side is substring of the other (len > 6) | Catches domain-form names (`precisionagriculturalconsultants.com`) | binary | 0 |
| 9 | `name_ntok_diff` | abs difference in core-token counts | Cheap guard against "one extra word" near-twins (`…Overseas Limited`) | int ≥ 0 | 0 |
| 10 | `addr_len_diff` | abs(len) / max(len) of norm addresses | Truncation indicator; high LightGBM gain (helps read containment correctly) | float [0,1] | 1 when exactly one side empty; 0 when both empty |
| 11 | `locality_overlap` | Jaccard of last-3 alphabetic address tokens | City/state tail agreement (AUC 0.81 US / 0.75 India) | float [0,1] | 0 when empty |
| 12 | `street_num_match` | containment of non-pincode digit tokens | Redundant-ish with #4 (ρ 0.90) but cheap; keeps the near-twin signal if #4 is diluted by pincodes | float [0,1] | 0 |
| 13 | `addr_missing` | 1 if either address is empty/whitespace | Lets the model switch to name-only logic instead of reading 0-similarity as mismatch | binary | n/a (is the indicator) |
| 14 | `indic_script` | 1 if candidate name contains chars in U+0900–U+0D7F | Interaction handle: tells the model which name features to trust | binary | n/a |
| 15 | `src_s3` | 1 if candidate is S3 (from entity_id prefix) | S2/S3 behave similarly but not identically (S3 name signal slightly weaker) | binary | n/a |
| 16 | `s1_name_freq_log` | log1p(count of the S1 record's core name within FULL train+test S1 of same split) | Chain-ness. P(match \| identical name) falls 0.905 → 0.020 as freq goes 1 → 100+. See leakage note below | float ≥ 0 | log1p(1) for unseen names |
| 17 | `digit_exact_conflict` | 1 iff both normalized addresses contain ≥1 purely-numeric token AND the two digit-token sets are disjoint | Explicit house-number disagreement flag for near-twin FPs (added stage 3; validated on real candidates) | binary | 0 when either side has no digit tokens |

## OPTIONAL — include if free, drop first under pressure

| Feature | Definition | Why optional |
|---|---|---|
| `name_core_eq` | sorted core tokens identical | Only 15% precision alone; GBM importance rank last (75 gain); its information is inside `name_tok_jac`==1. Keep only as an interaction convenience |
| `country` | raw country string as LightGBM categorical (open set — never one-hot) | Slice AUCs differ (India digit-heavy, US locality-heavy), but with France unseen in training the model must not lean on it. Prefer `country_india`-style binary OFF and re-test; baseline used it with small gain |
| `n_cands` | candidate-set size for this S1 | High gain in baseline but **its value depends on the candidate generator**; recompute from the production blocking output, never reuse research values. Include only after re-validation on real candidates |
| `pin_eq` | pincode equality (last 5/6-digit token) | Pincodes present in ~12% US / ~0% India addresses; AUC 0.51. Only worth it if literally free |
| `addr_tok_jac` | Jaccard over address tokens | ρ = 0.99 with `addr_tok_cont`; swap in only if containment misbehaves on production candidates |

## DROP — redundant or empirically dead

| Feature | Reason |
|---|---|
| `name_tfidf_cos`, `addr_tfidf_cos` | ρ ≥ 0.90 with kept token features; needs a fitted vectorizer at inference (state to ship) for no AUC gain |
| `name_tok_cont` | ρ = 0.99 with `name_tok_jac` |
| `name_lev`, `name_jw` (raw) | ρ ≥ 0.86 with kept char features; the translit variants strictly dominate |
| `name_exact_norm`, `addr_exact_norm` | Subsumed by continuous features; exact name alone is 9% precision (danger) |
| `pin_prefix3`, `pin_present_both` | AUC ≤ 0.51; pincodes essentially absent from this dataset |
| `name_acronym` | Fires on 0.01% of pairs; no measurable lift |
| `name_len_diff` | Weakest length feature (AUC 0.30 inverse); `name_ntok_diff` covers it |
| `country_eq` | Use as a hard candidate filter (GT never crosses country), not a feature |

## Leakage warnings

- **`s1_name_freq_log`**: must be computed from S1 *input records only* (never from GT link
  counts, never from S2/S3 match counts). At test time compute it over test S1. Computing it
  over train S1 and applying to test S1 is also acceptable but same-split is cleaner.
  Never use "number of GT matches" or anything derived from labels as a feature.
- **`n_cands`**: reflects the blocking stage, which is label-free — no leakage — but it is
  distribution-sensitive: research-sample values (cap 150) differ from production blocking
  (top-k ≈ 100–120). Recompute per pipeline.
- **Entity-level splits only**: pairs of one S1 must never straddle train/validation
  (`model_baseline.py` splits by `s1_id`).

## Baseline evidence (entity-split validation, 1,200 S1 / 106K pairs)

Model: LightGBM, 400 trees, 63 leaves (small research config).
ROC-AUC **0.9987**, PR-AUC **0.976**. Without chain-ness: 0.9985 / 0.978 — chain-ness is
**not load-bearing for the model** (its signal is recoverable from other features) but is
kept for the same-name FP tail and interpretability; use `log1p`, never raw frequency
(distribution: median 1, p99 = 8, max 536; 48% of S1 rows share a non-unique name).
Best global threshold by macro-F0.5: **t = 0.7 → 0.9341** (t=0.5 → 0.9270).
Known model weakness: overconfidence in the 0.5–0.9 probability band (actual match rate
~0.53–0.71 vs predicted 0.59–0.81) → isotonic calibration before the decision stage.

**Stage-3 update (real blocking candidates — see docs/REAL_BLOCKING_MODEL_RESULTS.md):**
schema confirmed on 2.32M real candidate pairs (17 core features incl.
digit_exact_conflict + src_s3/country_india/n_cands). End-to-end validation macro-F0.5
**0.9246 at raw t=0.70** (blocking misses counted). Overconfidence largely disappeared on
the real negative distribution; isotonic calibration is a no-op for the threshold rule.
Candidate new feature for the next iteration: candidate-address token count (to discount
containment on very short addresses — the main high-confidence FP pattern).
