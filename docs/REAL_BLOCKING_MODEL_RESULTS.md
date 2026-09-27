# REAL-BLOCKING MODEL RESULTS — End-to-end TRAIN validation

Session 2 stage 3, 2026-09-25. Script: `scripts/real_blocking_validation.py`
(stages `candidates` / `features` / `train`). Metrics:
`experiments/real_blocking_validation_metrics.csv`, `threshold_results.csv`,
`calibration_results.csv`; errors: `experiments/model_error_samples.csv`.
TRAIN data only; test set untouched; no submission files generated.

## 0. TL;DR

Real blocking (locked Session-1 spec) → 19 features → LightGBM → threshold 0.70 gives
**end-to-end validation macro-F0.5 = 0.9246** (blocking misses counted as FNs).
Isotonic calibration and the per-S1 expected-F0.5 rule change nothing material;
one-to-one found 1 conflict in 18,220 selected pairs. The model is no longer
overconfident on the real candidate distribution. Biggest remaining lever: India
(0.8951 vs US 0.9447), driven by blocking misses and address-over-trust FPs.

## 1. Validation split

20,000 stratified S1 entities from the Session-1 blocker sample (12k US / 8k India,
`random_state=42`), split at the entity level with seed 21:
**fit 11,200 S1** (1,289,899 pairs, 37,826 positives) / **calibration 2,800 S1**
(327,199 pairs) / **validation 6,000 S1** (698,465 pairs, 20,251 candidate positives
+ 414 blocking-missed positives = 20,665 true matches). Val country mix: 3,567 US /
2,433 India. No S1 appears in more than one part; multi-match structure preserved;
GT used for labels only.

## 2. Actual candidate-generation recall (locked spec: US w50+c50+e, India w100+c100+e, exact-name cap 200)

| Slice | Pair recall |
|---|---:|
| Overall | **0.9794** |
| US | 0.9943 |
| India | 0.9571 |
| S2 / S3 | 0.9804 / 0.9784 |
| Entity all-match recall | 0.9400 |

Matches the Session-1 budget study exactly (asym config row) — the reconstruction is
faithful.

## 3. Candidate volume

2,315,563 pairs for 20k S1. Per S1: mean 115.8, median 92, p95 182, max 291
(US 89.8, India 154.7). Extrapolation to full test S1 (1.73M): ≈ 200M pairs.

## 4. Feature set

The 16 KEEP features from `docs/MODEL_FEATURE_SPEC.md` + **digit_exact_conflict**
(= 1 iff both normalized addresses contain ≥ 1 purely-numeric token and the sets are
disjoint; 0 otherwise, including when either side has no digits) + `src_s3`,
`country_india`, `n_cands` (recomputed from the real blocker). No DROP features
reintroduced. Top gain: name_jw_translit, n_cands, name_3gram_jac_translit,
addr_3gram_jac, addr_len_diff, addr_lev, s1_name_freq_log, addr_tok_cont.

## 5. LightGBM configuration

`LGBMClassifier(n_estimators=500, num_leaves=63, learning_rate=0.07,
min_child_samples=60, subsample=0.9, colsample_bytree=0.9, random_state=0)`.
All real blocking negatives kept — no downsampling (1.29M training pairs is cheap).

## 6. Raw vs calibrated probabilities

Raw: ROC-AUC **0.9982**, PR-AUC 0.9691. Calibrated (isotonic, fit on the separate
2,800-entity calibration fold, never on validation): ROC 0.9982, PR 0.9700.
Calibration table: raw is only mildly overconfident on real candidates
(0.5–0.7 bin: mean raw 0.597 vs actual 0.563; isotonic corrects to 0.568).
The research-sample overconfidence largely disappeared once the model saw real
blocking negatives. Downstream best macro-F0.5: raw 0.9246 vs calibrated 0.9240 —
**tie; calibration not required for the threshold rule** (keep it available for any
future probability-consuming rule).

## 7. Threshold results (raw probs; full grid in threshold_results.csv)

| t | P | R | macro-F0.5 | avg pred/S1 | zero-pred S1 | multi-pred S1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.813 | 0.938 | 0.8312 | 3.97 | 162 | 5497 |
| 0.30 | 0.915 | 0.907 | 0.8993 | 3.42 | 255 | 5221 |
| 0.50 | 0.951 | 0.883 | 0.9201 | 3.20 | 330 | 5063 |
| 0.65 | 0.967 | 0.863 | 0.9242 | 3.07 | 367 | 4959 |
| **0.70** | 0.971 | 0.857 | **0.9246** | 3.04 | 377 | 4931 |
| 0.75 | 0.975 | 0.848 | 0.9234 | 3.00 | 391 | 4896 |
| 0.90 | 0.989 | 0.786 | 0.9036 | 2.74 | 485 | 4610 |

Recall here counts blocking misses as FNs. The optimum is a plateau (0.60–0.75 all
within 0.001) — threshold choice is robust. **Best: t = 0.70.**

## 8. Per-S1 decision rule

Expected-F0.5 prefix selection: raw 0.9245, calibrated 0.9244 — statistical tie with
the global threshold (0.9246). It does not help on this dataset; keep the simpler
global threshold.

## 9. One-to-one post-processing

At the best config, **1 conflict among 18,220 selected pairs**; macro-F0.5 unchanged
(0.9246 → 0.9246). Recorded explicitly: with a 20k-entity sample vs a 10.3M-record
pool, cross-S1 competition is essentially absent. At full scale conflicts will be more
common; the constraint is structurally safe (0 GT violations in 7.64M links) and
resolves by highest model score. Verdict: harmless, include as a cheap final-pipeline
safety step, but expect no measurable validation gain.

## 10. Error analysis (best config: 521 FP / 2,552 in-candidate FN / 414 blocking-miss FN)

FP causes (overlapping): 43% address-similar-different-name, 28% near-identical name
(chains — a third of these have S1 name freq > 10), 15% digit-conflict pairs the model
overrode, 42% are high-confidence (p > 0.9). The worst FPs are **India pairs where a
short truncated candidate address coincides on tokens** (`H.no 130-H, South West Delhi`)
and the name is weak/other-script — the model over-trusts address containment when the
candidate address is short.
FN causes: 41% at p 0.3–0.7 (threshold zone), 34% at p < 0.1 (feature-blind: DBA
renames, domain names, digits-only overlap), 18% empty address, 10% Indic script.

Classification of all 3,487 errors:

| Class | Count | Share | Fix path |
|---|---:|---:|---|
| A — blocking (missed candidates) | 414 | 12% | bigger India k-budget / rescue channels |
| B — feature (FN with p < 0.3) | 1,508 | 43% | new signals: candidate-address-shortness interaction, better transliteration, digit-position features |
| C — calibration/threshold (FN p ≥ 0.3 + FP p < 0.9) | 1,348 | 39% | operating-point tuning; partially irreducible precision-recall trade |
| D — inherently ambiguous (FP p ≥ 0.9) | 217 | 6% | near-twins by design; accept |

## 11. Final validation configuration

Blocking per locked spec → 19 features (spec §4) → LightGBM (§5) → **raw probability
threshold 0.70** → optional one-to-one (no-op at this scale). End-to-end validation
macro-F0.5 **0.9246** (US 0.9447, India 0.8951). Singletons: 91.1% correctly empty
(316 in val). Ambiguous high-name/high-addr quadrant: P 0.983 / R 0.945 — solved.
Indic-script pairs: P 0.936 / R 0.798 — weakest slice.

## 12. Remaining risks

1. **India gap (−5 pp macro-F0.5)**: 3/4 of blocking misses are India; Indic-script
   recall 0.80. France is untested by construction — Latin script suggests US-like
   behavior, but there is zero validation evidence.
2. **Address-over-trust FPs** on short candidate addresses — candidate for one new
   feature (address length/token count of the *candidate* side as an explicit input,
   letting the model discount containment on 3-token addresses).
3. **n_cands feature** is blocker-dependent: if final blocking budgets change, retrain.
4. Sample is 20k of 2.2M entities; full-train retraining will see ~100× more pairs —
   expect equal or slightly better model, but rerun the threshold sweep at that scale.
5. Runtime path to full pipeline: retrieval extrapolates to ~10 h (train) + ~8 h (test)
   on this machine; feature computation ~200M pairs needs multiprocessing (~1–2 h).
