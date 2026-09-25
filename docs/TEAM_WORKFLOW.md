# 48-Hour Team Workflow

## Team roles

### Person 1 — Data + Normalization + Blocking
Own:
- profiling
- normalization
- exact blocking
- token blocking
- character n-gram retrieval
- TF-IDF retrieval
- candidate recall

### Person 2 — Matching + ML
Own:
- pairwise features
- Logistic Regression baseline
- gradient boosting models
- threshold optimization
- entity-level macro F0.5

### Person 3 — Error Analysis + Integration
Own:
- experiment tracking
- false-positive/false-negative analysis
- singleton handling
- multiple-match handling
- country/source analysis
- final integration and audit

## 48-hour sequence
- Hours 0–3: profile data and establish baseline
- Hours 3–8: normalization + baseline matcher
- Hours 8–16: blocking experiments
- Hours 16–24: matching models/features
- Hours 24–30: error analysis
- Hours 30–36: threshold/singleton/multi-match optimization
- Hours 36–40: controlled advanced/embedding experiments
- Hours 40–44: freeze best architecture and train full data
- Hours 44–47: test generation + adversarial audit
- Hours 47–48: submission validation and packaging

## Experiment rule
Every experiment must report:
- candidate recall
- precision
- recall
- macro F0.5
- singleton performance
- multi-match performance
- runtime
- memory/candidate volume where relevant

Never keep a change merely because it is more sophisticated. Keep it if validation shows improvement.
