# DATA PROFILE — Amazon ML Challenge 2026 (Business Entity Resolution)

Measured 2026-09-25 by `scripts/profile_data.py` (raw numbers in `docs/data_profile.json`).
All row counts exclude the header line. Dataset was read-only; nothing was modified.

## 1. Files and sizes

| File | Rows | Size |
|---|---:|---:|
| train_source1.tsv | 2,206,821 | 210 MB |
| train_source2.tsv | 5,034,616 | 489 MB |
| train_source3.tsv | 5,285,603 | 504 MB |
| train_ground_truth.tsv | 2,206,821 | 127 MB |
| test_source1.tsv | 1,732,544 | 175 MB |
| test_source2.tsv | 4,887,273 | 509 MB |
| test_source3.tsv | 5,082,316 | 506 MB |
| **Total** | ~24.2M | **2.3 GB** |

Columns in every source file: `entity_id`, `business_name`, `business_address`, `country`.
`entity_id` is unique in every file. Ground truth has exactly one row per train S1 entity.

## 2. Country distribution

| File | US | India | France |
|---|---:|---:|---:|
| train S1 | 1,323,633 (60.0%) | 883,188 (40.0%) | — |
| train S2 | 3,016,817 (59.9%) | 2,017,799 (40.1%) | — |
| train S3 | 3,170,056 (60.0%) | 2,115,547 (40.0%) | — |
| test S1 | 663,106 (38.3%) | 809,986 (46.7%) | **259,452 (15.0%)** |
| test S2 | 1,871,330 (38.3%) | 2,312,565 (47.3%) | 703,378 (14.4%) |
| test S3 | 1,945,701 (38.3%) | 2,405,000 (47.3%) | 731,615 (14.4%) |

France appears **only in test** (15% of test S1 entities) — zero training signal for it.
Train is US-majority; test is India-majority. Country distribution shift is real.

## 3. Missingness

- **S1 (train and test): zero missing values in any column.** S1 is the clean reference source.
- **S2/S3: `business_address` empty in ~2.6–3.4% of rows** (train S2 3.36%, train S3 3.33%, test S2 2.65%, test S3 2.68%). Roughly uniform across countries.
- `business_name` and `country` are never empty anywhere.

## 4. Text characteristics

Name length ~24–26 chars everywhere (max ~123). Address length: US ~32–39 chars,
India ~59–78 chars, France ~39–50 chars. India S1 addresses (mean 78) are much longer
than India S2/S3 addresses (59–69) — S2/S3 drop components.

**Script / Unicode (200k-row samples):**

- S1 names are 100% ASCII in both train and test (test S1 has 2.4% non-ASCII — French accents only).
- **S2 India: 13.3–13.4% of names are in Devanagari** (e.g. `राम मार्केटिंग प्राइवेट लिमिटेड`); S3 India: 7.5–7.6%. S1 never uses Devanagari → those matches require transliteration handling.
- Overall non-ASCII names: S2 15–19%, S3 11–15% (Devanagari + accented Latin).
- Junk prefixes exist (`-- Holloway Peak Inc Seafood`, `<< Team Ecole`), domain-style names (`wilfordhancock.com`), legal suffixes in many variants (Inc, Corp, Pvt Ltd, प्राइवेट लिमिटेड, SARL, S.A.S, SCI, LLC — sometimes prefixed: `LLC Moncada Léarning Center`).
- Addresses contain typos by design (`RUE JEN ZAY` ≈ Rue Jean Zay, `Rue Icmre`), khasra numbers (`KH NO. -570/13`), landmark references, reordered components (`IA, Iowa City, 1064 Newton Rd, Unit 11`).

## 5. Duplicates

| File | Exact dup rows (name+addr+country) | Dup names | Dup non-empty addresses |
|---|---:|---:|---:|
| train S1 | **0** | 667,592 (30%) | 76,215 (3.5%) |
| train S2 | 25,873 | 632,607 | 528,388 (10.5%) |
| train S3 | 18,860 | 633,994 | 476,923 (9.0%) |
| test S1 | **0** | 493,677 (28%) | 55,061 |
| test S2 | 22,641 | 576,232 | 533,082 |
| test S3 | 16,293 | 560,387 | 489,783 |

- S1 is deduplicated (as documented): no exact duplicate rows.
- S2/S3 contain exact duplicate rows (distinct entity_ids, identical content) — these are
  distinct records that may *both* be true matches for the same S1 entity.
- Duplicate business names are massive (~30% of S1): **name alone is nowhere near a unique key**; chains/franchises share names at different addresses.

## 6. Ground truth structure

2,206,821 S1 entities; 7,638,365 matched IDs total (avg **3.461 matches per S1**, max 11).

| Matches per S1 | Count | % |
|---:|---:|---:|
| 0 (singleton) | 123,247 | **5.59%** |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.05% |
| 4 | 484,115 | 21.94% |
| 5 | 321,957 | 14.59% |
| 6 | 164,868 | 7.47% |
| 7+ | 87,424 | 3.96% |

**S2 vs S3 split:** 3,693,619 S2 IDs vs 3,944,746 S3 IDs. Per S1: 87.0% have ≥1 S2 match,
87.9% have ≥1 S3 match, 80.5% have both, 6.5% S2-only, 7.5% S3-only.
Max 5 matches from S2, max 6 from S3 per entity.

**Clean partition:** all 7,638,365 matched IDs are unique — **each S2/S3 record matches at
most one S1 entity**. This is a strong constraint usable at inference (one-to-at-most-one
assignment from the S2/S3 side).

**Distractors:** 73.4% of train S2 records and 74.6% of train S3 records are matched to
some S1 entity → **~26% of S2/S3 rows match nothing** and exist purely as precision traps.

## 7. Source differences (S2 vs S3)

- S2 has more Devanagari names (13.4% vs 7.6% of India rows) — heavier transliteration noise.
- S2 India addresses are longer (mean 68–69) than S3 India (59) — S3 truncates more.
- S3 contributes slightly more matches overall and allows up to 6 per entity (S2 max 5).
- Both have ~3% empty addresses and similar duplicate profiles. Differences are moderate —
  same pipeline should handle both, possibly with a source-ID feature.

## 8. Computational profile

- Machine: 48 GB RAM, 14 cores (Apple Silicon Mac).
- Each source file loads in pandas (object dtype) using roughly 2–4 GB peak; the full
  dataset can be held in memory, but per-file processing is safer. No chunking required.
- **All-pairs is impossible:** test S1 × (S2+S3) = 1.73M × 9.97M ≈ 1.7×10¹³ pairs.
  Blocking by country first (largest block: India 1.73M×4.7M) still ≈ 3.8×10¹² — blocking
  must reduce candidates to ~10–100 per S1 entity (≈ 10⁷–10⁸ pairs scored) to be tractable.
- Expected bottlenecks: (1) candidate generation over ~10M records (needs vectorized
  TF-IDF/ANN or inverted-index approach, not per-row Python loops), (2) pairwise feature
  computation at 10⁷–10⁸ scale (needs batching/multiprocessing), (3) memory for TF-IDF
  matrices over 10M strings (sparse, manageable).
- Rough recall ceiling math: at avg 3.46 true matches/entity, blocking recall of 95%
  caps F_0.5 recall term; with F_0.5 precision-weighted, precision of the final
  classifier matters ~2× more.

## 9. Notable per-entity quirks observed

- Test S1 address formats include shuffled component order (`IA, Iowa City, 1064 Newton Rd, Unit 11`).
- French legal forms: SARL, S.A.S, SCI; regions in addresses (Nouvelle-Aquitaine, Hauts-de-France).
- India addresses use state abbreviations in some sources (`HR`, `DL`) vs full names in others (`Madhya Pradesh`).
- US state given as abbreviation in S1/S2 (`NC`, `TX`) but sometimes full (`Texas`) in S3.
