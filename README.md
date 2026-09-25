# Amazon ML Challenge 2026 — Business Entity Resolution

Personal competition repository for Jyati Agarwal.

## Goal
Resolve noisy business records across Source 1, Source 2, and Source 3. For every Source 1 entity, identify all matching Source 2/Source 3 records.

## Repository policy
This repository contains CODE and DOCUMENTATION only. The competition dataset stays local and is intentionally ignored by Git.

## Planned pipeline
1. Data profiling
2. Name/address normalization
3. Candidate generation / blocking
4. Pairwise similarity features
5. ML matching
6. Threshold and singleton optimization
7. Error analysis
8. Final prediction
9. Submission validation

## Evaluation
The challenge uses macro F0.5, so precision is weighted more heavily than recall. The system must correctly handle entities with no match as well as entities with multiple matches.

## Local structure
```
code/
  business_entity_resolution/
    src/
experiments/
docs/
output/              # local only; ignored
dataset/             # local only; ignored
```

## Safety
See CLAUDE.md for Git, dataset, and competition-boundary rules.
