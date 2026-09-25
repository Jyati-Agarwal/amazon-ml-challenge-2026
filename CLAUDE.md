# Claude Code Rules — Amazon ML Challenge 2026

## Project boundary
This repository is the ONLY GitHub repository Claude Code may use for this project:
- Account: Jyati-Agarwal
- Repository: Jyati-Agarwal/amazon-ml-challenge-2026

This project is being developed on a work laptop. Protect work/company repositories and credentials.

## Git safety
- Never inspect, modify, push to, pull from, fetch from, or clone a company/work repository.
- Never change global Git configuration.
- Never change SSH keys, credential-manager settings, or company authentication.
- Use only this repository for Git operations.
- Before any push, show the current branch, `git remote -v`, and files to be pushed, then ask for explicit approval.
- Never force-push.
- Never run destructive commands such as `git reset --hard` or `git clean -fd` without explicit approval.
- Never commit secrets, tokens, credentials, .env files, or competition data.

## Dataset safety
The competition dataset is local-only. NEVER commit or upload:
- dataset/
- raw TSV/CSV data
- generated candidate datasets
- model binaries
- credentials or environment files

## Competition constraints
- No external business-data lookup, enrichment, geocoding, commercial ER APIs, or external datasets.
- Respect the challenge model-size and license constraints.
- Optimize for the official macro F0.5 metric.
- Preserve zero, one, and multiple matches per Source 1 entity.
- Final predicted matches must be contained in candidate_pairs.tsv.

## Working style
- Inspect before changing.
- Keep experiments reproducible.
- Measure every change against validation macro F0.5.
- Preserve the previous best experiment.
- Prefer simple, measurable improvements over unnecessary complexity.
