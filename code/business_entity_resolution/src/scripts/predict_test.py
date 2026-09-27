#!/usr/bin/env python3
"""V0 TEST inference: feature parquets -> official submission files.

Implements EXACTLY the locked D15 decision pipeline validated at macro-F0.5
0.9322 (train_final_model.py stage_validate):
  LightGBM probabilities -> threshold (0.625) -> one-to-one by model score
  (sort prob desc, drop duplicate cand_id keep-first), made fully
  deterministic here with a stable sort and (s1_id, cand_id) tie-break.

Inputs
  --features  dir of feat_*.parquet from feature_pipeline.py `features`
              (run on the MERGED test candidate dir with an isolated
              --cache-dir; each S1's candidates live in exactly one file).
  --cands     the merged candidate dir (s1_index/ gives the complete S1
              universe, including zero-candidate S1s).
  --model     models/lgbm_final.txt (feature order verified against
              MODEL_FEATURES and the booster's own header before scoring).

Outputs (sorted by s1_id, UTF-8, tab-separated, validator-conformant)
  <out>/matching_results.tsv   one row per S1 (empty second column = no match)
  <out>/candidate_pairs.tsv    the COMPLETE candidate set actually scored by
                               the model (every pair, not just predictions)

Guarantees: predictions are a subset of candidate_pairs.tsv by construction
(only scored candidate pairs can be selected); duplicate pairs are rejected;
IDs pass through verbatim. Never reads dataset/ or the model training data.

Usage (.venv312):
  python scripts/predict_test.py \
      --features experiments/test_features \
      --cands experiments/candidates/test_sharded/merged \
      --out output
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from feature_pipeline import log                     # noqa: E402
from train_final_model import MODEL_FEATURES        # noqa: E402  canonical order

# D15 locked decision threshold (== best_threshold in
# experiments/train_final/validation_summary.json). Do not retune.
THRESHOLD_D15 = 0.625


def load_all_s1(cand_dir):
    """Complete S1 universe from the merged s1_index (covers zero-candidate
    S1s). Returns a sorted list; fails on duplicates (shards must be disjoint)."""
    files = sorted((Path(cand_dir) / 's1_index').glob('*.parquet'))
    if not files:
        sys.exit(f"no s1_index parquets under {cand_dir}")
    ids = pd.concat([pd.read_parquet(f, columns=['s1_id']) for f in files],
                    ignore_index=True)['s1_id']
    if ids.duplicated().any():
        sys.exit("duplicate s1_id across s1_index files — shards not disjoint?")
    return sorted(ids.tolist())


def joined_lists(df):
    """(s1_id, cand_id) frame -> {s1_id: 'id1,id2,...'} with ids sorted
    (deterministic within-row order)."""
    d = df[['s1_id', 'cand_id']].sort_values(
        ['s1_id', 'cand_id'], kind='stable')
    return d.groupby('s1_id', sort=False)['cand_id'].agg(','.join).to_dict()


def write_tsv(path, second_col, all_s1, mapping):
    """One row per S1 in sorted order; empty second column when no ids."""
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(f"source1_entity_id\t{second_col}\n")
        for s1 in all_s1:
            f.write(f"{s1}\t{mapping.get(s1, '')}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features', required=True,
                    help='dir with feat_*.parquet (unlabeled test features)')
    ap.add_argument('--cands', required=True,
                    help='merged candidate dir (s1_index/ = full S1 universe)')
    ap.add_argument('--model', default=str(ROOT / 'models' / 'lgbm_final.txt'))
    ap.add_argument('--threshold', type=float, default=THRESHOLD_D15,
                    help='decision threshold (default: locked D15 value)')
    ap.add_argument('--out', default=str(ROOT / 'output'))
    ap.add_argument('--chunk', type=int, default=5_000_000,
                    help='prediction chunk rows')
    args = ap.parse_args()
    if args.threshold != THRESHOLD_D15:
        log(f"WARNING: threshold {args.threshold} != locked D15 "
            f"{THRESHOLD_D15} — only use for experiments, never submission")

    import lightgbm as lgb
    booster = lgb.Booster(model_file=args.model)
    # The model's own header is the source of truth for feature order. Any
    # accepted model must start with the exact V0 MODEL_FEATURES order and may
    # only EXTEND it (e.g. H1 adds digit_lev_best, digit_prefix_cont); the
    # feature parquets must then contain those extra columns too.
    feats = booster.feature_name()
    if feats[:len(MODEL_FEATURES)] != MODEL_FEATURES:
        sys.exit("FATAL: booster features do not start with MODEL_FEATURES:\n"
                 f"  booster: {feats}\n"
                 f"  code:    {MODEL_FEATURES}")
    extra = feats[len(MODEL_FEATURES):]
    log(f"model {args.model}: {len(feats)} features verified"
        + (f" (V0 base + {extra})" if extra else " (== V0 MODEL_FEATURES)")
        + f"; threshold {args.threshold}")

    all_s1 = load_all_s1(args.cands)
    log(f"S1 universe: {len(all_s1):,} entities (from merged s1_index)")

    feat_files = sorted(Path(args.features).glob('feat_*.parquet'))
    if not feat_files:
        sys.exit(f"no feat_*.parquet under {args.features}")

    cand_lists = {}     # s1_id -> comma list: the COMPLETE scored candidate set
    kept = []           # pairs with prob >= threshold
    t0, total = time.time(), 0
    for f in feat_files:
        df = pd.read_parquet(f, columns=['s1_id', 'cand_id'] + feats)
        if df.duplicated(['s1_id', 'cand_id']).any():
            sys.exit(f"FATAL: duplicate (s1_id, cand_id) pairs in {f.name}")
        X = df[feats].to_numpy(dtype=np.float32)
        probs = np.empty(len(df), dtype=np.float32)
        for lo in range(0, len(df), args.chunk):
            probs[lo:lo + args.chunk] = booster.predict(X[lo:lo + args.chunk])
        del X
        part = joined_lists(df)
        clash = cand_lists.keys() & part.keys()
        if clash:
            sys.exit(f"FATAL: S1 ids appear in multiple feature files "
                     f"(e.g. {next(iter(clash))}) — candidates must be "
                     "grouped per file (merged generate_candidates output)")
        cand_lists.update(part)
        m = probs >= args.threshold
        if m.any():
            kept.append(pd.DataFrame({
                's1_id': df['s1_id'].to_numpy()[m],
                'cand_id': df['cand_id'].to_numpy()[m],
                'prob': probs[m]}))
        total += len(df)
        log(f"{f.name}: {len(df):,} pairs, {int(m.sum()):,} >= t "
            f"(cum {total:,}, {total / max(time.time() - t0, 1e-9):,.0f} pairs/s)")

    unknown = cand_lists.keys() - set(all_s1)
    if unknown:
        sys.exit(f"FATAL: {len(unknown)} scored S1 ids missing from s1_index "
                 f"(e.g. {next(iter(unknown))})")

    # one-to-one: exact stage_validate semantics (prob desc, keep-first per
    # cand_id) with an explicit deterministic tie-break on (s1_id, cand_id).
    if kept:
        sel = pd.concat(kept, ignore_index=True)
        sel = sel.sort_values(['prob', 's1_id', 'cand_id'],
                              ascending=[False, True, True], kind='stable')
        dropped = int(sel.duplicated('cand_id', keep='first').sum())
        sel = sel[~sel.duplicated('cand_id', keep='first')]
    else:
        sel = pd.DataFrame(columns=['s1_id', 'cand_id', 'prob'])
        dropped = 0
    match_lists = joined_lists(sel) if len(sel) else {}
    log(f"selected {len(sel):,} pairs after o2o (dropped {dropped:,}); "
        f"{len(match_lists):,} S1 with >=1 match, "
        f"{len(all_s1) - len(match_lists):,} empty")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_tsv(out / 'matching_results.tsv', 'matched_entity_ids',
              all_s1, match_lists)
    write_tsv(out / 'candidate_pairs.tsv', 'candidate_entity_ids',
              all_s1, cand_lists)
    log(f"wrote {out / 'matching_results.tsv'} + {out / 'candidate_pairs.tsv'} "
        f"({len(all_s1):,} rows each, {total:,} candidate pairs) in "
        f"{time.time() - t0:.0f}s")
    log("next: run utils/validate_submission.py before zipping")


if __name__ == '__main__':
    main()
