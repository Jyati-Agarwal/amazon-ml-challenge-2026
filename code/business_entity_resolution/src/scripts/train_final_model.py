#!/usr/bin/env python3
"""Production training pipeline: shard candidates -> features -> LightGBM -> validation.

Consumes the outputs of scripts/generate_candidates.py (shard dirs with
parts/*.parquet + s1_index/*.parquet), one part file at a time — the full
200M+ pair set is never in RAM as raw pairs/text. Feature computation is
scripts/feature_pipeline.py (locked schema; run its `prep` stage first).

Stages (each resumable):
  featurize — per shard part: compute features, attach GT labels, write
              <out>/features/<shard>/feat_*.parquet (manifest-skips done parts).
              Also writes <out>/s1_meta.parquet (every S1: country, n_cands,
              n_true) — the macro-F0.5 denominator, incl. zero-candidate S1s.
  train     — entity-level split (MD5(s1_id)%100 < --val-pct => validation;
              deterministic, machine-independent, no S1 straddles), trains
              LightGBM on ALL train-split pairs (no negative downsampling),
              saves models/lgbm_final.txt + .meta.json, writes the val-split
              feature rows to <out>/val_features.parquet for `validate`.
  validate  — raw probabilities, threshold grid (0.60–0.75 focus), macro-F0.5
              with blocking misses counted as FN, singleton accounting,
              optional one-to-one post-processing comparison, country slices.

GT is used ONLY for labels/evaluation. TEST data is never touched here.

Usage (.venv312):
  python scripts/feature_pipeline.py prep --workers 4          # once
  python scripts/train_final_model.py featurize \
      --cands experiments/candidates/train_sharded/shard_0 \
              experiments/candidates/train_sharded/shard_1 \
              experiments/candidates/train_sharded/shard_2 \
      --out experiments/train_final --workers 6
  python scripts/train_final_model.py train    --out experiments/train_final
  python scripts/train_final_model.py validate --out experiments/train_final
"""
import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from feature_pipeline import FEATURES, FeatureComputer, load_n_cands_index, log  # noqa: E402

MODELS = ROOT / 'models'
GT_DEFAULT = ROOT / 'dataset' / 'train' / 'train_ground_truth.tsv'
# cand_addr_ntok is computed and stored in the feature parquet but EXCLUDED from
# the model: measured on the 20k smoke set it costs ~1.3pp macro-F0.5
# (0.8983 w/ vs 0.9113 w/o; PR-AUC 0.8798 vs 0.9192).
MODEL_FEATURES = [f for f in FEATURES if f != 'cand_addr_ntok']
LGB_PARAMS = dict(n_estimators=500, num_leaves=63, learning_rate=0.07,
                  min_child_samples=60, subsample=0.9, colsample_bytree=0.9,
                  random_state=0, verbose=-1)          # validated D9 config


def val_bucket(s1_ids):
    """Deterministic entity bucket 0-99 (same scheme family as D11 sharding)."""
    return np.fromiter(
        (int(hashlib.md5(e.encode()).hexdigest(), 16) // 1000 % 100
         for e in s1_ids), dtype=np.int64, count=len(s1_ids))
    # //1000 decorrelates from the %3 shard assignment on the same digest


def read_gt(path):
    """-> (exploded (s1_id, cand_id, label=1) frame, s1_id -> n_true Series)."""
    import csv
    csv.field_size_limit(10**7)
    s1s, cands, counts_id, counts_n = [], [], [], []
    with open(path) as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            m = row[1].split(',') if len(row) > 1 and row[1] else []
            counts_id.append(row[0])
            counts_n.append(len(m))
            s1s.extend([row[0]] * len(m))
            cands.extend(m)
    pos = pd.DataFrame({'s1_id': s1s, 'cand_id': cands})
    pos['label'] = np.uint8(1)
    n_true = pd.Series(counts_n, index=pd.Index(counts_id, name='s1_id'),
                       name='n_true', dtype=np.int32)
    return pos, n_true


# ---------------------------------------------------------------- featurize
def stage_featurize(args):
    out = Path(args.out)
    (out / 'features').mkdir(parents=True, exist_ok=True)
    man_path = out / 'featurize_manifest.json'
    man = json.loads(man_path.read_text()) if man_path.exists() else {}

    pos, n_true = read_gt(args.gt)
    log(f"GT: {len(n_true):,} S1, {len(pos):,} positive links")

    # per-S1 metadata over ALL shards (zero-candidate S1s included)
    metas = []
    for d in args.cands:
        idx = Path(d) / 's1_index'
        for f in sorted(idx.glob('*.parquet')):
            metas.append(pd.read_parquet(f))
    meta = pd.concat(metas, ignore_index=True) \
        .groupby(['s1_id', 'country'], as_index=False)['n_cands'].sum()
    meta = meta.merge(n_true.reset_index(), on='s1_id', how='left')
    meta['n_true'] = meta['n_true'].fillna(0).astype(np.int32)
    meta.to_parquet(out / 's1_meta.parquet', index=False)
    log(f"s1_meta: {len(meta):,} S1 across {len(args.cands)} shard dir(s); "
        f"{int((meta.n_cands == 0).sum()):,} with zero candidates")

    fc = FeatureComputer(workers=args.workers)
    t0, total = time.time(), 0
    for d in args.cands:
        shard = Path(d).name
        ncm = load_n_cands_index([d])
        odir = out / 'features' / shard
        odir.mkdir(parents=True, exist_ok=True)
        files = sorted((Path(d) / 'parts').glob('*.parquet'))
        for f in files:
            key = f"{shard}/{f.name}"
            if man.get(key) == 'done':
                continue
            pairs = pd.read_parquet(
                f, columns=['s1_id', 'cand_id', 'source', 'country'])
            chunks = []
            for lo in range(0, len(pairs), args.pair_chunk):
                chunk = pairs.iloc[lo:lo + args.pair_chunk]
                feats = fc.compute(chunk, n_cands_map=ncm)
                lab = chunk[['s1_id', 'cand_id']].merge(
                    pos, on=['s1_id', 'cand_id'], how='left')['label'] \
                    .fillna(0).astype(np.uint8)
                res = pd.concat(
                    [chunk[['s1_id', 'cand_id']].reset_index(drop=True),
                     feats.reset_index(drop=True)], axis=1)
                res['label'] = lab.to_numpy()
                chunks.append(res)
            res = pd.concat(chunks, ignore_index=True)
            bad = int((~np.isfinite(res[FEATURES].to_numpy())).sum())
            if bad:
                raise ValueError(f"{bad} non-finite feature values in {key}")
            res.to_parquet(odir / ('feat_' + f.name), index=False)
            total += len(res)
            man[key] = 'done'
            man_path.write_text(json.dumps(man, indent=1))
            log(f"{key}: {len(res):,} pairs, {int(res.label.sum()):,} pos "
                f"(cum {total:,}, "
                f"{total / max(time.time() - t0, 1e-9):,.0f} pairs/s)")
    fc.close()
    log(f"featurize done: {total:,} new pairs in {time.time() - t0:.0f}s")


# -------------------------------------------------------------------- train
def feature_files(out):
    return sorted((Path(out) / 'features').rglob('feat_*.parquet'))


def stage_train(args):
    import lightgbm as lgb
    import pyarrow.parquet as pq

    out = Path(args.out)
    MODELS.mkdir(exist_ok=True)
    model_path = MODELS / args.model_name
    if model_path.exists() and not args.force:
        sys.exit(f"{model_path} exists — pass --force to retrain")
    files = feature_files(out)
    if not files:
        sys.exit("no feature files — run featurize first")

    # pass 1: split masks per file + row counts (only s1_id column is read)
    masks, n_train = [], 0
    for f in files:
        ids = pq.read_table(f, columns=['s1_id'])['s1_id'].to_numpy(
            zero_copy_only=False)
        m = val_bucket(ids) >= args.val_pct           # True -> train row
        masks.append(m)
        n_train += int(m.sum())
    log(f"pass 1: {len(files)} files, {n_train:,} train pairs, "
        f"{sum(len(m) for m in masks) - n_train:,} val pairs "
        f"(val_pct={args.val_pct})")
    est_gb = n_train * len(MODEL_FEATURES) * 4 / 1e9
    log(f"train matrix estimate: {est_gb:.1f} GB float32")

    # pass 2: fill preallocated train matrix; stream val rows to parquet
    X = np.empty((n_train, len(MODEL_FEATURES)), dtype=np.float32)
    y = np.empty(n_train, dtype=np.uint8)
    val_parts, pos_at = [], 0
    for f, m in zip(files, masks):
        df = pd.read_parquet(f)
        fx = df[MODEL_FEATURES].to_numpy(dtype=np.float32)
        lb = df['label'].to_numpy()
        k = int(m.sum())
        X[pos_at:pos_at + k] = fx[m]
        y[pos_at:pos_at + k] = lb[m]
        pos_at += k
        if (~m).any():
            val_parts.append(df.loc[~m])
        del df, fx
    if val_parts:
        pd.concat(val_parts, ignore_index=True).to_parquet(
            out / 'val_features.parquet', index=False)
    del val_parts
    log(f"pass 2 done: X {X.shape}, positives {int(y.sum()):,} "
        f"({y.mean():.4%}); val rows -> val_features.parquet")

    params = dict(LGB_PARAMS)
    params['n_estimators'] = args.n_estimators
    clf = lgb.LGBMClassifier(**params, n_jobs=args.n_jobs)
    t0 = time.time()
    clf.fit(X, y, feature_name=MODEL_FEATURES)
    log(f"LightGBM trained in {time.time() - t0:.0f}s")
    clf.booster_.save_model(str(model_path))
    imp = dict(zip(MODEL_FEATURES, clf.feature_importances_.tolist()))
    meta = dict(features=MODEL_FEATURES, params=params, val_pct=args.val_pct,
                n_train_pairs=n_train, n_pos=int(y.sum()),
                feature_files=len(files), importances=imp,
                trained=time.strftime('%Y-%m-%d %H:%M:%S'),
                peak_gb=round(resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss / 1e9, 1))
    (MODELS / (args.model_name + '.meta.json')).write_text(
        json.dumps(meta, indent=1))
    log(f"saved {model_path} + meta; top importances: "
        f"{dict(sorted(imp.items(), key=lambda kv: -kv[1])[:6])}")


# ----------------------------------------------------------------- validate
def macro_f05(pred_counts, tp_counts, meta_val):
    """Vectorized macro-F0.5 over ALL validation S1 (misses/zero-cand incl.).
    pred_counts/tp_counts: Series indexed by s1_id (missing => 0)."""
    n_true = meta_val.set_index('s1_id')['n_true']
    npred = pred_counts.reindex(n_true.index).fillna(0).to_numpy(dtype=float)
    tp = tp_counts.reindex(n_true.index).fillna(0).to_numpy(dtype=float)
    nt = n_true.to_numpy(dtype=float)
    score = np.zeros(len(nt))
    score[(npred == 0) & (nt == 0)] = 1.0
    ok = (npred > 0) & (nt > 0) & (tp > 0)
    p, r = tp[ok] / npred[ok], tp[ok] / nt[ok]
    score[ok] = 1.25 * p * r / (0.25 * p + r)
    return float(score.mean())


def stage_validate(args):
    import lightgbm as lgb

    out = Path(args.out)
    booster = lgb.Booster(model_file=str(MODELS / args.model_name))
    val = pd.read_parquet(out / 'val_features.parquet')
    meta = pd.read_parquet(out / 's1_meta.parquet')
    meta['bucket'] = val_bucket(meta['s1_id'].to_numpy())
    meta_val = meta[meta.bucket < args.val_pct]
    log(f"validate: {len(val):,} pairs, {len(meta_val):,} val S1 "
        f"({int((meta_val.n_true == 0).sum()):,} singletons), "
        f"total true links {int(meta_val.n_true.sum()):,}")

    probs = np.empty(len(val), dtype=np.float32)
    for lo in range(0, len(val), 5_000_000):
        probs[lo:lo + 5_000_000] = booster.predict(
            val[MODEL_FEATURES].iloc[lo:lo + 5_000_000].to_numpy(dtype=np.float32))
    val = val[['s1_id', 'cand_id', 'label']].copy()
    val['prob'] = probs

    grid = sorted(set([0.50, 0.55] + list(np.round(
        np.arange(0.60, 0.7501, 0.025), 3)) + [0.80, 0.85]))
    rows = []
    total_true = int(meta_val.n_true.sum())
    for t in grid:
        sel = val[val.prob >= t]
        pc = sel.groupby('s1_id').size()
        tc = sel[sel.label == 1].groupby('s1_id').size()
        mf = macro_f05(pc, tc, meta_val)
        tp = int(sel.label.sum())
        rows.append(dict(rule=f'thr={t}', tp=tp, fp=len(sel) - tp,
                         fn=total_true - tp,
                         precision=round(tp / max(len(sel), 1), 4),
                         recall=round(tp / max(total_true, 1), 4),
                         macro_f05=round(mf, 4)))
        log(f"t={t:5.3f} P={rows[-1]['precision']:.4f} "
            f"R={rows[-1]['recall']:.4f} macroF05={mf:.4f}")
    res = pd.DataFrame(rows)
    best = res.loc[res.macro_f05.idxmax()]
    bt = float(best.rule.split('=')[1])

    # one-to-one at best threshold (keep each cand only for top-prob S1)
    sel = val[val.prob >= bt].sort_values('prob', ascending=False)
    dup = sel.duplicated('cand_id', keep='first')
    sel2 = sel[~dup]
    mf_o2o = macro_f05(sel2.groupby('s1_id').size(),
                       sel2[sel2.label == 1].groupby('s1_id').size(), meta_val)
    rows.append(dict(rule=f'thr={bt}+o2o', tp=int(sel2.label.sum()),
                     fp=int(len(sel2) - sel2.label.sum()),
                     fn=total_true - int(sel2.label.sum()),
                     precision=round(sel2.label.mean(), 4),
                     recall=round(sel2.label.sum() / max(total_true, 1), 4),
                     macro_f05=round(mf_o2o, 4)))
    log(f"BEST t={bt}: macroF05={best.macro_f05:.4f}; o2o dropped "
        f"{int(dup.sum()):,} of {len(sel):,} pairs -> {mf_o2o:.4f}")

    # singletons + country slices at best threshold
    pm = val[val.prob >= bt].groupby('s1_id').size()
    sing = meta_val[meta_val.n_true == 0]
    sing_ok = float((~sing.s1_id.isin(pm.index)).mean()) if len(sing) else 1.0
    log(f"singletons predicted empty correctly: {sing_ok:.4f}")
    cmap = meta_val.set_index('s1_id')['country']
    selb = val[val.prob >= bt]
    for cc, g in meta_val.groupby('country'):
        s = selb[selb.s1_id.map(cmap).eq(cc)]
        pc, tc = s.groupby('s1_id').size(), s[s.label == 1].groupby('s1_id').size()
        log(f"  {cc}: macroF05={macro_f05(pc, tc, g):.4f} over {len(g):,} S1")

    pd.DataFrame(rows).to_csv(
        ROOT / 'experiments' / 'final_validation_metrics.csv', index=False)
    summary = dict(best_threshold=bt, best_macro_f05=float(best.macro_f05),
                   o2o_macro_f05=mf_o2o, o2o_dropped=int(dup.sum()),
                   singleton_correct_empty=sing_ok,
                   n_val_s1=len(meta_val), n_val_pairs=len(val))
    (out / 'validation_summary.json').write_text(json.dumps(summary, indent=1))
    log("saved experiments/final_validation_metrics.csv + validation_summary.json")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('featurize')
    p.add_argument('--cands', nargs='+', required=True,
                   help='shard dirs from generate_candidates.py')
    p.add_argument('--out', required=True)
    p.add_argument('--gt', default=str(GT_DEFAULT))
    p.add_argument('--pair-chunk', type=int, default=1_000_000)
    p.add_argument('--workers', type=int, default=4)
    for name in ('train', 'validate'):
        p = sub.add_parser(name)
        p.add_argument('--out', required=True)
        p.add_argument('--val-pct', type=int, default=10,
                       help='validation entity percentage (MD5 bucket < pct)')
        p.add_argument('--model-name', default='lgbm_final.txt')
        if name == 'train':
            p.add_argument('--n-estimators', type=int,
                           default=LGB_PARAMS['n_estimators'])
            p.add_argument('--n-jobs', type=int, default=8)
            p.add_argument('--force', action='store_true')
    args = ap.parse_args()
    {'featurize': stage_featurize, 'train': stage_train,
     'validate': stage_validate}[args.stage](args)


if __name__ == '__main__':
    main()
