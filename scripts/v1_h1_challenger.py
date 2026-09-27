#!/usr/bin/env python3
"""H1 CHALLENGER experiment: soft digit/house-number similarity features.

V0 (models/lgbm_final.txt, t=0.625, o2o, D14 split) is LOCKED and never touched.
This script trains a challenger = V0's exact methodology + 2 extra features,
both derived ONLY from the already-cached prep digit-token sets
(experiments/cache/prep_*.parquet 'digits' column — same preprocessing as V0):

  digit_lev_best   — max over cross-pairs of S1/candidate digit tokens of
                     Levenshtein normalized similarity (captures 1-digit typos:
                     "4804" vs "2804" -> 0.75). 0 if either side has no digits.
  digit_prefix_cont — 1 if any cross-pair where one token is a proper prefix
                     of the other (shorter side >= 2 chars; captures
                     truncations: "3859" vs "385"). Else 0.

These are the H1 features proposed in experiments/error_analysis/ERROR_ANALYSIS.md
(the third idea there — |a-b| of LEADING house numbers — is not implementable
from the cached sorted digit SETS without changing preprocessing, so it is
deliberately out of scope).

Everything identical to V0 otherwise: same 255.7M candidate pairs (feature
parquets under experiments/train_final/features/), same D14 MD5 split
(val_pct 10), same LGB_PARAMS, same validation (grid + t=0.625 + one-to-one).

Stages:
  featurize — per existing feature file, compute the 2 H1 columns (row-aligned)
              -> experiments/v1_h1/h1cols/<shard>/<file>. Resumable.
  train     — 2-pass train exactly like train_final_model.stage_train but with
              21 features -> models/lgbm_v1_h1.txt (+ meta). Never overwrites V0.
  validate  — identical procedure -> experiments/v1_h1/{threshold_analysis.csv,
              validation_summary.json, val_probs.npy}
  compare   — side-by-side V0 vs H1 at t=0.625+o2o, digit-conflict FN recovery,
              new-FP accounting -> experiments/v1_h1/compare.json
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from feature_pipeline import log                                # noqa: E402
from train_final_model import (LGB_PARAMS, MODEL_FEATURES,      # noqa: E402
                               macro_f05, val_bucket)

TF = ROOT / 'experiments' / 'train_final'
V1 = ROOT / 'experiments' / 'v1_h1'
CACHE = ROOT / 'experiments' / 'cache'
EA = ROOT / 'experiments' / 'error_analysis'
V0_MODEL = ROOT / 'models' / 'lgbm_final.txt'          # read-only, never written
H1_MODEL = ROOT / 'models' / 'lgbm_v1_h1.txt'
H1_COLS = ['digit_lev_best', 'digit_prefix_cont']
H1_FEATURES = MODEL_FEATURES + H1_COLS
THR = 0.625
GRID = [0.50, 0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75]


def cumsum0(x):
    out = np.zeros(len(x) + 1, dtype=np.int64)
    np.cumsum(x, out=out[1:])
    return out


def _explode_cap(strings, n, cap=6):
    """digit-set strings -> (row_ids, tokens) with <=cap tokens per row."""
    s = pd.Series(strings).str.split(' ').explode()
    s = s[s != ''].dropna()
    rows = s.index.to_numpy(dtype=np.int64)
    toks = s.to_numpy()
    cnt = np.bincount(rows, minlength=n)
    starts = cumsum0(cnt)
    within = np.arange(len(rows)) - starts[rows]
    keep = within < cap
    return rows[keep], toks[keep]


def h1_features(dig_a, dig_b, rf_workers=8):
    """The 2 H1 features for aligned arrays of digit-set strings."""
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein
    n = len(dig_a)
    out_lev = np.zeros(n, dtype=np.float32)
    out_pre = np.zeros(n, dtype=np.float32)
    ra, ta = _explode_cap(dig_a, n)
    rb, tb = _explode_cap(dig_b, n)
    na = np.bincount(ra, minlength=n)
    nb = np.bincount(rb, minlength=n)
    sa, sb = cumsum0(na), cumsum0(nb)
    cross = (na * nb).astype(np.int64)
    total = int(cross.sum())
    if total == 0:
        return out_lev, out_pre
    row_of = np.repeat(np.arange(n, dtype=np.int64), cross)
    cstart = cumsum0(cross)
    k = np.arange(total, dtype=np.int64) - cstart[row_of]
    nb_r = nb[row_of]
    A = ta[sa[row_of] + k // nb_r]
    B = tb[sb[row_of] + k % nb_r]
    scores = process.cpdist(A, B, scorer=Levenshtein.normalized_similarity,
                            workers=rf_workers).astype(np.float32)
    AU = A.astype('U18')
    BU = B.astype('U18')
    la = np.char.str_len(AU)
    lb = np.char.str_len(BU)
    pre = (np.char.startswith(BU, AU) | np.char.startswith(AU, BU)) \
        & (np.minimum(la, lb) >= 2) & (la != lb)
    rows_nz = np.nonzero(cross > 0)[0]
    seg = cstart[rows_nz]
    out_lev[rows_nz] = np.maximum.reduceat(scores, seg)
    out_pre[rows_nz] = np.maximum.reduceat(pre.astype(np.float32), seg)
    return out_lev, out_pre


def paired_files():
    """[(base_feat_file, h1_file)] over the V0 feature parquets, sorted."""
    base = sorted((TF / 'features').rglob('feat_*.parquet'))
    out = []
    for f in base:
        rel = f.relative_to(TF / 'features')
        out.append((f, V1 / 'h1cols' / rel))
    return out


# ------------------------------------------------------------------ featurize
def stage_featurize(args):
    """Default (no args): TRAIN overlay files, exactly as used for H1 training.
    For TEST inference pass --features-dir/--cache-dir/--out plus --merge:
    --merge writes FULL parquets (all base columns + the 2 H1 columns) so
    predict_test.py can score an H1 booster from a single --features dir."""
    cache = Path(args.cache_dir)
    feats_dir = Path(args.features_dir)
    out_dir = Path(args.out)
    dig_s1 = pd.read_parquet(cache / 'prep_s1.parquet',
                             columns=['entity_id', 'digits'])
    dig_c = pd.read_parquet(cache / 'prep_cand.parquet',
                            columns=['entity_id', 'digits'])
    idx_s1 = pd.Index(dig_s1['entity_id'])
    idx_c = pd.Index(dig_c['entity_id'])
    dg_s1 = dig_s1['digits'].to_numpy()
    dg_c = dig_c['digits'].to_numpy()
    t0, total = time.time(), 0
    for base in sorted(feats_dir.rglob('feat_*.parquet')):
        h1f = out_dir / base.relative_to(feats_dir)
        if h1f.exists():
            continue
        h1f.parent.mkdir(parents=True, exist_ok=True)
        pairs = pd.read_parquet(base) if args.merge else \
            pd.read_parquet(base, columns=['s1_id', 'cand_id'])
        lev = np.empty(len(pairs), dtype=np.float32)
        pre = np.empty(len(pairs), dtype=np.float32)
        for lo in range(0, len(pairs), args.chunk):
            hi = min(lo + args.chunk, len(pairs))
            ap = idx_s1.get_indexer(pairs['s1_id'].iloc[lo:hi])
            bp = idx_c.get_indexer(pairs['cand_id'].iloc[lo:hi])
            assert (ap >= 0).all() and (bp >= 0).all()
            lev[lo:hi], pre[lo:hi] = h1_features(dg_s1[ap], dg_c[bp],
                                                 args.rf_workers)
        tmp = h1f.with_suffix('.tmp')
        if args.merge:
            pairs['digit_lev_best'] = lev
            pairs['digit_prefix_cont'] = pre
            pairs.to_parquet(tmp, index=False)
        else:
            pd.DataFrame({'digit_lev_best': lev,
                          'digit_prefix_cont': pre}).to_parquet(tmp, index=False)
        tmp.rename(h1f)
        total += len(pairs)
        log(f"{h1f.parent.name}/{h1f.name}: {len(pairs):,} rows "
            f"(cum {total:,}, {total / (time.time() - t0):,.0f} rows/s)")
    log(f"featurize done: {total:,} new rows in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------- train
def stage_train(args):
    import lightgbm as lgb
    import pyarrow.parquet as pq
    assert not H1_MODEL.exists() or args.force, \
        f"{H1_MODEL} exists — pass --force"
    files = paired_files()
    for _, h1f in files:
        assert h1f.exists(), f"missing {h1f} — run featurize first"

    masks, n_train = [], 0
    for base, _ in files:
        ids = pq.read_table(base, columns=['s1_id'])['s1_id'].to_numpy(
            zero_copy_only=False)
        m = val_bucket(ids) >= 10                       # same D14 split as V0
        masks.append(m)
        n_train += int(m.sum())
    log(f"pass 1: {len(files)} files, {n_train:,} train / "
        f"{sum(len(m) for m in masks) - n_train:,} val pairs")

    X = np.empty((n_train, len(H1_FEATURES)), dtype=np.float32)
    y = np.empty(n_train, dtype=np.uint8)
    val_parts, pos_at = [], 0
    for (base, h1f), m in zip(files, masks):
        df = pd.read_parquet(base,
                             columns=['s1_id', 'cand_id', 'label']
                             + MODEL_FEATURES)
        h1 = pd.read_parquet(h1f)
        assert len(h1) == len(df)
        for c in H1_COLS:
            df[c] = h1[c].to_numpy()
        fx = df[H1_FEATURES].to_numpy(dtype=np.float32)
        k = int(m.sum())
        X[pos_at:pos_at + k] = fx[m]
        y[pos_at:pos_at + k] = df['label'].to_numpy()[m]
        pos_at += k
        if (~m).any():
            val_parts.append(df.loc[~m])
        del df, fx, h1
    V1.mkdir(parents=True, exist_ok=True)
    pd.concat(val_parts, ignore_index=True).to_parquet(
        V1 / 'val_features.parquet', index=False)
    del val_parts
    log(f"pass 2 done: X {X.shape}, positives {int(y.sum()):,}")

    clf = lgb.LGBMClassifier(**LGB_PARAMS, n_jobs=8)    # identical D9 config
    t0 = time.time()
    clf.fit(X, y, feature_name=H1_FEATURES)
    log(f"trained in {time.time() - t0:.0f}s")
    clf.booster_.save_model(str(H1_MODEL))
    imp = dict(zip(H1_FEATURES, clf.feature_importances_.tolist()))
    (ROOT / 'models' / (H1_MODEL.name + '.meta.json')).write_text(json.dumps(
        dict(features=H1_FEATURES, params=LGB_PARAMS, val_pct=10,
             n_train_pairs=n_train, n_pos=int(y.sum()), importances=imp,
             baseline='V0 models/lgbm_final.txt (LOCKED, unchanged)',
             trained=time.strftime('%Y-%m-%d %H:%M:%S'),
             peak_gb=round(resource.getrusage(
                 resource.RUSAGE_SELF).ru_maxrss / 1e9, 1)), indent=1))
    log(f"saved {H1_MODEL}; H1 importances: "
        f"{ {c: imp[c] for c in H1_COLS} }; top: "
        f"{dict(sorted(imp.items(), key=lambda kv: -kv[1])[:5])}")


# ------------------------------------------------------------------- validate
def o2o(sel):
    s = sel.sort_values('prob', ascending=False, kind='stable')
    dup = s.duplicated('cand_id', keep='first')
    return s[~dup], s[dup]


def f05_stats(sel, meta_val, label):
    pc = sel.groupby('s1_id').size()
    tc = sel[sel.label == 1].groupby('s1_id').size()
    tp = int(sel.label.sum())
    tot = int(meta_val.n_true.sum())
    sing = meta_val[meta_val.n_true == 0]
    sing_ok = float((~sing.s1_id.isin(pc.index)).mean()) if len(sing) else 1.0
    return dict(rule=label, n_pred=len(sel), tp=tp, fp=len(sel) - tp,
                fn=tot - tp, precision=round(tp / max(len(sel), 1), 4),
                recall=round(tp / max(tot, 1), 4),
                macro_f05=round(macro_f05(pc, tc, meta_val), 4),
                singleton_empty_acc=round(sing_ok, 4))


def load_meta_val():
    meta = pd.read_parquet(TF / 's1_meta.parquet')
    meta['bucket'] = val_bucket(meta['s1_id'].to_numpy())
    return meta[meta.bucket < 10].reset_index(drop=True)


def stage_validate(_):
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(H1_MODEL))
    assert booster.feature_name() == H1_FEATURES
    val = pd.read_parquet(V1 / 'val_features.parquet')
    meta_val = load_meta_val()
    probs = np.empty(len(val), dtype=np.float32)
    for lo in range(0, len(val), 5_000_000):
        probs[lo:lo + 5_000_000] = booster.predict(
            val[H1_FEATURES].iloc[lo:lo + 5_000_000].to_numpy(dtype=np.float32))
    np.save(V1 / 'val_probs.npy', probs)
    val['prob'] = probs
    rows = [f05_stats(val[val.prob >= t], meta_val, f'thr={t}') for t in GRID]
    kept, _ = o2o(val[val.prob >= THR])
    rows.append(f05_stats(kept, meta_val, f'thr={THR}+o2o (H1 challenger)'))
    df = pd.DataFrame(rows)
    df.to_csv(V1 / 'threshold_analysis.csv', index=False)
    print(df.to_string(index=False))
    cmap = load_meta_val().set_index('s1_id')['country']
    for cc in ('US', 'India'):
        g = meta_val[meta_val.country == cc]
        s = kept[kept.s1_id.map(cmap).eq(cc)]
        print(f05_stats(s, g, f'H1 {cc} @{THR}+o2o'))
    (V1 / 'validation_summary.json').write_text(
        df.to_json(orient='records', indent=1))


def stage_compare(_):
    v0 = pd.read_parquet(TF / 'val_features.parquet',
                         columns=['s1_id', 'cand_id', 'label',
                                  'digit_exact_conflict'])
    v0['prob'] = np.load(EA / 'val_probs.npy')
    h1 = pd.read_parquet(V1 / 'val_features.parquet',
                         columns=['s1_id', 'cand_id', 'label',
                                  'digit_exact_conflict',
                                  'digit_lev_best', 'digit_prefix_cont'])
    h1['prob'] = np.load(V1 / 'val_probs.npy')
    assert len(v0) == len(h1)
    assert (v0.s1_id.to_numpy() == h1.s1_id.to_numpy()).all()
    assert (v0.cand_id.to_numpy() == h1.cand_id.to_numpy()).all()
    meta_val = load_meta_val()
    cmap = meta_val.set_index('s1_id')['country']

    res = {}
    sets = {}
    for name, df in (('V0', v0), ('H1', h1)):
        kept, _ = o2o(df[df.prob >= THR])
        st = f05_stats(kept, meta_val, f'{name} @{THR}+o2o')
        for cc in ('US', 'India'):
            g = meta_val[meta_val.country == cc]
            s = kept[kept.s1_id.map(cmap).eq(cc)]
            st[f'{cc}_macro_f05'] = f05_stats(s, g, cc)['macro_f05']
        res[name] = st
        sets[name] = kept
        print(st)

    key = ['s1_id', 'cand_id']
    v0k = sets['V0'].set_index(key)
    h1k = sets['H1'].set_index(key)
    v0_fn1_dc = v0[(v0.label == 1) & (v0.prob < THR)
                   & (v0.digit_exact_conflict == 1)].set_index(key)
    recovered = v0_fn1_dc.index.isin(h1k.index).sum()
    v0_fp = v0k[v0k.label == 0]
    h1_fp = h1k[h1k.label == 0]
    new_fp = int((~h1_fp.index.isin(v0_fp.index)).sum())
    fixed_fp = int((~v0_fp.index.isin(h1_fp.index)).sum())
    new_tp = int((~h1k[h1k.label == 1].index.isin(
        v0k[v0k.label == 1].index)).sum())
    lost_tp = int((~v0k[v0k.label == 1].index.isin(
        h1k[h1k.label == 1].index)).sum())
    out = dict(
        v0=res['V0'], h1=res['H1'],
        delta_macro_f05=round(res['H1']['macro_f05'] - res['V0']['macro_f05'], 4),
        delta_precision=round(res['H1']['precision'] - res['V0']['precision'], 4),
        delta_recall=round(res['H1']['recall'] - res['V0']['recall'], 4),
        delta_us=round(res['H1']['US_macro_f05'] - res['V0']['US_macro_f05'], 4),
        delta_india=round(res['H1']['India_macro_f05']
                          - res['V0']['India_macro_f05'], 4),
        v0_digit_conflict_fns=int(len(v0_fn1_dc)),
        digit_conflict_fns_recovered=int(recovered),
        new_fps_introduced=new_fp, v0_fps_fixed=fixed_fp,
        new_tps=new_tp, tps_lost=lost_tp)
    (V1 / 'compare.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['featurize', 'train', 'validate',
                                      'compare'])
    ap.add_argument('--chunk', type=int, default=2_000_000)
    ap.add_argument('--rf-workers', type=int, default=8)
    ap.add_argument('--force', action='store_true')
    # featurize-only overrides (defaults reproduce the TRAIN H1 run exactly)
    ap.add_argument('--features-dir', default=str(TF / 'features'),
                    help='dir of base feat_*.parquet (TEST: experiments/test_features)')
    ap.add_argument('--cache-dir', default=str(CACHE),
                    help='prep cache with prep_s1/prep_cand (TEST: experiments/cache_test)')
    ap.add_argument('--out', default=str(V1 / 'h1cols'),
                    help='output dir (TEST: experiments/test_features_h1 with --merge)')
    ap.add_argument('--merge', action='store_true',
                    help='write full parquets (base cols + 2 H1 cols) for predict_test')
    args = ap.parse_args()
    {'featurize': stage_featurize, 'train': stage_train,
     'validate': stage_validate, 'compare': stage_compare}[args.stage](args)


if __name__ == '__main__':
    main()
