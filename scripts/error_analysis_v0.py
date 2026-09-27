#!/usr/bin/env python3
"""Deep error analysis of the LOCKED V0 final model (READ-ONLY on all V0 artifacts).

V0 = models/lgbm_final.txt, threshold 0.625, one-to-one by model score,
19 MODEL_FEATURES, D14 val split (val_pct 10). Nothing here modifies the model,
threshold, blocking, candidates, or any production artifact. TEST is never read.

Stages:
  score    — predict raw probabilities for experiments/train_final/val_features.parquet
             -> experiments/error_analysis/val_probs.npy (row-aligned). Run once.
  analyze  — everything else (thresholds, FP/FN taxonomy, blocking misses,
             feature separation, country slices) -> CSVs + printed report.

Usage:
  .venv312/bin/python scripts/error_analysis_v0.py score
  .venv312/bin/python scripts/error_analysis_v0.py analyze
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from feature_pipeline import FEATURES, log                      # noqa: E402
from train_final_model import (MODEL_FEATURES, macro_f05,       # noqa: E402
                               read_gt, val_bucket)

TF = ROOT / 'experiments' / 'train_final'
EA = ROOT / 'experiments' / 'error_analysis'
CACHE = ROOT / 'experiments' / 'cache'
MODEL = ROOT / 'models' / 'lgbm_final.txt'
GT = ROOT / 'dataset' / 'train' / 'train_ground_truth.tsv'
THR = 0.625                                     # LOCKED production threshold
GRID = [0.50, 0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75]


# -------------------------------------------------------------------- helpers
def load_val():
    val = pd.read_parquet(TF / 'val_features.parquet')
    probs = np.load(EA / 'val_probs.npy')
    assert len(probs) == len(val)
    val['prob'] = probs
    meta = pd.read_parquet(TF / 's1_meta.parquet')
    meta['bucket'] = val_bucket(meta['s1_id'].to_numpy())
    meta_val = meta[meta.bucket < 10].drop(columns='bucket').reset_index(drop=True)
    return val, meta_val


def prep_lookup(ids, table):
    """Row-position lookup of prep records for the given entity ids."""
    idx = pd.Index(table['entity_id'])
    pos = idx.get_indexer(ids)
    assert (pos >= 0).all()
    return table.iloc[pos].reset_index(drop=True)


def f05_stats(sel, meta_val, label='') -> dict:
    pc = sel.groupby('s1_id').size()
    tc = sel[sel.label == 1].groupby('s1_id').size()
    mf = macro_f05(pc, tc, meta_val)
    tp = int(sel.label.sum())
    tot = int(meta_val.n_true.sum())
    sing = meta_val[meta_val.n_true == 0]
    sing_ok = float((~sing.s1_id.isin(pc.index)).mean()) if len(sing) else 1.0
    return dict(rule=label, n_pred=len(sel), tp=tp, fp=len(sel) - tp,
                fn=tot - tp, precision=round(tp / max(len(sel), 1), 4),
                recall=round(tp / max(tot, 1), 4), macro_f05=round(mf, 4),
                singleton_empty_acc=round(sing_ok, 4))


def o2o(sel):
    """Production one-to-one: keep each cand only for its top-prob S1."""
    s = sel.sort_values('prob', ascending=False, kind='stable')
    return s[~s.duplicated('cand_id', keep='first')], s[s.duplicated('cand_id', keep='first')]


def auc(x, y):
    """ROC-AUC of x separating y (1 vs 0) via rank statistic."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --------------------------------------------------------------------- score
def stage_score(_):
    import lightgbm as lgb
    EA.mkdir(parents=True, exist_ok=True)
    booster = lgb.Booster(model_file=str(MODEL))
    assert booster.feature_name() == MODEL_FEATURES
    val = pd.read_parquet(TF / 'val_features.parquet',
                          columns=MODEL_FEATURES)
    log(f"scoring {len(val):,} val pairs with {MODEL.name}")
    probs = np.empty(len(val), dtype=np.float32)
    t0 = time.time()
    for lo in range(0, len(val), 5_000_000):
        probs[lo:lo + 5_000_000] = booster.predict(
            val[MODEL_FEATURES].iloc[lo:lo + 5_000_000].to_numpy(dtype=np.float32))
        log(f"  {min(lo + 5_000_000, len(val)):,} done")
    np.save(EA / 'val_probs.npy', probs)
    log(f"saved val_probs.npy in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------- analyze
def stage_analyze(_):
    t0 = time.time()
    val, meta_val = load_val()
    total_true = int(meta_val.n_true.sum())
    log(f"val: {len(val):,} pairs, {len(meta_val):,} S1, "
        f"{total_true:,} true links, positives in candidates: "
        f"{int(val.label.sum()):,}")
    cmap = meta_val.set_index('s1_id')['country']
    val['country'] = val['s1_id'].map(cmap)

    # ---------------- C. threshold analysis
    rows = []
    for t in GRID:
        rows.append(f05_stats(val[val.prob >= t], meta_val, f'thr={t}'))
    sel625 = val[val.prob >= THR]
    kept, dropped = o2o(sel625)
    rows.append(f05_stats(kept, meta_val, f'thr={THR}+o2o (PRODUCTION V0)'))
    thr_df = pd.DataFrame(rows)
    thr_df.to_csv(EA / 'threshold_analysis.csv', index=False)
    print('\n== C. THRESHOLDS ==')
    print(thr_df.to_string(index=False))

    # per-country at production rule
    print('\n== production rule by country ==')
    ctry_rows = []
    for cc in ('US', 'India'):
        g = meta_val[meta_val.country == cc]
        s = kept[kept.country == cc]
        st = f05_stats(s, g, f'V0 {cc}')
        ctry_rows.append(st)
        print(st)
    pd.DataFrame(ctry_rows).to_csv(EA / 'country_production.csv', index=False)

    # ---------------- oracle upper bounds
    print('\n== ORACLE BOUNDS (what each error class costs) ==')
    oracle = []
    tp_only = kept[kept.label == 1]
    oracle.append(f05_stats(tp_only, meta_val, 'oracle: V0 minus all FPs'))
    all_pos = val[val.label == 1]
    oracle.append(f05_stats(all_pos, meta_val,
                            'oracle: perfect on candidates (blocking ceiling)'))
    fix_fn = pd.concat([kept[kept.label == 1], all_pos]).drop_duplicates(
        ['s1_id', 'cand_id'])
    v0_fp = kept[kept.label == 0]
    oracle.append(f05_stats(pd.concat([fix_fn, v0_fp]), meta_val,
                            'oracle: V0 plus all candidate FNs recovered'))
    for o in oracle:
        print(o)
    pd.DataFrame(oracle).to_csv(EA / 'oracle_bounds.csv', index=False)

    # ---------------- error sets
    fp = kept[kept.label == 0].copy()
    fn1 = val[(val.label == 1) & (val.prob < THR)].copy()
    fn2 = dropped[dropped.label == 1].copy()
    tp = kept[kept.label == 1]
    n_miss = total_true - int(val.label.sum())
    print(f"\nV0 errors: FP={len(fp):,}  FN-lowscore={len(fn1):,}  "
          f"FN-o2o={len(fn2):,}  FN-blockingmiss={n_miss:,} "
          f"({n_miss / total_true:.2%} of true links)")

    prep_s1 = pd.read_parquet(CACHE / 'prep_s1.parquet')
    prep_c = pd.read_parquet(CACHE / 'prep_cand.parquet')

    # ---------------- A. FP taxonomy
    def taxonomy(df, name):
        a = prep_lookup(df['s1_id'].to_numpy(), prep_s1)
        b = prep_lookup(df['cand_id'].to_numpy(), prep_c)
        t = pd.DataFrame({
            'country': df['country'].to_numpy(),
            'src_s3': df['src_s3'].to_numpy(),
            'prob': df['prob'].to_numpy(),
            'exact_core_name': (a.core.to_numpy() == b.core.to_numpy())
                               & (a.core.to_numpy() != ''),
            'name_sim_high': df['name_tok_jac'].to_numpy() >= 0.7,
            'name_sim_low': df['name_tok_jac'].to_numpy() < 0.3,
            'addr_sim_high': df['addr_3gram_jac'].to_numpy() >= 0.5,
            'addr_sim_low': df['addr_3gram_jac'].to_numpy() < 0.2,
            'chain_freq5': df['s1_name_freq_log'].to_numpy() >= np.log1p(5),
            'chain_freq20': df['s1_name_freq_log'].to_numpy() >= np.log1p(20),
            'digit_conflict': df['digit_exact_conflict'].to_numpy() == 1,
            'addr_missing': df['addr_missing'].to_numpy() == 1,
            'short_addr': (b.addr_ntok.to_numpy() <= 3)
                          & (df['addr_missing'].to_numpy() == 0),
            'indic': df['indic_script'].to_numpy() == 1,
            'dba_domain': np.array([('www' in n.split() or 'com' in n.split())
                                    for n in np.concatenate(
                                        [a.nname.to_numpy(), b.nname.to_numpy()])
                                    ]).reshape(2, -1).any(axis=0),
            'marginal_prob': (df['prob'].to_numpy() < 0.75),
        })
        t['s1_nname'] = a.nname.to_numpy()
        t['cand_nname'] = b.nname.to_numpy()
        t['s1_naddr'] = a.naddr.to_numpy()
        t['cand_naddr'] = b.naddr.to_numpy()
        flags = ['exact_core_name', 'name_sim_high', 'name_sim_low',
                 'addr_sim_high', 'addr_sim_low', 'chain_freq5', 'chain_freq20',
                 'digit_conflict', 'addr_missing', 'short_addr', 'indic',
                 'dba_domain', 'marginal_prob']
        rows = []
        for fl in flags:
            m = t[fl].to_numpy()
            rows.append(dict(set=name, category=fl, count=int(m.sum()),
                             pct=round(m.mean() * 100, 1),
                             pct_india=round((t.country[m] == 'India').mean()
                                             * 100, 1) if m.any() else 0.0))
        # compound: name-driven vs address-driven FPs
        nm = t.name_sim_high & t.addr_sim_low
        ad = t.addr_sim_high & t.name_sim_low
        for lab, m in (('COMPOUND name_high_addr_low', nm),
                       ('COMPOUND addr_high_name_low', ad)):
            rows.append(dict(set=name, category=lab, count=int(m.sum()),
                             pct=round(m.mean() * 100, 1),
                             pct_india=round((t.country[m] == 'India').mean()
                                             * 100, 1) if m.any() else 0.0))
        return t, pd.DataFrame(rows)

    fp_t, fp_cat = taxonomy(fp, 'FP')
    fn1_t, fn1_cat = taxonomy(fn1, 'FN_lowscore')
    print('\n== A. FP CATEGORIES (overlapping flags) ==')
    print(fp_cat.to_string(index=False))
    print('\n== B. FN(score<0.625) CATEGORIES ==')
    print(fn1_cat.to_string(index=False))
    pd.concat([fp_cat, fn1_cat]).to_csv(EA / 'error_categories.csv', index=False)

    # FN1 recoverability: prob bands
    bands = pd.cut(fn1['prob'], [0, 0.05, 0.1, 0.25, 0.5, 0.575, 0.625])
    print('\nFN(lowscore) probability bands:')
    print(bands.value_counts().sort_index().to_string())

    # feature means TP vs FP vs FN1
    fm = pd.DataFrame({
        'TP_mean': tp[MODEL_FEATURES].mean(),
        'FP_mean': fp[MODEL_FEATURES].mean(),
        'FN1_mean': fn1[MODEL_FEATURES].mean(),
    })
    fm['auc_tp_vs_fp'] = [auc(pd.concat([tp[f], fp[f]]).to_numpy(),
                              np.r_[np.ones(len(tp)), np.zeros(len(fp))])
                          for f in MODEL_FEATURES]
    fm['auc_tp_vs_fn1'] = [auc(pd.concat([tp[f], fn1[f]]).to_numpy(),
                               np.r_[np.ones(len(tp)), np.zeros(len(fn1))])
                           for f in MODEL_FEATURES]
    fm = fm.round(4)
    fm.to_csv(EA / 'feature_separation.csv')
    print('\n== D. FEATURE SEPARATION (AUC of feature separating TP from errors) ==')
    print(fm.sort_values('auc_tp_vs_fp').to_string())

    # feature correlation (2M sample) on candidates
    samp = val[MODEL_FEATURES].sample(2_000_000, random_state=0)
    corr = samp.corr().round(3)
    corr.to_csv(EA / 'feature_correlation.csv')
    hi = [(a_, b_, corr.loc[a_, b_]) for i, a_ in enumerate(MODEL_FEATURES)
          for b_ in MODEL_FEATURES[i + 1:] if abs(corr.loc[a_, b_]) >= 0.8]
    print('\nhighly correlated pairs (|r|>=0.8):', hi)

    # ---------------- B3. blocking misses
    pos, _ = read_gt(GT)
    vset = set(meta_val.s1_id)
    posv = pos[pos.s1_id.isin(vset)]
    have = val.loc[val.label == 1, ['s1_id', 'cand_id']]
    miss = posv.merge(have, on=['s1_id', 'cand_id'], how='left',
                      indicator=True)
    miss = miss[miss._merge == 'left_only'].drop(columns=['_merge', 'label'])
    assert len(miss) == n_miss, (len(miss), n_miss)
    a = prep_lookup(miss['s1_id'].to_numpy(), prep_s1)
    b = prep_lookup(miss['cand_id'].to_numpy(), prep_c)
    ntl = []
    for x, yv in zip(a.core.to_numpy(), b.core.to_numpy()):
        sx, sy = set(x.split()), set(yv.split())
        sx.discard(''); sy.discard('')
        ntl.append(len(sx & sy) / max(len(sx | sy), 1))
    ntl = np.array(ntl)
    mt = pd.DataFrame({
        'country': miss['s1_id'].map(cmap).to_numpy(),
        'src': [('S3' if e.startswith('S3') else 'S2') for e in miss.cand_id],
        'zero_name_overlap': ntl == 0,
        'cand_addr_empty': (b.naddr.to_numpy() == ''),
        's1_indic': a.indic.to_numpy().astype(bool),
        'cand_indic': b.indic.to_numpy().astype(bool),
        'dba_domain': np.array([('www' in n.split() or 'com' in n.split())
                                for n in np.concatenate(
                                    [a.nname.to_numpy(), b.nname.to_numpy()])
                                ]).reshape(2, -1).any(axis=0),
    })
    # was the S1 partially served anyway (some true pair found)?
    s1_pos_found = have.groupby('s1_id').size()
    mt['s1_has_other_tp_cand'] = miss['s1_id'].map(s1_pos_found).fillna(0).to_numpy() > 0
    mrows = []
    for c in mt.columns:
        if c in ('country', 'src'):
            continue
        m = mt[c].to_numpy()
        mrows.append(dict(set='BLOCKING_MISS', category=c, count=int(m.sum()),
                          pct=round(m.mean() * 100, 1),
                          pct_india=round((mt.country[m] == 'India').mean()
                                          * 100, 1) if m.any() else 0.0))
    mdf = pd.DataFrame(mrows)
    print(f'\n== B3. BLOCKING MISSES ({n_miss:,} = {n_miss / total_true:.2%} '
          f'of true links) ==')
    print('by country:', mt.country.value_counts().to_dict(),
          ' by source:', mt.src.value_counts().to_dict())
    print(mdf.to_string(index=False))
    mdf.to_csv(EA / 'blocking_miss_categories.csv', index=False)
    print('name-overlap jaccard of misses: mean %.3f, median %.3f, ==0: %.1f%%'
          % (ntl.mean(), np.median(ntl), (ntl == 0).mean() * 100))

    # ---------------- E. country slices of error composition
    print('\n== E. US vs INDIA ==')
    erow = []
    for cc in ('US', 'India'):
        g = meta_val[meta_val.country == cc]
        cval = val[val.country == cc]
        ck = kept[kept.country == cc]
        cfp = ck[ck.label == 0]
        cfn1 = fn1[fn1.country == cc]
        cmiss = mt[mt.country == cc]
        ctrue = int(g.n_true.sum())
        erow.append(dict(
            country=cc, n_s1=len(g), true_links=ctrue,
            cands_per_s1=round(len(cval) / len(g), 1),
            fp=len(cfp), fp_rate_pct=round(len(cfp) / max(len(ck), 1) * 100, 2),
            fn_lowscore=len(cfn1),
            fn_lowscore_pct_true=round(len(cfn1) / ctrue * 100, 2),
            blocking_miss=len(cmiss),
            miss_pct_true=round(len(cmiss) / ctrue * 100, 2),
            fp_indic_pct=round(fp_t[fp_t.country == cc].indic.mean() * 100, 1),
            fn1_addrmiss_pct=round(
                fn1_t[fn1_t.country == cc].addr_missing.mean() * 100, 1),
        ))
    edf = pd.DataFrame(erow)
    print(edf.to_string(index=False))
    edf.to_csv(EA / 'country_error_composition.csv', index=False)

    # ---------------- examples (normalized strings; git-ignored dir)
    with open(EA / 'examples.txt', 'w') as fh:
        for name, t in (('FP', fp_t), ('FN_lowscore', fn1_t)):
            for cat in ('exact_core_name', 'digit_conflict', 'addr_missing',
                        'indic', 'dba_domain', 'short_addr',
                        'chain_freq20'):
                sub = t[t[cat]]
                if not len(sub):
                    continue
                fh.write(f"\n===== {name} / {cat} (n={len(sub)}) =====\n")
                for _, r in sub.sample(min(6, len(sub)), random_state=0).iterrows():
                    fh.write(f"p={r.prob:.3f} [{r.country}]\n"
                             f"  S1  : {r.s1_nname[:70]} | {r.s1_naddr[:80]}\n"
                             f"  cand: {r.cand_nname[:70]} | {r.cand_naddr[:80]}\n")
    log(f"analysis done in {time.time() - t0:.0f}s; outputs in {EA}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['score', 'analyze'])
    args = ap.parse_args()
    {'score': stage_score, 'analyze': stage_analyze}[args.stage](args)


if __name__ == '__main__':
    main()
