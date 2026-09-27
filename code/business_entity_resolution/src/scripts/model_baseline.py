#!/usr/bin/env python3
"""Small LightGBM research baseline + decision-rule study (Session 2, stage 2).

Uses the sampled pair set from scripts/feature_signal_research.py
(experiments/cache/pairs_features.parquet). Adds the chain-ness feature,
trains a small LightGBM on an entity-level split, sweeps thresholds for
macro-F0.5, and tests per-S1 expected-F0.5 selection and one-to-one
post-processing.

Stages:
    python scripts/model_baseline.py chainness   # S1 name-frequency feature + eval
    python scripts/model_baseline.py train       # LightGBM + thresholds + rules
Read-only w.r.t. dataset/. Research baseline only — not the production pipeline.
"""
import csv
import os
import re
import sys
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

csv.field_size_limit(10**7)
ROOT = os.path.join(os.path.dirname(__file__), '..')
TRAIN = os.path.join(ROOT, 'dataset', 'train')
CACHE = os.path.join(ROOT, 'experiments', 'cache')

SUFFIX = {'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private',
          'co', 'company', 'llp', 'sarl', 'sas', 'sci', 'sa', 'plc', 'and', 'the',
          'of', 'com', 'www', 'pc', 'inc.'}
NON_WORD = re.compile(r'[^a-z0-9ऀ-ൿ ]+')


def norm(s):
    s = unicodedata.normalize('NFKD', s.lower())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return ' '.join(NON_WORD.sub(' ', s).split())


def core_name(s):
    return ' '.join(t for t in norm(s).split() if t not in SUFFIX)


# ------------------------------------------------------------- stage: chainness
def stage_chainness():
    # Frequency of the suffix-stripped normalized name within FULL train S1,
    # overall and per country. Input-data statistic only — no ground truth used.
    cnt, cnt_country = Counter(), Counter()
    with open(os.path.join(TRAIN, 'train_source1.tsv')) as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            cn = core_name(row[1])
            cnt[cn] += 1
            cnt_country[(cn, row[3])] += 1
    freq = pd.DataFrame([(k, v) for k, v in cnt.items()],
                        columns=['core_name', 's1_name_freq'])
    freq.to_parquet(os.path.join(CACHE, 's1_name_freq.parquet'))
    print(f"unique core names in S1: {len(cnt):,}")
    v = np.array(list(cnt.values()))
    print("freq distribution:", {q: int(np.quantile(v, q)) for q in
                                 [0.5, 0.9, 0.99, 0.999]}, "max", v.max())
    print(f"names with freq>1: {(v > 1).sum():,} ({(v > 1).mean():.1%}); "
          f"S1 rows covered by freq>1 names: {v[v > 1].sum() / v.sum():.1%}")

    # attach to sampled pairs and measure association with false positives
    out = pd.read_parquet(os.path.join(CACHE, 'pairs_features.parquet'))
    out['s1_core_name'] = out.s1_name.map(core_name)
    out['s1_name_freq'] = out.s1_core_name.map(cnt).fillna(1).astype(int)
    out['s1_name_freq_log'] = np.log1p(out.s1_name_freq)
    out.to_parquet(os.path.join(CACHE, 'pairs_features2.parquet'))

    from sklearn.metrics import roc_auc_score
    print(f"\nchain-ness alone AUC (expect <0.5, i.e. high freq => negative): "
          f"{roc_auc_score(out.label, out.s1_name_freq_log):.3f}")
    same = out[out.name_core_eq == 1]
    print("\nP(match | identical core name) by S1 name frequency bucket:")
    for lo, hi in [(1, 1), (2, 3), (4, 10), (11, 100), (101, 10**9)]:
        m = same[(same.s1_name_freq >= lo) & (same.s1_name_freq <= hi)]
        if len(m):
            print(f"  freq {lo:>4}-{hi if hi < 10**9 else 'max':>4}: "
                  f"n={len(m):>6}  P(match)={m.label.mean():.3f}")


# ----------------------------------------------------------------- stage: train
FEATURES = [
    # name
    'name_tok_jac', 'name_3gram_jac_translit', 'name_jw_translit',
    'name_squash_cont', 'name_core_eq', 'name_ntok_diff',
    # address
    'addr_tok_cont', 'addr_3gram_jac', 'addr_lev', 'digit_jac',
    'street_num_match', 'locality_overlap', 'addr_missing', 'addr_len_diff',
    # context
    'indic_script', 'src_s3', 'country_india', 's1_name_freq_log', 'n_cands',
]


def macro_f05(val, pred_col):
    """Per-S1 F0.5 averaged over entities (empty-empty = 1.0)."""
    scores = []
    for _, g in val.groupby('s1_id', sort=False):
        true = set(g.loc[g.label == 1, 'cand_id'])
        pred = set(g.loc[g[pred_col] == 1, 'cand_id'])
        if not pred and not true:
            scores.append(1.0)
        elif not pred or not true:
            scores.append(0.0)
        else:
            tp = len(pred & true)
            p, r = tp / len(pred), tp / len(true)
            scores.append(1.25 * p * r / (0.25 * p + r) if tp else 0.0)
    return float(np.mean(scores))


def expected_f05_select(probs):
    """Return boolean mask: prefix of sorted probs maximizing E[F0.5]
    under independence (E[empty] = P(no true matches) approximated by prod(1-p))."""
    order = np.argsort(-probs)
    p = probs[order]
    best_k, best_val = 0, float(np.prod(1 - p))   # E[score | predict empty]
    # E[F0.5 | predict top-k] via simple expectation of tp: sum p_i, approximate
    # F0.5 at expected counts (adequate for a research baseline).
    exp_total = p.sum()
    for k in range(1, len(p) + 1):
        tp = p[:k].sum()
        prec = tp / k
        rec = tp / max(exp_total, 1e-9)
        f = 1.25 * prec * rec / (0.25 * prec + rec) if tp > 0 else 0.0
        # account for the possibility the entity is a true singleton
        f = f * (1 - np.prod(1 - p))
        if f > best_val:
            best_k, best_val = k, f
    mask = np.zeros(len(p), bool)
    mask[order[:best_k]] = True
    return mask


def stage_train():
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, average_precision_score

    out = pd.read_parquet(os.path.join(CACHE, 'pairs_features2.parquet'))
    out['src_s3'] = (out.source == 'S3').astype(int)
    out['country_india'] = (out.s1_country == 'India').astype(int)
    out['n_cands'] = out.groupby('s1_id').s1_id.transform('size')

    # entity-level split: all pairs of an S1 stay on one side
    ents = out.s1_id.drop_duplicates().sample(frac=1, random_state=11)
    val_ents = set(ents[:1200])
    tr = out[~out.s1_id.isin(val_ents)]
    val = out[out.s1_id.isin(val_ents)].copy()
    print(f"train: {len(tr)} pairs / {tr.s1_id.nunique()} entities "
          f"({tr.label.sum()} pos); val: {len(val)} pairs / {val.s1_id.nunique()} "
          f"entities ({val.label.sum()} pos)")

    clf = lgb.LGBMClassifier(n_estimators=400, num_leaves=63, learning_rate=0.08,
                             min_child_samples=40, subsample=0.9,
                             colsample_bytree=0.9, random_state=0, n_jobs=12,
                             verbose=-1)
    clf.fit(tr[FEATURES], tr.label)
    val['prob'] = clf.predict_proba(val[FEATURES])[:, 1]

    print(f"\nROC-AUC: {roc_auc_score(val.label, val.prob):.4f}   "
          f"PR-AUC: {average_precision_score(val.label, val.prob):.4f}")

    # ablation: without chain-ness
    f2 = [f for f in FEATURES if f != 's1_name_freq_log']
    clf2 = lgb.LGBMClassifier(**clf.get_params()).fit(tr[f2], tr.label)
    p2 = clf2.predict_proba(val[f2])[:, 1]
    print(f"without chain-ness: ROC-AUC {roc_auc_score(val.label, p2):.4f}   "
          f"PR-AUC {average_precision_score(val.label, p2):.4f}")

    imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nfeature importances (gain-ranked):")
    print(imp.to_string())

    # ---- threshold sweep ----
    rows = []
    print("\nthreshold sweep (pair-level P/R + entity macro-F0.5):")
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        val['pred'] = (val.prob >= t).astype(int)
        tp = int(((val.pred == 1) & (val.label == 1)).sum())
        fp = int(((val.pred == 1) & (val.label == 0)).sum())
        fn = int(((val.pred == 0) & (val.label == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f05 = 1.25 * prec * rec / max(0.25 * prec + rec, 1e-9)
        mf = macro_f05(val, 'pred')
        rows.append(dict(rule=f'thr={t}', tp=tp, fp=fp, fn=fn,
                         pair_precision=round(prec, 4), pair_recall=round(rec, 4),
                         pair_f05=round(f05, 4), macro_f05=round(mf, 4)))
        print(f"  t={t:.1f}  P={prec:.4f} R={rec:.4f} pairF05={f05:.4f} "
              f"macroF05={mf:.4f}  (tp={tp} fp={fp} fn={fn})")

    # ---- per-S1 expected-F0.5 selection ----
    val['pred_ef'] = 0
    for s1, g in val.groupby('s1_id', sort=False):
        mask = expected_f05_select(g.prob.values)
        val.loc[g.index[mask], 'pred_ef'] = 1
    mf_ef = macro_f05(val, 'pred_ef')
    tp = int(((val.pred_ef == 1) & (val.label == 1)).sum())
    fp = int(((val.pred_ef == 1) & (val.label == 0)).sum())
    fn = int(((val.pred_ef == 0) & (val.label == 1)).sum())
    rows.append(dict(rule='perS1-expectedF05', tp=tp, fp=fp, fn=fn,
                     pair_precision=round(tp / max(tp + fp, 1), 4),
                     pair_recall=round(tp / max(tp + fn, 1), 4),
                     pair_f05='', macro_f05=round(mf_ef, 4)))
    print(f"\nper-S1 expected-F0.5 rule: macroF05={mf_ef:.4f} (tp={tp} fp={fp} fn={fn})")

    # ---- one-to-one post-processing on best threshold ----
    best_t = max((r for r in rows if r['rule'].startswith('thr')),
                 key=lambda r: r['macro_f05'])
    bt = float(best_t['rule'].split('=')[1])
    for base_col, base_name in [('pred_bt', f'thr={bt}'), ('pred_ef', 'perS1-expectedF05')]:
        if base_col == 'pred_bt':
            val['pred_bt'] = (val.prob >= bt).astype(int)
        sel = val[val[base_col] == 1].sort_values('prob', ascending=False)
        keep = set()
        seen_cand = set()
        for t in sel.itertuples():
            key = (t.s1_id, t.cand_id)
            if t.cand_id in seen_cand:
                continue
            seen_cand.add(t.cand_id)
            keep.add(key)
        col = base_col + '_o2o'
        val[col] = [(1 if (a, b) in keep else 0)
                    for a, b in zip(val.s1_id, val.cand_id)] * np.array(val[base_col])
        mf = macro_f05(val, col)
        dropped = int(val[base_col].sum() - val[col].sum())
        rows.append(dict(rule=base_name + ' +one-to-one', tp='', fp='', fn='',
                         pair_precision='', pair_recall='', pair_f05='',
                         macro_f05=round(mf, 4)))
        print(f"one-to-one on {base_name}: macroF05={mf:.4f} "
              f"(pairs dropped by o2o: {dropped})")

    pd.DataFrame(rows).to_csv(
        os.path.join(ROOT, 'experiments', 'model_baseline_metrics.csv'), index=False)

    # ---- slice metrics at best threshold ----
    val['pred'] = (val.prob >= bt).astype(int)
    print(f"\nslices at t={bt} (pair precision / recall / n_pos):")
    slices = {'US': val.s1_country == 'US', 'India': val.s1_country == 'India',
              'S2': val.source == 'S2', 'S3': val.source == 'S3',
              'Indic-script': val.indic_script == 1,
              'ambiguous(nameJ>=.5 & addrC>=.5)':
                  (val.name_tok_jac >= 0.5) & (val.addr_tok_cont >= 0.5)}
    for nm, m in slices.items():
        g = val[m]
        tp = ((g.pred == 1) & (g.label == 1)).sum()
        fp = ((g.pred == 1) & (g.label == 0)).sum()
        fn = ((g.pred == 0) & (g.label == 1)).sum()
        print(f"  {nm:34s} P={tp/max(tp+fp,1):.4f} R={tp/max(tp+fn,1):.4f} "
              f"pos={int(tp+fn)} n={len(g)}")

    # ---- error samples ----
    val['err'] = np.where((val.pred == 1) & (val.label == 0), 'FP',
                 np.where((val.pred == 0) & (val.label == 1), 'FN', 'ok'))
    errs = val[val.err != 'ok'][
        ['s1_id', 'cand_id', 'err', 'prob', 's1_name', 'c_name', 's1_addr',
         'c_addr', 's1_country', 'source', 'name_tok_jac', 'addr_tok_cont',
         'digit_jac', 'indic_script', 's1_name_freq', 'addr_missing']]
    errs.to_csv(os.path.join(ROOT, 'experiments', 'model_error_samples.csv'),
                index=False)
    print(f"\nerrors saved: {len(errs)} rows -> experiments/model_error_samples.csv")
    val.to_parquet(os.path.join(CACHE, 'val_scored.parquet'))


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage in ('chainness', 'all'):
        stage_chainness()
    if stage in ('train', 'all'):
        stage_train()
