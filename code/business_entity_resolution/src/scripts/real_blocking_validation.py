#!/usr/bin/env python3
"""End-to-end TRAIN validation on REAL blocking candidates (Session 2, stage 3).

Reuses the Session-1 production-style blocker output
(experiments/blocking/cand_sample/{word,char}_{US,India}.parquet — top-200 cosine
ranks per channel for 20k stratified S1 vs the FULL same-country S2+S3 corpus),
assembles the locked union spec (BLOCKING_IMPLEMENTATION_PLAN.md: US w50+c50+exact,
default w100+c100+exact, exact-name block cap 200), then:
features -> LightGBM -> isotonic calibration -> threshold grid -> per-S1 rule ->
one-to-one -> error analysis. TRAIN data only; no test files touched.

Stages: candidates | features | train   (each caches to experiments/cache/)
"""
import csv
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

csv.field_size_limit(10**7)
ROOT = os.path.join(os.path.dirname(__file__), '..')
TRAIN = os.path.join(ROOT, 'dataset', 'train')
CACHE = os.path.join(ROOT, 'experiments', 'cache')
BLK = os.path.join(ROOT, 'experiments', 'blocking', 'cand_sample')

BUDGETS = {'US': (50, 50)}
DEFAULT_BUDGET = (100, 100)
EXACT_CAP = 200
SEED = 21
VAL_FRAC = 0.30

SUFFIX = {'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private',
          'co', 'company', 'llp', 'sarl', 'sas', 'sci', 'sa', 'plc', 'and', 'the',
          'of', 'com', 'www', 'pc'}
INDIC = re.compile(r'[ऀ-ൿ]')


def block_norm(s):
    """Blocking-plan normalization: latin accent strip, lower, punct->space."""
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if not (unicodedata.category(c) == 'Mn' and ord(c) < 0x0900))
    s = unicodedata.normalize('NFC', s).lower()
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def norm(s):
    s = unicodedata.normalize('NFKD', s.lower())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return ' '.join(re.sub(r'[^a-z0-9ऀ-ൿ ]+', ' ', s).split())


def core_tokens(s):
    return [t for t in norm(s).split() if t not in SUFFIX]


def core_name(s):
    return ' '.join(core_tokens(s))


def squash(s):
    return re.sub(r'[^a-z0-9]', '', norm(s).replace(' ', ''))


def read_rows(path):
    with open(path) as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        yield from r


# ------------------------------------------------------------ stage: candidates
def stage_candidates():
    # union of word/char channels at spec budgets
    parts = []
    for country in ('US', 'India'):
        kw, kc = BUDGETS.get(country, DEFAULT_BUDGET)
        w = pd.read_parquet(os.path.join(BLK, f'word_{country}.parquet'))
        c = pd.read_parquet(os.path.join(BLK, f'char_{country}.parquet'))
        w = w[w['rank'] < kw].rename(columns={'rank': 'word_rank'})
        c = c[c['rank'] < kc].rename(columns={'rank': 'char_rank'})
        u = w.merge(c, on=['s1_id', 'cand_id'], how='outer')
        u['country'] = country
        parts.append(u)
    cand = pd.concat(parts, ignore_index=True)
    s1_ids = set(cand.s1_id)
    print(f"retrieval union: {len(cand):,} pairs for {len(s1_ids):,} S1")

    # exact normalized-name channel (cap per name block), against full S2+S3
    s1_name_map = defaultdict(list)          # block_norm(name) -> [s1_id]
    s1rec = {}
    for row in read_rows(os.path.join(TRAIN, 'train_source1.tsv')):
        if row[0] in s1_ids:
            s1rec[row[0]] = row
            s1_name_map[(block_norm(row[1]), row[3])].append(row[0])
    exact_pairs = defaultdict(list)          # s1_id -> [cand_id]
    for src in ('train_source2.tsv', 'train_source3.tsv'):
        for row in read_rows(os.path.join(TRAIN, src)):
            key = (block_norm(row[1]), row[3])
            if key in s1_name_map:
                for s1 in s1_name_map[key]:
                    if len(exact_pairs[s1]) < EXACT_CAP:
                        exact_pairs[s1].append(row[0])
    ex = pd.DataFrame([(s1, c) for s1, lst in exact_pairs.items() for c in lst],
                      columns=['s1_id', 'cand_id'])
    ex['exact_name'] = 1
    cand = cand.merge(ex, on=['s1_id', 'cand_id'], how='outer')
    cand['exact_name'] = cand.exact_name.fillna(0).astype(int)
    cand['word_rank'] = cand.word_rank.fillna(-1).astype(int)
    cand['char_rank'] = cand.char_rank.fillna(-1).astype(int)
    cand['country'] = cand.groupby('s1_id').country.transform('first')
    cand['country'] = cand.country.fillna(
        cand.s1_id.map(lambda k: s1rec[k][3] if k in s1rec else None))

    # labels from GT + blocking-miss accounting
    gt = {}
    for row in read_rows(os.path.join(TRAIN, 'train_ground_truth.tsv')):
        if row[0] in s1_ids:
            gt[row[0]] = set(row[1].split(',')) if len(row) > 1 and row[1] else set()
    cand['label'] = [int(c in gt[s]) for s, c in zip(cand.s1_id, cand.cand_id)]

    total_true = sum(len(v) for v in gt.values())
    found = cand.groupby('s1_id').apply(
        lambda g: g.label.sum(), include_groups=False)
    pair_recall = cand.label.sum() / total_true
    ent_all = np.mean([found.get(s, 0) == len(gt[s]) for s in gt if gt[s]])
    percs = cand.groupby('s1_id').size()
    print(f"REAL BLOCKING: pair recall {pair_recall:.4f}; entity all-match recall "
          f"{ent_all:.4f}; cand/S1 mean {percs.mean():.1f} median "
          f"{percs.median():.0f} p95 {percs.quantile(.95):.0f} max {percs.max()}")
    for cc in ('US', 'India'):
        sub = cand[cand.country == cc]
        tt = sum(len(gt[s]) for s in gt if s1rec[s][3] == cc)
        print(f"  {cc}: pair recall {sub.label.sum()/tt:.4f}; "
              f"cand/S1 {sub.groupby('s1_id').size().mean():.1f}")
    for pfx in ('S2', 'S3'):
        tt = sum(1 for v in gt.values() for m in v if m.startswith(pfx))
        ff = cand[cand.cand_id.str.startswith(pfx)].label.sum()
        print(f"  {pfx}: pair recall {ff/tt:.4f}")

    # store misses (needed for honest validation FN accounting)
    found_map = defaultdict(set)
    for s, c, l in zip(cand.s1_id, cand.cand_id, cand.label):
        if l:
            found_map[s].add(c)
    miss = [(s, m) for s, v in gt.items() for m in v if m not in found_map[s]]
    pd.DataFrame(miss, columns=['s1_id', 'missed_id']).to_parquet(
        os.path.join(CACHE, 'rb_blocking_misses.parquet'))
    cand.to_parquet(os.path.join(CACHE, 'rb_candidates.parquet'))
    pd.Series({s: len(v) for s, v in gt.items()}).rename('n_true').to_frame() \
        .to_parquet(os.path.join(CACHE, 'rb_gt_counts.parquet'))
    print(f"saved: {len(cand):,} candidate pairs, {len(miss):,} blocking misses")


# -------------------------------------------------------------- stage: features
def stage_features():
    from rapidfuzz import distance
    from unidecode import unidecode

    cand = pd.read_parquet(os.path.join(CACHE, 'rb_candidates.parquet'))
    need_s1, need_c = set(cand.s1_id), set(cand.cand_id)
    rec = {}
    for fn, need in (('train_source1.tsv', need_s1),
                     ('train_source2.tsv', need_c), ('train_source3.tsv', need_c)):
        for row in read_rows(os.path.join(TRAIN, fn)):
            if row[0] in need:
                rec[row[0]] = (row[1], row[2] if len(row) > 2 else '')
    print(f"texts fetched: {len(rec):,}")

    freq = pd.read_parquet(os.path.join(CACHE, 's1_name_freq.parquet'))
    freq_map = dict(zip(freq.core_name, freq.s1_name_freq))

    def ngrams(s, n=3):
        s = s.replace(' ', '')
        return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else ({s} if s else set())

    def jac(a, b):
        if not a or not b:
            return 0.0
        a, b = set(a), set(b)
        return len(a & b) / len(a | b)

    def cont(a, b):
        if not a or not b:
            return 0.0
        a, b = set(a), set(b)
        return len(a & b) / min(len(a), len(b))

    # per-S1 precomputation
    s1_cache = {}
    for s1 in need_s1:
        n1, a1 = rec[s1]
        nn1 = norm(n1)
        ct1 = core_tokens(n1)
        na1 = norm(a1)
        at1 = na1.split()
        s1_cache[s1] = (nn1, ct1, na1, at1, squash(n1),
                        {t for t in at1 if t.isdigit()},
                        [t for t in at1 if t.isalpha()][-3:],
                        float(np.log1p(freq_map.get(' '.join(ct1), 1))))

    rows = np.empty((len(cand), 18), dtype=np.float32)
    cols = ['addr_tok_cont', 'addr_3gram_jac', 'addr_lev', 'digit_jac',
            'name_tok_jac', 'name_3gram_jac_translit', 'name_jw_translit',
            'name_squash_cont', 'name_ntok_diff', 'addr_len_diff',
            'locality_overlap', 'street_num_match', 'addr_missing',
            'indic_script', 's1_name_freq_log', 'digit_exact_conflict',
            'src_s3', 'country_india']
    lev = distance.Levenshtein.normalized_similarity
    jw = distance.JaroWinkler.similarity
    for i, t in enumerate(cand.itertuples(index=False)):
        nn1, ct1, na1, at1, sq1, d1, loc1, fql = s1_cache[t.s1_id]
        n2, a2 = rec[t.cand_id]
        nn2 = norm(n2)
        ct2 = core_tokens(n2)
        na2 = norm(a2)
        at2 = na2.split()
        indic = bool(INDIC.search(n2))
        nn2t = norm(unidecode(n2)) if indic else nn2
        ct2t = [x for x in nn2t.split() if x not in SUFFIX] if indic else ct2
        sq2 = squash(n2)
        d2 = {x for x in at2 if x.isdigit()}
        loc2 = [x for x in at2 if x.isalpha()][-3:]
        rows[i] = (
            cont(at1, at2), jac(ngrams(na1), ngrams(na2)), lev(na1, na2),
            jac(d1, d2), jac(ct1, ct2),
            jac(ngrams(nn1), ngrams(nn2t)), jw(nn1, nn2t),
            float(bool(sq1) and bool(sq2) and len(min(sq1, sq2, key=len)) > 6
                  and (sq1 in sq2 or sq2 in sq1)),
            abs(len(ct1) - len(ct2)),
            abs(len(na1) - len(na2)) / max(len(na1), len(na2), 1),
            jac(loc1, loc2), cont(d1, d2),
            float(not na1 or not na2), float(indic), fql,
            float(bool(d1) and bool(d2) and not (d1 & d2)),
            float(t.cand_id.startswith('S3')), float(t.country == 'India'))
        if i % 500000 == 0:
            print(f"  {i:,}/{len(cand):,}")
    feat = pd.DataFrame(rows, columns=cols)
    out = pd.concat([cand.reset_index(drop=True), feat], axis=1)
    out['n_cands'] = out.groupby('s1_id').s1_id.transform('size').astype(np.float32)
    out['s1_name'] = out.s1_id.map(lambda k: rec[k][0])
    out['s1_addr'] = out.s1_id.map(lambda k: rec[k][1])
    out['c_name'] = out.cand_id.map(lambda k: rec[k][0])
    out['c_addr'] = out.cand_id.map(lambda k: rec[k][1])
    out.to_parquet(os.path.join(CACHE, 'rb_features.parquet'))
    print(f"features for {len(out):,} pairs saved")


# ----------------------------------------------------------------- stage: train
FEATURES = ['addr_tok_cont', 'addr_3gram_jac', 'addr_lev', 'digit_jac',
            'name_tok_jac', 'name_3gram_jac_translit', 'name_jw_translit',
            'name_squash_cont', 'name_ntok_diff', 'addr_len_diff',
            'locality_overlap', 'street_num_match', 'addr_missing',
            'indic_script', 's1_name_freq_log', 'digit_exact_conflict',
            'src_s3', 'country_india', 'n_cands']


def macro_f05_sets(pred_map, gt_counts, label_map):
    """pred_map: s1 -> set(pred ids); label_map: s1 -> set(true ids in/out of cands);
    gt_counts includes blocked-missed truths (denominator of recall)."""
    sc = []
    for s1, n_true in gt_counts.items():
        pred = pred_map.get(s1, set())
        true = label_map.get(s1, set())
        if not pred and n_true == 0:
            sc.append(1.0)
        elif not pred or n_true == 0:
            sc.append(0.0)
        else:
            tp = len(pred & true)
            p, r = tp / len(pred), tp / n_true
            sc.append(1.25 * p * r / (0.25 * p + r) if tp else 0.0)
    return float(np.mean(sc))


def stage_train():
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score, average_precision_score

    out = pd.read_parquet(os.path.join(CACHE, 'rb_features.parquet'))
    gtc = pd.read_parquet(os.path.join(CACHE, 'rb_gt_counts.parquet'))
    gt_counts_all = gtc.n_true.to_dict()

    rng = np.random.RandomState(SEED)
    ents = pd.Series(sorted(set(out.s1_id)))
    ents = ents.sample(frac=1, random_state=SEED).tolist()
    n_val = int(len(ents) * VAL_FRAC)
    val_ents = set(ents[:n_val])
    fit_cal = ents[n_val:]
    cal_ents = set(fit_cal[:int(len(fit_cal) * 0.2)])   # calibration fold
    fit_ents = set(fit_cal[int(len(fit_cal) * 0.2):])

    tr = out[out.s1_id.isin(fit_ents)]
    ca = out[out.s1_id.isin(cal_ents)]
    val = out[out.s1_id.isin(val_ents)].copy()
    print(f"fit {len(fit_ents)} S1 / {len(tr):,} pairs ({tr.label.sum():,} pos); "
          f"cal {len(cal_ents)} S1 / {len(ca):,} pairs; "
          f"val {len(val_ents)} S1 / {len(val):,} pairs ({val.label.sum():,} pos)")
    vc = out[out.s1_id.isin(val_ents)].groupby('s1_id').first().country
    print(f"val country: {vc.value_counts().to_dict()}")

    clf = lgb.LGBMClassifier(n_estimators=500, num_leaves=63, learning_rate=0.07,
                             min_child_samples=60, subsample=0.9,
                             colsample_bytree=0.9, random_state=0, n_jobs=12,
                             verbose=-1)
    clf.fit(tr[FEATURES], tr.label)
    val['prob_raw'] = clf.predict_proba(val[FEATURES])[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(clf.predict_proba(ca[FEATURES])[:, 1], ca.label)
    val['prob_cal'] = iso.predict(val.prob_raw)

    print(f"\nraw:  ROC {roc_auc_score(val.label, val.prob_raw):.4f}  "
          f"PR {average_precision_score(val.label, val.prob_raw):.4f}")
    print(f"cal:  ROC {roc_auc_score(val.label, val.prob_cal):.4f}  "
          f"PR {average_precision_score(val.label, val.prob_cal):.4f}")
    bins = pd.cut(val.prob_raw, [0, .1, .3, .5, .7, .9, 1.0])
    caltab = val.groupby(bins, observed=True).agg(
        raw=('prob_raw', 'mean'), cal=('prob_cal', 'mean'),
        actual=('label', 'mean'), n=('label', 'size'))
    print("\ncalibration table (raw bin -> mean raw / mean cal / actual):")
    print(caltab.to_string())
    caltab.to_csv(os.path.join(ROOT, 'experiments', 'calibration_results.csv'))

    # entity bookkeeping for macro-F0.5 (blocking misses count as FN via n_true)
    gt_counts = {s: gt_counts_all[s] for s in val_ents}
    label_map = defaultdict(set)
    for s, c, l in zip(val.s1_id, val.cand_id, val.label):
        if l:
            label_map[s].add(c)

    imp = pd.Series(clf.feature_importances_, index=FEATURES) \
        .sort_values(ascending=False)
    print("\ntop feature importances:", dict(imp.head(8)))

    def eval_threshold(pcol, t):
        sel = val[val[pcol] >= t]
        pred_map = defaultdict(set)
        for s, c in zip(sel.s1_id, sel.cand_id):
            pred_map[s].add(c)
        mf = macro_f05_sets(pred_map, gt_counts, label_map)
        tp = int(sel.label.sum())
        fp = len(sel) - tp
        fn = sum(gt_counts.values()) - tp
        n_zero = sum(1 for s in val_ents if not pred_map.get(s))
        n_multi = sum(1 for s in val_ents if len(pred_map.get(s, ())) > 1)
        return dict(tp=tp, fp=fp, fn=fn,
                    precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                    macro_f05=mf, avg_pred=len(sel) / len(val_ents),
                    n_zero_pred=n_zero, n_multi_pred=n_multi), pred_map

    grid = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65,
            0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    rows = []
    for pcol in ('prob_raw', 'prob_cal'):
        print(f"\nthreshold sweep [{pcol}]:")
        for t in grid:
            m, _ = eval_threshold(pcol, t)
            rows.append(dict(prob=pcol, rule=f'thr={t}', **{k: (round(v, 4)
                        if isinstance(v, float) else v) for k, v in m.items()}))
            print(f"  t={t:.2f} P={m['precision']:.4f} R={m['recall']:.4f} "
                  f"macroF05={m['macro_f05']:.4f} avg_pred={m['avg_pred']:.2f} "
                  f"zero={m['n_zero_pred']} multi={m['n_multi_pred']}")

    # per-S1 expected-F0.5 rule (calibrated probs)
    def exp_f05_rule(pcol):
        pred_map = {}
        for s1, g in val.groupby('s1_id', sort=False):
            p = np.sort(g[pcol].values)[::-1]
            ids = g.cand_id.values[np.argsort(-g[pcol].values)]
            best_k, best = 0, float(np.prod(1 - p))
            exp_tot = p.sum()
            for k in range(1, len(p) + 1):
                tp = p[:k].sum()
                pr, rc = tp / k, tp / max(exp_tot, 1e-9)
                f = 1.25 * pr * rc / (0.25 * pr + rc) * (1 - np.prod(1 - p))
                if f > best:
                    best_k, best = k, f
            pred_map[s1] = set(ids[:best_k])
        return pred_map

    for pcol in ('prob_raw', 'prob_cal'):
        pm = exp_f05_rule(pcol)
        mf = macro_f05_sets(pm, gt_counts, label_map)
        npred = sum(len(v) for v in pm.values())
        tp = sum(len(pm[s] & label_map.get(s, set())) for s in pm)
        rows.append(dict(prob=pcol, rule='perS1-expF05', tp=tp, fp=npred - tp,
                         fn=sum(gt_counts.values()) - tp,
                         precision=round(tp / max(npred, 1), 4),
                         recall=round(tp / max(sum(gt_counts.values()), 1), 4),
                         macro_f05=round(mf, 4), avg_pred=round(npred / len(val_ents), 2),
                         n_zero_pred=sum(1 for v in pm.values() if not v),
                         n_multi_pred=sum(1 for v in pm.values() if len(v) > 1)))
        print(f"per-S1 expF05 [{pcol}]: macroF05={mf:.4f}")

    # one-to-one on the best config
    res = pd.DataFrame(rows)
    best = res.loc[res.macro_f05.idxmax()]
    print(f"\nBEST so far: {best['prob']} {best['rule']} macroF05={best.macro_f05}")
    bt = float(best['rule'].split('=')[1]) if 'thr' in best['rule'] else None
    pcol = best['prob']
    if bt is None:
        pred_map = exp_f05_rule(pcol)
        sel = val[[a in pred_map.get(s, set())
                   for s, a in zip(val.s1_id, val.cand_id)]].copy()
    else:
        sel = val[val[pcol] >= bt].copy()
    # o2o: keep each cand_id only for its highest-prob S1
    sel = sel.sort_values(pcol, ascending=False)
    dup_mask = sel.duplicated('cand_id', keep='first')
    print(f"one-to-one conflicts among selected pairs: {int(dup_mask.sum())} "
          f"of {len(sel)}")
    pm_o2o = defaultdict(set)
    for s, c in zip(sel[~dup_mask].s1_id, sel[~dup_mask].cand_id):
        pm_o2o[s].add(c)
    mf_o2o = macro_f05_sets(pm_o2o, gt_counts, label_map)
    pm_base = defaultdict(set)
    for s, c in zip(sel.s1_id, sel.cand_id):
        pm_base[s].add(c)
    mf_base = macro_f05_sets(pm_base, gt_counts, label_map)
    print(f"best config without o2o: {mf_base:.4f}; with o2o: {mf_o2o:.4f}")
    rows.append(dict(prob=pcol, rule=str(best['rule']) + '+o2o', tp='', fp='', fn='',
                     precision='', recall='', macro_f05=round(mf_o2o, 4),
                     avg_pred='', n_zero_pred='', n_multi_pred=''))
    pd.DataFrame(rows).to_csv(os.path.join(
        ROOT, 'experiments', 'real_blocking_validation_metrics.csv'), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(
        ROOT, 'experiments', 'threshold_results.csv'), index=False)

    # slices at best config
    val['pred'] = (val[pcol] >= bt).astype(int) if bt is not None else \
        [int(a in pred_map.get(s, set())) for s, a in zip(val.s1_id, val.cand_id)]
    print("\nslices at best config (pair P/R):")
    for nm, m in {'US': val.country == 'US', 'India': val.country == 'India',
                  'S2': val.src_s3 == 0, 'S3': val.src_s3 == 1,
                  'Indic': val.indic_script == 1,
                  'ambig': (val.name_tok_jac >= .5) & (val.addr_tok_cont >= .5)}.items():
        g = val[m]
        tp = ((g.pred == 1) & (g.label == 1)).sum()
        fp = ((g.pred == 1) & (g.label == 0)).sum()
        fn = ((g.pred == 0) & (g.label == 1)).sum()
        print(f"  {nm:6s} P={tp/max(tp+fp,1):.4f} R={tp/max(tp+fn,1):.4f} "
              f"pos={int(tp+fn)}")
    # singletons
    sing = [s for s in val_ents if gt_counts[s] == 0]
    pm = defaultdict(set)
    for s, c in zip(val[val.pred == 1].s1_id, val[val.pred == 1].cand_id):
        pm[s].add(c)
    print(f"singletons: {len(sing)}; predicted empty correctly: "
          f"{np.mean([not pm.get(s) for s in sing]):.3f}")

    # error export
    val['err'] = np.where((val.pred == 1) & (val.label == 0), 'FP',
                 np.where((val.pred == 0) & (val.label == 1), 'FN', 'ok'))
    errs = val[val.err != 'ok'][
        ['s1_id', 'cand_id', 'err', pcol, 's1_name', 'c_name', 's1_addr', 'c_addr',
         'country', 'src_s3', 'name_tok_jac', 'addr_tok_cont', 'digit_jac',
         'digit_exact_conflict', 'indic_script', 's1_name_freq_log', 'addr_missing']]
    errs.to_csv(os.path.join(ROOT, 'experiments', 'model_error_samples.csv'),
                index=False)
    val.to_parquet(os.path.join(CACHE, 'rb_val_scored.parquet'))
    print(f"\nerrors: {len(errs)} -> experiments/model_error_samples.csv")


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage in ('candidates', 'all'):
        stage_candidates()
    if stage in ('features', 'all'):
        stage_features()
    if stage in ('train', 'all'):
        stage_train()
