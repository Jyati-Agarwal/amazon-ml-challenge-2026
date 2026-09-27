#!/usr/bin/env python3
"""Feature-signal research for entity resolution (Session 2).

Builds a sampled research set of true matches + hard negatives from the
training data, computes candidate pairwise features, and emits summary
statistics used by docs/FEATURE_SIGNAL_RESEARCH.md.

Stages (run in order, each caches to experiments/cache/):
    python scripts/feature_signal_research.py sample    # sample S1 + candidate gen
    python scripts/feature_signal_research.py features  # compute pair features
    python scripts/feature_signal_research.py analyze   # per-feature stats -> CSV

Read-only w.r.t. dataset/. Sampled: 4,000 S1 entities, candidates capped at
150 per S1. Designed for 48 GB RAM (peak use well under 10 GB).
"""
import csv
import heapq
import os
import random
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
os.makedirs(CACHE, exist_ok=True)

N_SAMPLE = 4000
CAND_CAP = 150
SEED = 7

SUFFIX = {'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private',
          'co', 'company', 'llp', 'sarl', 'sas', 'sci', 'sa', 'plc', 'and', 'the',
          'of', 'com', 'www', 'pc', 'inc.'}
INDIC = re.compile(r'[ऀ-ൿ]')
NON_WORD = re.compile(r'[^a-z0-9ऀ-ൿ ]+')


def norm(s):
    s = unicodedata.normalize('NFKD', s.lower())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return ' '.join(NON_WORD.sub(' ', s).split())


def core_tokens(s):
    return [t for t in norm(s).split() if t not in SUFFIX]


def squash(s):
    return re.sub(r'[^a-z0-9]', '', norm(s).replace(' ', ''))


def read_rows(path):
    with open(path) as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            yield row


# ---------------------------------------------------------------- stage: sample
def stage_sample():
    rng = random.Random(SEED)
    gt = {}
    with open(os.path.join(TRAIN, 'train_ground_truth.tsv')) as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        all_rows = [(row[0], row[1] if len(row) > 1 else '') for row in r]
    sampled = rng.sample(all_rows, N_SAMPLE)
    gt = {s1: set(m.split(',')) if m else set() for s1, m in sampled}
    print(f"sampled {len(gt)} S1 entities; "
          f"{sum(len(v) for v in gt.values())} true matches; "
          f"{sum(1 for v in gt.values() if not v)} singletons")
    del all_rows

    s1rec = {}
    for row in read_rows(os.path.join(TRAIN, 'train_source1.tsv')):
        if row[0] in gt:
            s1rec[row[0]] = row
    s1_ids = list(gt)

    # pass 1: token document frequencies over S2+S3
    name_df, addr_df = Counter(), Counter()
    for src in ('train_source2.tsv', 'train_source3.tsv'):
        for row in read_rows(os.path.join(TRAIN, src)):
            name_df.update(set(norm(row[1]).split()))
            if len(row) > 2 and row[2]:
                addr_df.update(set(norm(row[2]).split()))
    print(f"DF built: {len(name_df)} name tokens, {len(addr_df)} addr tokens")

    # postings: indexed tokens of sampled S1 records
    name_post, addr_post = defaultdict(list), defaultdict(list)
    squash_post = defaultdict(list)
    for i, s1 in enumerate(s1_ids):
        row = s1rec[s1]
        for t in set(core_tokens(row[1])):
            if name_df[t] <= 3000:
                name_post[t].append(i)
        for t in set(norm(row[2]).split()):
            if addr_df[t] <= 2000:
                addr_post[t].append(i)
        sq = squash(row[1])
        if len(sq) > 6:
            squash_post[sq].append(i)

    # pass 2: candidate generation over S2+S3
    heaps = [[] for _ in s1_ids]          # per-S1 heap of (overlap, cand_id)
    pos_found = set()
    gt_ids = {m for v in gt.values() for m in v}
    for src in ('train_source2.tsv', 'train_source3.tsv'):
        for row in read_rows(os.path.join(TRAIN, src)):
            rid = row[0]
            hits = Counter()
            rare = set()
            ntoks = set(norm(row[1]).split()) - SUFFIX
            for t in ntoks:
                for i in name_post.get(t, ()):
                    hits[i] += 1
                    if name_df[t] <= 300:
                        rare.add(i)
            atoks = set(norm(row[2]).split()) if len(row) > 2 and row[2] else set()
            for t in atoks:
                for i in addr_post.get(t, ()):
                    hits[i] += 1
                    if addr_df[t] <= 300:
                        rare.add(i)
            sq = squash(row[1])
            if len(sq) > 6 and sq in squash_post:
                for i in squash_post[sq]:
                    hits[i] += 2
            if rid in gt_ids:
                pos_found.add(rid)
            for i, c in hits.items():
                if c >= 2 or i in rare:
                    h = heaps[i]
                    if len(h) < CAND_CAP:
                        heapq.heappush(h, (c, rid))
                    elif c > h[0][0]:
                        heapq.heapreplace(h, (c, rid))
    pairs = []
    for i, s1 in enumerate(s1_ids):
        cands = {rid for _, rid in heaps[i]}
        blocked = cands & gt[s1]
        for rid in cands | gt[s1]:          # always include true matches
            pairs.append((s1, rid, int(rid in gt[s1]), int(rid in blocked)))
    dfp = pd.DataFrame(pairs, columns=['s1_id', 'cand_id', 'label', 'found_by_blocking'])
    print(f"pairs: {len(dfp)}; positives: {dfp.label.sum()}; "
          f"pos found by blocking: {dfp[dfp.label == 1].found_by_blocking.mean():.3f}")

    # pass 3: fetch candidate texts
    need = set(dfp.cand_id)
    rec = {}
    for src in ('train_source2.tsv', 'train_source3.tsv'):
        for row in read_rows(os.path.join(TRAIN, src)):
            if row[0] in need:
                rec[row[0]] = row
    for col, idx in (('name', 1), ('addr', 2), ('country', 3)):
        dfp[f's1_{col}'] = dfp.s1_id.map(lambda k: s1rec[k][idx])
        dfp[f'c_{col}'] = dfp.cand_id.map(lambda k: rec[k][idx] if len(rec[k]) > idx else '')
    dfp['source'] = dfp.cand_id.str[:2]
    dfp.to_parquet(os.path.join(CACHE, 'pairs_raw.parquet'))
    print("saved pairs_raw.parquet")


# -------------------------------------------------------------- stage: features
def stage_features():
    from rapidfuzz import distance, fuzz
    from unidecode import unidecode

    dfp = pd.read_parquet(os.path.join(CACHE, 'pairs_raw.parquet'))

    def ngrams(s, n=3):
        s = s.replace(' ', '')
        return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s} if s else set()

    def jac(a, b):
        if not a or not b:
            return 0.0
        a, b = set(a), set(b)
        return len(a & b) / len(a | b)

    def containment(a, b):
        if not a or not b:
            return 0.0
        a, b = set(a), set(b)
        return len(a & b) / min(len(a), len(b))

    def pincode(addr, country):
        pat = r'\b\d{6}\b' if country == 'India' else r'\b\d{5}\b'
        m = re.findall(pat, addr)
        return m[-1] if m else ''

    def digit_toks(addr):
        return {t for t in norm(addr).split() if t.isdigit()}

    def alpha_toks(addr):
        return [t for t in norm(addr).split() if t.isalpha()]

    rows = []
    for t in dfp.itertuples(index=False):
        n1, n2 = t.s1_name, t.c_name
        a1, a2 = t.s1_addr, t.c_addr
        nn1, nn2 = norm(n1), norm(n2)
        ct1, ct2 = core_tokens(n1), core_tokens(n2)
        na1, na2 = norm(a1), norm(a2)
        at1, at2 = na1.split(), na2.split()
        sq1, sq2 = squash(n1), squash(n2)
        indic = bool(INDIC.search(n2))
        n2t = unidecode(n2) if indic else n2       # transliterated candidate name
        nt2t = core_tokens(n2t)

        pin1 = pincode(a1, t.s1_country)
        pin2 = pincode(a2, t.s1_country)
        d1, d2 = digit_toks(a1), digit_toks(a2)
        d1x, d2x = d1 - {pin1}, d2 - {pin2}
        loc1, loc2 = alpha_toks(a1)[-3:], alpha_toks(a2)[-3:]

        f = dict(
            # --- name ---
            name_exact_norm=float(nn1 == nn2 and nn1 != ''),
            name_core_eq=float(bool(ct1) and sorted(ct1) == sorted(ct2)),
            name_tok_jac=jac(ct1, ct2),
            name_tok_cont=containment(ct1, ct2),
            name_lev=distance.Levenshtein.normalized_similarity(nn1, nn2),
            name_jw=distance.JaroWinkler.similarity(nn1, nn2),
            name_3gram_jac=jac(ngrams(nn1), ngrams(nn2)),
            name_squash_cont=float(bool(sq1) and bool(sq2) and
                                   (sq1 in sq2 or sq2 in sq1)),
            name_acronym=float(len(sq2) >= 2 and
                               sq2 == ''.join(w[0] for w in ct1) or
                               (len(sq1) >= 2 and sq1 == ''.join(w[0] for w in ct2))),
            name_len_diff=abs(len(nn1) - len(nn2)) / max(len(nn1), len(nn2), 1),
            name_ntok_diff=abs(len(ct1) - len(ct2)),
            # --- transliteration ---
            indic_script=float(indic),
            name_tok_jac_translit=jac(ct1, nt2t),
            name_3gram_jac_translit=jac(ngrams(nn1), ngrams(norm(n2t))),
            name_jw_translit=distance.JaroWinkler.similarity(nn1, norm(n2t)),
            # --- address ---
            addr_exact_norm=float(na1 == na2 and na1 != ''),
            addr_tok_jac=jac(at1, at2),
            addr_tok_cont=containment(at1, at2),
            addr_3gram_jac=jac(ngrams(na1), ngrams(na2)),
            addr_lev=distance.Levenshtein.normalized_similarity(na1, na2),
            addr_missing=float(not a2.strip() or not a1.strip()),
            pin_present_both=float(bool(pin1) and bool(pin2)),
            pin_eq=float(bool(pin1) and pin1 == pin2),
            pin_prefix3=float(bool(pin1) and bool(pin2) and pin1[:3] == pin2[:3]),
            street_num_match=containment(d1x, d2x),
            digit_jac=jac(d1, d2),
            locality_overlap=jac(loc1, loc2),
            addr_len_diff=abs(len(na1) - len(na2)) / max(len(na1), len(na2), 1),
            # --- combined / other ---
            country_eq=float(t.s1_country == t.c_country),
            name_addr_prod=0.0,        # filled below
            name_addr_mean=0.0,
        )
        f['name_addr_prod'] = f['name_tok_jac'] * f['addr_tok_jac']
        f['name_addr_mean'] = 0.5 * (f['name_tok_jac'] + f['addr_tok_jac'])
        rows.append(f)

    feat = pd.DataFrame(rows)
    out = pd.concat([dfp.reset_index(drop=True), feat], axis=1)

    # word TF-IDF cosine (names, addresses) over the sample universe
    from sklearn.feature_extraction.text import TfidfVectorizer
    for col, s1c, cc in (('name_tfidf_cos', 's1_name', 'c_name'),
                         ('addr_tfidf_cos', 's1_addr', 'c_addr')):
        corpus = pd.concat([out[s1c], out[cc]]).map(norm)
        v = TfidfVectorizer(analyzer='word').fit(corpus)
        A = v.transform(out[s1c].map(norm))
        B = v.transform(out[cc].map(norm))
        out[col] = np.asarray(A.multiply(B).sum(axis=1)).ravel()
    out.to_parquet(os.path.join(CACHE, 'pairs_features.parquet'))
    print(f"features computed for {len(out)} pairs -> pairs_features.parquet")


# --------------------------------------------------------------- stage: analyze
def stage_analyze():
    from sklearn.metrics import roc_auc_score

    out = pd.read_parquet(os.path.join(CACHE, 'pairs_features.parquet'))
    meta = ['s1_id', 'cand_id', 'label', 'found_by_blocking', 's1_name', 'c_name',
            's1_addr', 'c_addr', 's1_country', 'c_country', 'source']
    feats = [c for c in out.columns if c not in meta]
    y = out.label.values

    recs = []
    for f in feats:
        x = out[f].fillna(0).values
        try:
            auc = roc_auc_score(y, x)
        except ValueError:
            auc = float('nan')
        # precision if used alone at a "high" cutoff (>= 0.9 of its scale, or ==1 for flags)
        hi = x >= (0.9 if x.max() <= 1 else np.quantile(x, 0.99))
        prec_hi = y[hi].mean() if hi.sum() else float('nan')
        cov_hi = hi.mean()
        recs.append(dict(feature=f, auc=round(auc, 4),
                         pos_mean=round(x[y == 1].mean(), 4),
                         neg_mean=round(x[y == 0].mean(), 4),
                         prec_at_high=round(prec_hi, 4) if prec_hi == prec_hi else '',
                         cov_at_high=round(cov_hi, 4)))
    res = pd.DataFrame(recs).sort_values('auc', ascending=False)
    res.to_csv(os.path.join(ROOT, 'experiments', 'feature_signal_results.csv'), index=False)
    print(res.to_string(index=False))
    print(f"\nbase rate P(match) in sample: {y.mean():.4f}   pairs: {len(y)}")


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage in ('sample', 'all'):
        stage_sample()
    if stage in ('features', 'all'):
        stage_features()
    if stage in ('analyze', 'all'):
        stage_analyze()
