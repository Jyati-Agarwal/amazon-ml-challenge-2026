#!/usr/bin/env python3
"""Production feature computation for the pairwise matching model (Session 2).

Computes the locked feature set (docs/MODEL_FEATURE_SPEC.md: 16 KEEP +
digit_exact_conflict + src_s3 + country_india + n_cands) plus the stage-3
candidate feature cand_addr_ntok (candidate-address token count).

Design for 200M+ pairs (never loads all pairs into RAM):
  1. `prep`     — per-RECORD normalization done ONCE (not per pair): S1 and
                  S2+S3 records -> derived-string parquet tables. This removes
                  ~90% of the old per-pair cost (norm/core_tokens/squash were
                  recomputed for every pair).
  2. FeatureComputer.compute(pairs_chunk) — joins derived strings by position
                  (pd.Index.get_indexer, hash table built once), then:
                  - rapidfuzz.process.cpdist (C++, multithreaded) for
                    addr_lev + name_jw_translit
                  - numpy vectorized for all length/flag/frequency features
                  - the 9 token/ngram set features fully vectorized: tokens/
                    3-grams factorized to integer ids per slice, per-pair
                    intersections via scipy CSR multiply (no Python loop).
  3. `features` — stream a candidate parquet file/dir in --pair-chunk chunks,
                  write feature parquet parts + manifest (resumable).
  4. `bench`    — old per-pair loop vs new implementation on the validated
                  20k-sample artifacts; correctness vs rb_features.parquet.

Feature semantics are byte-for-byte the validated stage-3 implementation
(scripts/real_blocking_validation.py stage_features); `bench` asserts equality.

Usage (.venv312):
  python scripts/feature_pipeline.py prep --workers 4
  python scripts/feature_pipeline.py features --pairs <file-or-dir> --out <dir>
  python scripts/feature_pipeline.py bench --workers 4
"""
import argparse
import json
import os
import re
import resource
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "dataset" / "train"
CACHE = ROOT / "experiments" / "cache"
PREP_S1 = CACHE / "prep_s1.parquet"
PREP_CAND = CACHE / "prep_cand.parquet"


def set_cache_dir(path):
    """Redirect ALL cache artifacts (prep parquets + setmat npz) to `path`.

    The default (experiments/cache) holds the TRAIN caches. Any non-train run
    (e.g. TEST inference) MUST pass --cache-dir <other dir>: the set matrices
    are row-aligned to the prep tables they were built with, so loading a
    cached setmat against a different prep table silently produces garbage
    features. Isolating the whole cache directory makes that impossible.
    """
    global CACHE, PREP_S1, PREP_CAND
    CACHE = Path(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    PREP_S1 = CACHE / "prep_s1.parquet"
    PREP_CAND = CACHE / "prep_cand.parquet"

SUFFIX = {'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private',
          'co', 'company', 'llp', 'sarl', 'sas', 'sci', 'sa', 'plc', 'and', 'the',
          'of', 'com', 'www', 'pc'}
INDIC = re.compile(r'[ऀ-ൿ]')
NONNORM = re.compile(r'[^a-z0-9ऀ-ൿ ]+')
NONALNUM = re.compile(r'[^a-z0-9]')

# computed feature columns, fixed order (19 validated + cand_addr_ntok; the
# model excludes cand_addr_ntok — see MODEL_FEATURES in train_final_model.py)
FEATURES = ['addr_tok_cont', 'addr_3gram_jac', 'addr_lev', 'digit_jac',
            'name_tok_jac', 'name_3gram_jac_translit', 'name_jw_translit',
            'name_squash_cont', 'name_ntok_diff', 'addr_len_diff',
            'locality_overlap', 'street_num_match', 'addr_missing',
            'indic_script', 's1_name_freq_log', 'digit_exact_conflict',
            'src_s3', 'country_india', 'n_cands', 'cand_addr_ntok']


def log(*a):
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    print("[%s] (peak %.1fGB)" % (time.strftime("%H:%M:%S"), mem), *a, flush=True)


# ------------------------------------------------------- record normalization
# EXACT copies of the validated stage-3 functions (real_blocking_validation.py)
def norm(s):
    s = unicodedata.normalize('NFKD', s.lower())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return ' '.join(NONNORM.sub(' ', s).split())


def core_tokens(s):
    return [t for t in norm(s).split() if t not in SUFFIX]


def squash(s):
    return NONALNUM.sub('', norm(s).replace(' ', ''))


def derive_records(names, addrs, is_cand):
    """One record -> derived strings. Runs in worker processes for `prep`."""
    from unidecode import unidecode
    out = {k: [] for k in ('nname', 'core', 'naddr', 'digits', 'loc', 'squash',
                           'n_core', 'alen', 'addr_ntok', 'indic')}
    for nm, ad in zip(names, addrs):
        nn = norm(nm)
        ct = [t for t in nn.split() if t not in SUFFIX]
        na = norm(ad)
        at = na.split()
        indic = bool(INDIC.search(nm))
        if is_cand and indic:
            nn_t = norm(unidecode(nm))          # translit view for 3gram/JW
        else:
            nn_t = nn
        out['nname'].append(nn_t if is_cand else nn)
        out['core'].append(' '.join(ct))
        out['naddr'].append(na)
        out['digits'].append(' '.join(sorted({t for t in at if t.isdigit()})))
        out['loc'].append(' '.join([t for t in at if t.isalpha()][-3:]))
        out['squash'].append(NONALNUM.sub('', nn.replace(' ', '')))
        out['n_core'].append(len(ct))
        out['alen'].append(len(na))
        out['addr_ntok'].append(len(at))
        out['indic'].append(int(indic))
    return out


def _prep_one(path, is_cand, workers, chunk=400_000):
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, quoting=3,
                     usecols=['entity_id', 'business_name', 'business_address'])
    names = df['business_name'].to_numpy()
    addrs = df['business_address'].to_numpy()
    parts = []
    ctx = get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = [ex.submit(derive_records, names[i:i + chunk].tolist(),
                          addrs[i:i + chunk].tolist(), is_cand)
                for i in range(0, len(df), chunk)]
        for f in futs:
            parts.append(pd.DataFrame(f.result()))
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, 'entity_id', df['entity_id'].to_numpy())
    for c, t in (('n_core', np.int16), ('alen', np.int32),
                 ('addr_ntok', np.int16), ('indic', np.uint8)):
        out[c] = out[c].astype(t)
    return out


def stage_prep(args):
    """Build prep_s1.parquet / prep_cand.parquet + S1 core-name frequency."""
    t0 = time.time()
    s1 = _prep_one(args.s1, is_cand=False, workers=args.workers)
    # chain-ness over the FULL S1 input of this split (leakage-free: inputs only)
    freq = s1['core'].value_counts()
    s1['freq_log'] = np.log1p(
        s1['core'].map(freq).fillna(1).to_numpy()).astype(np.float32)
    s1.to_parquet(PREP_S1, index=False)
    log(f"prep S1: {len(s1):,} records -> {PREP_S1.name} "
        f"({time.time() - t0:.0f}s)")
    parts = []
    for p in (args.s2, args.s3):
        t = time.time()
        d = _prep_one(p, is_cand=True, workers=args.workers)
        log(f"prep {Path(p).name}: {len(d):,} records ({time.time() - t:.0f}s)")
        parts.append(d)
    cand = pd.concat(parts, ignore_index=True)
    cand.to_parquet(PREP_CAND, index=False)
    log(f"prep cand: {len(cand):,} records -> {PREP_CAND.name}; "
        f"total {time.time() - t0:.0f}s")


# ------------------------------------------------- vectorized set operations
def _ngrams(s, n=3):
    """Reference scalar implementation (kept for the old-impl benchmark)."""
    s = s.replace(' ', '')
    if len(s) >= n:
        return {s[i:i + n] for i in range(len(s) - n + 1)}
    return {s} if s else set()


def _token_rowcodes(strings, offset=0):
    """Space-joined token strings -> (row_ids, token_strings) exploded arrays.
    offset shifts row ids (used to stack S1 and cand uniques in one vocab)."""
    ser = pd.Series(strings)
    ex = ser.str.split(' ').explode()
    ex = ex[ex.notna() & (ex != '')]
    return ex.index.to_numpy() + offset, ex.to_numpy()


def _gram_rowcodes(strings, offset=0):
    """Char-3-gram codes per string (spaces removed), vectorized via UTF-32.

    Gram id packs 3 codepoints into an int64 (21 bits each). Strings of
    length 1-2 (after space removal) contribute a single short-string code,
    matching the scalar _ngrams fallback {s}. Empty strings contribute
    nothing. Codes cannot collide across lengths: 3-gram codes are >= 2^42,
    2-gram codes in [2^21, 2^42), 1-gram codes < 2^21, and \\x00 never
    appears inside a gram."""
    big = ('\x00'.join(strings)).replace(' ', '')
    a = np.frombuffer(big.encode('utf-32-le'), dtype=np.uint32).astype(np.int64)
    if len(a) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    row = np.cumsum(a == 0)                       # seps strictly before pos
    if len(a) >= 3:
        ok = (a[:-2] != 0) & (a[1:-1] != 0) & (a[2:] != 0)
        codes = ((a[:-2] << 42) | (a[1:-1] << 21) | a[2:])[ok]
        rows = row[:-2][ok]
    else:
        codes = np.empty(0, np.int64)
        rows = np.empty(0, np.int64)
    # short strings (1-2 chars): one code from the whole string
    z = np.flatnonzero(a == 0)
    starts = np.r_[0, z + 1]
    ends = np.r_[z, len(a)]
    lens = ends - starts
    short = np.flatnonzero((lens > 0) & (lens < 3))
    if len(short):
        c1 = a[starts[short]]
        c2 = np.where(lens[short] == 2, a[np.minimum(starts[short] + 1,
                                                     len(a) - 1)], 0)
        sc = np.where(lens[short] == 1, c1, (c1 << 21) | c2)
        rows = np.concatenate([rows, short])
        codes = np.concatenate([codes, sc])
    return rows + offset, codes


# stacked row-set matrices: rows 0..n_s1-1 = S1 records, then cand records
SET_FIELDS = (('naddr', 'tok'), ('naddr', 'gram'), ('digits', 'tok'),
              ('core', 'tok'), ('nname', 'gram'), ('loc', 'tok'))


def build_setmat(strings, kind, block=2_000_000):
    """All records' token/3-gram SETS as one binary CSR (record x vocab-id).
    Built once per field, cached to disk; duplicate (row, id) entries are
    merged by the COO->CSR conversion (set semantics)."""
    from scipy.sparse import csr_matrix
    rows_l, codes_l = [], []
    for lo in range(0, len(strings), block):
        chunk = strings[lo:lo + block]
        if kind == 'tok':
            r, toks = _token_rowcodes(chunk, offset=lo)
            rows_l.append(r)
            codes_l.append(toks)
        else:
            r, g = _gram_rowcodes(chunk, offset=lo)
            rows_l.append(r)
            codes_l.append(g)
    rows = np.concatenate(rows_l)
    raw = np.concatenate(codes_l)
    del rows_l, codes_l
    codes, uniq = pd.factorize(raw)
    del raw
    M = csr_matrix((np.ones(len(rows), np.int8),
                    (rows, codes.astype(np.int64))),
                   shape=(len(strings), max(len(uniq), 1)))
    M.data[:] = 1
    return M


def _pair_inter(M, a_rows, b_rows):
    """Per-pair |A∩B|, |A|, |B| for row pairs of a stacked row-set CSR."""
    A = M[a_rows]
    B = M[b_rows]
    inter = A.multiply(B).getnnz(axis=1).astype(np.float32)
    na = np.diff(A.indptr).astype(np.float32)
    nb = np.diff(B.indptr).astype(np.float32)
    return inter, na, nb


def _jac(inter, na, nb):
    return np.where((na > 0) & (nb > 0),
                    inter / np.maximum(na + nb - inter, 1), 0.0)


def _cont(inter, na, nb):
    return np.where((na > 0) & (nb > 0),
                    inter / np.maximum(np.minimum(na, nb), 1), 0.0)


# ------------------------------------------------------------ FeatureComputer
class FeatureComputer:
    """Chunk-oriented featurizer. Loads the prep tables once; call
    compute(pairs_df) per chunk. Thread/process pools are per-call."""

    def __init__(self, prep_s1=None, prep_cand=None, workers=4,
                 rf_workers=None, set_slice=500_000):
        # resolve at call time so set_cache_dir() (--cache-dir) takes effect
        prep_s1 = prep_s1 or PREP_S1
        prep_cand = prep_cand or PREP_CAND
        self.workers = workers
        self.rf_workers = rf_workers or workers
        self.set_slice = set_slice
        self.s1 = pd.read_parquet(prep_s1)
        self.s1_idx = pd.Index(self.s1['entity_id'])
        self.cand = pd.read_parquet(prep_cand)
        self.cand_idx = pd.Index(self.cand['entity_id'])
        self._s1_cols = {c: self.s1[c].to_numpy()
                         for c in ('nname', 'core', 'naddr', 'digits', 'loc',
                                   'squash', 'n_core', 'alen', 'freq_log')}
        self._c_cols = {c: self.cand[c].to_numpy()
                        for c in ('nname', 'core', 'naddr', 'digits', 'loc',
                                  'squash', 'n_core', 'alen', 'addr_ntok',
                                  'indic')}
        self._n_s1 = len(self.s1)
        self._load_setmats()

    def _load_setmats(self):
        """Load (or build once + cache) the stacked record set matrices."""
        from scipy import sparse
        self._mats = {}
        for field, kind in SET_FIELDS:
            p = CACHE / f'setmat_{field}_{kind}.npz'
            if p.exists():
                M = sparse.load_npz(p)
            else:
                t0 = time.time()
                strings = np.concatenate(
                    [self._s1_cols[field], self._c_cols[field]])
                M = build_setmat(strings, kind)
                sparse.save_npz(p, M)
                log(f"built set matrix {field}/{kind}: nnz={M.nnz:,} "
                    f"({time.time() - t0:.0f}s, cached to {p.name})")
            self._mats[(field, kind)] = M

    def close(self):
        pass                                       # kept for API stability

    def compute(self, pairs, n_cands_map=None):
        """pairs: DataFrame with s1_id, cand_id, country (+optional source).
        Returns float32 DataFrame with FEATURES columns, aligned to pairs.

        n_cands_map: Series entity_id -> global candidate count. If None,
        counts within `pairs` (only correct when pairs holds ALL candidates
        of each S1 — true for whole part files / the benchmark artifacts)."""
        from rapidfuzz import process as rf_process
        from rapidfuzz.distance import JaroWinkler, Levenshtein

        n = len(pairs)
        s1_ids = pairs['s1_id'].to_numpy()
        c_ids = pairs['cand_id'].to_numpy()
        sp = self.s1_idx.get_indexer(s1_ids)
        cp = self.cand_idx.get_indexer(c_ids)
        if (sp < 0).any() or (cp < 0).any():
            raise KeyError(f"{int((sp < 0).sum())} s1 / {int((cp < 0).sum())} "
                           "cand ids missing from prep tables")

        F = np.zeros((n, len(FEATURES)), dtype=np.float32)
        col = {name: i for i, name in enumerate(FEATURES)}

        # ---- vectorized block
        s_alen = self._s1_cols['alen'][sp].astype(np.float32)
        c_alen = self._c_cols['alen'][cp].astype(np.float32)
        F[:, col['addr_len_diff']] = (np.abs(s_alen - c_alen)
                                      / np.maximum(np.maximum(s_alen, c_alen), 1))
        F[:, col['addr_missing']] = ((s_alen == 0) | (c_alen == 0))
        F[:, col['name_ntok_diff']] = np.abs(
            self._s1_cols['n_core'][sp].astype(np.int32)
            - self._c_cols['n_core'][cp].astype(np.int32))
        F[:, col['indic_script']] = self._c_cols['indic'][cp]
        F[:, col['s1_name_freq_log']] = self._s1_cols['freq_log'][sp]
        F[:, col['cand_addr_ntok']] = self._c_cols['addr_ntok'][cp]
        if 'source' in pairs.columns:
            F[:, col['src_s3']] = (pairs['source'].to_numpy().astype(int) == 3)
        else:
            F[:, col['src_s3']] = np.char.startswith(c_ids.astype(str), 'S3')
        F[:, col['country_india']] = (pairs['country'].to_numpy() == 'India')
        if n_cands_map is not None:
            F[:, col['n_cands']] = n_cands_map.reindex(
                s1_ids).to_numpy(dtype=np.float32)
        else:
            F[:, col['n_cands']] = pd.Series(s1_ids).map(
                pd.Series(s1_ids).value_counts()).to_numpy(dtype=np.float32)

        # ---- rapidfuzz batch block (C++ threads)
        s_naddr = self._s1_cols['naddr'][sp]
        c_naddr = self._c_cols['naddr'][cp]
        F[:, col['addr_lev']] = rf_process.cpdist(
            s_naddr, c_naddr, scorer=Levenshtein.normalized_similarity,
            workers=self.rf_workers, dtype=np.float32)
        F[:, col['name_jw_translit']] = rf_process.cpdist(
            self._s1_cols['nname'][sp], self._c_cols['nname'][cp],
            scorer=JaroWinkler.similarity, workers=self.rf_workers,
            dtype=np.float32)

        # ---- set-op block: global row-set matrices, per-pair CSR multiply
        b_rows_all = cp + self._n_s1
        for lo in range(0, n, self.set_slice):
            hi = min(lo + self.set_slice, n)
            a, b = sp[lo:hi], b_rows_all[lo:hi]
            i, na, nb = _pair_inter(self._mats[('naddr', 'tok')], a, b)
            F[lo:hi, col['addr_tok_cont']] = _cont(i, na, nb)
            i, na, nb = _pair_inter(self._mats[('naddr', 'gram')], a, b)
            F[lo:hi, col['addr_3gram_jac']] = _jac(i, na, nb)
            i, na, nb = _pair_inter(self._mats[('digits', 'tok')], a, b)
            F[lo:hi, col['digit_jac']] = _jac(i, na, nb)
            F[lo:hi, col['street_num_match']] = _cont(i, na, nb)
            F[lo:hi, col['digit_exact_conflict']] = (na > 0) & (nb > 0) & (i == 0)
            i, na, nb = _pair_inter(self._mats[('core', 'tok')], a, b)
            F[lo:hi, col['name_tok_jac']] = _jac(i, na, nb)
            i, na, nb = _pair_inter(self._mats[('nname', 'gram')], a, b)
            F[lo:hi, col['name_3gram_jac_translit']] = _jac(i, na, nb)
            i, na, nb = _pair_inter(self._mats[('loc', 'tok')], a, b)
            F[lo:hi, col['locality_overlap']] = _jac(i, na, nb)

        # name_squash_cont: substring containment (only cheap per-pair loop)
        sq1 = self._s1_cols['squash'][sp]
        sq2 = self._c_cols['squash'][cp]
        res = F[:, col['name_squash_cont']]
        for k in range(n):
            q1, q2 = sq1[k], sq2[k]
            if q1 and q2:
                short, lng = (q1, q2) if len(q1) <= len(q2) else (q2, q1)
                if len(short) > 6 and short in lng:
                    res[k] = 1.0

        return pd.DataFrame(F, columns=FEATURES)


# ----------------------------------------------------------- stage: features
def list_pair_files(path):
    p = Path(path)
    if p.is_dir():
        parts = p / 'parts'
        base = parts if parts.is_dir() else p
        return sorted(base.glob('*.parquet'))
    return [p]


def load_n_cands_index(cand_dirs):
    """Global per-S1 candidate count from generate_candidates.py s1_index
    outputs (covers zero-candidate S1s; each S1 lives in exactly one chunk)."""
    frames = []
    for d in cand_dirs:
        idx = Path(d) / 's1_index'
        if idx.is_dir():
            for f in sorted(idx.glob('*.parquet')):
                frames.append(pd.read_parquet(f, columns=['s1_id', 'n_cands']))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    return df.groupby('s1_id')['n_cands'].sum()


def stage_features(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / 'manifest.json'
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    files = list_pair_files(args.pairs)
    ncm = load_n_cands_index([args.pairs]) if Path(args.pairs).is_dir() else None
    fc = FeatureComputer(workers=args.workers)
    log(f"featurizing {len(files)} file(s); n_cands source: "
        f"{'s1_index' if ncm is not None else 'within-file counts'}")
    total = 0
    t0 = time.time()
    for f in files:
        if man.get(f.name) == 'done':
            continue
        pairs = pd.read_parquet(f)
        file_counts = (None if ncm is not None else
                       pairs['s1_id'].map(pairs['s1_id'].value_counts())
                       .to_numpy(dtype=np.float32))
        chunks = []
        for lo in range(0, len(pairs), args.pair_chunk):
            chunk = pairs.iloc[lo:lo + args.pair_chunk]
            feats = fc.compute(chunk, n_cands_map=ncm)
            if file_counts is not None:
                # counts must span the whole file, not one chunk
                feats['n_cands'] = file_counts[lo:lo + len(chunk)]
            chunks.append(pd.concat(
                [chunk[['s1_id', 'cand_id']].reset_index(drop=True),
                 feats.reset_index(drop=True)], axis=1))
        res = pd.concat(chunks, ignore_index=True)
        bad = int((~np.isfinite(res[FEATURES].to_numpy())).sum())
        if bad:
            raise ValueError(f"{bad} non-finite feature values in {f.name}")
        res.to_parquet(out / ('feat_' + f.name), index=False)
        total += len(res)
        man[f.name] = 'done'
        man_path.write_text(json.dumps(man, indent=1))
        log(f"{f.name}: {len(res):,} pairs "
            f"(cum {total:,}, {total / max(time.time() - t0, 1e-9):,.0f} pairs/s)")
    fc.close()
    log(f"done: {total:,} pairs in {time.time() - t0:.0f}s")


# --------------------------------------------------------------- stage: bench
def old_impl_features(pairs, rec, freq_map):
    """The validated stage-3 per-pair loop, verbatim (for benchmarking only)."""
    from rapidfuzz import distance
    from unidecode import unidecode

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

    s1_cache = {}
    for s1 in set(pairs.s1_id):
        n1, a1 = rec[s1]
        nn1 = norm(n1)
        ct1 = core_tokens(n1)
        na1 = norm(a1)
        at1 = na1.split()
        s1_cache[s1] = (nn1, ct1, na1, at1, squash(n1),
                        {t for t in at1 if t.isdigit()},
                        [t for t in at1 if t.isalpha()][-3:],
                        float(np.log1p(freq_map.get(' '.join(ct1), 1))))
    rows = np.empty((len(pairs), 18), dtype=np.float32)
    lev = distance.Levenshtein.normalized_similarity
    jw = distance.JaroWinkler.similarity
    for i, t in enumerate(pairs.itertuples(index=False)):
        nn1, ct1, na1, at1, sq1, d1, loc1, fql = s1_cache[t.s1_id]
        n2, a2 = rec[t.cand_id]
        nn2 = norm(n2)
        ct2 = core_tokens(n2)
        na2 = norm(a2)
        at2 = na2.split()
        indic = bool(INDIC.search(n2))
        nn2t = norm(unidecode(n2)) if indic else nn2
        sq2 = squash(n2)
        d2 = {x for x in at2 if x.isdigit()}
        loc2 = [x for x in at2 if x.isalpha()][-3:]
        rows[i] = (
            cont(at1, at2), jac(_ngrams(na1), _ngrams(na2)), lev(na1, na2),
            jac(d1, d2), jac(ct1, ct2),
            jac(_ngrams(nn1), _ngrams(nn2t)), jw(nn1, nn2t),
            float(bool(sq1) and bool(sq2) and len(min(sq1, sq2, key=len)) > 6
                  and (sq1 in sq2 or sq2 in sq1)),
            abs(len(ct1) - len(ct2)),
            abs(len(na1) - len(na2)) / max(len(na1), len(na2), 1),
            jac(loc1, loc2), cont(d1, d2),
            float(not na1 or not na2), float(indic), fql,
            float(bool(d1) and bool(d2) and not (d1 & d2)),
            float(t.cand_id.startswith('S3')), float(t.country == 'India'))
    return rows


def stage_bench(args):
    """Old vs new on the validated 20k-sample candidates (2.32M pairs)."""
    import csv
    csv.field_size_limit(10**7)
    cand = pd.read_parquet(CACHE / 'rb_candidates.parquet')
    ref = pd.read_parquet(CACHE / 'rb_features.parquet')
    log(f"benchmark input: {len(cand):,} pairs")
    results = {}

    # ---- OLD implementation on a slice (too slow for the full file)
    n_old = min(args.old_sample, len(cand))
    sample = cand.iloc[:n_old]
    need = set(sample.s1_id) | set(sample.cand_id)
    rec = {}
    for fn in ('train_source1.tsv', 'train_source2.tsv', 'train_source3.tsv'):
        with open(TRAIN / fn) as fh:
            r = csv.reader(fh, delimiter='\t')
            next(r)
            for row in r:
                if row[0] in need:
                    rec[row[0]] = (row[1], row[2] if len(row) > 2 else '')
    freq = pd.read_parquet(CACHE / 's1_name_freq.parquet')
    freq_map = dict(zip(freq.core_name, freq.s1_name_freq))
    t0 = time.time()
    old_rows = old_impl_features(sample, rec, freq_map)
    t_old = time.time() - t0
    results['old_pairs_per_sec'] = round(n_old / t_old)
    log(f"OLD per-pair loop: {n_old:,} pairs in {t_old:.1f}s "
        f"= {n_old / t_old:,.0f} pairs/s")

    # ---- NEW implementation on the FULL file
    t0 = time.time()
    fc = FeatureComputer(workers=args.workers)
    t_load = time.time() - t0
    log(f"prep tables loaded in {t_load:.1f}s")
    t0 = time.time()
    feats = []
    for lo in range(0, len(cand), args.pair_chunk):
        feats.append(fc.compute(cand.iloc[lo:lo + args.pair_chunk]))
    new = pd.concat(feats, ignore_index=True)
    # n_cands must be counted over the whole file, not per chunk
    new['n_cands'] = cand['s1_id'].map(
        cand['s1_id'].value_counts()).to_numpy(dtype=np.float32)
    t_new = time.time() - t0
    fc.close()
    results['new_pairs_per_sec'] = round(len(cand) / t_new)
    results['speedup'] = round(results['new_pairs_per_sec']
                               / results['old_pairs_per_sec'], 1)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    rss_kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e9
    results['peak_gb_parent'] = round(rss, 2)
    results['peak_gb_worker_max'] = round(rss_kids, 2)
    log(f"NEW batched: {len(cand):,} pairs in {t_new:.1f}s "
        f"= {len(cand) / t_new:,.0f} pairs/s (x{results['speedup']} vs old); "
        f"peak RSS parent {rss:.1f}GB / worker {rss_kids:.1f}GB")

    # ---- correctness vs the validated rb_features.parquet (full 2.32M)
    shared = [c for c in FEATURES if c in ref.columns]
    diffs = {}
    for c in shared:
        d = np.abs(new[c].to_numpy() - ref[c].to_numpy(dtype=np.float32))
        diffs[c] = float(d.max())
    worst = max(diffs, key=diffs.get)
    nbad = int((~np.isfinite(new[FEATURES].to_numpy())).sum())
    results['n_features_compared'] = len(shared)
    results['max_abs_diff'] = diffs[worst]
    results['max_abs_diff_feature'] = worst
    results['nan_inf_count'] = nbad
    # old slice vs new head slice too (proves old_impl copy is faithful)
    d2 = float(np.abs(old_rows - new[FEATURES[:18]].iloc[:n_old].to_numpy()).max())
    results['old_vs_new_slice_max_diff'] = d2
    log(f"correctness: {len(shared)} features vs rb_features.parquet, "
        f"max|diff|={diffs[worst]:.2e} ({worst}); NaN/inf={nbad}; "
        f"old-loop vs new on {n_old:,}-pair slice max|diff|={d2:.2e}")
    per_feat = ", ".join(f"{c}={v:.1e}" for c, v in sorted(
        diffs.items(), key=lambda kv: -kv[1])[:5])
    log("top per-feature diffs:", per_feat)
    out = ROOT / 'experiments' / 'feature_pipeline_benchmark.json'
    out.write_text(json.dumps({**results, 'per_feature_max_diff': diffs,
                               'workers': args.workers,
                               'note': 'run concurrent with candidate-gen shard 0'
                               }, indent=1))
    log(f"saved {out.relative_to(ROOT)}")


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='stage', required=True)
    cache_help = ('cache dir for prep parquets + setmat npz (default: the '
                  'TRAIN cache experiments/cache). Non-train runs (TEST) MUST '
                  'pass an isolated dir, e.g. experiments/cache_test — see '
                  'set_cache_dir().')
    p = sub.add_parser('prep')
    p.add_argument('--s1', default=str(TRAIN / 'train_source1.tsv'))
    p.add_argument('--s2', default=str(TRAIN / 'train_source2.tsv'))
    p.add_argument('--s3', default=str(TRAIN / 'train_source3.tsv'))
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--cache-dir', default=str(CACHE), help=cache_help)
    p = sub.add_parser('features')
    p.add_argument('--pairs', required=True,
                   help='candidate parquet file, or shard dir with parts/')
    p.add_argument('--out', required=True)
    p.add_argument('--pair-chunk', type=int, default=1_000_000)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--cache-dir', default=str(CACHE), help=cache_help)
    p = sub.add_parser('bench')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--pair-chunk', type=int, default=1_000_000)
    p.add_argument('--old-sample', type=int, default=200_000)
    args = ap.parse_args()
    if getattr(args, 'cache_dir', None):
        set_cache_dir(args.cache_dir)
        log(f"cache dir: {CACHE}")
    {'prep': stage_prep, 'features': stage_features,
     'bench': stage_bench}[args.stage](args)


if __name__ == '__main__':
    main()
