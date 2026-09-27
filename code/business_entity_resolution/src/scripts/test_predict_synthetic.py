#!/usr/bin/env python3
"""Synthetic dry test for the two pre-TEST fixes. NO real data touched:
every input (model, features, s1_index, source TSVs) is generated here in a
temp dir; asserts afterwards that the TRAIN cache was not modified.

Covers:
 1. model loading via predict_test (real LightGBM booster, trained on noise)
 2. feature ordering respected (one feat file written with shuffled columns)
 3. thresholding at 0.625
 4. one-to-one keep-best-prob semantics
 5. deterministic (s1_id, cand_id) tie-break on equal probabilities
 6. predictions subset of candidate_pairs
 7. zero-match / zero-candidate S1 rows emitted
 8. duplicate pairs impossible (dup input rejected; outputs dup-free)
 9. candidate_pairs.tsv = COMPLETE scored candidate set (incl. below-threshold)
10. official validator (utils/validate_submission.py) passes on the outputs
 +  feature_pipeline --cache-dir isolation (prep + setmats land in the
    supplied dir; experiments/cache untouched)

Run: .venv312/bin/python scripts/test_predict_synthetic.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from feature_pipeline import FEATURES  # noqa: E402
from train_final_model import MODEL_FEATURES  # noqa: E402

PY = sys.executable
CHECKS = []


def check(name, ok, detail=''):
    CHECKS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        sys.exit(f"SYNTHETIC TEST FAILED: {name} {detail}")


def snapshot_cache():
    return {p.name: p.stat().st_mtime_ns
            for p in (ROOT / 'experiments' / 'cache').glob('*')}


def make_model(tmp):
    """Tiny booster over MODEL_FEATURES where P(match) ~= name_tok_jac."""
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    n = 20000
    X = np.zeros((n, len(MODEL_FEATURES)), dtype=np.float32)
    ntj = MODEL_FEATURES.index('name_tok_jac')
    X[:, ntj] = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < X[:, ntj]).astype(np.uint8)
    clf = lgb.LGBMClassifier(n_estimators=30, num_leaves=15,
                             min_child_samples=5, random_state=0, verbose=-1)
    clf.fit(X, y, feature_name=MODEL_FEATURES)
    path = tmp / 'synthetic_model.txt'
    clf.booster_.save_model(str(path))
    return path, ntj


def make_row(s1, cand, ntj_val, ntj_col):
    r = {'s1_id': s1, 'cand_id': cand}
    for i, f in enumerate(FEATURES):
        r[f] = np.float32(ntj_val if i == ntj_col else 0.0)
    return r


def main():
    cache_before = snapshot_cache()
    tmp = Path(tempfile.mkdtemp(prefix='synthtest_'))
    print(f"synthetic workspace: {tmp}")

    model_path, ntj = make_model(tmp)

    # ---- synthetic candidate world (see expected results below)
    # S1-A: S2-1 (.95 match), S2-2 (.15 no), S2-5 (.90 tie -> A wins tie-break)
    # S1-B: S2-1 (.85 match but loses o2o to A), S2-5 (.90 identical-row tie,
    #        loses to A on s1_id), S2-7 (.90 match kept)
    # S1-C: S3-9 (.95 match)  [separate feat file, shuffled column order]
    # S1-D: zero candidates (s1_index only)
    f1 = pd.DataFrame([
        make_row('S1-A', 'S2-1', 0.95, ntj),
        make_row('S1-A', 'S2-2', 0.15, ntj),
        make_row('S1-A', 'S2-5', 0.90, ntj),
        make_row('S1-B', 'S2-1', 0.85, ntj),
        make_row('S1-B', 'S2-5', 0.90, ntj),   # identical features to A/S2-5
        make_row('S1-B', 'S2-7', 0.90, ntj),
    ])
    f2 = pd.DataFrame([make_row('S1-C', 'S3-9', 0.95, ntj)])
    f2 = f2[list(f2.columns[::-1])]            # shuffled column order (check 2)

    feat_dir = tmp / 'features'
    feat_dir.mkdir()
    f1.to_parquet(feat_dir / 'feat_shard0_cand_X_00000.parquet', index=False)
    f2.to_parquet(feat_dir / 'feat_shard1_cand_Y_00000.parquet', index=False)

    cand_dir = tmp / 'merged'
    (cand_dir / 's1_index').mkdir(parents=True)
    pd.DataFrame({'s1_id': ['S1-A', 'S1-B', 'S1-C', 'S1-D'],
                  'country': ['US'] * 4,
                  'n_cands': [3, 3, 1, 0]}).to_parquet(
        cand_dir / 's1_index' / 'idx.parquet', index=False)

    # ---- run predict_test twice (determinism check)
    outs = []
    for run in (1, 2):
        out = tmp / f'out{run}'
        r = subprocess.run(
            [PY, str(ROOT / 'scripts' / 'predict_test.py'),
             '--features', str(feat_dir), '--cands', str(cand_dir),
             '--model', str(model_path), '--out', str(out)],
            capture_output=True, text=True)
        check(f'predict_test run {run} exits 0', r.returncode == 0,
              r.stdout[-400:] + r.stderr[-400:] if r.returncode else '')
        outs.append((out / 'matching_results.tsv').read_bytes()
                    + (out / 'candidate_pairs.tsv').read_bytes())
    check('deterministic outputs (byte-identical reruns)', outs[0] == outs[1])

    out = tmp / 'out1'
    match = dict(line.split('\t') for line in
                 (out / 'matching_results.tsv').read_text()
                 .rstrip('\n').split('\n')[1:])
    cands = dict(line.split('\t') for line in
                 (out / 'candidate_pairs.tsv').read_text()
                 .rstrip('\n').split('\n')[1:])
    header_m = (out / 'matching_results.tsv').read_text().split('\n')[0]
    header_c = (out / 'candidate_pairs.tsv').read_text().split('\n')[0]

    check('matching header', header_m == 'source1_entity_id\tmatched_entity_ids')
    check('candidate header', header_c == 'source1_entity_id\tcandidate_entity_ids')
    check('every S1 emitted incl. zero-candidate',
          list(match) == ['S1-A', 'S1-B', 'S1-C', 'S1-D'] == list(cands),
          f'match rows={list(match)}')
    check('rows sorted by s1_id', list(match) == sorted(match))
    check('zero-candidate S1-D empty in both files',
          match['S1-D'] == '' and cands['S1-D'] == '')
    check('candidate_pairs = COMPLETE scored set (incl. below-threshold S2-2)',
          cands['S1-A'] == 'S2-1,S2-2,S2-5' and cands['S1-B'] == 'S2-1,S2-5,S2-7'
          and cands['S1-C'] == 'S3-9', json.dumps(cands))
    check('threshold 0.625 applied (S2-2 @~0.15 not matched)',
          'S2-2' not in match['S1-A'])
    check('o2o keeps best prob (S2-1 -> S1-A, not S1-B @0.85)',
          'S2-1' in match['S1-A'] and 'S2-1' not in match['S1-B'])
    check('o2o deterministic tie-break (equal prob S2-5 -> smaller s1_id A)',
          'S2-5' in match['S1-A'] and 'S2-5' not in match['S1-B'])
    check('feature order respected (shuffled-column file scored: C matched)',
          match['S1-C'] == 'S3-9', f"C={match['S1-C']!r}")
    check('B keeps its own uncontested match', match['S1-B'] == 'S2-7')
    all_matched = [m for v in match.values() if v for m in v.split(',')]
    check('no duplicate cand across S1 after o2o',
          len(all_matched) == len(set(all_matched)))
    check('predictions subset of candidates',
          all(set(match[k].split(',')) <= set(cands[k].split(','))
              for k in match if match[k]))

    # duplicate input pairs must be rejected
    dup = pd.concat([f1, f1.iloc[[0]]], ignore_index=True)
    dup.to_parquet(feat_dir / 'feat_shard0_cand_X_00000.parquet', index=False)
    r = subprocess.run(
        [PY, str(ROOT / 'scripts' / 'predict_test.py'),
         '--features', str(feat_dir), '--cands', str(cand_dir),
         '--model', str(model_path), '--out', str(tmp / 'outdup')],
        capture_output=True, text=True)
    check('duplicate input pairs rejected (nonzero exit)',
          r.returncode != 0 and 'duplicate' in (r.stdout + r.stderr))
    f1.to_parquet(feat_dir / 'feat_shard0_cand_X_00000.parquet', index=False)

    # ---- official validator on the synthetic outputs (synthetic test dir)
    fake_test = tmp / 'fake_test'
    fake_test.mkdir()
    with open(fake_test / 'test_source1.tsv', 'w') as f:
        f.write('entity_id\tbusiness_name\tbusiness_address\tcountry\n')
        for s1 in ('S1-A', 'S1-B', 'S1-C', 'S1-D'):
            f.write(f'{s1}\tn\ta\tUS\n')
    r = subprocess.run(
        [PY, str(ROOT / 'utils' / 'validate_submission.py'),
         '--matching', str(out / 'matching_results.tsv'),
         '--candidate', str(out / 'candidate_pairs.tsv'),
         '--test-dir', str(fake_test)],
        capture_output=True, text=True)
    check('official validator PASS on synthetic outputs',
          r.returncode == 0 and 'PASS' in r.stdout, r.stdout[-500:])

    # ---- cache isolation: prep + setmats into an isolated --cache-dir
    src = tmp / 'fake_sources'
    src.mkdir()
    rng = np.random.default_rng(1)
    for name, n in (('s1.tsv', 30), ('s2.tsv', 40), ('s3.tsv', 40)):
        with open(src / name, 'w') as f:
            f.write('entity_id\tbusiness_name\tbusiness_address\n')
            for i in range(n):
                f.write(f'{name[:2].upper()}-{i}\tShop {rng.integers(1e4)}'
                        f'\t{rng.integers(1e3)} Main St\n')
    iso = tmp / 'cache_iso'
    r = subprocess.run(
        [PY, str(ROOT / 'scripts' / 'feature_pipeline.py'), 'prep',
         '--s1', str(src / 's1.tsv'), '--s2', str(src / 's2.tsv'),
         '--s3', str(src / 's3.tsv'), '--workers', '1',
         '--cache-dir', str(iso)],
        capture_output=True, text=True)
    check('prep --cache-dir exits 0', r.returncode == 0, r.stderr[-300:])
    check('prep parquets in isolated dir',
          (iso / 'prep_s1.parquet').exists() and (iso / 'prep_cand.parquet').exists())
    # FeatureComputer under the isolated dir builds setmats there
    code = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "import pandas as pd\n"
        "import feature_pipeline as fp\n"
        "fp.set_cache_dir(r'%s')\n"
        "fc = fp.FeatureComputer(workers=1)\n"
        "pairs = pd.DataFrame({'s1_id': ['S1-0'], 'cand_id': ['S2-0'],"
        " 'country': ['US']})\n"
        "f = fc.compute(pairs)\n"
        "assert list(f.columns) == fp.FEATURES\n"
        "print('OK', len(f))\n" % (ROOT / 'scripts', iso))
    r = subprocess.run([PY, '-c', code], capture_output=True, text=True)
    check('FeatureComputer works under isolated cache', 'OK 1' in r.stdout,
          r.stderr[-300:])
    check('setmats created in isolated dir',
          len(list(iso.glob('setmat_*.npz'))) == 6)
    check('TRAIN cache experiments/cache untouched',
          snapshot_cache() == cache_before)

    print(f"\nALL {len(CHECKS)} CHECKS PASSED (synthetic data only; "
          f"workspace {tmp} can be deleted)")


if __name__ == '__main__':
    main()
