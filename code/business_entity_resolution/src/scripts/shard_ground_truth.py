#!/usr/bin/env python3
"""Shard train_ground_truth.tsv (or any TSV keyed by an S1 entity_id column)
using the SAME deterministic assignment as generate_candidates.py:
MD5(entity_id UTF-8) -> int -> % num_shards.

Evaluation/training convenience only — the candidate generator stays GT-free.
Verifies: every row lands in exactly one shard, shards are disjoint, and
shard_0 ∪ ... ∪ shard_{N-1} == the original file.

Example:
  .venv312/bin/python scripts/shard_ground_truth.py \
      --input dataset/train/train_ground_truth.tsv \
      --id-col source1_entity_id --num-shards 3 \
      --out-dir experiments/candidates/train_sharded/gt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_candidates import read_tsv, shard_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="dataset/train/train_ground_truth.tsv")
    ap.add_argument("--id-col", default="source1_entity_id")
    ap.add_argument("--num-shards", type=int, default=3)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = read_tsv(args.input)
    ids = df[args.id_col].to_numpy()
    assign = np.fromiter((shard_of(e, args.num_shards) for e in ids),
                         dtype=np.int64, count=len(ids))

    n_total = 0
    for sid in range(args.num_shards):
        part = df[assign == sid]
        fn = out / ("%s.shard_%d.tsv" % (Path(args.input).stem, sid))
        part.to_csv(fn, sep="\t", index=False)
        n_total += len(part)
        print("shard %d: %d rows -> %s" % (sid, len(part), fn))

    # completeness/disjointness: assignment is a function of entity_id, so each
    # row gets exactly one shard; verify counts and id-set union explicitly.
    assert n_total == len(df), "union size mismatch"
    assert ((assign >= 0) & (assign < args.num_shards)).all()
    uniq = df[args.id_col].nunique()
    union_uniq = sum(df.loc[assign == s, args.id_col].nunique()
                     for s in range(args.num_shards))
    assert union_uniq == uniq, "duplicated/missing S1 ids across shards"
    print("OK: %d rows, %d unique %s, shards disjoint and complete"
          % (len(df), uniq, args.id_col))


if __name__ == "__main__":
    main()
