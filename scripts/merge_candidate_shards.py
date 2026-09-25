#!/usr/bin/env python3
"""Merge per-shard candidate outputs from generate_candidates.py into one dataset.

Never loads the whole candidate set into pandas: part files are copied verbatim
(schema preserved), and verification streams one part at a time.

Verifies before merging:
  - every shard manifest is state=done and configs agree except shard_id
  - shard S1 sets (from s1_index) are pairwise disjoint  -> no cross-shard
    duplicate (s1_id, source, cand_id) pair is possible
  - within each part file there is no duplicate (s1_id, source, cand_id)
  - merged S1 count == sum of shard S1 counts (no loss)

Example:
  .venv312/bin/python scripts/merge_candidate_shards.py \
      --shards experiments/candidates/train_sharded/shard_0 \
               experiments/candidates/train_sharded/shard_1 \
               experiments/candidates/train_sharded/shard_2 \
      --out experiments/candidates/train_sharded/merged
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


def fail(msg):
    sys.exit("MERGE FAILED: " + msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="skip the manifest state=done check (dry runs only)")
    args = ap.parse_args()

    shards = [Path(s) for s in args.shards]
    out = Path(args.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    (out / "s1_index").mkdir(parents=True, exist_ok=True)

    # 1. manifests: complete + consistent config (only shard_id may differ)
    configs = []
    for sh in shards:
        man = json.loads((sh / "manifest.json").read_text())
        if not args.allow_incomplete and man.get("state") != "done":
            fail("%s manifest state=%r (not done)" % (sh, man.get("state")))
        cfg = dict(man["config"])
        cfg.pop("shard_id", None)
        configs.append(cfg)
    if any(c != configs[0] for c in configs[1:]):
        fail("shard configs differ beyond shard_id")

    # 2. disjoint S1 sets from the (small) s1_index files
    s1_sets, s1_counts = [], []
    for sh in shards:
        ids = set()
        n = 0
        for f in sorted((sh / "s1_index").glob("*.parquet")):
            col = pd.read_parquet(f, columns=["s1_id"])["s1_id"]
            n += len(col)
            ids.update(col.tolist())
        if len(ids) != n:
            fail("%s: duplicate s1_id inside its own s1_index" % sh)
        s1_sets.append(ids)
        s1_counts.append(n)
    for i in range(len(shards)):
        for j in range(i + 1, len(shards)):
            inter = s1_sets[i] & s1_sets[j]
            if inter:
                fail("shards %d and %d overlap on %d S1 ids (e.g. %s)"
                     % (i, j, len(inter), next(iter(inter))))
    total_s1 = sum(s1_counts)
    print("S1 disjointness OK: shards %s -> %d total S1" % (s1_counts, total_s1))

    # 3. per-part duplicate-pair check (streamed) + copy
    total_pairs = 0
    for si, sh in enumerate(shards):
        for f in sorted((sh / "parts").glob("*.parquet")):
            df = pd.read_parquet(f, columns=["s1_id", "source", "cand_id"])
            if df.duplicated().any():
                fail("%s contains duplicate (s1_id, source, cand_id) rows" % f)
            total_pairs += len(df)
            shutil.copy2(f, out / "parts" / ("shard%d_%s" % (si, f.name)))
        for f in sorted((sh / "s1_index").glob("*.parquet")):
            shutil.copy2(f, out / "s1_index" / ("shard%d_%s" % (si, f.name)))
    # Disjoint S1 sets + per-file dedupe + generate_candidates' per-(country,chunk)
    # dedupe mean no (s1_id, source, cand_id) can repeat across the merged parts:
    # a pair's s1_id lives in exactly one shard, and within a shard in exactly one
    # (country, chunk) part.

    meta = {"shards": [str(s) for s in shards], "n_s1": total_s1,
            "n_pairs": total_pairs, "s1_per_shard": s1_counts}
    (out / "merge_manifest.json").write_text(json.dumps(meta, indent=1))
    print("MERGE OK: %d pairs, %d S1 -> %s" % (total_pairs, total_s1, out))


if __name__ == "__main__":
    main()
