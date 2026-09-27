"""Small environment benchmarks for the entity-resolution pipeline.

Uses a 200k-row sample of train_source2 names (read-only) plus synthetic data.
Never touches the full 24M-record dataset. Run:  .venv312/bin/python scripts/benchmark_env.py
"""
import resource
import time

import numpy as np


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def bench(label, fn):
    t0, m0 = time.perf_counter(), rss_gb()
    out = fn()
    dt, m1 = time.perf_counter() - t0, rss_gb()
    print(f"{label:44s} {dt:8.2f}s   peak-RSS {m1:5.2f} GB (+{m1 - m0:4.2f})")
    return out


def main():
    import pandas as pd

    names = bench(
        "load 200k sampled names (pandas, nrows)",
        lambda: pd.read_csv(
            "dataset/train/train_source2.tsv", sep="\t", nrows=200_000, dtype=str
        )["business_name"].fillna("").tolist(),
    )

    from sklearn.feature_extraction.text import TfidfVectorizer

    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), dtype=np.float32)
    X_char = bench("char TF-IDF 2-4gram fit_transform (200k)", lambda: char_vec.fit_transform(names))
    print(f"    -> shape {X_char.shape}, nnz {X_char.nnz:,}, "
          f"CSR size {(X_char.data.nbytes + X_char.indices.nbytes + X_char.indptr.nbytes) / 1e6:.0f} MB")

    word_vec = TfidfVectorizer(analyzer="word", dtype=np.float32)
    X_word = bench("word TF-IDF fit_transform (200k)", lambda: word_vec.fit_transform(names))
    print(f"    -> shape {X_word.shape}, nnz {X_word.nnz:,}")

    Q = X_char[:10_000]
    sim = bench("sparse cosine: 10k x 200k full product", lambda: Q @ X_char.T)
    print(f"    -> product nnz {sim.nnz:,} ({sim.nnz * 8 / 1e6:.0f} MB) — why top-k matters")

    from sparse_dot_topn import sp_matmul_topn

    topk = bench(
        "sparse_dot_topn top-30: 10k x 200k (14 thr)",
        lambda: sp_matmul_topn(Q, X_char.T.tocsr(), top_n=30, threshold=0.3, n_threads=14, sort=True),
    )
    print(f"    -> kept nnz {topk.nnz:,}")

    from rapidfuzz import process, fuzz

    a, b = names[:2000], names[2000:4000]
    bench(
        "rapidfuzz token_sort_ratio cdist 2k x 2k (4M pairs, 14 workers)",
        lambda: process.cdist(a, b, scorer=fuzz.token_sort_ratio, workers=14),
    )

    import lightgbm as lgb

    rng = np.random.default_rng(0)
    Xf = rng.random((100_000, 20), dtype=np.float32)
    y = (Xf[:, 0] + Xf[:, 1] > 1).astype(int)
    model = bench(
        "LightGBM train 100k x 20, 100 trees",
        lambda: lgb.train({"objective": "binary", "verbose": -1, "num_threads": 14},
                          lgb.Dataset(Xf, y), num_boost_round=100),
    )
    bench("LightGBM predict 100k", lambda: model.predict(Xf))

    print(f"\ntotal peak RSS: {rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
