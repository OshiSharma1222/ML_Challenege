"""Enlarge the out-of-sample stage-1 score set used to train/validate stage 2.

Stage 1 was trained on 500k queries only, so most other train queries are
out-of-sample for it. We pick an extra held-out entity set W (disjoint from the
first validation set V, and with no true-match query used in stage-1 training),
score every query that truly matches W or has a W entity among its top blocking
candidates, and save the pruned stage-1 scores like val_scores_s1.parquet.
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import config
import features as F
import stage2
from pipeline import load_records, gt_pairs

FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.08


def main():
    t = time.time()
    rec = load_records("train")
    gt = gt_pairs(rec)
    train_q = pl.scan_parquet(config.work("feat_tr_*.parquet")).select("q_rid").unique().collect()
    val_e = pl.read_parquet(config.work("val_entities.parquet"))
    tainted = gt.join(train_q, on="q_rid")["true_e"].unique()
    s1 = rec.filter(pl.col("src") == 1).select("rid")
    pool = s1.filter(~pl.col("rid").is_in(val_e["rid"].implode())
                     & ~pl.col("rid").is_in(tainted.implode()))
    W = pool.sample(fraction=FRAC / (pool.height / s1.height), seed=5)["rid"]
    cand = pl.scan_parquet(config.work("train_cand.parquet"))
    near = (cand.filter(pl.col("e_rid").is_in(W.implode())
                        & ((pl.col("br_f") <= 2) | (pl.col("br_n") <= 1)))
            .select("q_rid").collect()["q_rid"])
    old_q = pl.scan_parquet(config.work("val_scores_s1.parquet")).select("q_rid").unique().collect()
    wq = (pl.concat([gt.filter(pl.col("true_e").is_in(W.implode()))["q_rid"], near]).unique()
          .to_frame("q_rid").join(train_q, on="q_rid", how="anti").join(old_q, on="q_rid", how="anti"))
    print(f"W entities {W.len():,}  new queries {wq.height:,}  {time.time() - t:.0f}s", flush=True)
    pairs = cand.join(wq.lazy(), on="q_rid").collect()
    views = F.SparseViews(rec.filter(pl.col("src") == 1))
    m1 = lgb.Booster(model_file=config.work("model_s1.txt"))
    qids = wq["q_rid"]
    out = []
    step = 150_000
    for i in range(0, qids.len(), step):
        X = F.build(pairs.join(qids.slice(i, step).to_frame(), on="q_rid"), rec, views)
        p = m1.predict(X.select(F.FEATURES).to_numpy().astype(np.float32),
                       num_threads=config.N_JOBS)
        out.append(X.select("q_rid", "e_rid", *stage2.CARRY)
                   .with_columns(pl.Series("p", p.astype(np.float32)))
                   .filter(pl.col("p") >= stage2.PRUNE))
        print(f"[ext] {min(i + step, qids.len()):,}/{qids.len():,}  {time.time() - t:.0f}s",
              flush=True)
    pl.concat(out).write_parquet(config.work("val2_scores_s1.parquet"))
    W.to_frame("rid").write_parquet(config.work("val2_entities.parquet"))
    print(f"done {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
