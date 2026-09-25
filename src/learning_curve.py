"""Stage-1 learning curve: does more training data help?

Retrains the stage-1 model on a fraction of the existing training queries (same
parameters, same early-stopping set) and reports validation macro F0.5 at the best
threshold, to compare with the full model (work/thr_stage1.json).

usage: python learning_curve.py 0.5
"""
import gc
import glob
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import config
import features as F
from pipeline import load_records, gt_pairs, read_feats, assign, f05_macro
from train import PARAMS, label

FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5


def main():
    t = time.time()
    rec = load_records("train").select("rid", "entity_id")
    gt = gt_pairs(rec)
    del rec
    val_e = pl.read_parquet(config.work("val_entities.parquet"))["rid"]
    tr = read_feats("tr")
    qs = tr.select("q_rid").unique().sample(fraction=FRAC, seed=21)
    Xtr = label(tr.join(qs, on="q_rid"), gt)
    del tr
    gc.collect()
    va_q = pl.scan_parquet(config.work("feat_va_*.parquet")).select("q_rid").unique().collect()
    es_q = va_q["q_rid"].sample(min(120_000, va_q.height), seed=3).implode()
    Xes = label(read_feats("va").filter(pl.col("q_rid").is_in(es_q)), gt)
    dtr = lgb.Dataset(Xtr.select(F.FEATURES).to_numpy().astype(np.float32), label=Xtr["y"].to_numpy(),
                      feature_name=F.FEATURES, free_raw_data=True)
    print(f"frac {FRAC}: {qs.height:,} train queries, {Xtr.height:,} pairs  {time.time() - t:.0f}s",
          flush=True)
    del Xtr
    dva = lgb.Dataset(Xes.select(F.FEATURES).to_numpy().astype(np.float32),
                      label=Xes["y"].to_numpy(), reference=dtr)
    m = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    del dtr, dva, Xes
    gc.collect()
    parts = []
    for f in sorted(glob.glob(config.work("feat_va_*.parquet"))):
        X = pl.read_parquet(f)
        p = m.predict(X.select(F.FEATURES).to_numpy().astype(np.float32), num_threads=config.N_JOBS)
        parts.append(X.select("q_rid", "e_rid").with_columns(pl.Series("p", p)))
    sc = pl.concat(parts)
    gt_v = gt.filter(pl.col("true_e").is_in(val_e.implode()))
    best = max((f05_macro(assign(sc, thr), gt_v, val_e)[0], thr) for thr in np.arange(0.4, 0.9, 0.05))
    full = json.load(open(config.work("thr_stage1.json")))
    print(f"[curve] frac {FRAC}: best iter {m.best_iteration}  val F0.5 {best[0]:.5f} @ {best[1]:.2f}  "
          f"(full 500k model: {full['f05']:.5f})  {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
