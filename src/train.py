"""Stage 4: train the LightGBM pair matcher and tune the decision threshold.

Validation: 2% of Source 1 entities are held out. None of their true-match
queries is used for training. Every query that truly matches a held-out entity,
or has one as a top blocking candidate, is scored against all of its candidates,
assigned (argmax + threshold) and evaluated with macro F0.5 over the held-out
entities, singletons included, exactly like the leaderboard.
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
import stage2
from pipeline import load_records, gt_pairs, featurize, read_feats, assign, f05_macro

N_TRAIN_Q = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 600_000
VAL_FRAC = 0.02
SKIP_FEAT = "--skip-feat" in sys.argv

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, num_threads=config.N_JOBS, verbose=-1)


def splits(rec, cand, gt):
    s1 = rec.filter(pl.col("src") == 1).select("rid")
    val_e = s1.sample(fraction=VAL_FRAC, seed=42)["rid"]
    gt_v = gt.filter(pl.col("true_e").is_in(val_e.implode()))
    near = cand.filter(pl.col("e_rid").is_in(val_e.implode())
                       & ((pl.col("br_f") <= 2) | (pl.col("br_n") <= 1))).select("q_rid").collect()
    val_q = pl.concat([gt_v["q_rid"], near["q_rid"]]).unique()
    rest = (cand.select("q_rid").unique()
            .filter(~pl.col("q_rid").is_in(val_q.implode())).collect())
    train_q = rest.sample(min(N_TRAIN_Q, rest.height), seed=1)["q_rid"]
    return val_e, val_q, train_q


def label(X, gt):
    return (X.join(gt.rename({"true_e": "e_rid"}).with_columns(pl.lit(1, pl.Int8).alias("y")),
                   on=["q_rid", "e_rid"], how="left")
            .with_columns(pl.col("y").fill_null(0)))


def main():
    t = time.time()
    rec = load_records("train")
    gt = gt_pairs(rec)
    cand = pl.scan_parquet(config.work("train_cand.parquet"))
    val_e, val_q, train_q = splits(rec, cand, gt)
    print(f"val entities {val_e.len():,}  val queries {val_q.len():,}  "
          f"train queries {train_q.len():,}", flush=True)
    if not SKIP_FEAT:
        views = F.SparseViews(rec.filter(pl.col("src") == 1))
        for qs, tag in ((train_q, "tr"), (val_q, "va")):
            featurize(cand.filter(pl.col("q_rid").is_in(qs.implode())).collect(), rec, views, tag)
        del views
    del rec, cand
    gc.collect()
    val_e.to_frame().write_parquet(config.work("val_entities.parquet"))

    Xtr = label(read_feats("tr"), gt)
    ytr = Xtr["y"].to_numpy()
    Atr = Xtr.select(F.FEATURES).to_numpy().astype(np.float32)
    del Xtr
    gc.collect()
    # early-stopping set: pairs of 120k validation queries
    es_q = val_q.sample(min(120_000, val_q.len()), seed=3).implode()
    Xes = label(read_feats("va").filter(pl.col("q_rid").is_in(es_q)), gt)
    print(f"train pairs {len(ytr):,} pos {int(ytr.sum()):,} | es pairs {Xes.height:,}  "
          f"{time.time() - t:.0f}s", flush=True)
    dtr = lgb.Dataset(Atr, label=ytr, feature_name=F.FEATURES, free_raw_data=True)
    dva = lgb.Dataset(Xes.select(F.FEATURES).to_numpy().astype(np.float32),
                      label=Xes["y"].to_numpy(), reference=dtr)
    model = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    model.save_model(config.work("model_s1.txt"))
    del dtr, dva, Atr, Xes
    gc.collect()

    parts = []
    for fpath in sorted(glob.glob(config.work("feat_va_*.parquet"))):
        X = pl.read_parquet(fpath)
        p = model.predict(X.select(F.FEATURES).to_numpy().astype(np.float32),
                          num_threads=config.N_JOBS)
        parts.append(X.select("q_rid", "e_rid", *stage2.CARRY)
                     .with_columns(pl.Series("p", p.astype(np.float32))))
    sc = pl.concat(parts)
    sc.write_parquet(config.work("val_scores_s1.parquet"))
    report(sc, gt, val_e, "stage1")
    imp = sorted(zip(F.FEATURES, model.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(k, int(v)) for k, v in imp[:25]])
    print(f"done {time.time() - t:.0f}s")


def report(sc, gt, val_e, tag):
    gt_v = gt.filter(pl.col("true_e").is_in(val_e.implode()))
    best = None
    for thr in np.arange(0.2, 0.96, 0.05):
        f, _ = f05_macro(assign(sc, thr), gt_v, val_e)
        print(f"[{tag}] thr={thr:.2f}  F0.5={f:.5f}")
        if best is None or f > best[1]:
            best = (float(thr), f)
    print(f"[{tag}] best thr={best[0]:.2f} F0.5={best[1]:.5f}")
    json.dump({"thr": best[0], "f05": best[1]}, open(config.work(f"thr_{tag}.json"), "w"))
    return best


if __name__ == "__main__":
    main()
