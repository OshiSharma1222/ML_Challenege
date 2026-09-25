"""Stage-2 re-scorer training (5-fold CV over the held-out validation entities).

Rows are the stage-1-scored validation pairs. Training rows are restricted to pairs
whose S1 entity is a held-out entity, because only for those entities were all
plausible queries scored (so entity-side context is complete, as it is at test time).
"""
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

import config
import fine
import stage2
from pipeline import load_records, gt_pairs, assign, f05_macro
from train import report

TAG = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--tag=')), '')

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=config.N_JOBS, verbose=-1)
ROUNDS = 600


def main():
    rec = load_records("train").select("rid", "entity_id", "src", "core", "core_sk",
                                       "name_full", "nums")
    gt = gt_pairs(rec)
    idf = fine.init_idf(rec.filter(pl.col("src") == 1)["core_sk"])
    val_e = pl.read_parquet(config.work("val_entities.parquet"))["rid"]
    sc = pl.read_parquet(config.work("val_scores_s1.parquet"))
    if "--ext" in sys.argv:  # extra out-of-sample entities from extend_val.py
        val_e = pl.concat([val_e, pl.read_parquet(config.work("val2_entities.parquet"))["rid"]])
        sc = pl.concat([sc, pl.read_parquet(config.work("val2_scores_s1.parquet"))
                        .select(sc.columns)])
    sc = sc.filter(pl.col("p") >= stage2.PRUNE)
    X = stage2.build(sc, rec, idf)
    del rec
    X = (X.join(gt.rename({"true_e": "e_rid"}).with_columns(pl.lit(1, pl.Int8).alias("y")),
                on=["q_rid", "e_rid"], how="left").with_columns(pl.col("y").fill_null(0)))
    tq = X.join(gt, on="q_rid", how="left").select("q_rid", "true_e").unique("q_rid")
    fold = tq.with_columns((pl.coalesce("true_e", "q_rid").hash(7) % 5).alias("fold"))
    X = X.join(fold.select("q_rid", "fold"), on="q_rid")
    inV = X["e_rid"].is_in(val_e.implode()).to_numpy()
    A = X.select(stage2.FEATURES).to_numpy().astype(np.float32)
    y = X["y"].to_numpy()
    fo = X["fold"].to_numpy()
    oof = np.zeros(len(y), dtype=np.float32)
    for k in range(5):
        tr = (fo != k) & inV
        m = lgb.train(PARAMS, lgb.Dataset(A[tr], label=y[tr], feature_name=stage2.FEATURES),
                      num_boost_round=ROUNDS)
        oof[fo == k] = m.predict(A[fo == k])
        print(f"[s2] fold {k} done", flush=True)
    s1_best = json.load(open(config.work("thr_stage1.json")))
    s2 = X.select("q_rid", "e_rid").with_columns(pl.Series("p", oof))
    s2.write_parquet(config.work(f"s2_oof{TAG}.parquet"))
    best2 = report(s2, gt, val_e, "stage2")
    m = lgb.train(PARAMS, lgb.Dataset(A[inV], label=y[inV], feature_name=stage2.FEATURES),
                  num_boost_round=ROUNDS)
    m.save_model(config.work(f"model_s2{TAG}.txt"))
    stage = 2 if best2[1] > s1_best["f05"] else 1
    thr = best2[0] if stage == 2 else s1_best["thr"]
    json.dump({"stage": stage, "thr": thr, "f05_stage1": s1_best["f05"], "f05_stage2": best2[1]},
              open(config.work(f"final{TAG}.json"), "w"), indent=1)
    print(f"[s2] stage1 F0.5={s1_best['f05']:.5f}  stage2 F0.5={best2[1]:.5f} -> use stage {stage}")


if __name__ == "__main__":
    main()
