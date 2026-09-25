"""Stage-2 re-scorer training (5-fold CV over the held-out validation entities).

Rows are the stage-1-scored validation pairs. Training rows are restricted to pairs
whose S1 entity is a held-out entity, because only for those entities were all
plausible queries scored (so entity-side context is complete, as it is at test time).

--dupconf=N simulates the test set's confuser density: the test set has about twice as
many unmatched (confuser) queries per Source 1 entity as train. Every confuser query is
copied N times under a new query id *before* the stage-2 features are built, so the
entity-side context features see test-like crowding. Two models are then compared on
that test-like validation: A, trained on train-like rows, and B, trained on test-like rows.
"""
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

import config
import fine
import stage2
from decide import decide
from pipeline import load_records, gt_pairs, assign, f05_macro
from train import report

TAG = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--tag=')), '')
DUP = int(next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--dupconf=')), 0))

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=config.N_JOBS, verbose=-1)
ROUNDS = 600
SHIFTS = (1.0, 0.7, 0.5, 0.35, 0.25)
# --noent drops the entity-side context features, the only ones that change when an
# entity is crowded by more confusers (test has ~2x as many per entity as train)
ENT = ["e_nwin_other", "e_psum", "e_rank", "e_pmax"]
FEATS = [f for f in stage2.FEATURES if f not in ENT] if "--noent" in sys.argv else stage2.FEATURES


def with_copies(sc, gt, n):
    """Add n copies of every confuser query's pairs; `orig` keeps the source query id."""
    base = int(sc["q_rid"].max()) + 1
    sc = sc.with_columns(pl.col("q_rid").alias("orig"))
    un = sc.join(gt.select("q_rid"), on="q_rid", how="anti")
    return pl.concat([sc] + [un.with_columns(pl.col("q_rid") + base * (i + 1)) for i in range(n)])


def build(sc, rec, idf, gt, val_e):
    """Stage-2 feature matrix with label, CV fold (grouped by true entity / source query)."""
    copies = sc.filter(pl.col("q_rid") != pl.col("orig")).select("q_rid", "orig").unique()
    if copies.height:  # copied queries borrow their source query's record (name, numbers)
        rec = pl.concat([rec, copies.join(rec, left_on="orig", right_on="rid")
                         .drop("orig").rename({"q_rid": "rid"}).select(rec.columns)])
    X = stage2.build(sc.drop("orig"), rec, idf).join(sc.select("q_rid", "e_rid", "orig"),
                                                    on=["q_rid", "e_rid"])
    X = (X.join(gt.rename({"true_e": "e_rid"}).with_columns(pl.lit(1, pl.Int8).alias("y")),
                on=["q_rid", "e_rid"], how="left").with_columns(pl.col("y").fill_null(0)))
    tq = (X.select("q_rid", "orig").unique("q_rid")
          .join(gt.rename({"q_rid": "orig"}), on="orig", how="left"))
    fold = tq.with_columns((pl.coalesce("true_e", "orig").hash(7) % 5).alias("fold"))
    X = X.join(fold.select("q_rid", "fold"), on="q_rid")
    return (X.select("q_rid", "e_rid"), X.select(FEATS).to_numpy().astype(np.float32),
            X["y"].to_numpy(), X["fold"].to_numpy(), X["e_rid"].is_in(val_e.implode()).to_numpy())


def cv(train, pred):
    """5-fold OOF predictions on `pred` from models trained on `train` (same fold ids)."""
    _, A, y, fo, inV = train
    ids, Ap, _, fp, _ = pred
    oof = np.zeros(len(fp), dtype=np.float32)
    for k in range(5):
        tr = (fo != k) & inV
        m = lgb.train(PARAMS, lgb.Dataset(A[tr], label=y[tr], feature_name=FEATS),
                      num_boost_round=ROUNDS)
        oof[fp == k] = m.predict(Ap[fp == k])
        print(f"[s2] fold {k} done", flush=True)
    return ids.with_columns(pl.Series("p", oof))


def shifted(s2, r):
    return s2.with_columns((r * pl.col("p") / (r * pl.col("p") + 1 - pl.col("p"))).alias("p"))


def main():
    rec = load_records("train").select("rid", "entity_id", "src", "core", "core_sk",
                                       "name_full", "nums")
    gt = gt_pairs(rec)
    idf = fine.init_idf(rec.filter(pl.col("src") == 1)["core_sk"])
    val_e = pl.read_parquet(config.work("val_entities.parquet"))["rid"]
    sc = pl.read_parquet(config.work("val_scores_s1.parquet"))
    v0 = val_e  # the original held-out entities, kept for like-for-like comparisons
    if any(a.startswith("--ext") for a in sys.argv):  # extra out-of-sample entities from extend_val.py
        n_ext = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--ext=")), 2))
        for v in [f"val{i}" for i in range(2, n_ext + 1)]:
            val_e = pl.concat([val_e, pl.read_parquet(config.work(f"{v}_entities.parquet"))["rid"]])
            sc = pl.concat([sc, pl.read_parquet(config.work(f"{v}_scores_s1.parquet"))
                            .select(sc.columns)])
    sc = sc.filter(pl.col("p") >= stage2.PRUNE)
    gt_all, gt_v0 = (gt.filter(pl.col("true_e").is_in(v.implode())) for v in (val_e, v0))
    s1_best = json.load(open(config.work("thr_stage1.json")))

    base = build(with_copies(sc, gt, 0), rec, idf, gt, val_e)
    if not DUP:
        s2 = cv(base, base)
        s2.write_parquet(config.work(f"s2_oof{TAG}.parquet"))
        best2 = report(s2, gt, val_e, "stage2")
        f_rule = f05_macro(decide(s2), gt_all, val_e)[0]
        print(f"[s2] expected-F rule F0.5={f_rule:.5f} vs threshold {best2[1]:.5f}")
        print(f"[s2] original val entities: threshold "
              f"{f05_macro(assign(s2, best2[0]), gt_v0, v0)[0]:.5f}  expected-F rule "
              f"{f05_macro(decide(s2), gt_v0, v0)[0]:.5f}", flush=True)
        # test-like check by copying confuser queries' prediction rows: exact for a model
        # without entity-side features (the query-side features of a copy are unchanged)
        top = int(s2["q_rid"].max()) + 1
        sim = pl.concat([s2, s2.join(gt.select("q_rid"), on="q_rid", how="anti")
                         .with_columns(pl.col("q_rid") + top)])
        sims = {r: f05_macro(decide(shifted(sim, r)), gt_all, val_e)[0] for r in SHIFTS}
        for r, f in sims.items():
            print(f"[s2] test-like val (row copies) shift {r}: expected-F rule F0.5={f:.5f}"
                  f"{'' if '--noent' in sys.argv else '  (optimistic: entity features not rebuilt)'}",
                  flush=True)
        final = base
        shift = max(sims, key=sims.get) if "--noent" in sys.argv else 1.0
        stage = 2 if best2[1] > s1_best["f05"] else 1
        thr = best2[0] if stage == 2 else s1_best["thr"]
        rule = "expf" if stage == 2 and f_rule > best2[1] else "thr"
        extra = {"f05_stage2": best2[1], "f05_stage2_expf": f_rule}
    else:
        dup = build(with_copies(sc, gt, DUP), rec, idf, gt, val_e)
        del rec
        results = {}
        for name, train in (("A_trainlike", base), ("B_testlike", dup)):
            s2 = cv(train, dup)
            s2.write_parquet(config.work(f"s2_oof{TAG}_{name}.parquet"))
            for r in SHIFTS:
                f = f05_macro(decide(shifted(s2, r)), gt_all, val_e)[0]
                results[(name, r)] = f
                print(f"[s2] test-like val ({DUP} extra confuser copies) model {name} "
                      f"shift {r}: expected-F rule F0.5={f:.5f}", flush=True)
        (name, shift), f_best = max(results.items(), key=lambda kv: kv[1])
        print(f"[s2] best: model {name} shift {shift}  F0.5={f_best:.5f}")
        final = base if name == "A_trainlike" else dup
        stage, thr, rule = 2, 0.7, "expf"
        extra = {"f05_testlike": f_best, "model": name,
                 "all": {f"{k[0]}@{k[1]}": v for k, v in results.items()}}
    _, A, y, _, inV = final
    m = lgb.train(PARAMS, lgb.Dataset(A[inV], label=y[inV], feature_name=FEATS),
                  num_boost_round=ROUNDS)
    m.save_model(config.work(f"model_s2{TAG}.txt"))
    json.dump({"stage": stage, "thr": thr, "rule": rule, "shift": shift,
               "f05_stage1": s1_best["f05"], **extra},
              open(config.work(f"final{TAG}.json"), "w"), indent=1)
    print(f"[s2] saved model_s2{TAG}.txt  (stage {stage}, rule {rule}, shift {shift})")


if __name__ == "__main__":
    main()
