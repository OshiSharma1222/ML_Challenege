"""Validation error analysis: how much macro F0.5 each error type costs.

Each query of the held-out evaluation is put in one bucket (correct, confuser matched,
wrong entity, below threshold, pruned by stage 1, missed by blocking). The F0.5 gain from
fixing a bucket is measured by repairing only that bucket and re-scoring.

usage: python analyze.py [scores.parquet] [thr]   (defaults: stage-2 OOF scores, final thr)
"""
import json
import sys

import polars as pl

import config
from pipeline import load_records, gt_pairs, assign, f05_macro

TAG = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--tag=')), '_fine')
ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]


def main():
    path = ARGS[0] if ARGS else config.work(f"s2_oof{TAG}.parquet")
    thr = float(ARGS[1]) if len(ARGS) > 1 else json.load(open(config.work(f"final{TAG}.json")))["thr"]
    rec = load_records("train").select("rid", "entity_id", "src", "business_name", "addr", "country")
    gt = gt_pairs(rec)
    val_e = pl.read_parquet(config.work("val_entities.parquet"))["rid"]
    sc = pl.read_parquet(path).select("q_rid", "e_rid", "p")
    gt_v = gt.filter(pl.col("true_e").is_in(val_e.implode()))
    pred = assign(sc, thr)
    base, _ = f05_macro(pred, gt_v, val_e)

    # the evaluation only sees predictions that land on held-out entities
    pv = pred.filter(pl.col("e_rid").is_in(val_e.implode()))
    q = (pl.concat([gt_v["q_rid"], pv["q_rid"]]).unique().to_frame()
         .join(gt, on="q_rid", how="left")                       # true_e (null: matches nothing)
         .join(pred.rename({"e_rid": "pred_e"}), on="q_rid", how="left"))
    true_p = sc.rename({"e_rid": "true_e", "p": "p_true"})
    q = q.join(true_p, on=["q_rid", "true_e"], how="left")
    cand = (pl.scan_parquet(config.work("train_cand.parquet")).select("q_rid", "e_rid")
            .join(gt_v.lazy().rename({"true_e": "e_rid"}), on=["q_rid", "e_rid"]).collect()
            .with_columns(pl.lit(True).alias("in_cand")).rename({"e_rid": "true_e"}))
    q = q.join(cand, on=["q_rid", "true_e"], how="left")
    top = (sc.sort("p", descending=True).group_by("q_rid", maintain_order=True).first()
           .rename({"e_rid": "top_e", "p": "p_top"}))
    q = q.join(top, on="q_rid", how="left")
    q = q.with_columns(
        pl.when(pl.col("pred_e").is_not_null() & (pl.col("pred_e") == pl.col("true_e"))).then(pl.lit("ok"))
        .when(pl.col("true_e").is_null()).then(pl.lit("confuser_matched"))
        .when(pl.col("pred_e").is_not_null()).then(pl.lit("wrong_entity"))
        .when(pl.col("in_cand").is_null()).then(pl.lit("blocking_miss"))
        .when(pl.col("p_true").is_null()).then(pl.lit("pruned_s1"))
        .when(pl.col("top_e") != pl.col("true_e")).then(pl.lit("outranked_below_thr"))
        .otherwise(pl.lit("below_thr")).alias("bucket"))
    print(f"scores {path}  thr {thr:.2f}  base macro F0.5 {base:.5f}  "
          f"({val_e.len():,} entities, {q.height:,} queries)")

    def fixed(bucket):
        """Predictions with only this bucket repaired."""
        b = q.filter(pl.col("bucket") == bucket)
        keep = pred.join(b.select("q_rid"), on="q_rid", how="anti")
        add = (b.filter(pl.col("true_e").is_not_null())
               .select("q_rid", pl.col("true_e").alias("e_rid"), pl.lit(1.0).alias("p")))
        return pl.concat([keep, add.cast(keep.schema)])

    for bucket, n in q["bucket"].value_counts(sort=True).iter_rows():
        gain = 0.0 if bucket == "ok" else f05_macro(fixed(bucket), gt_v, val_e)[0] - base
        print(f"  {bucket:22s} {n:8,} queries   fixing it: {gain:+.5f}")
    q.join(rec.select("rid", pl.col("business_name").alias("q_name"), pl.col("addr").alias("q_addr"),
                      "country"), left_on="q_rid", right_on="rid", how="left") \
     .join(rec.select("rid", pl.col("business_name").alias("true_name")), left_on="true_e",
           right_on="rid", how="left") \
     .join(rec.select("rid", pl.col("business_name").alias("pred_name")), left_on="pred_e",
           right_on="rid", how="left") \
     .filter(pl.col("bucket") != "ok") \
     .write_parquet(config.work(f"errors{TAG}.parquet"))


if __name__ == "__main__":
    main()
