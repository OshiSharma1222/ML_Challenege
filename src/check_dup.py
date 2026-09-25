"""Does the model trained on duplicated confusers (B) generalise, or learn the copy artifact?

Scores B (trained with confuser copies) on the ordinary validation (no copies) and
compares with A (trained and scored without copies). If B only exploited the exact
duplicates, it should do worse than A here.
"""
import sys

import polars as pl

import config
import fine
from decide import decide
from pipeline import load_records, gt_pairs, f05_macro
import stage2
from train_s2 import with_copies, build, cv, shifted

sys.argv += ["--ext"]


def main():
    rec = load_records("train").select("rid", "entity_id", "src", "core", "core_sk",
                                       "name_full", "nums")
    gt = gt_pairs(rec)
    idf = fine.init_idf(rec.filter(pl.col("src") == 1)["core_sk"])
    val_e = pl.concat([pl.read_parquet(config.work("val_entities.parquet"))["rid"],
                       pl.read_parquet(config.work("val2_entities.parquet"))["rid"]])
    v0 = pl.read_parquet(config.work("val_entities.parquet"))["rid"]
    sc = pl.read_parquet(config.work("val_scores_s1.parquet"))
    sc = pl.concat([sc, pl.read_parquet(config.work("val2_scores_s1.parquet")).select(sc.columns)])
    sc = sc.filter(pl.col("p") >= stage2.PRUNE)
    base = build(with_copies(sc, gt, 0), rec, idf, gt, val_e)
    dup = build(with_copies(sc, gt, 1), rec, idf, gt, val_e)
    del rec
    gt_all = gt.filter(pl.col("true_e").is_in(val_e.implode()))
    gt_v0 = gt.filter(pl.col("true_e").is_in(v0.implode()))
    s = cv(dup, base)
    for r in (1.0, 0.7, 0.5):
        print(f"[check] model B on train-like val, shift {r}: all {f05_macro(decide(shifted(s, r)), gt_all, val_e)[0]:.5f}"
              f"  original entities {f05_macro(decide(shifted(s, r)), gt_v0, v0)[0]:.5f}", flush=True)
    print("[check] reference: model A on train-like val = 0.97693 (all), 0.97704 (original)")


if __name__ == "__main__":
    main()
