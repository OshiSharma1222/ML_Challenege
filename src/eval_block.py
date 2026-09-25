"""Blocking recall@K against the training ground truth."""
import sys
import polars as pl
import config


def gt_pairs():
    rec = pl.read_parquet(config.work("train_records.parquet"), columns=["rid", "entity_id"])
    gt = pl.read_csv(config.DATA_DIR + "/train/train_ground_truth.tsv", separator="\t",
                     quote_char=None, infer_schema=False)
    ex = (gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m", empty_as_null=True)
          .filter(pl.col("m").is_not_null() & (pl.col("m") != "")))
    ids = rec.rename({"entity_id": "m", "rid": "q_rid"})
    ex = ex.join(ids, on="m").join(rec.rename({"entity_id": "source1_entity_id", "rid": "true_e"}),
                                   on="source1_entity_id")
    return ex.select("q_rid", "true_e")


if __name__ == "__main__":
    cand = pl.read_parquet(config.work(sys.argv[1]))
    gt = gt_pairs()
    qs = cand.select("q_rid").unique()
    g = gt.join(qs, on="q_rid")
    print("queries", qs.height, "matched queries", g.height, f"({g.height/qs.height:.3f})")
    hit = g.join(cand, left_on=["q_rid", "true_e"], right_on=["q_rid", "e_rid"], how="left")
    print("pairs/query", cand.height / qs.height, " any-candidate recall", hit["bs_f"].is_not_null().mean())
    for k in (1, 2, 3, 5, 10, 20):
        f = hit["br_f"].fill_null(255) <= k
        n = hit["br_n"].fill_null(255) <= k
        print(f"recall@{k}: full {f.mean():.4f}  name {n.mean():.4f}  union {(f | n).mean():.4f}")
