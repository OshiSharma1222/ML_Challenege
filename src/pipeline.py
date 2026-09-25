"""Shared helpers: loading, ground truth, chunked featurisation, assignment, scoring."""
import glob
import os
import time

import numpy as np
import polars as pl

import config
import features as F


LOAD_COLS = ["rid", "entity_id", "src", "country", "business_name", "core", "core_sk",
             "name_full", "alt", "legal", "addr", "nums", "state"]


def load_records(split):
    return pl.read_parquet(config.work(f"{split}_records.parquet"), columns=LOAD_COLS)


def gt_pairs(rec):
    """(q_rid, true_e) for every matched S2/S3 record in train."""
    gt = pl.read_csv(os.path.join(config.DATA_DIR, "train", "train_ground_truth.tsv"),
                     separator="\t", quote_char=None, infer_schema=False)
    ex = (gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("m"))
          .explode("m", empty_as_null=True)
          .filter(pl.col("m").is_not_null() & (pl.col("m") != "")))
    ids = rec.select("rid", "entity_id")
    ex = (ex.join(ids.rename({"entity_id": "m", "rid": "q_rid"}), on="m")
          .join(ids.rename({"entity_id": "source1_entity_id", "rid": "true_e"}),
                on="source1_entity_id"))
    return ex.select("q_rid", "true_e")


def featurize(pairs, rec, views, tag, chunk_q=150_000):
    """Compute features chunk by chunk (grouped by query) and write parquet parts."""
    for f in glob.glob(config.work(f"feat_{tag}_*.parquet")):
        os.remove(f)
    qids = pairs["q_rid"].unique().shuffle(seed=0)
    t = time.time()
    n = 0
    for i in range(0, qids.len(), chunk_q):
        qq = pl.DataFrame({"q_rid": qids.slice(i, chunk_q)})
        p = pairs.join(qq, on="q_rid")
        X = F.build(p, rec, views)
        X.write_parquet(config.work(f"feat_{tag}_{i // chunk_q:04d}.parquet"))
        n += X.height
        print(f"[feat] {tag} {min(i + chunk_q, qids.len()):,}/{qids.len():,} queries, "
              f"{n:,} pairs, {time.time() - t:.0f}s", flush=True)


def read_feats(tag):
    return pl.read_parquet(config.work(f"feat_{tag}_*.parquet"))


def assign(scores, thr):
    """Each query goes to its single best S1 candidate if its probability >= thr."""
    best = (scores.sort("p", descending=True).group_by("q_rid", maintain_order=True).first())
    return best.filter(pl.col("p") >= thr).select("q_rid", "e_rid", "p")


def f05_macro(pred, gt, entities):
    """Macro F0.5 over the given S1 entity rids (singletons included)."""
    ents = pl.DataFrame({"e_rid": entities})
    tp = (pred.join(gt.rename({"true_e": "e_rid"}), on=["q_rid", "e_rid"])
          .group_by("e_rid").agg(pl.len().alias("tp")))
    npred = pred.group_by("e_rid").agg(pl.len().alias("np"))
    ntrue = gt.rename({"true_e": "e_rid"}).group_by("e_rid").agg(pl.len().alias("nt"))
    d = (ents.join(npred, on="e_rid", how="left").join(ntrue, on="e_rid", how="left")
         .join(tp, on="e_rid", how="left").fill_null(0))
    P = d["tp"] / d["np"].clip(1)
    R = d["tp"] / d["nt"].clip(1)
    f = (1.25 * P * R / (0.25 * P + R)).fill_nan(0).to_numpy()
    both_empty = ((d["np"] == 0) & (d["nt"] == 0)).to_numpy()
    f = np.where(both_empty, 1.0, np.nan_to_num(f))
    return float(f.mean()), d.with_columns(pl.Series("f", f))
