"""Record graph: links between Source 2/3 records, used as stage-2 evidence.

Records of one business in S2 and S3 often carry the same corruption (a transliterated name,
a joined-up web name, a missing address), so a record that is hard to match to Source 1 is
often very similar to a sibling record that matches easily. For every query record we find
its K most similar *other* S2/S3 records (same tokens as blocking, chunked index), and each
neighbour votes with its own stage-1 scores:

  nb_v1     stage-1 p of the closest neighbour for this entity
  nb_vmax   max over neighbours of their p for this entity
  nb_vw     max over neighbours of similarity * p
  nb_vsum   sum over neighbours of similarity * p
  nb_nv     number of neighbours with p > 0.5 for this entity
  nb_s1     similarity of the closest neighbour (query level)
  nb_vgap   nb_vw minus the best nb_vw of the query's other candidates

Entities voted for strongly (similarity * p >= NEW_MIN) but absent from the query's scored
candidates become new candidate pairs, scored by the stage-1 model like any other pair.
"""
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import blocking as B
import config
import features as F
import stage2
from pipeline import load_records

K = 5            # neighbours per record
N_CHUNKS = 4     # index chunks over the S2/S3 pool (bounds memory)
Q_CHUNK = 500_000
NEW_MIN = 0.35   # similarity * neighbour p needed to propose a new candidate
NEW_PER_Q = 2
NB_FEATURES = ["nb_v1", "nb_vmax", "nb_vw", "nb_vsum", "nb_nv", "nb_s1", "nb_vgap"]
ENABLED = os.environ.get("ER_NB", "0") == "1"
_COLS = ["rid", "src", "country", "core", "core_sk", "alt", "addr"]


def search(split, qrids, log="nb"):
    """Top-K most similar other S2/S3 records for each query rid: (q_rid, r_rid, s, r_rank).

    Cached in work/{split}_nb.parquet (delete it to recompute for a different query set).
    """
    path = config.work(f"{split}_nb.parquet")
    if os.path.exists(path):
        return pl.read_parquet(path)
    t = time.time()
    rec = pl.read_parquet(config.work(f"{split}_records.parquet"), columns=_COLS)
    pool = rec.filter(pl.col("src") != 1)
    del rec
    qs = pool.join(pl.DataFrame({"rid": np.asarray(qrids, dtype=np.uint32)}), on="rid", how="semi")
    bounds = np.linspace(0, pool.height, N_CHUNKS + 1).astype(int)
    best = None
    for c in range(N_CHUNKS):
        idx = B.Index(pool.slice(bounds[c], bounds[c + 1] - bounds[c]), True)
        for i in range(0, qs.height, Q_CHUNK):
            r = (idx.query(qs.slice(i, Q_CHUNK), K + 1, config.N_JOBS, "nb")
                 .select("q_rid", pl.col("e_rid").alias("r_rid"), pl.col("bs_nb").alias("s"))
                 .filter(pl.col("q_rid") != pl.col("r_rid")))
            best = r if best is None else pl.concat([best, r])
        best = best.sort("s", descending=True).group_by("q_rid", maintain_order=True).head(K)
        del idx
        print(f"[{log}] chunk {c + 1}/{N_CHUNKS}: {best.height:,} links  {time.time() - t:.0f}s", flush=True)
    nb = best.with_columns(pl.col("s").rank("ordinal", descending=True).over("q_rid")
                           .cast(pl.UInt8).alias("r_rank"))
    nb.write_parquet(path)
    return nb


def votes(nb, sc):
    """Vote table (q_rid, e_rid, nb_*) from neighbour links and all stage-1 scores `sc`."""
    v = (nb.join(sc.select(pl.col("q_rid").alias("r_rid"), "e_rid", pl.col("p").alias("pr")),
                 on="r_rid")
         .with_columns((pl.col("s") * pl.col("pr")).alias("w")))
    agg = v.group_by("q_rid", "e_rid").agg(
        pl.when(pl.col("r_rank") == 1).then(pl.col("pr")).max().fill_null(0).alias("nb_v1"),
        pl.col("pr").max().alias("nb_vmax"),
        pl.col("w").max().alias("nb_vw"),
        pl.col("w").sum().alias("nb_vsum"),
        (pl.col("pr") > 0.5).sum().cast(pl.Float32).alias("nb_nv"),
    )
    return agg


def _score_pairs(pairs, split):
    """Stage-1 score new (q_rid, e_rid) pairs; same output columns as the score caches."""
    rec = load_records(split)
    views = F.SparseViews(rec.filter(pl.col("src") == 1))
    m1 = lgb.Booster(model_file=config.work("model_s1.txt"))
    pairs = pairs.with_columns(pl.lit(0.0, pl.Float32).alias("bs_f"), pl.lit(99, pl.UInt8).alias("br_f"),
                               pl.lit(0.0, pl.Float32).alias("bs_n"), pl.lit(99, pl.UInt8).alias("br_n"))
    outs = []
    for i in range(0, pairs.height, 2_000_000):
        X = F.build(pairs.slice(i, 2_000_000), rec, views)
        p = m1.predict(X.select(F.FEATURES).to_numpy().astype(np.float32), num_threads=config.N_JOBS)
        outs.append(X.select("q_rid", "e_rid", *stage2.CARRY).with_columns(pl.Series("p", p.astype(np.float32))))
        del X
    return pl.concat(outs)


def augment(sc, split, all_sc=None, log="nb"):
    """Add neighbour-proposed candidates to `sc` and return (sc_plus, nb feature table).

    sc: the stage-1 pairs that go to stage 2. all_sc: every stage-1 score available for the
    split (the neighbours' votes); defaults to sc.
    """
    t = time.time()
    all_sc = sc if all_sc is None else all_sc
    nb = search(split, sc["q_rid"].unique().to_numpy(), log)
    nb = nb.join(sc.select("q_rid").unique(), on="q_rid", how="semi")
    vt = votes(nb, all_sc)
    new = (vt.filter(pl.col("nb_vw") >= NEW_MIN).join(sc.select("q_rid", "e_rid"), on=["q_rid", "e_rid"], how="anti")
           .sort("nb_vw", descending=True).group_by("q_rid", maintain_order=True).head(NEW_PER_Q)
           .select("q_rid", "e_rid"))
    print(f"[{log}] {vt.height:,} vote rows, {new.height:,} new candidate pairs  {time.time() - t:.0f}s", flush=True)
    if new.height:
        scored = _score_pairs(new, split).select(sc.columns)
        sc = pl.concat([sc, scored])
    s1 = nb.filter(pl.col("r_rank") == 1).select("q_rid", pl.col("s").alias("nb_s1"))
    f = (sc.select("q_rid", "e_rid").join(vt, on=["q_rid", "e_rid"], how="left").fill_null(0)
         .join(s1, on="q_rid", how="left").with_columns(pl.col("nb_s1").fill_null(0)))
    top = pl.col("nb_vw").max().over("q_rid")
    n_top = (pl.col("nb_vw") == top).sum().over("q_rid")
    second = pl.when(pl.col("nb_vw") < top).then(pl.col("nb_vw")).max().over("q_rid").fill_null(0)
    other = pl.when((pl.col("nb_vw") == top) & (n_top == 1)).then(second).otherwise(top)
    f = f.with_columns((pl.col("nb_vw") - other).alias("nb_vgap"))
    print(f"[{log}] augmented: {sc.height:,} pairs  {time.time() - t:.0f}s", flush=True)
    return sc, f.select("q_rid", "e_rid", *[c for c in NB_FEATURES]).with_columns(
        [pl.col(c).cast(pl.Float32) for c in NB_FEATURES])


if __name__ == "__main__":
    import sys
    split = sys.argv[1]
    if split == "train":  # the stage-2 training queries: every scored held-out set
        files = [f for f in ("val_scores_s1", "val2_scores_s1", "val3_scores_s1")
                 if os.path.exists(config.work(f + ".parquet"))]
        q = pl.concat([pl.read_parquet(config.work(f + ".parquet"), columns=["q_rid"]) for f in files])
    else:
        q = pl.read_parquet(config.work(f"{split}_records.parquet"), columns=["rid", "src"]).filter(
            pl.col("src") != 1).select(pl.col("rid").alias("q_rid"))
    search(split, q["q_rid"].unique().to_numpy(), log=f"nb-{split}")
