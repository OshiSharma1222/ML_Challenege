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

Cluster consensus (close neighbours, similarity >= CLOSE): a look-alike business is a perturbed
copy with its *own* records, which agree with each other, while noise in a true match's record
is random per record. So for a pair whose house numbers or name words differ, it matters which
side the cluster agrees with:
  nb_nclose  number of close neighbours
  nb_hn_e    close neighbours whose first house number equals the entity's
  nb_hn_q    close neighbours whose first house number equals the query's
  nb_xq      mean share of the query's extra name words (not in the entity) found in a neighbour
  nb_xe      mean share of the entity's extra name words (not in the query) found in a neighbour

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
from pipeline import LOAD_COLS

K = 5            # neighbours per record
N_CHUNKS = 4     # index chunks over the S2/S3 pool (bounds memory)
Q_CHUNK = 500_000
NEW_MIN = 0.35   # similarity * neighbour p needed to propose a new candidate
NEW_PER_Q = 2
CLOSE = 0.5      # neighbour similarity counted as "same cluster" for the consensus features
NB_FEATURES = ["nb_v1", "nb_vmax", "nb_vw", "nb_vsum", "nb_nv", "nb_s1", "nb_vgap",
               "nb_nclose", "nb_hn_e", "nb_hn_q", "nb_xq", "nb_xe"]
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
    """Vote table (q_rid, e_rid, nb_*) from neighbour links and stage-1 scores sc (r_rid, e_rid, p)."""
    v = (nb.join(sc.rename({"p": "pr"}), on="r_rid")
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
    qs = pairs.select(pl.col("q_rid").alias("rid")).unique()
    rec = (pl.scan_parquet(config.work(f"{split}_records.parquet")).select(LOAD_COLS)
           .filter((pl.col("src") == 1) | pl.col("rid").is_in(qs["rid"].implode())).collect())
    views = F.SparseViews(rec.filter(pl.col("src") == 1))
    m1 = lgb.Booster(model_file=config.work("model_s1.txt"))
    pairs = pairs.with_columns(pl.lit(0.0, pl.Float32).alias("bs_f"), pl.lit(99, pl.UInt8).alias("br_f"),
                               pl.lit(0.0, pl.Float32).alias("bs_n"), pl.lit(99, pl.UInt8).alias("br_n"))
    outs = []
    pairs = pairs.sort("q_rid")  # contiguous queries per chunk keep the per-chunk query set small
    for i in range(0, pairs.height, 300_000):
        X = F.build(pairs.slice(i, 300_000), rec, views)
        p = m1.predict(X.select(F.FEATURES).to_numpy().astype(np.float32), num_threads=config.N_JOBS)
        outs.append(X.select("q_rid", "e_rid", *stage2.CARRY).with_columns(pl.Series("p", p.astype(np.float32))))
        del X
        print(f"[nb-score] {min(i + 300_000, pairs.height):,}/{pairs.height:,} new pairs", flush=True)
    return pl.concat(outs)


def keys(split):
    """rid -> first house number and name-core word list, for the consensus features."""
    return (pl.read_parquet(config.work(f"{split}_records.parquet"), columns=["rid", "nums", "core"])
            .select("rid", pl.col("nums").fill_null("").str.split(" ").list.first().fill_null("").alias("hn"),
                    pl.col("core").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
                    .alias("tk")))


def consensus(pairs, nb, rk):
    """Cluster-consensus features for (q_rid, e_rid) pairs."""
    close = nb.filter(pl.col("s") >= CLOSE).select("q_rid", "r_rid")
    P = (pairs.join(rk.rename({"rid": "q_rid", "hn": "hq", "tk": "tq"}), on="q_rid", how="left")
         .join(rk.rename({"rid": "e_rid", "hn": "he", "tk": "te"}), on="e_rid", how="left")
         .with_columns(pl.col("tq").list.set_difference("te").alias("xq"),
                       pl.col("te").list.set_difference("tq").alias("xe")))
    Y = (P.select("q_rid", "e_rid", "hq", "he", "xq", "xe").join(close, on="q_rid")
         .join(rk.rename({"rid": "r_rid", "hn": "hr", "tk": "tr"}), on="r_rid", how="left"))
    Y = Y.with_columns(
        ((pl.col("hr") == pl.col("he")) & (pl.col("he") != "")).cast(pl.Float32).alias("a_e"),
        ((pl.col("hr") == pl.col("hq")) & (pl.col("hq") != "")).cast(pl.Float32).alias("a_q"),
        (pl.col("xq").list.set_intersection("tr").list.len() / pl.col("xq").list.len().clip(1)).alias("x_q"),
        (pl.col("xe").list.set_intersection("tr").list.len() / pl.col("xe").list.len().clip(1)).alias("x_e"))
    return Y.group_by("q_rid", "e_rid").agg(
        pl.len().cast(pl.Float32).alias("nb_nclose"), pl.col("a_e").sum().alias("nb_hn_e"),
        pl.col("a_q").sum().alias("nb_hn_q"), pl.col("x_q").mean().alias("nb_xq"),
        pl.col("x_e").mean().alias("nb_xe"))


def _features(nb, all_sc, sc_pairs, rk):
    """Votes for one chunk of queries: (new candidate pairs, nb features for sc_pairs + new)."""
    vt = votes(nb, all_sc)
    new = (vt.filter(pl.col("nb_vw") >= NEW_MIN).join(sc_pairs, on=["q_rid", "e_rid"], how="anti")
           .sort("nb_vw", descending=True).group_by("q_rid", maintain_order=True).head(NEW_PER_Q)
           .select("q_rid", "e_rid"))
    s1 = nb.filter(pl.col("r_rank") == 1).select("q_rid", pl.col("s").alias("nb_s1"))
    pairs = pl.concat([sc_pairs, new])
    f = (pairs.join(vt, on=["q_rid", "e_rid"], how="left").fill_null(0)
         .join(s1, on="q_rid", how="left").with_columns(pl.col("nb_s1").fill_null(0))
         .join(consensus(pairs, nb, rk), on=["q_rid", "e_rid"], how="left").fill_null(0))
    top = pl.col("nb_vw").max().over("q_rid")
    n_top = (pl.col("nb_vw") == top).sum().over("q_rid")
    second = pl.when(pl.col("nb_vw") < top).then(pl.col("nb_vw")).max().over("q_rid").fill_null(0)
    other = pl.when((pl.col("nb_vw") == top) & (n_top == 1)).then(second).otherwise(top)
    f = f.with_columns((pl.col("nb_vw") - other).alias("nb_vgap"))
    f = f.select("q_rid", "e_rid", *NB_FEATURES).with_columns([pl.col(c).cast(pl.Float32) for c in NB_FEATURES])
    return new, f


def augment(sc, split, all_sc=None, log="nb", q_chunk=1_000_000):
    """Add neighbour-proposed candidates to `sc` and return (sc_plus, nb feature table).

    sc: the stage-1 pairs that go to stage 2. all_sc: every stage-1 score available for the
    split (the neighbours' votes); defaults to sc. Votes are computed per chunk of queries to
    bound memory.
    """
    t = time.time()
    all_sc = (sc if all_sc is None else all_sc).select(pl.col("q_rid").alias("r_rid"), "e_rid", "p")
    qids = sc["q_rid"].unique().sort()
    nb = search(split, qids.to_numpy(), log).join(pl.DataFrame({"q_rid": qids}), on="q_rid", how="semi")
    rk = keys(split)
    news, feats = [], []
    for i in range(0, len(qids), q_chunk):
        lo, hi = qids[i], qids[min(i + q_chunk, len(qids)) - 1]
        inq = pl.col("q_rid").is_between(lo, hi)
        new, f = _features(nb.filter(inq), all_sc, sc.filter(inq).select("q_rid", "e_rid"), rk)
        news.append(new)
        feats.append(f)
    del nb, all_sc, rk
    new, f = pl.concat(news), pl.concat(feats)
    print(f"[{log}] {new.height:,} new candidate pairs  {time.time() - t:.0f}s", flush=True)
    if new.height:
        # scored new pairs are cached: `python neighbors.py <split> --score` runs this as its
        # own step, so the stage-2 process only reads the cache
        path = config.work(f"{split}_nb_new.parquet")
        if os.path.exists(path):
            scored = pl.read_parquet(path).join(new, on=["q_rid", "e_rid"], how="semi")
        else:
            scored = _score_pairs(new, split)
            scored.write_parquet(path)
        sc = pl.concat([sc, scored.select(sc.columns)])
    print(f"[{log}] augmented: {sc.height:,} pairs  {time.time() - t:.0f}s", flush=True)
    return sc, f


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
    if "--score" in sys.argv:  # also score the proposed new candidates (cached for augment)
        if split == "train":
            sc = pl.concat([pl.read_parquet(config.work(f + ".parquet")) for f in files], how="diagonal_relaxed")
        else:
            sc = pl.read_parquet(config.work("test_scores_s1.parquet"))
        sc = sc.select(pl.read_parquet_schema(config.work(f"{split}_scores_s1.parquet" if split == "test"
                                                           else "val_scores_s1.parquet")).names())
        augment(sc.filter(pl.col("p") >= stage2.PRUNE), split, log=f"nb-{split}")
