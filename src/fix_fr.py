"""Re-normalise the France test addresses and re-score only the affected queries.

normalize.py learned French regions/departments (a `state` for France) and "st" = saint.
Train has no France, so train data and all models are unchanged; on test only the
France records change. This rewrites their addr/nums/state in test_records.parquet,
re-scores stage 1 for every query touching a France record, and swaps those rows into
test_scores_s1.parquet. The previous files are kept as *_v2.parquet, and the stage-2
feature cache is moved aside so predict.py rebuilds it.

Usage: python fix_fr.py   then   python predict.py --tag=...
"""
import os
import shutil
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import config
import features as F
import stage2
from normalize import addr_parts
from pipeline import load_records


def backup(name):
    src, dst = config.work(f"{name}.parquet"), config.work(f"{name}_v2.parquet")
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)


def main(chunk_q=100_000):
    t = time.time()
    for n in ("test_records", "test_scores_s1"):
        backup(n)
    feats = config.work("test_s2_feats.parquet")
    if os.path.exists(feats):
        os.replace(feats, config.work("test_s2_feats_v2.parquet"))

    # 1. re-normalise France addresses (from the untouched backup, so reruns are idempotent)
    full = pl.read_parquet(config.work("test_records_v2.parquet"))
    fr = full.filter(pl.col("country") == "France")
    new = [addr_parts(a, c) for a, c in zip(fr["business_address"], fr["country"])]
    fr = fr.with_columns(pl.Series("addr", [x[0] for x in new], dtype=pl.Utf8),
                         pl.Series("nums", [" ".join(x[2]) for x in new], dtype=pl.Utf8),
                         pl.Series("state", [x[3] for x in new], dtype=pl.Utf8))
    full = full.update(fr.select("rid", "addr", "nums", "state"), on="rid")
    full.write_parquet(config.work("test_records.parquet"))
    print(f"[fr] {fr.height:,} France records re-normalised, "
          f"{(fr['state'] != '').mean():.1%} now with a state  {time.time() - t:.0f}s", flush=True)
    fr_rids = fr["rid"]
    del full, fr, new

    # 2. queries touching a France record (as query or as candidate entity)
    cand = pl.scan_parquet(config.work("test_cand.parquet"))
    qrids = (cand.filter(pl.col("q_rid").is_in(fr_rids.implode())
                         | pl.col("e_rid").is_in(fr_rids.implode()))
             .select("q_rid").unique().collect()["q_rid"].sort().to_numpy())
    print(f"[fr] {len(qrids):,} queries to re-score", flush=True)

    # 3. stage-1 re-scoring, checkpointed per chunk in test_s1_parts_fr/
    rec = load_records("test")
    views = F.SparseViews(rec.filter(pl.col("src") == 1))
    m1 = lgb.Booster(model_file=config.work("model_s1.txt"))
    part_dir = config.work("test_s1_parts_fr")
    os.makedirs(part_dir, exist_ok=True)
    for i in range(0, len(qrids), chunk_q):
        part = os.path.join(part_dir, f"part_{chunk_q}_{i:09d}.parquet")
        if os.path.exists(part):
            continue
        ids = pl.Series("q_rid", qrids[i:i + chunk_q])
        pairs = cand.filter(pl.col("q_rid").is_in(ids.implode())).collect()
        X = F.build(pairs, rec, views)
        p = m1.predict(X.select(F.FEATURES).to_numpy().astype(np.float32),
                       num_threads=config.N_JOBS)
        kept = (X.select("q_rid", "e_rid", *stage2.CARRY)
                .with_columns(pl.Series("p", p.astype(np.float32)))
                .filter(pl.col("p") >= stage2.PRUNE))
        kept.write_parquet(part + ".tmp")
        os.replace(part + ".tmp", part)
        del X, pairs, kept
        print(f"[fr-s1] {min(i + chunk_q, len(qrids)):,}/{len(qrids):,} queries, "
              f"{time.time() - t:.0f}s", flush=True)
    del views

    # 4. swap the re-scored queries into the stage-1 score cache
    old = pl.read_parquet(config.work("test_scores_s1_v2.parquet"))
    new = pl.read_parquet(os.path.join(part_dir, f"part_{chunk_q}_*.parquet")).select(old.columns)
    sc = pl.concat([old.join(pl.DataFrame({"q_rid": qrids}), on="q_rid", how="anti"), new])
    sc.write_parquet(config.work("test_scores_s1.parquet"))
    print(f"[fr] scores: {old.height:,} -> {sc.height:,} rows  done {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
