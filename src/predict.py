"""Stage 5: score test candidates, assign matches, write submission files.

Outputs (tab-separated, one row per Source 1 test entity):
  output/matching_results.tsv   source1_entity_id, matched_entity_ids
  output/candidate_pairs.tsv    source1_entity_id, candidate_entity_ids
candidate_pairs is exactly the set of pairs the model scored.
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import config
import features as F
from pipeline import load_records, assign
import fine
from decide import decide
import stage2

TAG = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--tag=')), '')


def score(cand_path, qrids, rec, views, model, chunk_q=100_000, log="score"):
    """Stage-1 scoring streamed over contiguous query-rid ranges; keeps p >= PRUNE.

    Each chunk is checkpointed to work/test_s1_parts/, so an interrupted run resumes
    where it stopped (the part files are keyed by chunk start and chunk size).
    """
    qrids = np.sort(qrids)
    lazy = pl.scan_parquet(cand_path)
    part_dir = config.work("test_s1_parts")
    os.makedirs(part_dir, exist_ok=True)
    t = time.time()
    for i in range(0, len(qrids), chunk_q):
        part = os.path.join(part_dir, f"part_{chunk_q}_{i:09d}.parquet")
        if os.path.exists(part):
            continue
        lo, hi = int(qrids[i]), int(qrids[min(i + chunk_q, len(qrids)) - 1])
        pairs = lazy.filter(pl.col("q_rid").is_between(lo, hi)).collect()
        if pairs.height == 0:
            continue
        X = F.build(pairs, rec, views)
        p = model.predict(X.select(F.FEATURES).to_numpy().astype(np.float32),
                          num_threads=config.N_JOBS)
        kept = (X.select("q_rid", "e_rid", *stage2.CARRY)
                .with_columns(pl.Series("p", p.astype(np.float32)))
                .filter(pl.col("p") >= stage2.PRUNE))
        kept.write_parquet(part + ".tmp")
        os.replace(part + ".tmp", part)  # atomic: a killed run never leaves a half part
        del X, pairs, kept
        print(f"[{log}] {min(i + chunk_q, len(qrids)):,}/{len(qrids):,} queries, "
              f"{time.time() - t:.0f}s", flush=True)
    return pl.read_parquet(os.path.join(part_dir, f"part_{chunk_q}_*.parquet"))


def write_lists(s1, pairs, col, path):
    """s1: rid, entity_id of all S1 entities. pairs: e_rid, q_rid."""
    ids = pairs.join(pl.DataFrame({"q_rid": RIDS, "qid": EIDS}), on="q_rid")
    lists = (ids.group_by("e_rid").agg(pl.col("qid").unique().sort().str.join(",").alias(col)))
    out = (s1.join(lists, left_on="rid", right_on="e_rid", how="left")
           .select(pl.col("entity_id").alias("source1_entity_id"), pl.col(col).fill_null("")))
    out.write_csv(path, separator="\t", quote_style="never")
    print(f"[write] {path}: {out.height:,} rows, "
          f"{(out[col] != '').sum():,} non-empty")


def main():
    t = time.time()
    rec = load_records("test")
    s1 = rec.filter(pl.col("src") == 1)
    global RIDS, EIDS
    RIDS, EIDS = rec["rid"], rec["entity_id"]
    cfg = json.load(open(config.work(f"final{TAG}.json")))
    if not os.path.exists(config.work("test_scores_s1.parquet")) or "--rescore" in sys.argv:
        views = F.SparseViews(s1)
        m1 = lgb.Booster(model_file=config.work("model_s1.txt"))
        qrids = rec.filter(pl.col("src") != 1)["rid"].to_numpy()
        sc = score(config.work("test_cand.parquet"), qrids, rec, views, m1, log="test-s1")
        sc.write_parquet(config.work("test_scores_s1.parquet"))
        del views
    sc = pl.read_parquet(config.work("test_scores_s1.parquet"))
    if cfg["stage"] == 2:
        m2 = lgb.Booster(model_file=config.work(f"model_s2{TAG}.txt"))
        # stage-2 features depend only on the cached stage-1 scores, so they are cached too:
        # a retrained stage-2 model or a new decision rule then takes minutes, not a rebuild
        fpath = config.work("test_s2_feats.parquet")
        X2 = pl.read_parquet(fpath) if os.path.exists(fpath) else None
        if X2 is None or not set(stage2.FEATURES) <= set(X2.columns):
            X2 = stage2.build(sc, rec, fine.init_idf(s1["core_sk"]))
            X2 = X2.select("q_rid", "e_rid", *stage2.FEATURES)
            X2.write_parquet(fpath)
        p = m2.predict(X2.select(stage2.FEATURES).to_numpy().astype(np.float32),
                       num_threads=config.N_JOBS)
        r = cfg.get("shift", 1.0)  # prior shift: odds multiplier for test's confuser density
        p = r * p / (r * p + 1 - p)
        sc = X2.select("q_rid", "e_rid").with_columns(pl.Series("p", p))
        del X2
    pred = decide(sc) if cfg.get("rule") == "expf" else assign(sc, cfg["thr"])
    print(f"[predict] {pred.height:,} matched queries of {sc['q_rid'].n_unique():,} "
          f"(rule {cfg.get('rule', 'thr')}, thr {cfg['thr']}, shift {cfg.get('shift', 1.0)}, "
          f"stage {cfg['stage']})  "
          f"{time.time() - t:.0f}s")
    s1ids = s1.select("rid", "entity_id")
    write_lists(s1ids, pred.select("e_rid", "q_rid"), "matched_entity_ids",
                os.path.join(config.OUT_DIR, "matching_results.tsv"))
    # candidate set = exactly the pairs the final (stage-2) model runs inference over
    write_lists(s1ids, sc.select("e_rid", "q_rid"), "candidate_entity_ids",
                os.path.join(config.OUT_DIR, "candidate_pairs.tsv"))
    print(f"done {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
