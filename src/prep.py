"""Stage 1: read raw TSVs and write normalised parquet files (one per split)."""
import sys
import time
from multiprocessing import Pool

import polars as pl

import config
from normalize import name_parts, addr_parts


def read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       null_values=[""])


def _norm_chunk(rows):
    out = []
    for name, addr, country in rows:
        full, core, sk, lk, alt = name_parts(name)
        atext, _, nums, state = addr_parts(addr, country)
        out.append((full, core, sk, lk, alt, atext, " ".join(nums), state))
    return out


COLS = ["name_full", "core", "core_sk", "legal", "alt", "addr", "nums", "state"]


def normalise(df, pool):
    rows = list(zip(df["business_name"].to_list(), df["business_address"].to_list(),
                    df["country"].to_list()))
    step = 20000
    chunks = [rows[i:i + step] for i in range(0, len(rows), step)]
    res = []
    for r in pool.imap(_norm_chunk, chunks, chunksize=1):
        res.extend(r)
    cols = list(zip(*res)) if res else [[] for _ in COLS]
    return df.with_columns([pl.Series(c, list(v), dtype=pl.Utf8) for c, v in zip(COLS, cols)])


def run(split):
    t = time.time()
    frames = []
    with Pool(config.N_JOBS) as pool:
        for src in (1, 2, 3):
            df = read_tsv(config.raw_path(split, src))
            df = df.with_columns(pl.lit(src, dtype=pl.Int8).alias("src"),
                                 pl.col("country").fill_null("").str.strip_chars())
            df = normalise(df, pool)
            frames.append(df)
            print(f"[prep] {split} source{src}: {df.height:,} rows  {time.time() - t:.0f}s",
                  flush=True)
    allr = pl.concat(frames)
    allr = allr.with_row_index("rid")  # dense integer id used everywhere downstream
    allr.write_parquet(config.work(f"{split}_records.parquet"))
    print(f"[prep] wrote {split}_records.parquet ({allr.height:,})  {time.time() - t:.0f}s")


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["train", "test"]):
        run(s)
