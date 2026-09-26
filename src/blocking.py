"""Stage 2: candidate generation.

Every record becomes a sparse IDF-weighted bag of blocking tokens:
  n|<skeleton word of name core>      (phonetic name words, legal suffixes removed)
  j|<despaced name skeleton>          (catches "colonialfoods.com" vs "Colonial Foods")
  a|<address word skeleton / number>  (non-generic address words)
  b|<adjacent address pair>           (e.g. "17560_ls" = house number + street)
all prefixed by the record's country so different countries never collide.
IDF comes from Source 1. For each Source 2/3 record (query) we take the top-K
Source 1 records by cosine similarity (sparse_dot_topn, multithreaded).
Because each S2/S3 record belongs to at most one S1 entity, blocking is
query-centric: K candidates per S2/S3 record.
"""
import os
import sys
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

import config
from normalize import skeleton, ADDR_GENERIC

MAX_DF = 3000  # tokens more frequent than this in S1 are dropped from the index
K_FULL = 20    # candidates per query from the name+address index
K_NAME = 10    # candidates per query from the name-only index
# exact-spelling name words next to the skeletons: the skeleton merges distinct names
# ("sweven", "seven", "shivam" -> "svn"), which crowds the true entity out of the top K
RAW_NAME = os.environ.get("ER_BLOCK_RAW", "0") == "1"


def _skel_map(words: pl.Series) -> pl.DataFrame:
    u = words.unique().drop_nulls().to_list()
    return pl.DataFrame({"w": u, "sk": [("#" + w) if w.isdigit() else skeleton(w) for w in u]},
                        schema={"w": pl.Utf8, "sk": pl.Utf8})


def _words(base, col):
    return (base.select("rid", "cty", pl.col(col).str.split(" ").alias("w"))
            .explode("w", empty_as_null=True)
            .filter(pl.col("w").is_not_null() & (pl.col("w").str.len_chars() > 0)))


def _bigrams(w, kind):
    return (w.with_columns(pl.col("sk").shift(-1).over("rid").alias("nx"))
            .filter(pl.col("nx").is_not_null())
            .select("rid", (pl.col("cty") + kind + pl.col("sk") + "_" + pl.col("nx")).alias("tok")))


def tokenize(df: pl.DataFrame, with_addr=True) -> pl.DataFrame:
    """df needs rid, country, core_sk, alt, addr. Returns (rid, tok)."""
    base = df.select("rid", pl.col("country").str.to_lowercase().alias("cty"),
                     "core_sk", "alt", "addr")
    nw = _words(base, "core_sk").with_columns(pl.col("w").alias("sk"))
    parts = [nw.select("rid", (pl.col("cty") + "|n|" + pl.col("sk")).alias("tok")),
             _bigrams(nw, "|m|")]
    altw = _words(base.filter(pl.col("alt").str.len_chars() > 0), "alt")
    if altw.height:
        altw = altw.join(_skel_map(altw["w"]), on="w", how="inner", maintain_order="left")
        parts += [altw.select("rid", (pl.col("cty") + "|n|" + pl.col("sk")).alias("tok")),
                  _bigrams(altw, "|m|")]
    parts.append(base.filter(pl.col("core_sk").str.contains(" "))
                 .select("rid", (pl.col("cty") + "|n|" + pl.col("core_sk").str.replace_all(" ", ""))
                         .alias("tok")))
    if with_addr:
        aw = _words(base, "addr").filter(~pl.col("w").is_in(list(ADDR_GENERIC)))
        aw = aw.join(_skel_map(aw["w"]), on="w", how="left", maintain_order="left")
        parts += [aw.select("rid", (pl.col("cty") + "|a|" + pl.col("sk")).alias("tok")),
                  _bigrams(aw, "|b|")]
    return pl.concat(parts).unique(["rid", "tok"])


def block_tokens(df: pl.DataFrame, with_addr=True) -> pl.DataFrame:
    """Blocking tokens: tokenize() plus, with RAW_NAME, exact name words and word pairs."""
    toks = tokenize(df, with_addr)
    if not RAW_NAME:
        return toks
    rw = _words(df.select("rid", pl.col("country").str.to_lowercase().alias("cty"), "core"),
                "core").with_columns(pl.col("w").alias("sk"))
    return pl.concat([toks, rw.select("rid", (pl.col("cty") + "|r|" + pl.col("w")).alias("tok")),
                      _bigrams(rw, "|s|")]).unique(["rid", "tok"])


class Index:
    def __init__(self, s1: pl.DataFrame, with_addr=True):
        t = time.time()
        self.with_addr = with_addr
        toks = block_tokens(s1, with_addr)
        n1 = s1.height
        df = toks.group_by("tok").agg(pl.len().alias("df")).filter(pl.col("df") <= MAX_DF)
        df = df.with_columns((np.log(n1 + 1) - pl.col("df").cast(pl.Float64).log()).alias("w"))
        self.vocab = df.with_row_index("col").select("tok", "col", "w")
        self.s1_rids = s1["rid"].to_numpy()
        self.rid2row = pl.DataFrame({"rid": self.s1_rids,
                                     "row": np.arange(n1, dtype=np.uint32)})
        self.ET = self._matrix(toks, self.rid2row, n1).T.tocsr()
        print(f"[index] addr={with_addr} S1={n1:,} vocab={self.vocab.height:,} "
              f"nnz={self.ET.nnz:,} {time.time() - t:.0f}s", flush=True)

    def _matrix(self, toks, rid2row, nrows):
        m = (toks.join(self.vocab, on="tok", how="inner").join(rid2row, on="rid", how="inner"))
        M = sp.csr_matrix((m["w"].to_numpy().astype(np.float32),
                           (m["row"].to_numpy(), m["col"].to_numpy().astype(np.int64))),
                          shape=(nrows, self.vocab.height))
        norm = np.sqrt(M.multiply(M).sum(axis=1)).A1
        norm[norm == 0] = 1
        return sp.diags(1 / norm).dot(M).tocsr().astype(np.float32)

    def query(self, q: pl.DataFrame, k: int, threads: int, tag: str):
        toks = block_tokens(q, self.with_addr)
        rid2row = pl.DataFrame({"rid": q["rid"].to_numpy(),
                                "row": np.arange(q.height, dtype=np.uint32)})
        Q = self._matrix(toks, rid2row, q.height)
        R = sp_matmul_topn(Q, self.ET, top_n=k, threshold=0.01, sort=True,
                           n_threads=threads).tocoo()
        out = pl.DataFrame({"q_rid": q["rid"].to_numpy()[R.row],
                            "e_rid": self.s1_rids[R.col],
                            f"bs_{tag}": R.data.astype(np.float32)})
        return out.with_columns(
            pl.col(f"bs_{tag}").rank("ordinal", descending=True).over("q_rid").cast(pl.UInt8)
            .alias(f"br_{tag}"))


def generate(s1, qs, chunk=500_000, log=""):
    """Union of name+address and name-only top-K candidates for every query record."""
    full = Index(s1, True)
    name = Index(s1, False)
    outs = []
    t = time.time()
    for i in range(0, qs.height, chunk):
        part = qs.slice(i, chunk)
        a = full.query(part, K_FULL, config.N_JOBS, "f")
        b = name.query(part, K_NAME, config.N_JOBS, "n")
        outs.append(a.join(b, on=["q_rid", "e_rid"], how="full", coalesce=True))
        print(f"[block] {log} {min(i + chunk, qs.height):,}/{qs.height:,} {time.time() - t:.0f}s",
              flush=True)
    c = pl.concat(outs)
    return c.with_columns(pl.col("bs_f").fill_null(0), pl.col("bs_n").fill_null(0),
                          pl.col("br_f").fill_null(99), pl.col("br_n").fill_null(99))


def load(split):
    return pl.read_parquet(config.work(f"{split}_records.parquet"))


def run(split, sample=None):
    rec = load(split)
    s1 = rec.filter(pl.col("src") == 1)
    qs = rec.filter(pl.col("src") != 1)
    if sample:
        qs = qs.sample(sample, seed=7)
    cand = generate(s1, qs, log=split)
    name = f"{split}_cand{'_sample' if sample else ''}.parquet"
    cand.write_parquet(config.work(name))
    print(f"[block] wrote {name}: {cand.height:,} pairs")
    return cand


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
