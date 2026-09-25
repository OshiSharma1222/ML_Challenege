"""Stage 2 re-scorer: stage-1 probabilities in context.

Query-side context:  how dominant this S1 candidate is among the query's candidates.
Entity-side context: how many other records point strongly at the same S1 entity and
                     how this record ranks among them.
"""
import polars as pl

import fine

PRUNE = 0.001  # stage-1 pairs below this probability are dropped before stage 2

CARRY = ["n_tset", "n_ratio", "sk_ratio", "cos_name", "cos_addr", "cos_num", "a_tset",
         "bs_f", "br_f", "q_noaddr", "q_native", "num_first_eq", "legal_eq", "q_ncand"]


def build(sc: pl.DataFrame, rec: pl.DataFrame, idf) -> pl.DataFrame:
    q, e = "q_rid", "e_rid"
    # q_max is materialised first: a window nested inside another window is quadratic in
    # polars (hours on the 10M test pairs), a window over a plain column is linear.
    X = sc.with_columns(pl.col("p").max().over(q).alias("q_max"))
    m1 = pl.col("q_max")
    n1 = (pl.col("p") == m1).sum().over(q)
    m2 = pl.when(pl.col("p") < m1).then(pl.col("p")).max().over(q).fill_null(0)
    other = pl.when((pl.col("p") == m1) & (n1 == 1)).then(m2).otherwise(m1)
    X = X.with_columns(
        (pl.col("p") - other).alias("q_gap"),
        pl.col("p").sum().over(q).alias("q_sum"),
        (pl.col("p") > 0.1).sum().over(q).cast(pl.Float32).alias("q_n10"),
        pl.col("p").rank("min", descending=True).over(q).cast(pl.Float32).alias("q_rank"),
    )
    X = X.with_columns(((pl.col("q_rank") == 1) & (pl.col("p") > 0.5)).alias("_win"))
    X = X.with_columns(
        (pl.col("_win").cast(pl.Float32).sum().over(e) - pl.col("_win").cast(pl.Float32))
        .alias("e_nwin_other"),
        pl.col("p").sum().over(e).alias("e_psum"),
        pl.len().over(e).cast(pl.Float32).alias("e_ncand"),
        pl.col("p").rank("min", descending=True).over(e).cast(pl.Float32).alias("e_rank"),
        pl.col("p").max().over(e).alias("e_pmax"),
    ).drop("_win")
    return X.join(fine.build(X.select("q_rid", "e_rid"), rec, idf), on=["q_rid", "e_rid"])


# e_ncand is computed but not used: during validation only a subset of an entity's
# candidate queries is scored, so its distribution would differ from test.
FEATURES = ["p", "q_gap", "q_max", "q_sum", "q_n10", "q_rank", "e_nwin_other", "e_psum",
            "e_rank", "e_pmax"] + CARRY + fine.FINE
