"""Stage 3: pairwise features for (query record, Source 1 record) candidate pairs.

All features are country-agnostic similarity measurements, so the model transfers
to countries unseen in training (France in the test set).
"""
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

import config
from blocking import tokenize

VIEWS = {"name": ("n", "m"), "addr": ("a", "b"), "num": ("a#",)}
REC_COLS = ["rid", "src", "business_name", "core", "core_sk", "name_full", "alt", "legal",
            "addr", "nums", "state"]


def _view_of(tok: pl.Expr) -> pl.Expr:
    kind = tok.str.split("|").list.get(1)
    val = tok.str.split("|").list.get(2)
    return (pl.when((kind == "a") & val.str.starts_with("#")).then(pl.lit("num"))
            .when(kind.is_in(["n", "m"])).then(pl.lit("name"))
            .otherwise(pl.lit("addr")))


class SparseViews:
    """IDF-weighted token matrices per view; Source 1 side precomputed."""

    def __init__(self, s1: pl.DataFrame):
        t = time.time()
        toks = tokenize(s1).with_columns(_view_of(pl.col("tok")).alias("view"))
        n1 = s1.height
        voc = (toks.group_by("tok", "view").agg(pl.len().alias("df"))
               .with_columns((np.log(n1 + 1) - pl.col("df").cast(pl.Float64).log())
                             .cast(pl.Float32).alias("w")))
        self.vocab = {}
        for v in ("name", "addr", "num"):
            self.vocab[v] = voc.filter(pl.col("view") == v).with_row_index("col").select(
                "tok", "col", "w")
        del voc
        self.s1_row = pl.DataFrame({"rid": s1["rid"], "row": np.arange(n1, dtype=np.int64)})
        self.E = {}
        for v in ("name", "addr", "num"):
            self.E[v] = self._mats(toks.filter(pl.col("view") == v), v, self.s1_row, n1)
        del toks
        print(f"[views] built {time.time() - t:.0f}s", flush=True)

    def _mats(self, toks, view, rowmap, nrows):
        voc = self.vocab[view]
        m = toks.filter(pl.col("view") == view).join(voc, on="tok").join(rowmap, on="rid")
        m = m.select("row", "col", "w")
        r, c = m["row"].to_numpy(), m["col"].to_numpy().astype(np.int64)
        W = sp.csr_matrix((m["w"].to_numpy(), (r, c)), shape=(nrows, voc.height), dtype=np.float32)
        B = W.copy()
        B.data[:] = 1
        return W, B, np.asarray(W.sum(1)).ravel(), np.sqrt(np.asarray(W.multiply(W).sum(1)).ravel())

    def query_mats(self, q: pl.DataFrame):
        toks = tokenize(q).with_columns(_view_of(pl.col("tok")).alias("view"))
        rowmap = pl.DataFrame({"rid": q["rid"], "row": np.arange(q.height, dtype=np.int64)})
        return {v: self._mats(toks, v, rowmap, q.height) for v in self.vocab}, rowmap


def _rowdot(A, ia, B, ib):
    return np.asarray(A[ia].multiply(B[ib]).sum(1)).ravel().astype(np.float32)


def _cd(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=config.N_JOBS, dtype=np.float32, **kw)


def _group(df, cols, key="q_rid"):
    """For each col: margin over the best *other* candidate of the same query."""
    ex = []
    for c in cols:
        m1 = pl.col(c).max().over(key)
        n1 = (pl.col(c) == m1).sum().over(key)
        m2 = pl.col(c).filter(pl.col(c) < m1).max().over(key).fill_null(0)
        other = pl.when((pl.col(c) == m1) & (n1 == 1)).then(m2).otherwise(m1)
        ex.append((pl.col(c) - other).alias(f"{c}_gap"))
        ex.append(pl.col(c).rank("min", descending=True).over(key).cast(pl.Float32)
                  .alias(f"{c}_rk"))
    return df.with_columns(ex)


GROUP_COLS = ["n_tset", "n_ratio", "cos_name", "cos_addr", "cos_num", "a_tset", "bs_f"]


def build(pairs: pl.DataFrame, rec: pl.DataFrame, views: SparseViews) -> pl.DataFrame:
    """pairs: q_rid, e_rid, bs_f, br_f, bs_n, br_n. rec: records table (indexed by rid)."""
    qids = pairs["q_rid"].unique().sort()
    q = rec.join(pl.DataFrame({"rid": qids}), on="rid").sort("rid")
    qm, qrow = views.query_mats(q)
    P = (pairs.join(qrow.rename({"rid": "q_rid", "row": "qi"}), on="q_rid")
         .join(views.s1_row.rename({"rid": "e_rid", "row": "ei"}), on="e_rid"))
    qi, ei = P["qi"].to_numpy(), P["ei"].to_numpy()
    feats = {}
    for v, (QW, QB, qsum, qnorm) in qm.items():
        EW, EB, esum, enorm = views.E[v]
        dw2 = _rowdot(QW, qi, EW, ei)
        dw = _rowdot(QW, qi, EB, ei)
        feats[f"cos_{v}"] = dw2 / np.maximum(qnorm[qi] * enorm[ei], 1e-6)
        feats[f"covq_{v}"] = dw / np.maximum(qsum[qi], 1e-6)
        feats[f"cove_{v}"] = np.asarray(EW[ei].multiply(QB[qi]).sum(1)).ravel() / np.maximum(
            esum[ei], 1e-6)
        feats[f"nq_{v}"] = np.diff(QB.indptr)[qi].astype(np.float32)
        feats[f"ne_{v}"] = np.diff(EB.indptr)[ei].astype(np.float32)
    P = P.with_columns([pl.Series(k, v) for k, v in feats.items()])

    cols = REC_COLS
    P = (P.join(rec.select(cols).rename({c: c + "_q" for c in cols}), left_on="q_rid",
                right_on="rid_q")
         .join(rec.select(cols).rename({c: c + "_e" for c in cols}), left_on="e_rid",
               right_on="rid_e"))

    def core(side):
        return pl.when(pl.col(f"core_{side}").str.len_chars() > 0).then(pl.col(f"core_{side}")) \
            .otherwise(pl.col(f"name_full_{side}"))
    P = P.with_columns(core("q").alias("cq"), core("e").alias("ce"))
    cq, ce = P["cq"].fill_null("").to_list(), P["ce"].fill_null("").to_list()
    sq, se = P["core_sk_q"].fill_null("").to_list(), P["core_sk_e"].fill_null("").to_list()
    fq, fe = P["name_full_q"].fill_null("").to_list(), P["name_full_e"].fill_null("").to_list()
    aq, ae = P["addr_q"].fill_null("").to_list(), P["addr_e"].fill_null("").to_list()
    dq = [x.replace(" ", "") for x in cq]
    de = [x.replace(" ", "") for x in ce]
    f = {
        "n_ratio": _cd(cq, ce, fuzz.ratio),
        "n_tsort": _cd(cq, ce, fuzz.token_sort_ratio),
        "n_tset": _cd(cq, ce, fuzz.token_set_ratio),
        "n_partial": _cd(cq, ce, fuzz.partial_ratio),
        "n_jw": _cd(cq, ce, JaroWinkler.normalized_similarity),
        "n_despace": _cd(dq, de, fuzz.ratio),
        "n_despace_part": _cd(dq, de, fuzz.partial_ratio),
        "sk_ratio": _cd(sq, se, fuzz.ratio),
        "sk_tset": _cd(sq, se, fuzz.token_set_ratio),
        "full_tsort": _cd(fq, fe, fuzz.token_sort_ratio),
        "a_ratio": _cd(aq, ae, fuzz.ratio),
        "a_tset": _cd(aq, ae, fuzz.token_set_ratio),
        "a_tsort": _cd(aq, ae, fuzz.token_sort_ratio),
        "a_partial_tset": _cd(aq, ae, fuzz.partial_token_set_ratio),
    }
    altq, alte = P["alt_q"].fill_null("").to_list(), P["alt_e"].fill_null("").to_list()
    alt = np.maximum(_cd(altq, ce, fuzz.token_set_ratio), _cd(cq, alte, fuzz.token_set_ratio))
    has_alt = (P["alt_q"].fill_null("").str.len_chars() > 0) | (
        P["alt_e"].fill_null("").str.len_chars() > 0)
    f["n_alt"] = np.where(has_alt.to_numpy(), alt, -1).astype(np.float32)
    P = P.with_columns([pl.Series(k, v) for k, v in f.items()])

    nq = pl.col("nums_q").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    ne = pl.col("nums_e").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    P = P.with_columns(nq.alias("NQ"), ne.alias("NE"))
    P = P.with_columns(
        pl.col("NQ").list.len().cast(pl.Float32).alias("num_nq"),
        pl.col("NE").list.len().cast(pl.Float32).alias("num_ne"),
        pl.col("NQ").list.set_intersection("NE").list.len().cast(pl.Float32).alias("num_shared"),
        (pl.col("NQ").list.first() == pl.col("NE").list.first()).cast(pl.Float32)
        .fill_null(-1).alias("num_first_eq"),
        pl.col("NQ").list.first().is_in(pl.col("NE")).cast(pl.Float32).fill_null(-1)
        .alias("num_first_in"),
        pl.when((pl.col("legal_q") == "") | (pl.col("legal_e") == "")).then(0)
        .when(pl.col("legal_q") == pl.col("legal_e")).then(1).otherwise(-1)
        .cast(pl.Float32).alias("legal_eq"),
        pl.when((pl.col("state_q") == "") | (pl.col("state_e") == "")).then(0)
        .when(pl.col("state_q") == pl.col("state_e")).then(1).otherwise(-1)
        .cast(pl.Float32).alias("state_eq"),
        (pl.col("addr_q").fill_null("").str.len_chars() == 0).cast(pl.Float32).alias("q_noaddr"),
        pl.col("business_name_q").str.contains(r"[^\x00-\x{024F}]").cast(pl.Float32)
        .alias("q_native"),
        pl.col("cq").str.len_chars().cast(pl.Float32).alias("len_cq"),
        pl.col("ce").str.len_chars().cast(pl.Float32).alias("len_ce"),
        pl.col("cq").str.count_matches(" ").cast(pl.Float32).alias("ntok_cq"),
        pl.col("addr_q").fill_null("").str.len_chars().cast(pl.Float32).alias("len_aq"),
        pl.col("addr_e").fill_null("").str.len_chars().cast(pl.Float32).alias("len_ae"),
        (pl.col("src_q") == 3).cast(pl.Float32).alias("q_src3"),
        pl.len().over("q_rid").cast(pl.Float32).alias("q_ncand"),
    )
    P = P.with_columns(
        (pl.col("num_shared") / pl.max_horizontal(pl.col("num_nq"), 1)).alias("num_covq"),
        (pl.col("num_nq") - pl.col("num_shared")).alias("num_qonly"),
    )
    P = _group(P, GROUP_COLS)
    return P.select(["q_rid", "e_rid"] + FEATURES)


FEATURES = (
    ["bs_f", "br_f", "bs_n", "br_n"]
    + [f"{k}_{v}" for v in ("name", "addr", "num") for k in ("cos", "covq", "cove", "nq", "ne")]
    + ["n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw", "n_despace", "n_despace_part",
       "sk_ratio", "sk_tset", "full_tsort", "a_ratio", "a_tset", "a_tsort", "a_partial_tset",
       "n_alt", "num_nq", "num_ne", "num_shared", "num_first_eq", "num_first_in", "legal_eq",
       "state_eq", "q_noaddr", "q_native", "len_cq", "len_ce", "ntok_cq", "len_aq", "len_ae",
       "q_src3", "q_ncand", "num_covq", "num_qonly"]
    + [f"{c}_{s}" for c in GROUP_COLS for s in ("gap", "rk")]
)
