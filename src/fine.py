"""Fine-grained pair features, computed only for pairs that survive stage-1 pruning.

Targeted at "confuser" records: same address as a real entity but one name word
swapped, a nearby house number, or a different legal form.
"""
import math
import re
from collections import Counter

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from normalize import LEGAL, skeleton

# transliterated abbreviations of legal words ("प्रा. लि." -> "pra li")
LEGAL_EXTRA = {"pra", "li", "lt", "limi", "pvt", "prai", "praivet", "limited", "limitet",
               "elelpi", "elelsi", "inkarporated", "korp", "kampani", "kampni"}
TITLES = {"md", "do", "dds", "dmd", "dls", "dvm", "phd", "cpa", "esq", "rn", "np", "pa", "od",
          "dc", "jr", "sr", "ii", "iii", "iv"}
_W = re.compile(r"[a-z0-9]+")

_IDF = None
_NDOC = 1


def init_idf(core_sk_series: pl.Series):
    """Skeleton-word document frequencies over Source 1 names."""
    c = Counter()
    for s in core_sk_series.to_list():
        if s:
            c.update(set(s.split()))
    return dict(c), len(core_sk_series)


def _set_idf(idf, n):
    global _IDF, _NDOC
    _IDF, _NDOC = idf, n


def _idf(sk):
    return math.log((_NDOC + 1) / (_IDF.get(sk, 0) + 1))


def _match(w, others):
    """Best similarity of word w to any word in others (0..100)."""
    best = 0.0
    for o in others:
        if w == o:
            return 100.0
        r = fuzz.ratio(w, o)
        if r > best:
            best = r
    return best


def _align(a, b):
    """a, b: lists of core words. Returns stats about unmatched words of a vs b."""
    if not a:
        return 0, 0.0, 0.0, 0, 100.0
    bsk = [skeleton(x) for x in b]
    un, un_idf, tot_idf, un_len, worst = 0, 0.0, 0.0, 0, 100.0
    for w in a:
        sk = skeleton(w)
        wi = _idf(sk)
        tot_idf += wi
        s = 100.0 if sk in bsk else _match(w, b)
        worst = min(worst, s)
        if s < 75:
            un += 1
            un_idf += wi
            un_len = max(un_len, len(w))
    return un, un_idf, un_idf / max(tot_idf, 1e-6), un_len, worst


def _nums(s):
    return [int(x) for x in (s or "").split() if x.isdigit() and len(x) < 10]


def _legal_toks(full):
    return {w for w in _W.findall(full or "") if w in LEGAL or w in LEGAL_EXTRA}


def _title_toks(full):
    return {w for w in _W.findall(full or "") if w in TITLES}


def _one(r):
    cq, ce, fq, fe, nq, ne = r
    wq = [w for w in (cq or "").split() if w not in LEGAL_EXTRA]
    we = [w for w in (ce or "").split() if w not in LEGAL_EXTRA]
    uq = _align(wq, we)
    ue = _align(we, wq)
    first_eq = float(bool(wq and we and (wq[0] == we[0] or skeleton(wq[0]) == skeleton(we[0]))))
    lq, le = _legal_toks(fq), _legal_toks(fe)
    tq, te = _title_toks(fq), _title_toks(fe)
    Nq, Ne = _nums(nq), _nums(ne)
    if Nq and Ne:
        h = Nq[0]
        dmin = min(abs(h - x) for x in Ne)
        hn = (float(h == Ne[0]), math.log1p(dmin), math.log1p(abs(h - Ne[0])),
              float(len(set(Ne) - set(Nq))))
    else:
        hn = (-1.0, -1.0, -1.0, float(len(Ne)))
    return (*uq, *ue, first_eq, float(len(lq - le)), float(len(le - lq)),
            float(len(tq ^ te)), float(bool(tq & te)), *hn)


FINE = ["fq_un", "fq_unidf", "fq_unfrac", "fq_unlen", "fq_worst",
        "fe_un", "fe_unidf", "fe_unfrac", "fe_unlen", "fe_worst",
        "f_first_eq", "f_legal_qonly", "f_legal_eonly", "f_title_diff", "f_title_shared",
        "f_hn_eq", "f_hn_dmin", "f_hn_dfirst", "f_num_eonly"]


def build(pairs: pl.DataFrame, rec: pl.DataFrame, idf) -> pl.DataFrame:
    """pairs: q_rid, e_rid. rec: records (rid, core, name_full, nums)."""
    cols = ["rid", "core", "name_full", "nums"]
    P = (pairs.select("q_rid", "e_rid")
         .join(rec.select(cols).rename({c: c + "_q" for c in cols}), left_on="q_rid",
               right_on="rid_q", how="left")
         .join(rec.select(cols).rename({c: c + "_e" for c in cols}), left_on="e_rid",
               right_on="rid_e", how="left"))
    _set_idf(*idf)
    # batched so only 500k pairs are ever held as Python objects (10M+ at once exhausts RAM)
    arr = np.empty((P.height, len(FINE)), dtype=np.float32)
    for s in range(0, P.height, 500_000):
        C = P.slice(s, 500_000)
        rows = zip(C["core_q"].to_list(), C["core_e"].to_list(), C["name_full_q"].to_list(),
                   C["name_full_e"].to_list(), C["nums_q"].to_list(), C["nums_e"].to_list())
        arr[s:s + C.height] = np.asarray([_one(r) for r in rows],
                                         dtype=np.float32).reshape(-1, len(FINE))
    return P.select("q_rid", "e_rid").with_columns(
        [pl.Series(n, arr[:, i]) for i, n in enumerate(FINE)])
