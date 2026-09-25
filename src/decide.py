"""Entity-level decision rule that maximises expected macro F0.5.

Every query first picks its best Source 1 candidate. For one entity, let its
queries have match probabilities p_1 >= p_2 >= ... Predicting the top k of them gives
    F0.5 = 1.25 * TP / (0.25 * NT + k)      (F = 1 when k = NT = 0)
where TP = true matches among the k and NT = all true matches of the entity. The
expectation of F under independent Bernoulli(p_i) outcomes is estimated by sampling,
and each entity keeps the k with the highest expected F (k = 0 included). Low
probabilities are shrunk by `alpha` (p ** alpha) to correct calibration.
"""
import numpy as np
import polars as pl

N_SAMPLES = 256
MAX_K = 24  # queries per entity considered; the rest are never predicted


def _best_k(P, rng):
    """P: (n_ent, m) probabilities sorted descending, 0-padded. Returns best k per entity."""
    n, m = P.shape
    best_f = np.zeros(n)
    best_k = np.zeros(n, dtype=np.int64)
    for s in range(0, n, 2048):
        p = P[s:s + 2048]
        X = rng.random((N_SAMPLES,) + p.shape) < p            # (S, n, m) outcomes
        nt = X.sum(-1)                                         # (S, n)
        tp = np.cumsum(X, -1)                                  # TP when predicting top k+1
        k = np.arange(1, m + 1)
        f = (1.25 * tp / (0.25 * nt[..., None] + k)).mean(0)  # (n, m) expected F for k=1..m
        f0 = (nt == 0).mean(0)                                 # k = 0: F = 1 only if no match
        f = np.where(p > 0, f, -1.0)                           # padded slots are not queries
        kk = f.argmax(-1)
        fb = f[np.arange(len(p)), kk]
        take = fb > f0
        best_f[s:s + 2048] = np.where(take, fb, f0)
        best_k[s:s + 2048] = np.where(take, kk + 1, 0)
    return best_k


def decide(scores: pl.DataFrame, alpha=1.0, floor=0.02, seed=0) -> pl.DataFrame:
    """scores: q_rid, e_rid, p (all scored pairs). Returns assigned (q_rid, e_rid, p)."""
    best = (scores.sort("p", descending=True).group_by("q_rid", maintain_order=True).first()
            .filter(pl.col("p") >= floor)
            .with_columns((pl.col("p") ** alpha).alias("pa")))
    g = (best.sort(["e_rid", "pa"], descending=[False, True])
         .with_columns(pl.int_range(pl.len()).over("e_rid").alias("r"))
         .filter(pl.col("r") < MAX_K))
    ents = g["e_rid"].unique(maintain_order=True)
    row = pl.DataFrame({"e_rid": ents, "i": np.arange(len(ents))})
    g = g.join(row, on="e_rid")
    m = int(g["r"].max()) + 1
    P = np.zeros((len(ents), m), dtype=np.float32)
    P[g["i"].to_numpy(), g["r"].to_numpy()] = g["pa"].to_numpy()
    k = _best_k(P, np.random.default_rng(seed))
    keep = g.join(pl.DataFrame({"i": np.arange(len(ents)), "k": k}), on="i")
    return keep.filter(pl.col("r") < pl.col("k")).select("q_rid", "e_rid", "p")
