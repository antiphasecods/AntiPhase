"""Balanced two-block (contextual) stochastic block model parametrised by the defect.

With blocks of equal size and expected degree d, the mean-field normalized
adjacency has eigenvalues {1, (p_in - p_out)/(p_in + p_out)} plus a bulk.  Fixing
the expected degree d and writing eps = 2 p_in / (p_in + p_out) gives

    p_in  = eps       * d / n,      p_out = (2 - eps) * d / n,
    lambda_n(E[P]) = -(1 - eps),

so eps is a direct experimental dial on the bipartiteness defect:
eps = 0 is an exact (random) bipartite graph, eps = 1 is Erdos-Renyi.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def sbm_probs(eps: float, davg: float, n: int):
    s = 2.0 * davg / n                    # p_in + p_out
    return eps * s / 2.0, (2.0 - eps) * s / 2.0


def sample_sbm(n: int, eps: float, davg: float, rng: np.random.Generator):
    """Sparse symmetric adjacency of a balanced 2-block SBM with defect eps."""
    p_in, p_out = sbm_probs(eps, davg, n)
    half = n // 2
    blocks = np.zeros(n, dtype=np.int64)
    blocks[half:] = 1
    rows, cols = [], []
    for (i0, i1), (j0, j1), p in (((0, half), (0, half), p_in),
                                  ((half, n), (half, n), p_in),
                                  ((0, half), (half, n), p_out)):
        ni, nj = i1 - i0, j1 - j0
        same = (i0 == j0)
        exp_edges = p * (ni * (ni - 1) / 2 if same else ni * nj)
        k = rng.poisson(exp_edges)
        r = rng.integers(i0, i1, size=k)
        c = rng.integers(j0, j1, size=k)
        rows.append(r); cols.append(c)
    r = np.concatenate(rows); c = np.concatenate(cols)
    keep = r != c
    r, c = r[keep], c[keep]
    A = sp.coo_matrix((np.ones(r.size), (r, c)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float64).tocsr()
    A.setdiag(0.0); A.eliminate_zeros()
    return A, blocks


def csbm_features(blocks, d: int, mu: float, rng: np.random.Generator):
    """Contextual features: x_i = mu * u * s_i + noise/sqrt(d), s_i = +-1 by block."""
    n = blocks.size
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    s = np.where(blocks == 0, 1.0, -1.0)
    return mu * np.outer(s, u) + rng.standard_normal((n, d)) / np.sqrt(d)


def edge_homophily(A, labels) -> float:
    A = sp.coo_matrix(sp.triu(A, k=1))
    if A.nnz == 0:
        return float("nan")
    return float((labels[A.row] == labels[A.col]).mean())
