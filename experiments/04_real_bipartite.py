"""Experiment 4 -- real graphs where the theory is exactly true, and a real-data
version of the central lifetime experiment.

Recommender GNNs (LightGCN, NGCF) propagate on user-item graphs with literally
P = D^-1/2 A D^-1/2, no self-loops and no non-linearity, i.e. X_t = P^t X_0.
Those graphs are bipartite by construction, so eps = 0 exactly.  Bipartite
spectra are symmetric (lambda_{n-1} = -lambda_2), hence

    delta^-  =  log(1 / lambda_2)  =  the ordinary spectral gap,

so the oscillation emerges in mixing time and then never decays.  We then
contaminate the graph with intra-side edges -- the "social recommendation"
setting -- which is the real-topology analogue of the CSBM defect dial, and
check the lifetime law t2 ~ log(.)/eps on real data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hetero_osc import diagnostics as dg, propagate as pg, spectra, theory  # noqa: E402
from hetero_osc.spectra import is_bipartite  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
T = 300
GAMMA, ETA, WIN = 0.10, 0.05, 4


def movielens():
    from torch_geometric.datasets import MovieLens100K
    d = MovieLens100K(str(ROOT / "data" / "ml100k"))[0]
    nu = d["user"].x.shape[0]
    nm = d["movie"].x.shape[0]
    ei = d["user", "rates", "movie"].edge_index.numpy()
    n = nu + nm
    rows = np.concatenate([ei[0], ei[1] + nu])
    cols = np.concatenate([ei[1] + nu, ei[0]])
    A = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64).tocsr()
    A.setdiag(0.0); A.eliminate_zeros()
    xu = d["user"].x.numpy().astype(np.float64)
    xm = d["movie"].x.numpy().astype(np.float64)
    X = np.zeros((n, xu.shape[1] + xm.shape[1]))
    X[:nu, :xu.shape[1]] = xu
    X[nu:, xu.shape[1]:] = xm
    side = np.zeros(n, dtype=np.int64); side[nu:] = 1
    return "MovieLens100K", A, X, side


def hetero_bipartite(name, cls, root, hub, dim=32, seed=0):
    """Any heterogeneous dataset whose edges all touch one 'hub' node type is
    bipartite: hub on one side, every other type on the other.

    DBLP  : all edges touch 'paper'  (author-paper, paper-term, paper-conference)
    IMDB  : all edges touch 'movie'  (movie-director, movie-actor)

    Per-type features are mapped to a common `dim`-dimensional space by a fixed
    random Gaussian projection (types without features get zeros); the operator,
    not the features, is what the theory is about.
    """
    d = cls(str(ROOT / root))[0]
    rng = np.random.default_rng(seed)
    offs, off = {}, 0
    for t in d.node_types:
        offs[t] = off
        off += int(d[t].num_nodes)
    n = off
    rows, cols, dropped = [], [], 0
    for et in d.edge_types:
        src, _, dst = et
        ei = d[et].edge_index.numpy()
        if (src == hub) == (dst == hub):        # not a cross-side edge
            dropped += ei.shape[1]
            continue
        rows.append(ei[0] + offs[src]); cols.append(ei[1] + offs[dst])
    r = np.concatenate(rows); c = np.concatenate(cols)
    A = sp.coo_matrix((np.ones(r.size), (r, c)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float64).tocsr()
    A.setdiag(0.0); A.eliminate_zeros()

    X = np.zeros((n, dim))
    for t in d.node_types:
        if "x" not in d[t]:
            continue
        xt = d[t].x.numpy().astype(np.float64)
        Wp = rng.standard_normal((xt.shape[1], dim)) / np.sqrt(xt.shape[1])
        X[offs[t]:offs[t] + xt.shape[0]] = xt @ Wp
    side = np.ones(n, dtype=np.int64)
    side[offs[hub]:offs[hub] + int(d[hub].num_nodes)] = 0
    if dropped:
        print(f"  [{name}] dropped {dropped} intra-side edges")
    ok, _ = is_bipartite(A)
    print(f"  [{name}] n={n} m={A.nnz//2} structurally bipartite: {ok}")
    return name, A, X, side


def synthetic_bipartite(n=3000, davg=20.0, seed=0):
    from hetero_osc import sbm
    rng = np.random.default_rng(seed)
    A, blocks = sbm.sample_sbm(n, 0.0, davg, rng)
    X = sbm.csbm_features(blocks, 16, 1.0, rng)
    return "SyntheticBipartite", A.tocsr(), X, blocks


def contaminate(A, side, q, rng):
    """Add q * m random intra-side edges (homophilic contamination)."""
    A = sp.csr_matrix(A).copy()
    m = int(A.nnz // 2)
    k = int(round(q * m))
    if k == 0:
        return A
    idx0 = np.flatnonzero(side == 0)
    idx1 = np.flatnonzero(side == 1)
    rows, cols = [], []
    for grp in (idx0, idx1):
        kk = k // 2
        r = rng.choice(grp, size=kk); c = rng.choice(grp, size=kk)
        rows.append(r); cols.append(c)
    r = np.concatenate(rows); c = np.concatenate(cols)
    keep = r != c
    B = sp.coo_matrix((np.ones(keep.sum()), (r[keep], c[keep])), shape=A.shape)
    A = ((A + B + B.T) > 0).astype(np.float64).tocsr()
    A.setdiag(0.0); A.eliminate_zeros()
    return A


def run(name, A, X, side, qs):
    rows = []
    rng = np.random.default_rng(0)
    for q in qs:
        Aq = contaminate(A, side, q, rng) if q > 0 else sp.csr_matrix(A)
        Aq, idx = spectra.largest_component(Aq)
        Xq = X[idx]
        P = spectra.normalized_adjacency(Aq)
        spec, _, vn = spectra.summarize_sparse(P)
        v1 = spectra.degree_mode(Aq)
        c1, cn, zn = spectra.decompose(Xq, v1, vn)

        diag = dg.stream_series(P, Xq, T, v1)
        t1, t2, life = dg.observation_window(diag, GAMMA, ETA)
        lo = int(t1) if np.isfinite(t1) else 10
        hi = int(t2) if np.isfinite(t2) else T - 5
        eps_ray = dg.eps_from_rayleigh(diag["R"], lo, max(hi, lo + 5))
        p1, p2, _ = theory.window(spec.eps, spec.rho, GAMMA, ETA,
                                  z_over_cn=zn / max(cn, 1e-12),
                                  cn_over_c1=cn / max(c1, 1e-12))
        rows.append({"dataset": name, "q": q, "n": spec.n, "m": int(Aq.nnz // 2),
                     "lam_n": spec.lam_n, "lam_2": spec.lam2, "eps": spec.eps,
                     "rho": spec.rho, "delta_minus": spec.delta_minus,
                     "bipartite_symmetry": abs(spec.lam_nm1 + spec.lam2),
                     "t1_obs": t1, "t2_obs": t2, "life": life,
                     "t1_pred": p1, "t2_pred": p2,
                     "eps_rayleigh": eps_ray, "O_max": float(np.nanmax(diag["O_prime"]))})
        print(f"{name:<18} q={q:<6} n={spec.n:>6} lam_n={spec.lam_n:+.6f} "
              f"eps={spec.eps:.5f} (Rayleigh {eps_ray:.5f}) lam2={spec.lam2:.4f} "
              f"d-={spec.delta_minus:+.4f} | t1={t1:>4} t2={t2:>4} "
              f"(pred {p1:5.1f},{p2:7.1f}) O={rows[-1]['O_max']:+.3f}", flush=True)
    return rows


if __name__ == "__main__":
    QS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
    out = []
    from functools import partial
    from torch_geometric.datasets import DBLP, IMDB
    loaders = [synthetic_bipartite, movielens,
               partial(hetero_bipartite, "DBLP", DBLP, "data/dblp", "paper"),
               partial(hetero_bipartite, "IMDB", IMDB, "data/imdb", "movie")]
    for loader in loaders:
        try:
            name, A, X, side = loader()
        except Exception as exc:                       # noqa: BLE001
            print(f"[skip] {getattr(loader, '__name__', loader)}: "
                  f"{type(exc).__name__}: {exc}")
            continue
        print(f"--- {name}: n={A.shape[0]} m={A.nnz//2} ---", flush=True)
        out += run(name, A, X, side, QS)
    (OUT / "04_real_bipartite.json").write_text(json.dumps(out, indent=1))
    print("\nwrote", OUT / "04_real_bipartite.json")
