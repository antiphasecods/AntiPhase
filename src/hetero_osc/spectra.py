"""Spectral quantities for the bipartiteness-defect analysis.

Everything is phrased in terms of the symmetric normalized adjacency
    P = D^{-1/2} A D^{-1/2},
whose eigenvalues satisfy 1 = lambda_1 >= ... >= lambda_n >= -1, with
lambda_n = -1 iff the (connected) graph is bipartite.

Defect / gap parameters (the two knobs the whole paper turns on):
    eps        = 1 - |lambda_n|        bipartiteness defect
    rho        = max(|lambda_2|, |lambda_{n-1}|)   bulk spectral radius
    delta_minus= log((1 - eps) / rho)  negative-end gap, in log units
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def normalized_adjacency(A, self_loops: bool = False):
    """P = D^{-1/2} A D^{-1/2}. Isolated nodes get a zero row/col."""
    A = sp.csr_matrix(A, dtype=np.float64)
    A = ((A + A.T) > 0).astype(np.float64)      # symmetrize, unweight
    A.setdiag(0.0)
    A.eliminate_zeros()
    if self_loops:
        A = A + sp.eye(A.shape[0], format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    dinv = np.zeros_like(deg)
    nz = deg > 0
    dinv[nz] = deg[nz] ** -0.5
    Dh = sp.diags(dinv)
    return (Dh @ A @ Dh).tocsr()


def largest_component(A):
    """Restrict to the largest connected component; returns (A_sub, node_idx)."""
    A = sp.csr_matrix(A)
    ncomp, labels = sp.csgraph.connected_components(A, directed=False)
    if ncomp == 1:
        return A, np.arange(A.shape[0])
    keep = np.flatnonzero(labels == np.bincount(labels).argmax())
    return A[keep][:, keep], keep


@dataclass
class Spectrum:
    n: int
    m: int
    lam1: float
    lam2: float
    lam_nm1: float
    lam_n: float
    eps: float           # 1 - |lam_n|
    rho: float           # max(|lam2|, |lam_{n-1}|)
    delta_minus: float   # log((1-eps)/rho); <= 0 means lam_n is NOT isolated
    isolated: bool       # rho < 1 - eps  (near-bipartite regime)

    def as_dict(self):
        return asdict(self)


def summarize(P, k: int = 6, dense_below: int = 4000) -> Spectrum:
    """Extreme eigenvalues of a symmetric P, plus the defect/gap parameters."""
    P = sp.csr_matrix(P)
    n = P.shape[0]
    m = int(P.nnz // 2)
    if n <= dense_below:
        lam = np.linalg.eigvalsh(P.toarray())
    else:
        top = spla.eigsh(P, k=k, which="LA", return_eigenvectors=False)
        bot = spla.eigsh(P, k=k, which="SA", return_eigenvectors=False)
        lam = np.sort(np.concatenate([bot, top]))
    lam1, lam2 = float(lam[-1]), float(lam[-2])
    lam_n, lam_nm1 = float(lam[0]), float(lam[1])
    eps = 1.0 - abs(lam_n)
    rho = max(abs(lam2), abs(lam_nm1))
    env = 1.0 - eps                       # = |lam_n|
    delta = float(np.log(env / rho)) if rho > 0 and env > 0 else float("inf")
    return Spectrum(n=n, m=m, lam1=lam1, lam2=lam2, lam_nm1=lam_nm1, lam_n=lam_n,
                    eps=eps, rho=rho, delta_minus=delta, isolated=bool(rho < env))


def trevisan_bounds(eps: float) -> tuple[float, float]:
    """Trevisan (2009) / dual-Cheeger sandwich for the bipartiteness ratio beta(G).

    With lambda_max(L) = 1 - lambda_n = 2 - eps, Trevisan's inequality reads
        eps / 2 <= beta(G) <= sqrt(2 * eps).
    So the analytic defect eps and the combinatorial bipartiteness ratio are
    equivalent up to a quadratic factor: beta^2/2 <= eps <= 2 beta.
    """
    return eps / 2.0, float(np.sqrt(2.0 * eps))


def extreme_pairs(P, k: int = 3, tol: float = 1e-6, maxiter: int | None = None):
    """Top-k and bottom-k eigenpairs of symmetric P (each ascending inside its end).

    ARPACK's "SA"/"LA" modes converge painfully slowly on large sparse graphs.
    Since spec(P) is contained in [-1, 1], the shifts P + I and P - I turn each
    end into a largest-magnitude problem, where Lanczos converges quickly.
    """
    P = sp.csr_matrix(P)
    n = P.shape[0]
    I = sp.eye(n, format="csr")
    k = min(k, n - 2)
    if maxiter is None:
        maxiter = 200 * n

    def solve(M):
        try:
            return spla.eigsh(M, k=k, which="LM", tol=tol, maxiter=maxiter)
        except spla.ArpackNoConvergence as exc:      # keep whatever converged
            if exc.eigenvalues.size < 2:
                raise
            return exc.eigenvalues, exc.eigenvectors

    wt, vt = solve(P + I)
    wb, vb = solve(P - I)
    wt, wb = wt - 1.0, wb + 1.0
    ot, ob = np.argsort(wt)[::-1], np.argsort(wb)
    return (wt[ot], vt[:, ot]), (wb[ob], vb[:, ob])


def summarize_sparse(P, k: int = 3, tol: float = 1e-6):
    """Spectrum summary from Lanczos only, plus the two extreme eigenvectors."""
    (wt, vt), (wb, vb) = extreme_pairs(P, k=k, tol=tol)
    n = P.shape[0]
    lam1, lam2 = float(wt[0]), float(wt[1])
    lam_n, lam_nm1 = float(wb[0]), float(wb[1])
    eps = 1.0 - abs(lam_n)
    rho = max(abs(lam2), abs(lam_nm1))
    env = 1.0 - eps
    delta = float(np.log(env / rho)) if rho > 0 and env > 0 else float("inf")
    s = Spectrum(n=n, m=int(sp.csr_matrix(P).nnz // 2), lam1=lam1, lam2=lam2,
                 lam_nm1=lam_nm1, lam_n=lam_n, eps=eps, rho=rho,
                 delta_minus=delta, isolated=bool(rho < env))
    return s, vt[:, 0], vb[:, 0]


def decompose(X0, v1, vn):
    """X0 = v1 c1^T + vn cn^T + Z  ->  (||c1||, ||cn||, ||Z||_F).

    v1 _|_ vn are unit vectors, so Pythagoras gives ||Z||^2 directly; forming Z
    explicitly is both wasteful and (with NumPy temporary elision) a way to
    silently modify the caller's X0.
    """
    c1 = np.linalg.norm(v1 @ X0)
    cn = np.linalg.norm(vn @ X0)
    z2 = float(np.linalg.norm(X0)) ** 2 - c1 ** 2 - cn ** 2
    return float(c1), float(cn), float(np.sqrt(max(z2, 0.0)))


def degree_mode(A):
    """The exact top eigenvector of P = D^-1/2 A D^-1/2: v1 = D^1/2 1 / ||.||.

    lambda_1 = 1 with this eigenvector for every connected graph, so the
    persistent component can be removed analytically -- no spectral solve, no
    running-mean estimate, and no assumption that the DC part is static.
    """
    deg = np.asarray(sp.csr_matrix(A).sum(axis=1)).ravel()
    v = np.sqrt(np.maximum(deg, 0.0))
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v


def is_bipartite(A):
    """Structural 2-colouring test; returns (bool, colour vector or None)."""
    A = sp.csr_matrix(A)
    n = A.shape[0]
    colour = np.full(n, -1, dtype=np.int8)
    for s in range(n):
        if colour[s] != -1:
            continue
        colour[s] = 0
        stack = [s]
        while stack:
            u = stack.pop()
            for v in A.indices[A.indptr[u]:A.indptr[u + 1]]:
                if colour[v] == -1:
                    colour[v] = 1 - colour[u]
                    stack.append(v)
                elif colour[v] == colour[u]:
                    return False, None
    return True, colour
