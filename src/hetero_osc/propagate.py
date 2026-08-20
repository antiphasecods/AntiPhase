"""Deep propagation X_{t+1} = sigma(P X_t W), linear and non-linear."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

ACTS = {
    "linear": lambda x: x,
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(x, 0.0),
    "lrelu": lambda x: np.where(x > 0, x, 0.01 * x),
}


def propagate(P, X0, T: int, act: str = "linear", W=None,
              renorm: bool = False, seed: int | None = None):
    """Return an array of shape (T+1, n, d) with the layer-wise features.

    act    : element-wise non-linearity (all options are 1-Lipschitz).
    W      : optional (d, d) channel mixing applied on the right each layer.
    renorm : rescale each layer to unit Frobenius norm (removes the global
             envelope; used only to check that *shape* dynamics are unchanged).
    """
    f = ACTS[act]
    P = sp.csr_matrix(P)
    X = np.asarray(X0, dtype=np.float64)
    out = np.empty((T + 1,) + X.shape)
    out[0] = X
    for t in range(1, T + 1):
        X = P @ X
        if W is not None:
            X = X @ W
        X = f(X)
        if renorm:
            nrm = np.linalg.norm(X)
            if nrm > 0:
                X = X / nrm
        out[t] = X
    return out


def spectral_init(P_eigvecs, coeffs, rng, n, d):
    """Build X0 = c1 v1 + cn vn + Z with prescribed coefficients.

    P_eigvecs : (v1, vn) columns.  coeffs : (c1, cn).  Z is isotropic Gaussian
    noise in the orthogonal complement, scaled to Frobenius norm coeffs[2].
    """
    v1, vn = P_eigvecs
    c1, cn, znorm = coeffs
    a = rng.standard_normal((1, d))
    b = rng.standard_normal((1, d))
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    X0 = c1 * np.outer(v1, a) + cn * np.outer(vn, b)
    Z = rng.standard_normal((n, d))
    for v in (v1, vn):
        Z -= np.outer(v, v @ Z)
    Z *= znorm / np.linalg.norm(Z)
    return X0 + Z, Z


def propagate_filter(P, X0, T: int, coeffs, act: str = "linear",
                     normalize_gain: bool = False, lam1: float = 1.0):
    """Iterate X_{t+1} = sigma(q(P) X_t) with q(z) = sum_k coeffs[k] z^k.

    coeffs is ordered [c_0, c_1, ...].  With normalize_gain the filter is
    rescaled by 1/|q(lam1)| so the persistent mode has unit gain and the
    trajectory neither blows up nor underflows.
    """
    import scipy.sparse as _sp
    f = ACTS[act]
    P = _sp.csr_matrix(P)
    c = np.asarray(coeffs, dtype=np.float64)
    if normalize_gain:
        g = abs(np.polyval(c[::-1], lam1))
        if g > 0:
            c = c / g

    def apply(Y):
        out = c[-1] * Y                      # Horner in P
        for k in range(len(c) - 2, -1, -1):
            out = P @ out + c[k] * Y
        return out

    X = np.asarray(X0, dtype=np.float64)
    res = np.empty((T + 1,) + X.shape)
    res[0] = X
    for t in range(1, T + 1):
        X = f(apply(X))
        res[t] = X
    return res


def filter_defect(coeffs, lam1: float, lam_n: float, rho: float):
    """Effective (eps, delta^-) after applying a polynomial filter q.

    eps_eff  = 1 - |q(lam_n)| / |q(lam_1)|      (envelope decay per layer)
    delta_eff= log(|q(lam_n)| / max_{|z|<=rho} |q(z)|)   (bulk isolation)
    A sign flip survives only if q(lam_n) < 0.
    """
    c = np.asarray(coeffs, dtype=np.float64)[::-1]
    q1, qn = np.polyval(c, lam1), np.polyval(c, lam_n)
    zs = np.linspace(-rho, rho, 2001)
    qb = np.abs(np.polyval(c, zs)).max()
    eps_eff = 1.0 - abs(qn) / abs(q1)
    delta_eff = float(np.log(abs(qn) / qb)) if qb > 0 else float("inf")
    return {"q1": float(q1), "qn": float(qn), "flips": bool(qn < 0),
            "eps_eff": float(eps_eff), "delta_eff": delta_eff}
