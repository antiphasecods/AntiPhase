"""Layer-wise diagnostics for the near-bipartite oscillation.

The old paper's detector
    O(X_t) = ||X_t - X_{t-1}|| / (||X_t|| + ||X_{t-1}||)
is bounded by 1 by the triangle inequality (its claimed limit of 2 is
impossible) and it cannot separate a sign flip from ordinary decay.  We use a
DC-subtracted flip detector instead: remove the persistent component with a
centred running mean over an EVEN window (which annihilates a period-2 signal
exactly), then measure anti-alignment of consecutive residuals.
"""
from __future__ import annotations

import numpy as np

FRO = lambda X: float(np.linalg.norm(X))


def old_detector(Xs):
    """The (bounded-by-1) detector from the previous version, for comparison."""
    return np.array([FRO(Xs[t] - Xs[t - 1]) / (FRO(Xs[t]) + FRO(Xs[t - 1]) + 1e-300)
                     for t in range(1, len(Xs))])


def dc_split(Xs, window: int = 4):
    """Centred running mean over an even window -> (mean, residual) sequences.

    Returns arrays indexed by t with NaN padding where the window does not fit.
    """
    assert window % 2 == 0, "window must be even to annihilate the period-2 mode"
    T = len(Xs) - 1
    h = window // 2
    mean = np.full_like(Xs, np.nan)
    resid = np.full_like(Xs, np.nan)
    for t in range(h, T - h + 2):
        lo, hi = t - h, t + h
        if hi > T + 1:
            break
        mean[t] = Xs[lo:hi].mean(axis=0)
        resid[t] = Xs[t] - mean[t]
    return mean, resid


def flip_series(Xs, window: int = 4):
    """Diagnostics per layer t.

    O_prime : -<Xt~, X(t-1)~> / (||Xt~|| ||X(t-1)~||)  in [-1, 1];
              -> +1 under a clean period-2 flip, ~0 under diffusion.
    amp     : ||Xt~||_F                      (the oscillation envelope)
    ratio   : ||Xt~||_F / ||Xbar_t||_F       (amplitude relative to the DC mode)
    """
    mean, resid = dc_split(Xs, window)
    T = len(Xs) - 1
    O = np.full(T + 1, np.nan)
    amp = np.full(T + 1, np.nan)
    ratio = np.full(T + 1, np.nan)
    for t in range(T + 1):
        if np.isnan(resid[t]).any():
            continue
        amp[t] = FRO(resid[t])
        mb = FRO(mean[t])
        ratio[t] = amp[t] / mb if mb > 0 else np.inf
        if t == 0 or np.isnan(resid[t - 1]).any():
            continue
        d = FRO(resid[t]) * FRO(resid[t - 1])
        if d > 0:
            O[t] = -float((resid[t] * resid[t - 1]).sum()) / d
    return {"O_prime": O, "amp": amp, "ratio": ratio}


def observation_window(diag, gamma: float = 0.1, eta: float = 0.05):
    """Measured [t1, t2].

    t1 = first layer whose flip alignment exceeds 1 - gamma (bulk has died);
    t2 = last layer with alignment above 1 - gamma AND relative amplitude >= eta.
    Returns (t1, t2, lifetime) with NaN if the oscillation is never clean.
    """
    O, ratio = diag["O_prime"], diag["ratio"]
    clean = (O >= 1.0 - gamma)
    if not clean.any():
        return (np.nan, np.nan, 0.0)
    t1 = int(np.flatnonzero(clean)[0])
    ok = np.flatnonzero(clean & (ratio >= eta))
    ok = ok[ok >= t1]
    if ok.size == 0:
        return (t1, t1, 0.0)
    # require contiguity from t1: stop at the first break
    t2 = t1
    for t in range(t1, len(O)):
        if t in ok:
            t2 = t
        elif not np.isnan(O[t]):
            break
    return (t1, t2, float(t2 - t1))


def decay_rate(diag, t1: int, t2: int):
    """Least-squares slope of log(amp) on [t1, t2] -> estimate of eps.

    Under the theory amp_t ~ (1 - eps)^t, so eps_hat = 1 - exp(slope).
    """
    amp = diag["amp"]
    ts = np.arange(t1, t2 + 1)
    ts = ts[np.isfinite(amp[ts]) & (amp[ts] > 0)]
    if ts.size < 3:
        return np.nan
    slope = np.polyfit(ts, np.log(amp[ts]), 1)[0]
    return float(1.0 - np.exp(slope))


def project_out(Xs, v1):
    """Remove the persistent (degree-mode) component exactly: X - v1 (v1' X)."""
    v1 = np.asarray(v1).ravel()
    out = np.empty_like(Xs)
    for t in range(len(Xs)):
        out[t] = Xs[t] - np.outer(v1, v1 @ Xs[t])
    return out


def rayleigh_series(P, Xs, window: int = 4, v1=None):
    """Residual Rayleigh quotient  R_t = tr(Xt~' P Xt~) / ||Xt~||_F^2.

    This is the diagnostic to prefer.  Under the two-timescale model the DC-
    subtracted residual aligns with v_n, so R_t -> lambda_n and the defect can
    be read off a single layer-wise scalar as eps_hat = 1 - |R_t|.  Unlike the
    anti-alignment detector it is invariant to any orthogonal channel mixing
    X -> X W (a trace of the form tr(W' X' P X W)) and to per-layer rescaling,
    so it survives realistic architectures and normalisation layers.

    Its complement 1 - R_t is the normalised Dirichlet energy of the residual,
    which tends to 2 - eps: the maximum of 2 is attained only at exact
    bipartiteness.
    """
    import scipy.sparse as _sp
    P = _sp.csr_matrix(P)
    if v1 is not None:
        resid = project_out(Xs, v1)          # exact, gauge-invariant
    else:
        _, resid = dc_split(Xs, window)      # estimated from a running mean
    T = len(Xs) - 1
    R = np.full(T + 1, np.nan)
    for t in range(T + 1):
        Rt = resid[t]
        if np.isnan(Rt).any():
            continue
        nrm = float((Rt * Rt).sum())
        if nrm > 0:
            R[t] = float((Rt * (P @ Rt)).sum()) / nrm
    return R


def eps_from_rayleigh(R, t1: int, t2: int) -> float:
    """Defect estimate from the residual Rayleigh quotient over [t1, t2]."""
    seg = R[t1:t2 + 1]
    seg = seg[np.isfinite(seg)]
    if seg.size == 0:
        return float("nan")
    return float(1.0 - abs(np.median(seg)))


def stream_series(P, X0, T: int, v1, act: str = "linear", W=None,
                  renorm: bool = True):
    """Layer-wise diagnostics without ever storing the trajectory.

    Because the persistent component is exactly v1 (v1' X_t) -- v1 = D^1/2 1
    being the analytic top eigenvector of P -- the DC/residual split needs only
    the current layer, and the flip alignment only the previous residual.  So
    all four series are computable in O(n d) memory instead of O(T n d), which
    is what makes graphs like DBLP (26k nodes, 300 layers) feasible.

    renorm rescales each layer to unit norm to prevent under/overflow; every
    quantity reported here is scale-free, and R_t is unaffected by it.
    """
    import scipy.sparse as _sp
    from .propagate import ACTS
    f = ACTS[act]
    P = _sp.csr_matrix(P)
    v1 = np.asarray(v1).ravel()

    X = np.asarray(X0, dtype=np.float64)
    O = np.full(T + 1, np.nan)
    amp = np.full(T + 1, np.nan)
    ratio = np.full(T + 1, np.nan)
    R = np.full(T + 1, np.nan)
    prev = None
    for t in range(T + 1):
        if t > 0:
            X = P @ X
            if W is not None:
                X = X @ W
            X = f(X)
            if renorm:
                nrm = FRO(X)
                if nrm > 0:
                    X = X / nrm
        dc = v1 @ X                              # (d,)
        res = X - np.outer(v1, dc)
        a = FRO(res)
        amp[t] = a
        dcn = float(np.linalg.norm(dc))
        ratio[t] = a / dcn if dcn > 0 else np.inf
        if a > 0:
            R[t] = float((res * (P @ res)).sum()) / (a * a)
        if prev is not None:
            d = a * FRO(prev)
            if d > 0:
                O[t] = -float((res * prev).sum()) / d
        prev = res
    return {"O_prime": O, "amp": amp, "ratio": ratio, "R": R}
