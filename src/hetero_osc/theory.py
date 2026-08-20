"""Closed-form predictions of the two-timescale theory."""
from __future__ import annotations

import numpy as np


def t1_pred(eps: float, rho: float, gamma: float, z_over_cn: float) -> float:
    """Emergence time: bulk contamination falls below gamma.

    t1 = log(||Z|| / (gamma |c_n|)) / log((1 - eps) / rho)
    """
    denom = np.log((1.0 - eps) / rho)
    if denom <= 0:
        return np.inf
    return float(np.log(max(z_over_cn, 1e-300) / gamma) / denom)


def t2_pred(eps: float, eta: float, cn_over_c1: float) -> float:
    """Extinction time: envelope (1-eps)^t drops below eta relative to the DC mode.

    t2 = log(|c_n| / (eta |c_1|)) / log(1 / (1 - eps))   ~   log(.)/eps
    """
    if eps <= 0:
        return np.inf
    return float(np.log(max(cn_over_c1, 1e-300) / eta) / np.log(1.0 / (1.0 - eps)))


def window(eps, rho, gamma=0.1, eta=0.05, z_over_cn=1.0, cn_over_c1=1.0):
    a = t1_pred(eps, rho, gamma, z_over_cn)
    b = t2_pred(eps, eta, cn_over_c1)
    return a, b, max(b - a, 0.0)


def depth_threshold(L: int, c: float = 1.0) -> float:
    """|lambda_n| needed for the envelope to survive L layers: 1 - c/L."""
    return 1.0 - c / L


def sbm_defect(p_in: float, p_out: float):
    """Expected (lambda_n, eps) for the balanced two-block SBM (mean-field)."""
    lam = -(p_out - p_in) / (p_out + p_in)
    return lam, 1.0 - abs(lam)


def bulk_radius(davg: float) -> float:
    """Alon-Boppana / Friedman-type bulk edge for the normalized adjacency."""
    return 2.0 * np.sqrt(max(davg - 1.0, 0.0)) / davg
