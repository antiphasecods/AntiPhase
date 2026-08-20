"""Experiment 3 -- (A) does a 1-Lipschitz non-linearity sustain the mode?
                   (B) the rectification rule: a negative self-loop of weight eps/2.

(A) The linearisation is where a referee pushes.  We re-run the sweep with
    tanh (odd, hence sign-equivariant) and ReLU (not odd) and compare the
    measured envelope decay eps_eff against the linear eps.

(B) Replacing P by q(P) = P - a I rescales the two modes by q(1) = 1 - a and
    q(lam_n) = -(1 - eps) - a, so

        eps_eff = 1 - (1 - eps + a) / (1 - a),

    which vanishes at a* = eps / 2:  a negative self-loop of half the defect
    exactly cancels it and makes the oscillation persistent.  The cost is
    bulk isolation, delta_eff = log((1 - eps + a) / (rho + a)), which shrinks
    in a -- so t1 grows.  We map that trade-off.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hetero_osc import diagnostics as dg, propagate as pg, sbm, spectra  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
N, DAVG, DIM, T = 3000, 20.0, 16, 200
GAMMA, ETA, WIN = 0.10, 0.05, 4


def make(eps_target, seed):
    rng = np.random.default_rng(7000 + 1000 * seed + int(round(1e4 * eps_target)))
    A, blocks = sbm.sample_sbm(N, eps_target, DAVG, rng)
    A, idx = spectra.largest_component(A)
    P = spectra.normalized_adjacency(A)
    spec, v1, vn = spectra.summarize_sparse(P)
    X0 = sbm.csbm_features(blocks[idx], DIM, 1.0, rng)
    X0 = X0 / np.abs(X0).std()          # unit-RMS entries: a fair scale for tanh
    return P, X0, spec, rng, spectra.degree_mode(A)


def measure(Xs, P=None, v1=None):
    d = dg.flip_series(Xs, WIN)
    t1, t2, life = dg.observation_window(d, GAMMA, ETA)
    e = dg.decay_rate(d, int(t1), int(t2)) if np.isfinite(t1) else np.nan
    out = {"t1": t1, "t2": t2, "life": life, "eps_eff": e,
           "O_max": float(np.nanmax(d["O_prime"]))}
    if P is not None and v1 is not None:
        R = dg.rayleigh_series(P, Xs, v1=v1)
        lo = int(t1) if np.isfinite(t1) else 20
        hi = int(t2) if np.isfinite(t2) else len(Xs) - 5
        out["eps_rayleigh"] = dg.eps_from_rayleigh(R, lo, max(hi, lo + 5))
        out["R_mid"] = float(R[len(Xs) // 2])
    return out


def part_a():
    rows = []
    for eps_t in [0.02, 0.05, 0.10, 0.20]:
        for seed in [0, 1]:
            P, X0, spec, rng, v1 = make(eps_t, seed)
            W = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0]   # orthogonal: ||W||=1
            for act, Wm, tag in [("linear", None, "linear"),
                                 ("linear", W, "linear+orthW"),
                                 ("tanh", None, "tanh"),
                                 ("relu", None, "relu"),
                                 ("tanh", None, "tanh+renorm")]:
                Xs = pg.propagate(P, X0, T, act=act, W=Wm,
                                  renorm=tag.endswith("renorm"))
                m = measure(Xs, P, v1)
                rows.append({"eps_target": eps_t, "seed": seed, "variant": tag,
                             "eps": spec.eps, "rho": spec.rho, **m})
                print(f"eps={spec.eps:.4f} s={seed} {tag:<13} "
                      f"t1={m['t1']:>4} t2={m['t2']:>4} eps_env={m['eps_eff']:.4f} "
                      f"eps_ray={m.get('eps_rayleigh', float('nan')):.4f} "
                      f"O_max={m['O_max']:+.3f}", flush=True)
    return rows


def part_b():
    rows = []
    for eps_t in [0.05, 0.15]:
        P, X0, spec, _, v1 = make(eps_t, 0)
        eps, rho = spec.eps, spec.rho
        for mult in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
            a = mult * eps / 2.0
            coeffs = [-a, 1.0]                       # q(P) = P - a I
            fd = pg.filter_defect(coeffs, spec.lam1, spec.lam_n, rho)
            Xs = pg.propagate_filter(P, X0, T, coeffs, normalize_gain=True,
                                     lam1=spec.lam1)
            m = measure(Xs, P, v1)
            rows.append({"eps": eps, "rho": rho, "a": a, "a_over_astar": mult,
                         **fd, **m})
            print(f"eps={eps:.4f} a={a:.4f} (={mult:.2f} a*) "
                  f"eps_eff={fd['eps_eff']:+.4f} delta_eff={fd['delta_eff']:+.3f} "
                  f"| t1={m['t1']:>4} t2={m['t2']:>4} life={m['life']:>5} "
                  f"measured_eps_eff={m['eps_eff']:+.4f}", flush=True)
    return rows


if __name__ == "__main__":
    print("=== A: non-linearities ===")
    a_rows = part_a()
    print("\n=== B: negative self-loop (rectification) ===")
    b_rows = part_b()
    (OUT / "03_nonlinear_and_filters.json").write_text(
        json.dumps({"activations": a_rows, "filters": b_rows}, indent=1))
    print("\nwrote", OUT / "03_nonlinear_and_filters.json")
