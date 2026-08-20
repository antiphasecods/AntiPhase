"""Experiment 1 -- the central claim: oscillation lifetime scales as Theta(1/eps).

Sweep the bipartiteness defect eps of a balanced disassortative SBM, propagate
CSBM features through L layers of P = D^-1/2 A D^-1/2, and measure

    t1  emergence time  (predicted Theta(1/delta^-), delta^- = log((1-eps)/rho))
    t2  extinction time (predicted Theta(1/eps))

against the closed-form predictions of theory.window().
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hetero_osc import diagnostics as dg, propagate as pg, sbm, spectra, theory  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

N = 3000
DAVG = 20.0
DIM = 16
T = 400
GAMMA = 0.10          # bulk-contamination tolerance defining a "clean" flip
ETA = 0.05            # amplitude floor relative to the persistent mode
WIN = 4               # even DC-removal window
EPS_GRID = [0.0, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10,
            0.15, 0.20, 0.30, 0.40, 0.55]
SEEDS = [0, 1, 2]


def run_one(eps_target, seed, keep_trace=False):
    rng = np.random.default_rng(1000 * seed + int(round(1e4 * eps_target)))
    A, blocks = sbm.sample_sbm(N, eps_target, DAVG, rng)
    A, idx = spectra.largest_component(A)
    blocks = blocks[idx]
    P = spectra.normalized_adjacency(A)
    spec, v1, vn = spectra.summarize_sparse(P)
    X0 = sbm.csbm_features(blocks, DIM, 1.0, rng)
    c1, cn, zn = spectra.decompose(X0, v1, vn)

    Xs = pg.propagate(P, X0, T)
    diag = dg.flip_series(Xs, WIN)
    t1, t2, life = dg.observation_window(diag, GAMMA, ETA)
    eps_hat = dg.decay_rate(diag, int(t1), int(t2)) if np.isfinite(t1) else np.nan

    p1, p2, pw = theory.window(spec.eps, spec.rho, GAMMA, ETA,
                               z_over_cn=zn / max(cn, 1e-12),
                               cn_over_c1=cn / max(c1, 1e-12))
    # consistency probe: the two-term model predicts ratio_t = (1-eps)^t cn/c1
    tp = min(40, T)
    pred_ratio = (1.0 - spec.eps) ** tp * cn / max(c1, 1e-12)
    out = {
        "eps_target": eps_target, "seed": seed,
        "n": spec.n, "h": sbm.edge_homophily(A, blocks),
        "lam_n": spec.lam_n, "eps": spec.eps, "rho": spec.rho,
        "delta_minus": spec.delta_minus, "isolated": spec.isolated,
        "c1": c1, "cn": cn, "znorm": zn,
        "t1_obs": t1, "t2_obs": t2, "life_obs": life,
        "t1_pred": p1, "t2_pred": p2, "life_pred": pw,
        "eps_hat_from_envelope": eps_hat,
        "O_prime_max": float(np.nanmax(diag["O_prime"])),
        "ratio_probe_obs": float(diag["ratio"][tp]),
        "ratio_probe_pred": float(pred_ratio),
    }
    if keep_trace:
        out["trace"] = {k: np.asarray(v).tolist() for k, v in diag.items()}
    return out


def main():
    rows, traces = [], {}
    for e in EPS_GRID:
        for s in SEEDS:
            r = run_one(e, s, keep_trace=(s == 0))
            if "trace" in r:
                traces[f"{e}"] = r.pop("trace")
            rows.append(r)
            print(f"eps_t={e:<6} s={s} | eps={r['eps']:.4f} d-={r['delta_minus']:.3f} "
                  f"| t1 {r['t1_obs']:>4} (pred {r['t1_pred']:5.1f}) "
                  f"| t2 {r['t2_obs']:>4} (pred {r['t2_pred']:6.1f}) "
                  f"| eps^={r['eps_hat_from_envelope']:.4f} "
                  f"| probe {r['ratio_probe_obs']:.3e} vs {r['ratio_probe_pred']:.3e}",
                  flush=True)
    (OUT / "01_csbm_lifetime.json").write_text(json.dumps(rows, indent=1))
    (OUT / "01_traces.json").write_text(json.dumps(traces))
    print("\nwrote", OUT / "01_csbm_lifetime.json")


if __name__ == "__main__":
    main()
