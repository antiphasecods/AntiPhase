"""Experiment 2 -- the negative half: where do real graphs sit in the (eps, delta^-) plane?

The theory predicts its own absence: a graph shows a visible oscillation only if
its bipartiteness defect eps = 1 - |lambda_n| is small compared with the
negative-end gap delta^- = log((1-eps)/rho).  We audit the standard heterophilic
benchmarks (plus homophilic controls) and report the predicted visibility depth.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hetero_osc import sbm, spectra, theory  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

ETA = 0.05
GAMMA = 0.10


def load_all():
    from torch_geometric.datasets import (Actor, HeterophilousGraphDataset,
                                          Planetoid, WebKB, WikipediaNetwork)
    specs = []
    for name in ["Texas", "Wisconsin", "Cornell"]:
        specs.append((name, lambda n=name: WebKB(DATA, n)[0]))
    for name in ["chameleon", "squirrel"]:
        specs.append((name, lambda n=name: WikipediaNetwork(DATA, n, geom_gcn_preprocess=True)[0]))
    specs.append(("actor", lambda: Actor(DATA / "actor")[0]))
    for name in ["Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"]:
        specs.append((name, lambda n=name: HeterophilousGraphDataset(DATA, n)[0]))
    for name in ["Cora", "CiteSeer", "PubMed"]:
        specs.append((name, lambda n=name: Planetoid(DATA, n)[0]))
    return specs


def audit(name, data):
    ei = data.edge_index.numpy()
    n = int(data.num_nodes)
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float64).tocsr()
    A.setdiag(0.0); A.eliminate_zeros()
    A, idx = spectra.largest_component(A)
    y = data.y.numpy()[idx]
    X = data.x.numpy()[idx].astype(np.float64)

    P = spectra.normalized_adjacency(A)
    spec, v1, vn = spectra.summarize_sparse(P, k=3)
    # GCN's renormalization trick: self-loops push lambda_n away from -1
    spec_sl, _, _ = spectra.summarize_sparse(spectra.normalized_adjacency(A, self_loops=True), k=3)
    c1, cn, zn = spectra.decompose(X, v1, vn)
    lo, hi = spectra.trevisan_bounds(max(spec.eps, 0.0))
    t1p, t2p, wid = theory.window(spec.eps, spec.rho, GAMMA, ETA,
                                  z_over_cn=zn / max(cn, 1e-12),
                                  cn_over_c1=cn / max(c1, 1e-12))
    return {
        "dataset": name, "n": spec.n, "m": int(A.nnz // 2),
        "h_edge": sbm.edge_homophily(A, y),
        "lam_n": spec.lam_n, "lam_2": spec.lam2, "lam_nm1": spec.lam_nm1,
        "eps": spec.eps, "rho": spec.rho, "delta_minus": spec.delta_minus,
        "isolated": spec.isolated, "eps_lt_delta": bool(spec.eps < spec.delta_minus),
        "beta_lower": lo, "beta_upper": hi,
        "c1": c1, "cn": cn, "znorm": zn,
        "t1_pred": t1p, "t2_pred": t2p, "window": wid,
        "half_life": float(np.log(2) / -np.log(1 - spec.eps)) if spec.eps > 0 else np.inf,
        "lam_n_selfloop": spec_sl.lam_n, "eps_selfloop": spec_sl.eps,
        "rho_selfloop": spec_sl.rho, "delta_minus_selfloop": spec_sl.delta_minus,
    }


def free_gb(path=".") -> float:
    import shutil
    return shutil.disk_usage(path).free / 2 ** 30


def main():
    rows = []
    for name, load in load_all():
        if free_gb() < 0.8:
            print(f"[skip] {name}: only {free_gb():.2f} GiB free")
            continue
        try:
            data = load()
        except Exception as exc:                       # noqa: BLE001
            print(f"[skip] {name}: {type(exc).__name__}: {exc}")
            continue
        r = audit(name, data)
        rows.append(r)
        (OUT / "02_real_spectra.json").write_text(json.dumps(rows, indent=1))
        print(f"{r['dataset']:<15} n={r['n']:>6} h={r['h_edge']:.3f} "
              f"lam_n={r['lam_n']:+.4f} eps={r['eps']:.4f} rho={r['rho']:.3f} "
              f"d-={r['delta_minus']:+.3f} eps<d-={str(r['eps_lt_delta']):>5} "
              f"| half-life={r['half_life']:6.2f} L  window=[{r['t1_pred']:.1f},{r['t2_pred']:.1f}]"
              f"  | +selfloops: lam_n={r['lam_n_selfloop']:+.4f} eps={r['eps_selfloop']:.4f}",
              flush=True)
    (OUT / "02_real_spectra.json").write_text(json.dumps(rows, indent=1))
    print("\nwrote", OUT / "02_real_spectra.json")


if __name__ == "__main__":
    main()
