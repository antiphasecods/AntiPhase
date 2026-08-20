# Near-bipartite oscillations in deep GNNs: a finite-depth theory

Rewrite of the rejected COLT submission for CODS 2026.  The knife-edge claim
(`lambda_n = -1` exactly, period-2 limit cycle, contradicted by the paper's own
non-bipartiteness assumption) is replaced by a **two-timescale finite-depth
window** governed by one parameter, the bipartiteness defect.

## The object

`P = D^-1/2 A D^-1/2`, eigenvalues `1 = lambda_1 > ... > lambda_n`, and

    eps        = 1 - |lambda_n|                  bipartiteness defect
    rho        = max(|lambda_2|, |lambda_{n-1}|) bulk radius
    delta^-    = log((1 - eps) / rho)            negative-end gap

Decomposing `X_0 = c_1 v_1 + c_n v_n + Z` gives the exact transient

    || X_t - (c_1 v_1 + (-1)^t (1-eps)^t c_n v_n) ||_F  <=  rho^t ||Z||_F,

so a clean sign-flip is visible on `[t1, t2]` with `t1 = Theta(1/delta^-)`
(bulk dies) and `t2 = Theta(1/eps)` (envelope dies).  The phenomenon is real
iff `eps << delta^-`: a band of width `~delta^-` in `lambda_n`, not a point.

## Results

| # | Experiment | Finding |
|---|---|---|
| 1 | `01_csbm_lifetime.py` | Observed `t2` matches the closed form to **1-2 layers** across `eps in [0, 0.55]` (`t2` from 399 down to 9).  Envelope decay recovers `eps` to 4 decimals.  Window closes exactly where predicted (`eps=0.46`, `delta^-=0.21`: `t1=6`, `t2=10`). |
| 2 | `02_real_spectra.py` | 14 benchmarks.  **Every one fails the isolation condition** (`rho in [0.92, 1.00]`, `delta^- <= 0`) -- the oscillation never *emerges*, rather than dying early.  `eps` is small on many of them (Roman-empire 0.026, Questions 0.017, Texas 0.062). |
| 2b | same | **`h` and `eps` are uncorrelated**: Pearson `+0.08`, Spearman `-0.35` over 14 graphs.  CiteSeer (`h=0.735`) is *more* near-bipartite than Texas (`h=0.061`).  Low homophily does not imply near-bipartite. |
| 2c | same | GCN's renormalisation trick (self-loops) inflates `eps` **5-10x** on every graph (Texas `0.062 -> 0.536`).  The standard operator destroys the mode heterophilic architectures exist to preserve. |
| 3A | `03_nonlinear_and_filters.py` | **tanh** (odd, 1-Lipschitz) leaves the rate essentially unchanged and slightly faster (`0.1749` vs `0.1743`): a 1-Lipschitz non-linearity can only *accelerate* the decay.  **ReLU** keeps the same envelope rate but **halves the window** (`t2` 34 -> 16) -- rectification pumps the oscillation into the DC component. |
| 3B | same | **Rectification rule**: with `q(P) = P - aI`, `eps_eff = 1 - (1-eps+a)/(1-a)`, which vanishes at **`a* = eps/2`**.  Measured `eps_eff` matches prediction to 4 decimals; at `a*` the oscillation becomes exactly persistent (`t2` censored at 199) while `delta^-_eff` falls only `0.688 -> 0.619`. |
| 4 | `04_real_bipartite.py` | **MovieLens100K**: `lambda_n = -1.000000` exactly, `delta^- = 0.42 > 0` (the only real graph in the observable region), oscillation emerges at layer 4 and survives all 299 layers.  Contamination sweep reproduces the lifetime law on real topology (`t2` observed/predicted: 179/179.7, 91/92.0, 39/39.8, 21/21.6, 11/12.1). |

## Diagnostics

The old detector `O(X_t) = ||X_t - X_{t-1}|| / (||X_t|| + ||X_{t-1}||)` is
bounded by 1 by the triangle inequality, so its claimed limit of 2 is
impossible.  Replaced by two:

1. **flip alignment** `O'_t = -<Xt~, X(t-1)~> / (||Xt~|| ||X(t-1)~||) -> +1`
   under a clean flip, `0` under diffusion.  Simple, but not invariant to
   channel mixing.
2. **residual Rayleigh quotient** `R_t = tr(Xt~' P Xt~) / ||Xt~||^2 -> lambda_n`
   after projecting out the *analytic* top eigenvector `v_1 = D^1/2 1 / ||.||`.
   Recovers `eps` to **six decimals** under linear, orthogonal channel mixing,
   tanh, ReLU, per-layer renormalisation, and every combination.  This is the
   one to put in the paper.

## Data

All 17 real-graph datasets are public benchmarks pulled automatically by
[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) on first use —
no manual download or preprocessing. Each `experiments/0N_*.py` script
downloads into `data/<name>/` (created on demand) the first time it runs.

| Source (`torch_geometric.datasets`) | Datasets | Used by |
|---|---|---|
| `Planetoid` | Cora, CiteSeer, PubMed | `02_real_spectra.py` |
| `WebKB` | Texas, Wisconsin, Cornell | `02_real_spectra.py` |
| `WikipediaNetwork` (`geom_gcn_preprocess=True`) | chameleon, squirrel | `02_real_spectra.py` |
| `Actor` | actor | `02_real_spectra.py` |
| `HeterophilousGraphDataset` | Roman-empire, Amazon-ratings, Minesweeper, Tolokers, Questions | `02_real_spectra.py` |
| `MovieLens100K` | ml100k | `04_real_bipartite.py` |
| `DBLP`, `IMDB` | dblp, imdb | `04_real_bipartite.py` |

`data/` is gitignored — datasets are not checked into this repo, only the
code that fetches and audits them. `results/` holds the derived numbers
(per-graph `eps`, `delta^-`, homophily, etc.) and logs from those runs, which
*are* checked in.

## Layout

    src/hetero_osc/   spectra, sbm, propagate, diagnostics, theory
    experiments/      01..04 + make_figures.py
    results/          json + logs      figures/  fig1..fig5
