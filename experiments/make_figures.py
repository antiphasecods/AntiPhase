"""Figures for the CODS 2026 submission."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 160, "axes.grid": True,
                     "grid.alpha": 0.3, "legend.frameon": False})

load = lambda f: json.loads((RES / f).read_text())


def fig1_lifetime():
    """The central experiment: t2 vs 1/eps, observed against predicted."""
    d = load("01_csbm_lifetime.json")
    bip = load("04_real_bipartite.json")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))

    ok = [r for r in d if r["eps"] > 1e-4 and r["t2_obs"] < 399]
    x = np.array([1 / r["eps"] for r in ok])
    ax[0].scatter(x, [r["t2_obs"] for r in ok], s=18, label="observed", zorder=3)
    ax[0].scatter(x, [r["t2_pred"] for r in ok], s=18, marker="x",
                  label="predicted", zorder=3)
    xs = np.linspace(x.min(), x.max(), 50)
    ax[0].plot(xs, np.log(1 / 0.05) * xs, "--", lw=1, c="gray",
               label=r"$\log(1/\eta)\,/\,\epsilon$")
    ax[0].set_xlabel(r"$1/\epsilon$"); ax[0].set_ylabel(r"extinction depth $t_2$")
    ax[0].set_title(r"CSBM: lifetime is $\Theta(1/\epsilon)$"); ax[0].legend()

    for name, mk in [("SyntheticBipartite", "o"), ("MovieLens100K", "s")]:
        rr = [r for r in bip if r["dataset"] == name and 0 < r["eps"]
              and r["t2_obs"] < 299]
        if not rr:
            continue
        ax[1].scatter([r["t2_pred"] for r in rr], [r["t2_obs"] for r in rr],
                      s=22, marker=mk, label=name, zorder=3)
    lim = [4, 400]
    ax[1].plot(lim, lim, "--", c="gray", lw=1, label="y = x")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel(r"predicted $t_2$"); ax[1].set_ylabel(r"observed $t_2$")
    ax[1].set_title("real topology (contamination sweep)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG / "fig1_lifetime.png"); plt.close(fig)


def fig2_plane():
    """Where graphs live in the (eps, delta^-) plane. The feasibility region."""
    real = load("02_real_spectra.json")
    csbm = load("01_csbm_lifetime.json")
    bip = load("04_real_bipartite.json")
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    e = np.linspace(1e-3, 1.0, 200)
    ax.fill_between(e, e, 1.2, color="tab:green", alpha=0.10,
                    label=r"observable: $\epsilon < \delta^-$")
    ax.plot(e, e, c="tab:green", lw=1, ls="--")
    ax.axhline(0, c="k", lw=0.8)

    ax.scatter([r["eps"] for r in csbm], [r["delta_minus"] for r in csbm],
               s=12, c="tab:blue", alpha=0.5, label="CSBM sweep")
    het = {"Texas", "Wisconsin", "Cornell", "chameleon", "squirrel", "actor",
           "Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"}
    # the benchmarks pile up at delta^- ~ 0, so label only the extremes and
    # let Table 1 carry the full list
    lab = {"Minesweeper": (6, 2), "Amazon-ratings": (-50, -11),
           "squirrel": (5, -8), "Roman-empire": (-54, -9), "Texas": (5, 5),
           "CiteSeer": (-32, 7)}
    for r in real:
        c = "tab:red" if r["dataset"] in het else "tab:orange"
        ax.scatter(r["eps"], r["delta_minus"], s=32, c=c, marker="v", zorder=4)
        if r["dataset"] in lab:
            ax.annotate(r["dataset"], (r["eps"], r["delta_minus"]), fontsize=6,
                        xytext=lab[r["dataset"]], textcoords="offset points",
                        zorder=6)
    ml = [r for r in bip if r["dataset"] == "MovieLens100K"]
    ax.scatter([r["eps"] for r in ml], [r["delta_minus"] for r in ml], s=34,
               c="tab:purple", marker="*", zorder=5, label="MovieLens100K")
    ax.scatter([], [], s=30, c="tab:red", marker="v", label="heterophilic benchmarks")
    ax.scatter([], [], s=30, c="tab:orange", marker="v", label="homophilic controls")
    ax.set_xlabel(r"bipartiteness defect $\epsilon = 1 - |\lambda_n|$")
    ax.set_ylabel(r"negative-end gap $\delta^- = \log\frac{1-\epsilon}{\rho}$")
    ax.set_xlim(-0.02, 0.62); ax.set_ylim(-0.7, 1.0)
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig2_plane.png"); plt.close(fig)


def fig3_traces():
    """Emergence and extinction, layer by layer."""
    tr = load("01_traces.json")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for key in ["0.0", "0.02", "0.075", "0.2", "0.4"]:
        if key not in tr:
            continue
        O = np.array(tr[key]["O_prime"], dtype=float)
        r = np.array(tr[key]["ratio"], dtype=float)
        t = np.arange(O.size)
        ax[0].plot(t, O, lw=1, label=rf"$\epsilon\approx{float(key):.3f}$")
        ax[1].semilogy(t, r, lw=1)
    ax[0].axhline(0.9, ls=":", c="gray", lw=1)
    ax[0].set_xlim(0, 120); ax[0].set_ylim(-0.1, 1.05)
    ax[0].set_xlabel("depth $t$"); ax[0].set_ylabel(r"flip alignment $O'_t$")
    ax[0].set_title(r"emergence: $O'_t\to 1$ after $t_1=\Theta(1/\delta^-)$")
    ax[0].legend(fontsize=6)
    ax[1].axhline(0.05, ls=":", c="gray", lw=1)
    ax[1].set_xlim(0, 200); ax[1].set_ylim(1e-4, 1e3)
    ax[1].set_xlabel("depth $t$")
    ax[1].set_ylabel(r"$\|\tilde X_t\| / \|\bar X_t\|$")
    ax[1].set_title(r"extinction: envelope $(1-\epsilon)^t$")
    fig.tight_layout(); fig.savefig(FIG / "fig3_traces.png"); plt.close(fig)


def fig4_rectify():
    """The rectification rule a* = eps/2 and its cost."""
    d = load("03_nonlinear_and_filters.json")["filters"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for eps in sorted({round(r["eps"], 4) for r in d}):
        rr = [r for r in d if round(r["eps"], 4) == eps]
        x = [r["a_over_astar"] for r in rr]
        ax[0].plot(x, [r["eps_eff"] for r in rr], "o-", ms=3, lw=1,
                   label=rf"$\epsilon={eps:.3f}$ predicted")
        ax[0].plot(x, [r["measured_eps_eff"] if "measured_eps_eff" in r
                       else r["eps_eff"] for r in rr], "x", ms=4)
        ax[1].plot(x, [r["delta_eff"] for r in rr], "s-", ms=3, lw=1)
    ax[0].axhline(0, c="k", lw=0.8); ax[0].axvline(1, ls=":", c="gray", lw=1)
    ax[0].set_xlabel(r"$a / a^*$,  $a^* = \epsilon/2$")
    ax[0].set_ylabel(r"$\epsilon_{\rm eff}$")
    ax[0].set_title(r"negative self-loop cancels the defect at $a^*$")
    ax[0].legend(fontsize=6)
    ax[1].axvline(1, ls=":", c="gray", lw=1)
    ax[1].set_xlabel(r"$a / a^*$"); ax[1].set_ylabel(r"$\delta^-_{\rm eff}$")
    ax[1].set_title("cost: bulk isolation shrinks")
    fig.tight_layout(); fig.savefig(FIG / "fig4_rectify.png"); plt.close(fig)


def fig5_h_vs_eps():
    """Homophily and near-bipartiteness are unrelated."""
    real = load("02_real_spectra.json")
    h = np.array([r["h_edge"] for r in real])
    e = np.array([r["eps"] for r in real])
    rho_p = float(np.corrcoef(h, e)[0, 1])
    from scipy.stats import spearmanr
    rs = float(spearmanr(h, e).statistic)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.scatter(h, e, s=28, c="tab:red", zorder=3)
    # label only the points the text argues from; the rest are in Table 1
    lab = {"Texas": (6, 2), "CiteSeer": (-36, 3), "Questions": (-14, 10),
           "Minesweeper": (-48, -2), "Roman-empire": (6, -10)}
    for r in real:
        if r["dataset"] in lab:
            ax.annotate(r["dataset"], (r["h_edge"], r["eps"]), fontsize=6.5,
                        xytext=lab[r["dataset"]], textcoords="offset points")
    ax.set_xlim(-0.03, 0.95)
    ax.set_ylim(-0.025, 0.52)
    ax.set_xlabel("edge homophily $h$")
    ax.set_ylabel(r"bipartiteness defect $\epsilon$")
    ax.set_title(rf"Pearson $r={rho_p:.2f}$,  Spearman $={rs:.2f}$  ($n={len(real)}$)",
                 fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig5_h_vs_eps.png"); plt.close(fig)
    return rho_p, rs


if __name__ == "__main__":
    fig1_lifetime(); print("fig1 ok")
    fig2_plane();    print("fig2 ok")
    fig3_traces();   print("fig3 ok")
    fig4_rectify();  print("fig4 ok")
    r, rs = fig5_h_vs_eps()
    print(f"fig5 ok   Pearson(h, eps) = {r:+.3f}   Spearman = {rs:+.3f}")
