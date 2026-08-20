"""Explanatory schematics for the method sections (vector output)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "legend.frameon": False})

C_DC   = "#1f77b4"   # persistent / degree mode
C_OSC  = "#d62728"   # alternating mode
C_BULK = "#8c8c8c"   # bulk
C_WIN  = "#2ca02c"   # observation window


def spectrum():
    """Where the three components live in the spectrum, and what each one does."""
    fig = plt.figure(figsize=(5.0, 3.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.55)
    ax = fig.add_subplot(gs[0])

    rho, lam_n = 0.45, -0.86      # schematic: eps drawn larger than typical
    ax.add_patch(Rectangle((-rho, -0.12), 2 * rho, 0.24, color=C_BULK,
                           alpha=0.22, zorder=1))
    ax.annotate("bulk: $|\\lambda| \\leq \\rho$", (0, 0.30), ha="center",
                fontsize=8, color="#4d4d4d")
    rng = np.random.default_rng(3)
    for v in rng.uniform(-rho, rho, 22):
        ax.plot([v, v], [-0.07, 0.07], color=C_BULK, lw=0.9, zorder=2)

    ax.plot([-1.06, 1.06], [0, 0], color="k", lw=1.0, zorder=0)
    for x, lab in [(-1, "$-1$"), (0, "$0$"), (1, "$+1$")]:
        ax.plot([x, x], [-0.13, 0.13], color="k", lw=1.0)
        ax.annotate(lab, (x, -0.34), ha="center", fontsize=8)

    ax.plot(1, 0, "o", ms=9, color=C_DC, zorder=4)
    ax.annotate("$\\lambda_1 = 1$\npersistent mode $v_1$", (1, 0.42),
                ha="center", fontsize=8, color=C_DC)
    ax.plot(lam_n, 0, "o", ms=9, color=C_OSC, zorder=4)
    ax.annotate("$\\lambda_n$\nalternating mode $v_n$", (lam_n, 0.42),
                ha="center", fontsize=8, color=C_OSC)

    # the two gaps are drawn at different heights so their labels do not collide
    ax.annotate("", (-1, -0.30), (lam_n, -0.30),
                arrowprops=dict(arrowstyle="<->", color=C_OSC, lw=1.3))
    ax.annotate("$\\epsilon$", ((-1 + lam_n) / 2, -0.40), ha="center",
                va="top", fontsize=11, color=C_OSC)
    ax.annotate("", (lam_n, -0.86), (-rho, -0.86),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.3))
    ax.annotate("$\\delta^-$", ((lam_n - rho) / 2, -0.96), ha="center",
                va="top", fontsize=11)

    ax.set_xlim(-1.30, 1.30); ax.set_ylim(-1.35, 0.85); ax.axis("off")

    # what each component does layer by layer
    ax2 = fig.add_subplot(gs[1])
    T = 8
    t = np.arange(T)
    rows = [("persistent  $v_1 c_1^\\top$", np.ones(T), C_DC),
            ("alternating  $\\lambda_n^t v_n c_n^\\top$",
             (-1.0) ** t * 0.93 ** t, C_OSC),
            ("bulk  $\\mathbf{P}^t Z$",
             0.44 ** t * rng.choice([-1, 1], T), C_BULK)]
    for i, (lab, vals, c) in enumerate(rows):
        y0 = -i * 1.6
        ax2.plot([-0.6, T - 0.4], [y0, y0], color="k", lw=0.5, alpha=0.4)
        ax2.bar(t, vals * 0.62, bottom=y0, width=0.42, color=c, zorder=3)
        ax2.annotate(lab, (-0.9, y0), ha="right", va="center", fontsize=8,
                     color=c)
    for tt in t:
        ax2.annotate(f"$t={tt}$", (tt, 0.95), ha="center", fontsize=7,
                     color="#4d4d4d")
    ax2.set_xlim(-4.0, T - 0.2); ax2.set_ylim(-4.0, 1.35); ax2.axis("off")
    fig.savefig(FIG / "fig_spectrum.pdf", bbox_inches="tight")
    plt.close(fig)


def window():
    """How the two crossings define the observation window."""
    eps, rho, gamma, eta = 0.08, 0.44, 0.1, 0.05
    c1, cn, Z = 1.0, 20.0, 400.0
    t = np.linspace(0, 100, 900)
    dc = np.full_like(t, c1)
    osc = cn * (1 - eps) ** t
    bulk = Z * rho ** t

    t1 = np.log(Z / (gamma * cn)) / np.log((1 - eps) / rho)
    t2 = np.log(cn / (eta * c1)) / np.log(1 / (1 - eps))

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.axvspan(t1, t2, color=C_WIN, alpha=0.10, zorder=0)
    ax.semilogy(t, bulk, color=C_BULK, lw=1.8, label=r"bulk  $\rho^t\|Z\|_F$")
    ax.semilogy(t, osc, color=C_OSC, lw=1.8,
                label=r"alternating  $(1-\epsilon)^t\|c_n\|$")
    ax.semilogy(t, dc, color=C_DC, lw=1.8, label=r"persistent  $\|c_1\|$")
    ax.semilogy(t, gamma * osc, color=C_OSC, lw=1.0, ls=":",
                label=r"$\gamma\times$ alternating")
    ax.semilogy(t, eta * dc, color=C_DC, lw=1.0, ls=":",
                label=r"$\eta\times$ persistent")

    for tc, lab, col in [(t1, "$t_1$", C_BULK), (t2, "$t_2$", C_OSC)]:
        ax.axvline(tc, color=col, lw=1.1, ls="--")
        ax.annotate(lab, (tc, 2.2e3), ha="center", fontsize=11, color=col)
    ax.annotate("observation window", ((t1 + t2) / 2, 1.6e2), ha="center",
                fontsize=9, color="#1a7a1a")
    ax.annotate("emergence: the bulk falls below the\n"
                "alternating mode, at rate $\\delta^-$",
                (t1 + 3.0, 1.2e-3), fontsize=7.5, color="#5f5f5f")
    ax.annotate("extinction: the alternating mode\n"
                "falls below the persistent one",
                (t2 - 33, 2.5e-4), fontsize=7.5, color=C_OSC)
    ax.annotate(r"slope $-\epsilon$  (sets $t_2$)", (46, 1.1), fontsize=8,
                color=C_OSC, rotation=-13)

    ax.set_xlabel("depth $t$")
    ax.set_ylabel("magnitude (log scale)")
    ax.set_xlim(0, 100)
    ax.set_ylim(1e-5, 1e4)
    ax.legend(loc="upper right", fontsize=6.8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig_window.pdf")
    plt.close(fig)


def diagnostic():
    """The measurement pipeline and what each branch is good for.

    Boxes are sized to the rendered text rather than to hand-picked constants,
    so no label can overflow its box when the font metrics change.
    """
    fig, ax = plt.subplots(figsize=(7.2, 2.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def boxed(x, y, text, fc, ec, fs, padx=0.012, pady=0.07):
        t = ax.annotate(text, (x, y), ha="center", va="center", fontsize=fs,
                        zorder=4)
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        (x0, y0), (x1, y1) = ax.transData.inverted().transform(bb)
        ax.add_patch(FancyBboxPatch(
            (x0 - padx, y0 - pady), (x1 - x0) + 2 * padx,
            (y1 - y0) + 2 * pady, boxstyle="round,pad=0.008",
            fc=fc, ec=ec, lw=1.2, zorder=2))
        return x0 - padx, y0 - pady, x1 + padx, y1 + pady

    def arrow(p0, p1):
        ax.annotate("", p1, p0, zorder=5,
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="k"))

    b_in = boxed(0.055, 0.50, r"$X_t$", "#eef3f8", C_DC, 11)
    b_mid = boxed(0.275, 0.50,
                  "remove persistent part\n"
                  r"$\widetilde X_t = X_t - v_1(v_1^\top X_t)$" "\n"
                  r"$v_1 = D^{1/2}\mathbf{1}/\|D^{1/2}\mathbf{1}\|$",
                  "#eef3f8", C_DC, 7.5)
    b_flip = boxed(0.60, 0.79,
                   r"$O'_t = -\dfrac{\langle \widetilde X_t,"
                   r"\widetilde X_{t-1}\rangle}"
                   r"{\|\widetilde X_t\|\,\|\widetilde X_{t-1}\|}$",
                   "#fdeeee", C_OSC, 7.5)
    b_ray = boxed(0.60, 0.21,
                  r"$R_t = \dfrac{\mathrm{tr}(\widetilde X_t^\top \mathbf{P}"
                  r"\widetilde X_t)}{\|\widetilde X_t\|_F^2}$",
                  "#eef7ee", C_WIN, 7.5)

    arrow((b_in[2], 0.50), (b_mid[0], 0.50))
    arrow((b_mid[2], 0.60), (b_flip[0], 0.74))
    arrow((b_mid[2], 0.40), (b_ray[0], 0.26))

    ax.annotate(r"$\to +1$ under a clean flip;" "\n"
                "reports the window endpoints,\n"
                "but not invariant to channel mixing",
                (b_flip[2] + 0.02, 0.79), ha="left", va="center",
                fontsize=8, color=C_OSC)
    ax.annotate(r"$\to \lambda_n$, so $\hat\epsilon = 1-|R_t|$;" "\n"
                "invariant to channel mixing,\n"
                "activation and renormalisation",
                (b_ray[2] + 0.02, 0.21), ha="left", va="center",
                fontsize=8, color="#1a7a1a")

    fig.savefig(FIG / "fig_diagnostic.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    spectrum();   print("fig_spectrum.pdf")
    window();     print("fig_window.pdf")
    diagnostic(); print("fig_diagnostic.pdf")
