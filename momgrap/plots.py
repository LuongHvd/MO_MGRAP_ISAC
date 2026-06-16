"""The three figures -- the entire results section.  [spec Sec 10]

* Fig 1 -- unified robust front (F_com vs F_sen) for adaptive / no-transfer /
           fixed-RMP + USPA/random markers; optional inset front-width vs delta_phi.
* Fig 2 -- HV over generations (adaptive / fixed-RMP / pooled) + RMP-vs-gen inset.
* Fig 3 -- HV-CDF over realisations (offline unified front) + online-reference band.

Uses a non-interactive matplotlib backend so it runs headless.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .experiments import ExperimentData  # noqa: E402

_LABELS = {
    "adaptive": "MO-MGRAP (adaptive)",
    "fixed_rmp": "fixed-RMP",
    "no_transfer": "no-transfer (indep. NSGA-II)",
    "pooled": "pooled single-task",
}
_COLORS = {
    "adaptive": "C0",
    "fixed_rmp": "C1",
    "no_transfer": "C2",
    "pooled": "C3",
}


def fig1_unified_front(data: ExperimentData, outdir: str = "figures") -> str:
    os.makedirs(outdir, exist_ok=True)
    has_inset = bool(data.delta_sweep)
    fig, ax = plt.subplots(figsize=(6, 4.2))

    for m in ("adaptive", "no_transfer", "fixed_rmp"):
        front = data.unified_fronts.get(m)
        if front is None or front.shape[0] == 0:
            continue
        ax.plot(front[:, 0], front[:, 1], "-o", ms=3, color=_COLORS[m], label=_LABELS[m])

    uspa = data.reference["uspa"]
    rnd = data.reference["random"]
    if uspa.shape[0]:
        ax.scatter(uspa[:, 0], uspa[:, 1], marker="*", s=140, color="k", label="USPA", zorder=5)
    if rnd.shape[0]:
        ax.scatter(rnd[:, 0], rnd[:, 1], marker="x", s=22, color="gray", alpha=0.6, label="random")

    ax.set_xlabel(r"$F_{\mathrm{com}}$  (worst-user rate, bps/Hz)")
    ax.set_ylabel(r"$F_{\mathrm{sen}}$  (worst-target beampattern, dB)")
    ax.set_title("Fig 1 — Unified robust front")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    if has_inset:
        axin = fig.add_axes([0.60, 0.20, 0.28, 0.28])
        deltas = sorted(data.delta_sweep)
        com_r = [data.delta_sweep[d]["com_range"] for d in deltas]
        sen_r = [data.delta_sweep[d]["sen_range"] for d in deltas]
        axin.plot(deltas, com_r, "-o", ms=2, label=r"$F_{com}$ range")
        axin.plot(deltas, sen_r, "-s", ms=2, label=r"$F_{sen}$ range")
        axin.set_xlabel(r"$\Delta\varphi$ (deg)", fontsize=7)
        axin.set_ylabel("front width", fontsize=7)
        axin.tick_params(labelsize=6)
        axin.legend(fontsize=6)

    path = os.path.join(outdir, "fig1_unified_front.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig2_hv_over_gen(data: ExperimentData, outdir: str = "figures") -> str:
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.2))

    for m in ("adaptive", "fixed_rmp", "pooled"):
        hv = data.hv_gen.get(m)
        if hv is None or hv.size == 0:
            continue
        mean = hv.mean(axis=0)
        std = hv.std(axis=0)
        gens = np.arange(1, mean.shape[0] + 1)
        ax.plot(gens, mean, color=_COLORS[m], label=_LABELS[m])
        ax.fill_between(gens, mean - std, mean + std, color=_COLORS[m], alpha=0.15)

    ax.set_xlabel("generation")
    ax.set_ylabel("hypervolume (mean ± std)")
    ax.set_title("Fig 2 — HV over generations")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # inset: RMP vs generation (mean ± std) -- lower-right, clear of the legend
    if data.rmp_gen.size:
        axin = fig.add_axes([0.60, 0.16, 0.30, 0.28])
        rmp_mean = data.rmp_gen.mean(axis=0)
        rmp_std = data.rmp_gen.std(axis=0)
        gens = np.arange(1, rmp_mean.shape[0] + 1)
        axin.plot(gens, rmp_mean, color="C0")
        axin.fill_between(gens, rmp_mean - rmp_std, rmp_mean + rmp_std, color="C0", alpha=0.2)
        axin.set_xlabel("gen", fontsize=7)
        axin.set_ylabel("RMP", fontsize=7)
        axin.tick_params(labelsize=6)
        axin.set_title("RMP vs gen", fontsize=7)

    path = os.path.join(outdir, "fig2_hv_over_gen.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig3_hv_cdf(data: ExperimentData, outdir: str = "figures") -> str:
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.2))

    offline = np.sort(data.fig3["offline_hv"])
    if offline.size:
        cdf = np.arange(1, offline.size + 1) / offline.size
        ax.plot(offline, cdf, color="C0", label="offline unified front")

    online = data.fig3["online_hv"]
    if online.size:
        lo, hi = np.percentile(online, [10, 90])
        med = np.median(online)
        ax.axvspan(lo, hi, color="C1", alpha=0.18, label="online reference (10–90%)")
        ax.axvline(med, color="C1", ls="--", lw=1, label="online median")

    ax.set_xlabel("hypervolume per realisation")
    ax.set_ylabel("CDF")
    ax.set_title("Fig 3 — HV-CDF over realisations")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    path = os.path.join(outdir, "fig3_hv_cdf.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig4_rmp_sweep(data: ExperimentData, outdir: str = "figures") -> str | None:
    """Inverted-U of HV vs fixed RMP, with adaptive shown as auto-tuning to the sweet spot."""
    if not data.rmp_sweep:
        return None
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.2))

    vals = sorted(data.rmp_sweep)
    hvs = [data.rmp_sweep[v] for v in vals]
    ax.plot(vals, hvs, "-o", color="C1", label="fixed-RMP (swept)")

    # adaptive result (mean over seeds), drawn as a horizontal line
    adaptive_hv = float(data.hv_gen["adaptive"].mean(axis=0)[-1])
    ax.axhline(adaptive_hv, color="C0", ls="--", lw=2, label="MO-MGRAP (adaptive)")

    ax.set_xlabel("fixed RMP value")
    ax.set_ylabel("final hypervolume (mean over seeds)")
    ax.set_title("Fig 4 — HV vs fixed RMP (transfer benefit) + adaptive operating point")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    path = os.path.join(outdir, "fig4_rmp_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_all_figures(data: ExperimentData, outdir: str = "figures") -> list[str]:
    figs = [
        fig1_unified_front(data, outdir),
        fig2_hv_over_gen(data, outdir),
        fig3_hv_cdf(data, outdir),
    ]
    f4 = fig4_rmp_sweep(data, outdir)
    if f4:
        figs.append(f4)
    return figs
