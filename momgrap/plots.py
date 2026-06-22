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
    "adaptive": "RAMP (proposed)",
    "fixed_rmp": "fixed-RMP",
    "no_transfer": "no-transfer (indep. NSGA-II)",
    "pooled": "pooled single-task",
    "mo_de": "MO-DE",
    "mopso": "MOPSO",
}
_COLORS = {
    "adaptive": "C0",
    "fixed_rmp": "C1",
    "no_transfer": "C2",
    "pooled": "C3",
    "mo_de": "C4",
    "mopso": "C5",
}


def fig1_unified_front(data: ExperimentData, outdir: str = "figures") -> str:
    os.makedirs(outdir, exist_ok=True)
    has_inset = bool(data.delta_sweep)
    fig, ax = plt.subplots(figsize=(6, 4.2))

    # Single curve = the robust front attained by the proposed MO-multitask
    # framework: the non-dominated envelope over its variants (adaptive / fixed /
    # no-transfer all share the framework). Plotting one framework front vs the
    # naive references keeps Fig 1 focused on "the product dominates references"
    # and the tradeoff; the algorithm comparison lives in Fig 2.
    import torch

    from .metrics import non_dominated

    parts = [data.unified_fronts[m] for m in ("adaptive", "no_transfer", "fixed_rmp")
             if data.unified_fronts.get(m) is not None and data.unified_fronts[m].shape[0] > 0]
    if parts:
        allpts = np.concatenate(parts, axis=0)
        env = non_dominated(torch.tensor(allpts)).cpu().numpy()
        env = env[np.argsort(env[:, 0])]
        ax.plot(env[:, 0], env[:, 1], "-o", ms=3, color="C0", label="RAMP (proposed)")

    uspa = data.reference["uspa"]
    rnd = data.reference["random"]
    if uspa.shape[0]:
        ax.scatter(uspa[:, 0], uspa[:, 1], marker="*", s=140, color="k", label="USPA", zorder=5)
    if rnd.shape[0]:
        ax.scatter(rnd[:, 0], rnd[:, 1], marker="x", s=22, color="gray", alpha=0.6, label="random")

    ax.set_xlabel(r"$F_{\mathrm{com}}$  (worst-user rate, bps/Hz)")
    ax.set_ylabel(r"$F_{\mathrm{sen}}$  (worst-target beampattern, dB)")
    # No in-figure title (IEEE convention): the LaTeX \caption{} provides the number
    # and description, so a baked-in "Fig N" would clash with the document numbering.
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

    # Four robust-HV-over-generation curves: proposed (multitask) vs the single-task
    # baselines. The gap to MO-DE/MOPSO reflects multitask structure + search engine;
    # the pooled (same-operator) baseline isolates the multitask effect. All curves
    # share population, generations, MC budget, eval set, and HV reference.
    # (No RMP-vs-gen inset: the paper adopts a fixed transfer rate — Framing C.)
    for m in ("adaptive", "pooled", "mo_de", "mopso"):
        hv = data.hv_gen.get(m)
        if hv is None or hv.size == 0:
            continue
        mean = hv.mean(axis=0)
        std = hv.std(axis=0)
        gens = np.arange(1, mean.shape[0] + 1)
        ax.plot(gens, mean, color=_COLORS[m], label=_LABELS[m])
        ax.fill_between(gens, mean - std, mean + std, color=_COLORS[m], alpha=0.15)

    ax.set_xlabel("generation")
    ax.set_ylabel("robust hypervolume (mean ± std)")
    # No in-figure title (IEEE convention) — see fig1.
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

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
        ax.plot(offline, cdf, color="C0", lw=2, label="offline RAMP design (one-time)")

    # real-time / online reference: re-optimised per realisation (upper bound).
    # Shown as its OWN empirical CDF for a direct distributional comparison, plus a
    # light 10-90% spread band.
    online = np.sort(data.fig3["online_hv"])
    if online.size:
        rt_cdf = np.arange(1, online.size + 1) / online.size
        ax.step(online, rt_cdf, where="post", color="C1", lw=2,
                label="real-time (per-realisation, upper bound)")
        lo, hi = np.percentile(online, [10, 90])
        ax.axvspan(lo, hi, color="C1", alpha=0.10)

    ax.set_xlabel("hypervolume per realisation")
    ax.set_ylabel("CDF")
    # No in-figure title (IEEE convention) — see fig1.
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
    ax.axhline(adaptive_hv, color="C0", ls="--", lw=2, label="RAMP (adaptive)")

    ax.set_xlabel("fixed RMP value")
    ax.set_ylabel("final hypervolume (mean over seeds)")
    ax.set_title("Fig 4 — HV vs fixed RMP (transfer benefit) + adaptive operating point")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    path = os.path.join(outdir, "fig4_rmp_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _sem(vals) -> float:
    v = np.asarray(vals, float)
    return float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0


def dfig1_hv_vs_delta(data, outdir: str = "figures_diag") -> str:
    """D-Fig 1 — unified HV vs delta_task: adaptive / oracle / single-fixed (protocol §9)."""
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    deltas = data.deltas

    am = [np.mean(data.adaptive_hv[d]) for d in deltas]
    ae = [_sem(data.adaptive_hv[d]) for d in deltas]
    ax.errorbar(deltas, am, yerr=ae, fmt="-o", color="C0", capsize=3, label="RAMP (adaptive)")

    sm = [np.mean(data.single_hv[d]) for d in deltas]
    se = [_sem(data.single_hv[d]) for d in deltas]
    ax.errorbar(deltas, sm, yerr=se, fmt="-s", color="C3", capsize=3,
                label=f"fixed RMP={data.single_fixed:g} (naïve)")

    od = [d for d in deltas if d in data.oracle_hv]
    if od:
        om = [np.mean(data.oracle_hv[d]) for d in od]
        oe = [_sem(data.oracle_hv[d]) for d in od]
        # markers only: the oracle is sampled at full_grid_deltas, so a connecting
        # line would misleadingly interpolate through deltas with no oracle data.
        ax.errorbar(od, om, yerr=oe, fmt="^", color="C2", capsize=3, linestyle="none",
                    label="best fixed per Δ (oracle, sampled)")

    ax.set_xlabel(r"$\Delta_{\mathrm{task}}$  (inter-task sector separation, deg)")
    ax.set_ylabel("unified robust HV (mean ± SEM)")
    ax.set_title("D-Fig 1 — HV vs task conflict")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    path = os.path.join(outdir, "dfig1_hv_vs_delta.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def dfig2_rmpstar_vs_delta(data, outdir: str = "figures_diag") -> str:
    """D-Fig 2 — oracle best-fixed RMP and adaptive's converged RMP vs delta_task (§9)."""
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    deltas = data.deltas

    od = sorted(data.rmp_star)
    if od:
        rstar = [data.rmp_star[d] for d in od]
        ax.plot(od, rstar, "^", color="C2", ms=10, linestyle="none",
                label="oracle best fixed RMP (sampled; noisy argmax)")

    am = [np.mean(data.adaptive_rmp[d]) for d in deltas]
    ae = [_sem(data.adaptive_rmp[d]) for d in deltas]
    ax.errorbar(deltas, am, yerr=ae, fmt="-o", color="C0", capsize=3,
                label="adaptive converged RMP")

    ax.set_xlabel(r"$\Delta_{\mathrm{task}}$  (inter-task sector separation, deg)")
    ax.set_ylabel("RMP")
    ax.set_title("D-Fig 2 — adaptive RMP tracks the oracle RMP")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    path = os.path.join(outdir, "dfig2_rmpstar_vs_delta.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def dfig3_paired_diff(data, outdir: str = "figures_diag") -> str:
    """D-Fig 3 — paired (adaptive - single-fixed) HV vs delta_task, with CI (§8/§9)."""
    from .diagnostic import paired_ci

    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    deltas = data.deltas
    means, lo_err, hi_err = [], [], []
    for d in deltas:
        diffs = np.asarray(data.adaptive_hv[d], float) - np.asarray(data.single_hv[d], float)
        m, lo, hi = paired_ci(diffs)
        means.append(m)
        lo_err.append(m - lo)
        hi_err.append(hi - m)

    ax.errorbar(deltas, means, yerr=[lo_err, hi_err], fmt="-o", color="C0", capsize=4,
                label=f"adaptive − fixed({data.single_fixed:g}), 95% paired CI")
    ax.axhline(0.0, color="k", lw=1, ls="--")
    ax.set_xlabel(r"$\Delta_{\mathrm{task}}$  (inter-task sector separation, deg)")
    ax.set_ylabel("paired HV difference")
    ax.set_title("D-Fig 3 — adaptive vs naïve fixed (paired)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    path = os.path.join(outdir, "dfig3_paired_diff.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_diagnostic_figures(data, outdir: str = "figures_diag") -> list[str]:
    return [
        dfig1_hv_vs_delta(data, outdir),
        dfig2_rmpstar_vs_delta(data, outdir),
        dfig3_paired_diff(data, outdir),
    ]


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
