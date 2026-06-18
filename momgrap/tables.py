"""Table I — method comparison on the robust (min-over-regimes) front.
[baselines-table spec §6]

All rows are scored identically: robust HV (mean ± std over seeds, from the same
RobustHVTracker), knee-point physical values from the seed-0 robust front, and an
optional normalized IGD against a reference front built from the union of all
methods' robust fronts. Emits a ready-to-paste IEEE-friendly LaTeX table.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .metrics import non_dominated

# two-sided 95% Student-t critical values for the paired CI
_T975 = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 14: 2.145, 15: 2.131, 20: 2.086,
         25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def _t975(df: int) -> float:
    if df <= 0:
        return float("inf")
    if df in _T975:
        return _T975[df]
    for k in sorted(_T975, reverse=True):
        if df >= k:
            return _T975[k]
    return _T975[min(_T975)]


def _paired_ci(diffs) -> tuple[float, float, float]:
    d = np.asarray(list(diffs), dtype=float)
    n = d.size
    m = float(d.mean()) if n else 0.0
    if n < 2:
        return m, m, m
    s = float(d.std(ddof=1))
    half = _t975(n - 1) * s / math.sqrt(n)
    return m, m - half, m + half


def paired_vs_proposed(data, proposed: str = "adaptive") -> dict:
    """Per-seed paired HV difference ``proposed - method`` with 95% CI.

    Returns ``{method: (mean, lo, hi, proposed_better)}`` where ``proposed_better``
    is True iff the CI excludes 0 on the positive side (proposed significantly higher).
    """
    out = {}
    if proposed not in data.hv_gen:
        return out
    base = np.asarray(data.hv_gen[proposed])[:, -1]
    for key, _ in _ROWS:
        if key == proposed or key not in data.hv_gen:
            continue
        other = np.asarray(data.hv_gen[key])[:, -1]
        n = min(base.shape[0], other.shape[0])
        m, lo, hi = _paired_ci(base[:n] - other[:n])
        out[key] = (m, lo, hi, lo > 0.0)
    return out


def paired_summary_text(data, proposed: str = "adaptive") -> str:
    """Human-readable paired-significance summary (proposed vs each method)."""
    sig = paired_vs_proposed(data, proposed)
    lbl = dict(_ROWS)
    lines = [f"Paired HV difference (proposed - method), 95% CI over {len(getattr(data,'seeds',[]) or [])} seeds:"]
    for key, (m, lo, hi, better) in sig.items():
        verdict = "proposed SIGNIFICANTLY better" if better else (
            "method significantly better" if hi < 0 else "not significant (tie)")
        lines.append(f"  vs {lbl.get(key, key):20s}: {m:+6.2f}  CI[{lo:+.2f}, {hi:+.2f}]  -> {verdict}")
    return "\n".join(lines)

# (data key, display label) in row order
_ROWS = [
    ("adaptive", "MO-MGRAP (proposed)"),
    ("pooled", "Pooled single-task"),
    ("mo_de", "MO-DE"),
    ("mopso", "MOPSO"),
    ("no_transfer", "No-transfer"),
]


def _final_hv_stats(data) -> dict:
    """mean/std of final robust HV per method (over seeds)."""
    out = {}
    for key, _ in _ROWS:
        arr = data.hv_gen.get(key)
        if arr is not None and getattr(arr, "size", 0):
            col = np.asarray(arr)[:, -1]
            out[key] = (float(col.mean()), float(col.std()))
    return out


def compute_igd(data, keys) -> dict:
    """Normalized IGD per method vs a reference front = non-dominated union of fronts."""
    fronts = {k: data.unified_fronts[k] for k in keys
              if data.unified_fronts.get(k) is not None and data.unified_fronts[k].shape[0] > 0}
    if len(fronts) < 1:
        return {}
    allpts = np.concatenate(list(fronts.values()), axis=0)
    R = non_dominated(torch.tensor(allpts)).cpu().numpy()
    mn = R.min(axis=0)
    rng = R.max(axis=0) - mn
    rng[rng < 1e-12] = 1.0
    Rn = (R - mn) / rng
    out = {}
    for k, F in fronts.items():
        Fn = (F - mn) / rng
        d = np.sqrt(((Rn[:, None, :] - Fn[None, :, :]) ** 2).sum(-1))  # (|R|,|F|)
        out[k] = float(d.min(axis=1).mean())
    return out


def table1_latex(data, with_igd: bool = True, include_no_transfer: bool = True,
                 mark_significance: bool = True, proposed: str = "adaptive") -> str:
    """Return a ready-to-paste LaTeX ``table`` (best HV bold, best IGD bold).

    With ``mark_significance``, a method's HV gets a ``\\dagger`` when the proposed
    method is significantly better on the per-seed paired 95% CI.
    """
    rows = [(k, lbl) for k, lbl in _ROWS if k in data.hv_gen
            and (include_no_transfer or k != "no_transfer")]
    hv = _final_hv_stats(data)
    keys = [k for k, _ in rows]
    igd = compute_igd(data, keys) if with_igd else {}
    sig = paired_vs_proposed(data, proposed) if mark_significance else {}

    best_hv = max((m for m, _ in hv.values()), default=None)
    best_igd = min(igd.values()) if igd else None
    n_seeds = len(getattr(data, "seeds", []) or [])
    any_dagger = any(v[3] for v in sig.values())

    ncol = 3 + (1 if igd else 0)
    colspec = "l" + "c" * ncol
    header = ["Method", "Robust HV", r"$F_{\mathrm{com}}$ knee (bps/Hz)", r"$F_{\mathrm{sen}}$ knee (dB)"]
    if igd:
        header.append("IGD")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison on the unified robust (worst-regime) front. All methods "
        r"share population, generations, Monte~Carlo budget, evaluation set, and HV "
        f"reference; HV is mean~$\\pm$~std over {n_seeds} seeds. MO-MGRAP is multitask; "
        r"Pooled, MO-DE, MOPSO are single-task (MO-DE: GDE3-style; MOPSO: Coello-style "
        r"with crowding-based leaders). Best HV/IGD in bold."
        + (r" $^{\dagger}$: proposed significantly better (per-seed paired 95\% CI)." if any_dagger else "")
        + r"}",
        r"\label{tab:methods}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\hline",
        " & ".join(header) + r" \\",
        r"\hline",
    ]

    for key, lbl in rows:
        cells = [lbl]
        if key in hv:
            m, s = hv[key]
            cell = f"{m:.1f} $\\pm$ {s:.1f}"
            if key in sig and sig[key][3]:
                cell += r"$^{\dagger}$"          # proposed significantly better
            cells.append(rf"\textbf{{{cell}}}" if (best_hv is not None and abs(m - best_hv) < 1e-9) else cell)
        else:
            cells.append("--")
        kv = getattr(data, "knee", {}).get(key)
        if kv is not None:
            cells.append(f"{kv[0]:.2f}")
            cells.append(f"{kv[1]:.2f}")
        else:
            cells += ["--", "--"]
        if igd:
            if key in igd:
                v = igd[key]
                cells.append(rf"\textbf{{{v:.3f}}}" if (best_igd is not None and abs(v - best_igd) < 1e-9) else f"{v:.3f}")
            else:
                cells.append("--")
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def save_table1(data, path: str = "results/table1.tex", **kw) -> str:
    import os

    tex = table1_latex(data, **kw)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
    return path
