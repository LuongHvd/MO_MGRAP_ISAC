"""Diagnostic protocol: does adaptive RMP actually help?  [Diagnostic_Protocol.md]

Parameterises a synergy<->conflict spectrum with the inter-task sector-separation
knob ``delta_task`` and reads the result through the pre-registered decision tree
(protocol §7).  Everything is measured with PAIRED seeds (§8): every method at a
given seed sees identical environments (paired_envs) and is scored by its final
cumulative-best robust HV on the same fixed eval set, so HV differences are
compared per-seed (tight CI), not via overlapping marginal bands.

Outputs a ``DiagnosticData`` object for the D-figures (plots.py) and a printed
gate/outcome verdict (``read_decision_tree``).
"""

from __future__ import annotations

import dataclasses
import math
import os
import pickle
from dataclasses import dataclass, field

import numpy as np

from . import baselines as B
from .config import Config
from .logging_utils import ensure_logging

# Two-sided 95% Student-t critical values (small-sample paired CI, §8).
_T975 = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086,
         30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def _t975(df: int) -> float:
    if df <= 0:
        return float("inf")
    if df in _T975:
        return _T975[df]
    for k in sorted(_T975, reverse=True):
        if df >= k:
            return _T975[k]
    return _T975[min(_T975)]


def paired_ci(diffs) -> tuple[float, float, float]:
    """Mean and 95% paired CI of per-seed differences.  Returns (mean, lo, hi)."""
    d = np.asarray(list(diffs), dtype=float)
    n = d.size
    m = float(d.mean()) if n else 0.0
    if n < 2:
        return m, m, m
    s = float(d.std(ddof=1))
    half = _t975(n - 1) * s / math.sqrt(n)
    return m, m - half, m + half


def _run_one(cfg: Config, seed: int, method: str, value: float | None = None) -> tuple[float, float]:
    """Run one method; return (final robust HV, converged RMP)."""
    tag = f"[{method}{'' if value is None else f'={value:g}'} d{cfg.delta_task_deg:g} s{seed}]"
    if method == "adaptive":
        res = B.run_adaptive(cfg, seed, tag=tag)
    elif method == "no_transfer":
        res = B.run_no_transfer(cfg, seed, tag=tag)
    elif method == "fixed":
        res = B.run_fixed_rmp(cfg, seed, value, tag=tag)
    else:
        raise ValueError(method)
    return float(res.hv_history[-1]), float(res.rmp_history[-1])


@dataclass
class DiagnosticData:
    deltas: list[float]
    seeds: list[int]
    single_fixed: float
    grid_values: list[float]
    full_grid_deltas: list[float]
    adaptive_hv: dict = field(default_factory=dict)       # delta -> [hv per seed]
    adaptive_rmp: dict = field(default_factory=dict)       # delta -> [converged rmp per seed]
    single_hv: dict = field(default_factory=dict)          # delta -> [hv per seed]
    grid_hv: dict = field(default_factory=dict)            # delta -> {value -> [hv per seed]}
    rmp_star: dict = field(default_factory=dict)           # delta -> oracle fixed RMP
    oracle_hv: dict = field(default_factory=dict)          # delta -> [hv per seed]
    subdiag: dict | None = None                            # {'delta','rank1','hv_contrib'}


def run_diagnostic(
    cfg_base: Config,
    deltas: list[float] | None = None,
    grid_values: list[float] | None = None,
    full_grid_deltas: list[float] | None = None,
    single_fixed: float = 0.3,
    seeds: list[int] | None = None,
    sub_diagnostic: bool = True,
    cache_path: str | None = "results/diagnostic_cache.pkl",
) -> DiagnosticData:
    """Run the delta_task sweep with paired seeds (protocol §6).

    All methods/values share the same ``seeds`` so every comparison is paired.
    The full fixed-RMP grid runs only at ``full_grid_deltas`` (compute trim §6);
    elsewhere only adaptive + single-fixed run.

    Checkpointing: each completed run is written to ``cache_path`` immediately, so a
    crash/disconnect loses nothing — re-running the SAME command resumes, skipping
    finished runs.  Pass ``cache_path=None`` to disable.  A config fingerprint guards
    the cache: changing N/G/S/grid starts fresh automatically.
    """
    log = ensure_logging()
    deltas = [0.0, 15.0, 30.0, 45.0, 60.0] if deltas is None else deltas
    grid_values = [0.05, 0.15, 0.3, 0.5, 0.9] if grid_values is None else grid_values
    full_grid_deltas = [0.0, 30.0, 60.0] if full_grid_deltas is None else full_grid_deltas
    seeds = list(range(cfg_base.n_seeds)) if seeds is None else seeds
    if single_fixed not in grid_values:
        grid_values = sorted(grid_values + [single_fixed])

    # ----- checkpoint cache (resume across breaks) -----
    fingerprint = (f"N{cfg_base.pop_size}-G{cfg_base.max_gen}-S{cfg_base.mc_samples}"
                   f"-D{cfg_base.D}-grid{grid_values}-paired{cfg_base.paired_envs}")
    cache: dict = {}
    if cache_path and os.path.exists(cache_path):
        try:
            blob = pickle.load(open(cache_path, "rb"))
            if blob.get("fingerprint") == fingerprint:
                cache = blob.get("runs", {})
                log.info(f"[resume] loaded {len(cache)} completed runs from {cache_path}")
            else:
                log.info("[cache] config fingerprint changed -> starting fresh")
        except Exception as e:  # corrupt/partial cache -> start fresh
            log.info(f"[cache] could not load ({e}) -> starting fresh")

    def _save_cache():
        if not cache_path:
            return
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        tmp = cache_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"fingerprint": fingerprint, "runs": cache}, f)
        os.replace(tmp, cache_path)  # atomic — never leaves a half-written cache

    def cached_run(cfg, seed, method, value=None):
        key = f"{method}|{value}|{cfg.delta_task_deg:g}|{cfg.survival_mode}|{seed}"
        if key in cache:
            return tuple(cache[key])
        out = _run_one(cfg, seed, method, value)
        cache[key] = out
        _save_cache()
        return out

    data = DiagnosticData(deltas=deltas, seeds=seeds, single_fixed=single_fixed,
                          grid_values=grid_values, full_grid_deltas=full_grid_deltas)

    for delta in deltas:
        log.info(f"=== delta_task = {delta:g} deg ===")
        cfg = dataclasses.replace(cfg_base, delta_task_deg=delta)

        data.adaptive_hv[delta], data.adaptive_rmp[delta] = [], []
        for sd in seeds:
            hv, rmp = cached_run(cfg, sd, "adaptive")
            data.adaptive_hv[delta].append(hv)
            data.adaptive_rmp[delta].append(rmp)

        data.single_hv[delta] = [cached_run(cfg, sd, "fixed", single_fixed)[0] for sd in seeds]

        if delta in full_grid_deltas:
            data.grid_hv[delta] = {}
            for v in grid_values:
                if v == single_fixed:
                    data.grid_hv[delta][v] = list(data.single_hv[delta])  # reuse
                else:
                    data.grid_hv[delta][v] = [cached_run(cfg, sd, "fixed", v)[0] for sd in seeds]
            # oracle: fixed value with best MEAN HV
            means = {v: float(np.mean(hvs)) for v, hvs in data.grid_hv[delta].items()}
            v_star = max(means, key=means.get)
            data.rmp_star[delta] = v_star
            data.oracle_hv[delta] = list(data.grid_hv[delta][v_star])

    # ----- sub-diagnostic (§7.3): rank1 vs hv_contrib at strongest conflict -----
    if sub_diagnostic and deltas:
        d_conf = max(deltas)
        log.info(f"=== sub-diagnostic at delta_task = {d_conf:g}: rank1 vs hv_contrib ===")
        cfg_r = dataclasses.replace(cfg_base, delta_task_deg=d_conf, survival_mode="rank1")
        cfg_h = dataclasses.replace(cfg_base, delta_task_deg=d_conf, survival_mode="hv_contrib")
        data.subdiag = {
            "delta": d_conf,
            "rank1": [cached_run(cfg_r, sd, "adaptive")[0] for sd in seeds],
            "hv_contrib": [cached_run(cfg_h, sd, "adaptive")[0] for sd in seeds],
        }
    return data


# --------------------------------------------------------------------------- #
# Decision tree (protocol §7) — read the result into a verdict                #
# --------------------------------------------------------------------------- #
def read_decision_tree(data: DiagnosticData, penalty_eps: float = 0.0) -> dict:
    """Apply the pre-registered decision tree and return a verdict dict + log it."""
    log = ensure_logging()
    gv = data.grid_values
    lo_d = min(data.full_grid_deltas)
    hi_d = max(data.full_grid_deltas)
    verdict: dict = {}

    # ---- §7.1 GATE: does the best fixed RMP shift with delta_task? ----
    rmp_lo = data.rmp_star.get(lo_d)
    rmp_hi = data.rmp_star.get(hi_d)
    shift_steps = abs(gv.index(rmp_hi) - gv.index(rmp_lo)) if (rmp_lo in gv and rmp_hi in gv) else 0
    # penalty of using synergy-optimal RMP at the conflict end (paired over seeds)
    penalty_diffs = []
    if hi_d in data.grid_hv and rmp_lo in data.grid_hv[hi_d]:
        a = np.asarray(data.grid_hv[hi_d][rmp_lo], float)   # rmp*(0) used at conflict end
        b = np.asarray(data.oracle_hv[hi_d], float)         # rmp*(60) at conflict end
        penalty_diffs = (b - a).tolist()
    p_mean, p_lo, p_hi = paired_ci(penalty_diffs)
    penalty_significant = (p_lo > penalty_eps)
    gate_pass = (shift_steps >= 1) and penalty_significant

    verdict["gate"] = {
        "rmp_star_lo": rmp_lo, "rmp_star_hi": rmp_hi, "shift_steps": shift_steps,
        "penalty_mean": p_mean, "penalty_ci": (p_lo, p_hi),
        "penalty_significant": penalty_significant, "pass": gate_pass,
    }
    log.info(f"[GATE] rmp*({lo_d:g})={rmp_lo}  rmp*({hi_d:g})={rmp_hi}  shift={shift_steps} step(s)")
    log.info(f"[GATE] conflict-end penalty of synergy-RMP: {p_mean:.3f} CI[{p_lo:.3f},{p_hi:.3f}]"
             f" -> {'significant' if penalty_significant else 'NOT significant'}")

    if not gate_pass:
        verdict["outcome"] = "GATE_FAIL"
        log.info("[OUTCOME] GATE FAILS — no validated conflict. Increase delta_task "
                 "(75-90 deg) then RIS-reliance (direct_atten_db) and re-run. "
                 "NOT evidence that adaptive fails.")
        return verdict

    # ---- §7.2: does adaptive track the oracle, and does single-fixed degrade? ----
    # adaptive vs oracle (paired) at full-grid deltas
    track = {}
    for d in data.full_grid_deltas:
        diffs = (np.asarray(data.oracle_hv[d], float) - np.asarray(data.adaptive_hv[d], float)).tolist()
        track[d] = paired_ci(diffs)  # (oracle - adaptive); ~0 => adaptive tracks oracle
    # adaptive vs single-fixed at the conflict end (paired)
    adv = (np.asarray(data.adaptive_hv[hi_d], float) - np.asarray(data.single_hv[hi_d], float)).tolist()
    a_mean, a_lo, a_hi = paired_ci(adv)
    single_degrades = (a_lo > 0.0)                      # adaptive significantly > single-fixed
    tracks_oracle = all(lo <= 0.0 <= hi or abs(m) < 1e-9 for (m, lo, hi) in track.values())
    # "tracks" = oracle-adaptive CI includes 0 (not significantly below oracle) at every full delta

    verdict["track_oracle"] = {str(d): track[d] for d in track}
    verdict["adaptive_vs_single_conflict"] = {"mean": a_mean, "ci": (a_lo, a_hi),
                                              "single_degrades": single_degrades}
    log.info(f"[7.2] adaptive vs single-fixed({data.single_fixed:g}) at delta={hi_d:g}: "
             f"{a_mean:.3f} CI[{a_lo:.3f},{a_hi:.3f}] -> "
             f"{'adaptive WINS' if single_degrades else 'not significant'}")
    for d, (m, lo, hi) in track.items():
        log.info(f"[7.2] oracle-adaptive at delta={d:g}: {m:.3f} CI[{lo:.3f},{hi:.3f}] -> "
                 f"{'tracks oracle' if (lo <= 0 <= hi) else 'adaptive BELOW oracle'}")

    if tracks_oracle and single_degrades:
        verdict["outcome"] = "A"
        log.info("[OUTCOME A] adaptive tracks the oracle AND beats single-fixed under conflict. "
                 "Paper framing A (robustness mechanism / when-transfer-helps).")
    elif single_degrades or _ge_best_single(data, hi_d):
        verdict["outcome"] = "B"
        log.info("[OUTCOME B] adaptive >= best single-fixed but below oracle. Try sub-diagnostic "
                 "(hv_contrib); else frame as no-tuning hyperparameter-robustness (framing B).")
    else:
        verdict["outcome"] = "C"
        log.info("[OUTCOME C] adaptive below best single-fixed even under validated conflict. "
                 "Do not force it — pivot to model+GPU framework (framing C).")

    # ---- §7.3 sub-diagnostic readout ----
    if data.subdiag:
        diffs = (np.asarray(data.subdiag["hv_contrib"], float)
                 - np.asarray(data.subdiag["rank1"], float)).tolist()
        m, lo, hi = paired_ci(diffs)
        verdict["subdiag"] = {"mean": m, "ci": (lo, hi), "hv_contrib_better": lo > 0.0}
        log.info(f"[7.3] (hv_contrib - rank1) at delta={data.subdiag['delta']:g}: "
                 f"{m:.3f} CI[{lo:.3f},{hi:.3f}] -> "
                 f"{'hv_contrib helps (signal was the culprit)' if lo > 0 else 'no signal effect'}")
    return verdict


def _ge_best_single(data: DiagnosticData, delta: float) -> bool:
    """True if adaptive mean HV >= best single fixed value's mean (at a full-grid delta)."""
    if delta not in data.grid_hv:
        return False
    best_single = max(float(np.mean(h)) for h in data.grid_hv[delta].values())
    return float(np.mean(data.adaptive_hv[delta])) >= best_single - 1e-9
