"""Experiment runner: run matrix, seeds, persistence.  [spec Sec 10, 11]

Gathers everything the three figures need:

* Fig 1 -- unified robust fronts (adaptive / no-transfer / fixed-RMP) + USPA/random
           markers, plus an optional front-width-vs-delta_phi sweep.
* Fig 2 -- HV over generations (adaptive / fixed-RMP / pooled) + RMP-vs-gen inset.
* Fig 3 -- HV-CDF over realisations (offline unified front) + online-reference band.

``run_all`` returns a plain-dict ``ExperimentData`` of CPU/numpy arrays, which can
be pickled with :func:`save` / :func:`load` and handed to ``plots.py``.
"""

from __future__ import annotations

import dataclasses
import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch

from . import baselines as B
from .config import Config, default_config
from .mfea import run_mfea


def _eval_generator(cfg: Config, seed: int) -> torch.Generator:
    g = torch.Generator(device=cfg.device)
    g.manual_seed(int(seed) + 314159)
    return g


def _front_to_np(front: torch.Tensor) -> np.ndarray:
    return front.detach().cpu().numpy()


def _unified_front_np(result, cfg: Config, seed: int, s_eval: int) -> np.ndarray:
    front, _ = result.archive.unified_robust_front(_eval_generator(cfg, seed), s_eval)
    # sort by F_com for a clean line
    front_np = _front_to_np(front)
    if front_np.shape[0] > 1:
        front_np = front_np[np.argsort(front_np[:, 0])]
    return front_np


@dataclass
class ExperimentData:
    cfg_repr: str
    seeds: list[int]
    hv_gen: dict                 # method -> (n_seeds, G) array
    rmp_gen: np.ndarray          # (n_seeds, G) adaptive RMP
    rho_gen: np.ndarray          # (n_seeds, G) survival rate
    unified_fronts: dict         # method -> (P,2) array (seed[0])
    reference: dict              # 'uspa' (1,2), 'random' (R,2)
    fig3: dict                   # 'offline_hv' (n_real,), 'online_hv' (n_real2,)
    delta_sweep: dict            # delta_phi -> {'com_range','sen_range','n_points','hv'}
    knee: dict                   # method -> (com, sen_db)
    rmp_sweep: dict = None       # fixed RMP value -> mean final HV (the inverted-U)


def run_all(
    cfg: Config | None = None,
    seeds: list[int] | None = None,
    s_eval: int | None = None,
    do_online: bool = True,
    n_real_offline: int = 200,
    n_real_online: int = 20,
    delta_phis: list[float] | None = None,
    rmp_sweep_values: list[float] | None = None,
) -> ExperimentData:
    """Run the full comparison matrix and assemble figure data."""
    cfg = default_config() if cfg is None else cfg
    seeds = list(range(cfg.n_seeds)) if seeds is None else seeds
    s_eval = (cfg.mc_samples * 2) if s_eval is None else s_eval

    from .logging_utils import ensure_logging

    log = ensure_logging()
    # MFEA methods (multitask) and single-task baselines (one MO problem each)
    methods = {
        "adaptive": B.run_adaptive,
        "fixed_rmp": B.run_fixed_rmp,
        "no_transfer": B.run_no_transfer,
    }
    single_task = {
        "pooled": B.run_pooled_single_task,
        "mo_de": B.run_mo_de,
        "mopso": B.run_mopso,
    }
    log.info(f"run_all: {len(seeds)} seeds x {len(methods) + len(single_task)} methods "
             f"(N={cfg.pop_size} G={cfg.max_gen} S={cfg.mc_samples} T={cfg.T}) "
             f"device={cfg.device}")

    hv_gen = {m: [] for m in list(methods) + list(single_task)}
    rmp_gen, rho_gen = [], []
    # Only seed[0]'s full results are needed for the fronts (Fig 1/3); other seeds
    # contribute only HV/RMP histories, so we do not retain their (GPU-resident)
    # archives -- this keeps GPU memory flat across many seeds.
    seed0_results: dict = {}
    seed0_single: dict = {}
    s0 = seeds[0]

    for si, sd in enumerate(seeds):
        log.info(f"=== seed {sd} ({si + 1}/{len(seeds)}) ===")
        for m, fn in methods.items():
            res = fn(cfg, sd, tag=f"[{m} s{sd}]")
            hv_gen[m].append(res.hv_history)
            if m == "adaptive":
                rmp_gen.append(res.rmp_history)
                rho_gen.append(res.rho_history)
            if sd == s0:
                seed0_results[m] = res
        for m, fn in single_task.items():
            res = fn(cfg, sd, tag=f"[{m} s{sd}]")
            hv_gen[m].append(res.hv_history)
            if sd == s0:
                seed0_single[m] = res
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    hv_gen_np = {m: np.array(v) for m, v in hv_gen.items()}

    # ----- Fig 1: unified robust fronts (seed[0]) + reference markers -----
    log.info("building unified robust fronts (seed 0) ...")
    unified = {m: _unified_front_np(seed0_results[m], cfg, s0, s_eval) for m in methods}
    for m, res in seed0_single.items():
        unified[m] = _unified_front_np(res, cfg, s0, s_eval)
    ref = B.reference_layouts(cfg, s0)
    reference = {"uspa": _front_to_np(ref["uspa"]), "random": _front_to_np(ref["random"])}

    # knee-point physical values per method (spec Sec 9)
    from .metrics import knee_point

    knee = {}
    for m, front in unified.items():
        if front.shape[0] >= 1:
            ft = torch.tensor(front)
            ki, kv = knee_point(ft)
            knee[m] = (float(kv[0]), float(kv[1]))

    # ----- Fig 3: robustness of the adaptive offline unified front -----
    log.info(f"Fig 3: offline HV over {n_real_offline} realisations"
             + (f" + online reference over {n_real_online}" if do_online else " (online skipped)"))
    adaptive0 = seed0_results["adaptive"]
    _, adaptive_front_genos = adaptive0.archive.unified_robust_front(_eval_generator(cfg, s0), s_eval)
    offline_hv = B.offline_hv_over_realisations(adaptive_front_genos, cfg, s0, n_real_offline)
    # USPA + random fixed-layout HV-over-realisations (same realisations as offline)
    ref_hv = B.ref_layout_hv_over_realisations(cfg, s0, n_real_offline)
    if do_online:
        online_hv, realtime_front = B.online_reference_hv(cfg, s0, n_real_online, collect_fronts=True)
        reference["realtime_front"] = _front_to_np(realtime_front)
    else:
        online_hv = torch.empty(0)
    fig3 = {
        "offline_hv": offline_hv.cpu().numpy(),
        "online_hv": online_hv.cpu().numpy(),
        "uspa_hv": ref_hv["uspa_hv"],
        "random_hv": ref_hv["random_hv"],
    }

    # ----- Fig 1 inset: front width vs delta_phi sweep -----
    delta_sweep: dict = {}
    if delta_phis:
        from .metrics import front_width, hv_reference, hypervolume_2d

        log.info(f"front-width vs delta_phi sweep over {len(delta_phis)} values: {delta_phis}")
        ref_pt = hv_reference(cfg).to(cfg.device)
        for dphi in delta_phis:
            cfg_d = dataclasses.replace(cfg, target_sep_deg=dphi)
            res_d = B.run_adaptive(cfg_d, s0, tag=f"[sweep dphi={dphi:g}]")
            front, _ = res_d.archive.unified_robust_front(_eval_generator(cfg_d, s0), s_eval)
            fw = front_width(front)
            fw["hv"] = hypervolume_2d(front.double(), ref_pt)
            delta_sweep[float(dphi)] = fw

    # ----- fixed-RMP sweep: shows the inverted-U + that adaptive auto-tunes -----
    rmp_sweep: dict = {}
    if rmp_sweep_values:
        import dataclasses as _dc

        log.info(f"fixed-RMP sweep over {rmp_sweep_values} ({len(seeds)} seeds each)")
        for v in rmp_sweep_values:
            cfg_v = _dc.replace(cfg, fixed_rmp=v, allow_transfer=(v > 0))
            vals = [run_mfea(cfg_v, sd, tag=f"[fixedRMP={v:g} s{sd}]").hv_history[-1] for sd in seeds]
            rmp_sweep[float(v)] = float(np.mean(vals))

    log.info("run_all complete")

    return ExperimentData(
        cfg_repr=repr(cfg),
        seeds=list(seeds),
        hv_gen=hv_gen_np,
        rmp_gen=np.array(rmp_gen),
        rho_gen=np.array(rho_gen),
        unified_fronts=unified,
        reference=reference,
        fig3=fig3,
        delta_sweep=delta_sweep,
        knee=knee,
        rmp_sweep=rmp_sweep,
    )


def save(data: ExperimentData, path: str = "results/experiment.pkl") -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


def load(path: str = "results/experiment.pkl") -> ExperimentData:
    with open(path, "rb") as f:
        return pickle.load(f)
