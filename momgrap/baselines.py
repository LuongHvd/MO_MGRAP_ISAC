"""Baselines that drive the three figures.  [spec Sec 8; mix]

* no-transfer        -> NSGA-II independently per regime (allow_transfer=False)
* fixed-RMP MO-MFEA  -> same framework, RMP held constant
* pooled single-task -> ONE MO problem, each eval draws from a mixture of regimes
* reference layouts  -> fixed USPA (no MA optimisation) + random layout
* online reference   -> per-realisation MO optimisation (upper bound, expensive)

The shared single-population NSGA-II driver (`single_pop_nsga`) is reused by the
pooled baseline and the online reference.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import torch

from .archive import Archive
from .channels import Environment, sample_environment
from .config import Config
from .mfea import MFEAResult, run_mfea
from .metrics import RobustHVTracker, hv_reference, hypervolume_2d, non_dominated
from .nsga import environmental_selection, fast_nondominated_sort
from .objectives import draw_generation_environments, evaluate_single_regime, robust_evaluate
from .operators import generate_offspring


# --------------------------------------------------------------------------- #
# Thin wrappers over the main loop (configuration-only variants)              #
# --------------------------------------------------------------------------- #
def run_adaptive(cfg: Config, seed: int | None = None, tag: str = "adaptive") -> MFEAResult:
    """MO-MGRAP: adaptive Pareto-survival RMP (the proposed method)."""
    return run_mfea(dataclasses.replace(cfg, fixed_rmp=None, allow_transfer=True), seed, tag)


def run_fixed_rmp(cfg: Config, seed: int | None = None, value: float = 0.3, tag: str = "fixed_rmp") -> MFEAResult:
    """Fixed-RMP MO-MFEA ablation (isolates the value of *adaptive* RMP)."""
    return run_mfea(dataclasses.replace(cfg, fixed_rmp=value, allow_transfer=True), seed, tag)


def run_no_transfer(cfg: Config, seed: int | None = None, tag: str = "no_transfer") -> MFEAResult:
    """No-transfer: NSGA-II independently per regime (isolates the value of transfer)."""
    return run_mfea(dataclasses.replace(cfg, allow_transfer=False, fixed_rmp=0.0), seed, tag)


# --------------------------------------------------------------------------- #
# Shared single-population NSGA-II driver                                     #
# --------------------------------------------------------------------------- #
@dataclass
class SinglePopResult:
    cfg: Config
    seed: int
    hv_history: list[float]
    final_pop: torch.Tensor
    archive: Archive = field(repr=False)


def single_pop_nsga(eval_fn, cfg: Config, gen: torch.Generator, n_gen: int, seed: int,
                    hv_envs, tag: str = "single_pop", log_progress: bool = True) -> SinglePopResult:
    """One MO problem, single population, NSGA-II selection.  ``eval_fn(genos)->F (N,2)``.

    HV is tracked on ``hv_envs`` (robust = min over regimes) via the same
    cumulative-best :class:`RobustHVTracker` used by MO-MFEA, so every method's
    HV-over-generation curve is measured identically.
    """
    import time

    from .logging_utils import get_logger

    device = cfg.device
    N, D = cfg.pop_size, cfg.D
    ref = hv_reference(cfg).to(device)
    pop = torch.rand(N, D, generator=gen, device=device, dtype=cfg.dtype_real)
    skill = torch.zeros(N, dtype=torch.long, device=device)   # single task
    archive = Archive(cfg)
    tracker = RobustHVTracker(cfg, hv_envs, ref)
    log = get_logger()
    t0 = time.perf_counter()

    for _g in range(n_gen):
        offspring, _, _, _ = generate_offspring(pop, skill, 0.0, cfg, gen)
        genos_all = torch.cat([pop, offspring], dim=0)
        F_all = eval_fn(genos_all)
        sel = environmental_selection(F_all, N)
        pop = genos_all[sel]
        F_next = F_all[sel]
        rank = fast_nondominated_sort(F_next)
        front_genos = pop[rank == 0]
        archive.update(front_genos, torch.zeros(front_genos.shape[0], dtype=torch.long, device=device))
        tracker.update(front_genos)
        if log_progress and cfg.log_every and ((_g + 1) % cfg.log_every == 0 or _g == n_gen - 1):
            elapsed = time.perf_counter() - t0
            rate = (_g + 1) / max(1e-9, elapsed)
            eta = (n_gen - (_g + 1)) / max(1e-9, rate)
            log.info(f"{tag} gen {_g + 1:>4d}/{n_gen}  HV={tracker.history[-1]:.4g}  "
                     f"({rate:.2f} gen/s, ETA {eta:5.0f}s)")

    return SinglePopResult(cfg=cfg, seed=seed, hv_history=tracker.history, final_pop=pop, archive=archive)


# --------------------------------------------------------------------------- #
# Evaluation helpers                                                          #
# --------------------------------------------------------------------------- #
def pooled_evaluate(genos: torch.Tensor, cfg: Config, gen: torch.Generator) -> torch.Tensor:
    """Pooled mixture objective: S MC samples split evenly across all regimes.  ``(N,2)``.

    This is the "is multitask just data augmentation?" strawman -- one population,
    one task, environment = mixture of all regimes.
    """
    from .channels import build_comm_channels
    from .comm import bd_precoding, evm_sinr_rate
    from .genotype import decode
    from .sensing import beampattern, effective_steering

    per_regime_S = max(1, cfg.mc_samples // cfg.T)
    P_ant, theta = decode(genos, cfg)
    a_eff = effective_steering(P_ant, theta, cfg)
    worst_rate_parts = []
    worst_beam_parts = []
    for reg in cfg.regimes:
        env = sample_environment(reg, per_regime_S, gen, cfg)
        Hh = build_comm_channels(P_ant, theta, env, cfg)
        W = bd_precoding(Hh, cfg)
        R = evm_sinr_rate(Hh, W, cfg)                 # (N, perS, K)
        B = beampattern(a_eff, W, cfg)                # (N, perS, Q)
        worst_rate_parts.append(R.min(dim=2).values)  # (N, perS)
        worst_beam_parts.append(B.min(dim=2).values)  # (N, perS)
    worst_rate = torch.cat(worst_rate_parts, dim=1)    # (N, T*perS)
    worst_beam = torch.cat(worst_beam_parts, dim=1)
    f_com = worst_rate.mean(dim=1)
    f_sen_db = 10.0 * torch.log10(worst_beam.mean(dim=1) + 1e-12)
    return torch.stack([f_com, f_sen_db], dim=-1)


def run_pooled_single_task(cfg: Config, seed: int | None = None, tag: str = "pooled") -> SinglePopResult:
    """Pooled single-task baseline (spec Sec 8.3 / Fig 2)."""
    from .objectives import draw_fixed_eval_environments

    seed = cfg.seed if seed is None else seed
    gen = torch.Generator(device=cfg.device)
    gen.manual_seed(int(seed) + 7777)
    # measure HV on the SAME fixed eval-set as the MO-MFEA methods (same seed)
    hv_envs = draw_fixed_eval_environments(cfg, seed)
    return single_pop_nsga(lambda g: pooled_evaluate(g, cfg, gen), cfg, gen, cfg.max_gen, seed,
                           hv_envs=hv_envs, tag=tag)


# --------------------------------------------------------------------------- #
# Reference layouts (Fig 1 markers)                                           #
# --------------------------------------------------------------------------- #
def uspa_genotype(cfg: Config) -> torch.Tensor:
    """USPA reference: antennas at cell centres (MA gene = 0.5 -> tanh(0)=0), zero RIS phase."""
    g = torch.full((1, cfg.D), 0.5, device=cfg.device, dtype=cfg.dtype_real)
    g[:, 2 * cfg.M:] = 0.0   # theta = 0
    return g


def reference_layouts(cfg: Config, seed: int | None = None, n_random: int = 32) -> dict:
    """Return robust-objective markers for USPA and random layouts."""
    seed = cfg.seed if seed is None else seed
    gen = torch.Generator(device=cfg.device)
    gen.manual_seed(int(seed) + 1234)
    envs = draw_generation_environments(cfg, gen, cfg.mc_samples)

    uspa = robust_evaluate(uspa_genotype(cfg), envs, cfg)                 # (1,2)
    rnd_g = torch.rand(n_random, cfg.D, generator=gen, device=cfg.device, dtype=cfg.dtype_real)
    rnd = robust_evaluate(rnd_g, envs, cfg)                              # (n_random,2)
    return {"uspa": uspa, "random": rnd}


# --------------------------------------------------------------------------- #
# Robustness over realisations (Fig 3)                                        #
# --------------------------------------------------------------------------- #
def offline_hv_over_realisations(
    front_genos: torch.Tensor, cfg: Config, seed: int, n_real: int = 200
) -> torch.Tensor:
    """HV of the (fixed) offline front under ``n_real`` independent realisations.  ``(n_real,)``."""
    gen = torch.Generator(device=cfg.device)
    gen.manual_seed(int(seed) + 555)
    ref = hv_reference(cfg).to(cfg.device)
    out = torch.empty(n_real, dtype=torch.float64)
    for r in range(n_real):
        envs = draw_generation_environments(cfg, gen, 1)
        pts = robust_evaluate(front_genos, envs, cfg).double()
        out[r] = hypervolume_2d(non_dominated(pts), ref)
    return out


def online_reference_hv(cfg: Config, seed: int, n_real: int = 20, n_gen: int | None = None) -> torch.Tensor:
    """Per-realisation online optimisation HV (upper bound).  Expensive -> few realisations.

    For each realisation a single environment per regime is fixed and a short
    single-population NSGA-II optimises the worst-regime objective for *that*
    realisation; its final-front HV is recorded.
    """
    from .logging_utils import get_logger

    n_gen = max(20, cfg.max_gen // 4) if n_gen is None else n_gen
    ref = hv_reference(cfg).to(cfg.device)
    log = get_logger()
    out = torch.empty(n_real, dtype=torch.float64)
    for r in range(n_real):
        gen = torch.Generator(device=cfg.device)
        gen.manual_seed(int(seed) + 90000 + r)
        envs = draw_generation_environments(cfg, gen, 1)
        res = single_pop_nsga(lambda g: robust_evaluate(g, envs, cfg), cfg, gen, n_gen,
                              seed + r, hv_envs=envs, log_progress=False)
        out[r] = res.hv_history[-1]
        if cfg.log_every:
            log.info(f"  online-ref realisation {r + 1}/{n_real}  HV={float(out[r]):.4g}")
    return out
