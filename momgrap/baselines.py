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
from .nsga import crowding_distance, environmental_selection, fast_nondominated_sort
from .objectives import draw_generation_environments, evaluate_single_regime, robust_evaluate
from .operators import gaussian_mutate, generate_offspring


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
                    hv_envs, tag: str = "single_pop", log_progress: bool = True,
                    variation=None) -> SinglePopResult:
    """One MO problem, single population, NSGA-II selection.  ``eval_fn(genos)->F (N,2)``.

    HV is tracked on ``hv_envs`` (robust = min over regimes) via the same
    cumulative-best :class:`RobustHVTracker` used by MO-MFEA, so every method's
    HV-over-generation curve is measured identically.

    ``variation(pop, gen) -> offspring (N,D)`` overrides the offspring step (e.g. DE
    variation for MO-DE); default ``None`` uses the standard SBX/arith + mutation.
    Only the variation engine changes — selection/archive/HV are identical, so the
    comparison is fair by construction.
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
        if variation is None:
            offspring, _, _, _ = generate_offspring(pop, skill, 0.0, cfg, gen)
        else:
            offspring = variation(pop, gen)
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
# MO-DE (GDE3-style) — single-task MO differential evolution                  #
# --------------------------------------------------------------------------- #
def _de_variation(pop: torch.Tensor, gen: torch.Generator, F: float = 0.5, CR: float = 0.9) -> torch.Tensor:
    """DE/rand/1 with binomial crossover.  ``pop (N,D) in [0,1]`` -> trials ``(N,D)``.

    Textbook defaults F=0.5, CR=0.9 (do not tune).
    """
    device = pop.device
    N, D = pop.shape
    # three mutually-distinct donors per target, all != the target index
    scores = torch.rand(N, N, generator=gen, device=device)
    scores[torch.arange(N, device=device), torch.arange(N, device=device)] = float("inf")
    order = scores.argsort(dim=1)
    r1, r2, r3 = order[:, 0], order[:, 1], order[:, 2]
    mutant = (pop[r1] + F * (pop[r2] - pop[r3])).clamp(0.0, 1.0)          # DE/rand/1
    cross = torch.rand(N, D, generator=gen, device=device) < CR
    j_rand = torch.randint(0, D, (N,), generator=gen, device=device)
    cross[torch.arange(N, device=device), j_rand] = True                 # force >=1 dim
    return torch.where(cross, mutant, pop).clamp(0.0, 1.0)               # binomial xover


def run_mo_de(cfg: Config, seed: int | None = None, tag: str = "mo_de") -> SinglePopResult:
    """MO-DE (GDE3-style) single-task baseline: NSGA-II (mu+lambda) truncation over
    pop u DE-trials.  Same problem/budget/eval-set/HV reference as the other methods;
    only the variation engine (DE/rand/1/bin) differs."""
    from .objectives import draw_fixed_eval_environments

    seed = cfg.seed if seed is None else seed
    gen = torch.Generator(device=cfg.device)
    gen.manual_seed(int(seed) + 24601)
    hv_envs = draw_fixed_eval_environments(cfg, seed)
    return single_pop_nsga(lambda g: pooled_evaluate(g, cfg, gen), cfg, gen, cfg.max_gen, seed,
                           hv_envs=hv_envs, tag=tag,
                           variation=lambda pop, g: _de_variation(pop, g))


# --------------------------------------------------------------------------- #
# MOPSO (Coello-style) — swarm + external crowding archive                    #
# --------------------------------------------------------------------------- #
def _pareto_dominates(F1: torch.Tensor, F2: torch.Tensor) -> torch.Tensor:
    """Row-wise maximisation dominance: does ``F1`` dominate ``F2``?  ``(N,m)`` -> ``(N,)``."""
    return (F1 >= F2).all(dim=-1) & (F1 > F2).any(dim=-1)


def _archive_nondominated(genos: torch.Tensor, objs: torch.Tensor, cap: int,
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-dominated set of (genos, objs); if larger than ``cap``, prune most-crowded."""
    rank = fast_nondominated_sort(objs)
    keep = rank == 0
    g, f = genos[keep], objs[keep]
    if g.shape[0] > cap:
        cd = crowding_distance(f, torch.zeros(f.shape[0], dtype=torch.long, device=f.device))
        order = torch.argsort(cd, descending=True)[:cap]
        g, f = g[order], f[order]
    return g, f


def run_mopso(cfg: Config, seed: int | None = None, tag: str = "mopso") -> SinglePopResult:
    """MOPSO (Coello-style) single-task baseline: fixed swarm + external archive with
    crowding-based leader selection.  HV measured identically (same RobustHVTracker,
    hv_envs, hv_reference) on the archive front each generation.  Defaults: w 0.9->0.4,
    c1=c2=1.49, v_max=0.2, archive cap=N, p_mut=0.1 (do not tune)."""
    import time

    from .logging_utils import get_logger
    from .objectives import draw_fixed_eval_environments

    seed = cfg.seed if seed is None else seed
    device = cfg.device
    N, D, n_gen = cfg.pop_size, cfg.D, cfg.max_gen
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + 1729)
    hv_envs = draw_fixed_eval_environments(cfg, seed)
    ref = hv_reference(cfg).to(device)

    def eval_fn(g):
        return pooled_evaluate(g, cfg, gen)

    w0, w1, c1, c2, vmax, p_mut, cap = 0.9, 0.4, 1.49, 1.49, 0.2, 0.1, N
    X = torch.rand(N, D, generator=gen, device=device, dtype=cfg.dtype_real)
    V = torch.zeros_like(X)
    Fx = eval_fn(X)
    Xp, Fp = X.clone(), Fx.clone()
    A_g, A_f = _archive_nondominated(X, Fx, cap)

    archive = Archive(cfg)
    tracker = RobustHVTracker(cfg, hv_envs, ref)
    log = get_logger()
    t0 = time.perf_counter()

    for _g in range(n_gen):
        w = w0 + (w1 - w0) * (_g / max(1, n_gen - 1))          # linear inertia 0.9 -> 0.4
        nA = A_g.shape[0]
        if nA > 0:                                             # crowding-tournament leaders
            cd = crowding_distance(A_f, torch.zeros(nA, dtype=torch.long, device=device))
            a = torch.randint(0, nA, (N,), generator=gen, device=device)
            b = torch.randint(0, nA, (N,), generator=gen, device=device)
            leaders = A_g[torch.where(cd[a] >= cd[b], a, b)]
        else:
            leaders = Xp
        r1 = torch.rand(N, D, generator=gen, device=device)
        r2 = torch.rand(N, D, generator=gen, device=device)
        V = w * V + c1 * r1 * (Xp - X) + c2 * r2 * (leaders - X)
        V = V.clamp(-vmax, vmax)
        Xnew = X + V
        hit = (Xnew < 0.0) | (Xnew > 1.0)
        V = torch.where(hit, torch.zeros_like(V), V)           # zero velocity at bounds
        X = Xnew.clamp(0.0, 1.0)
        mut = torch.rand(N, generator=gen, device=device) < p_mut   # turbulence
        if bool(mut.any()):
            X[mut] = gaussian_mutate(X[mut], cfg.mutation_sigma, gen).clamp(0.0, 1.0)
        Fx = eval_fn(X)
        new_dom = _pareto_dominates(Fx, Fp)                    # personal-best update
        mutual = (~new_dom) & (~_pareto_dominates(Fp, Fx))
        tie = torch.rand(N, generator=gen, device=device) < 0.5
        upd = (new_dom | (mutual & tie)).unsqueeze(-1)
        Xp = torch.where(upd, X, Xp)
        Fp = torch.where(upd, Fx, Fp)
        A_g, A_f = _archive_nondominated(torch.cat([A_g, X]), torch.cat([A_f, Fx]), cap)
        archive.update(A_g, torch.zeros(A_g.shape[0], dtype=torch.long, device=device))
        tracker.update(A_g)
        if cfg.log_every and ((_g + 1) % cfg.log_every == 0 or _g == n_gen - 1):
            elapsed = time.perf_counter() - t0
            rate = (_g + 1) / max(1e-9, elapsed)
            eta = (n_gen - (_g + 1)) / max(1e-9, rate)
            log.info(f"{tag} gen {_g + 1:>4d}/{n_gen}  HV={tracker.history[-1]:.4g}  "
                     f"|A|={A_g.shape[0]}  ({rate:.2f} gen/s, ETA {eta:5.0f}s)")

    return SinglePopResult(cfg=cfg, seed=seed, hv_history=tracker.history, final_pop=X, archive=archive)


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
