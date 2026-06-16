"""Main MO-MFEA loop with Pareto-transfer RMP.  [spec Sec 3.1; CHANGED]

Generalises MGRAP's Algorithm 1.  The three things that differ from the scalar
conference version (spec Sec 3):

* fitness is the 2-vector ``[F_com, F_sen]`` (objectives.py);
* environmental selection is per-regime NSGA-II (nsga.py) instead of "keep best";
* the adaptive RMP success signal is Pareto-survival of transfer offspring (rmp.py).

The same entry point covers the baselines by configuration:
``fixed_rmp`` set -> fixed-RMP MO-MFEA; ``allow_transfer=False`` -> no-transfer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .archive import Archive
from .config import Config
from .logging_utils import get_logger
from .metrics import RobustHVTracker, hv_contribution, hv_reference
from .nsga import environmental_selection, fast_nondominated_sort
from .objectives import (
    draw_fixed_eval_environments,
    draw_generation_environments,
    evaluate_population,
    evaluate_single_regime,
)
from .operators import generate_offspring
from .rmp import AdaptiveRMP, hv_survivor_count, rank1_survivor_count


@dataclass
class MFEAResult:
    cfg: Config
    seed: int
    rmp_history: list[float]
    rho_history: list[float]
    hv_history: list[float]                       # mean per-regime HV per generation
    final_pop: torch.Tensor
    final_skill: torch.Tensor
    per_regime_front: dict                         # t -> (front_points (P,2), genos (P,D))
    archive: Archive = field(repr=False)


def _regime_quotas(N: int, T: int) -> list[int]:
    """Split N survivors as evenly as possible across T regimes (N/T each)."""
    base, rem = divmod(N, T)
    return [base + (1 if t < rem else 0) for t in range(T)]


def _select_per_regime(
    F_all: torch.Tensor, skill_all: torch.Tensor, quotas: list[int], cfg: Config
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-regime NSGA-II environmental selection.

    Returns ``(selected_global_idx, front_rank_full)`` where ``front_rank_full`` is
    the within-regime non-domination rank for every pool member (-1 if unranked).
    """
    device = F_all.device
    total = F_all.shape[0]
    front_rank_full = torch.full((total,), -1, dtype=torch.long, device=device)
    selected = []
    for t in range(cfg.T):
        pool_idx = torch.nonzero(skill_all == t, as_tuple=False).flatten()
        if pool_idx.numel() == 0:
            continue
        Ft = F_all[pool_idx]
        rank_t = fast_nondominated_sort(Ft)
        front_rank_full[pool_idx] = rank_t
        sel_local = environmental_selection(Ft, quotas[t])
        selected.append(pool_idx[sel_local])
    sel_global = torch.cat(selected) if selected else torch.empty(0, dtype=torch.long, device=device)
    return sel_global, front_rank_full


def _count_group_survivors(
    sel_mask: torch.Tensor,
    front_rank_full: torch.Tensor,
    group_mask: torch.Tensor,
    F_all: torch.Tensor,
    skill_all: torch.Tensor,
    cfg: Config,
) -> int:
    """Count individuals in ``group_mask`` that 'survived' per the configured mode.

    ``rank1``: selected & in-group & in their regime's rank-1 front.
    ``hv_contrib``: in-group members of a regime's rank-1 front with positive HV
    contribution.
    """
    if cfg.survival_mode == "rank1":
        survived = sel_mask & group_mask
        return rank1_survivor_count(survived, front_rank_full)

    ref = hv_reference(cfg).to(F_all.device)
    n = 0
    for t in range(cfg.T):
        front_mask = sel_mask & (skill_all == t) & (front_rank_full == 0)
        idx = torch.nonzero(front_mask, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        contrib = hv_contribution(F_all[idx].double(), ref)
        n += hv_survivor_count(group_mask[idx], contrib)
    return n


def run_mfea(cfg: Config, seed: int | None = None, tag: str = "") -> MFEAResult:
    """Run one MO-MFEA optimisation and return per-regime fronts + archive.

    ``tag`` is a short context string (e.g. "adaptive s0") prefixed to progress logs.
    """
    cfg.validate()
    seed = cfg.seed if seed is None else seed
    device = cfg.device
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    log = get_logger()
    t_start = time.perf_counter()

    N, D, T = cfg.pop_size, cfg.D, cfg.T
    ref = hv_reference(cfg).to(device)

    # ----- init: P ~ U([0,1]^D), skill factors round-robin -----
    pop = torch.rand(N, D, generator=gen, device=device, dtype=cfg.dtype_real)
    skill = torch.arange(N, device=device) % T
    rmp = AdaptiveRMP(cfg)
    archive = Archive(cfg)

    # fixed evaluation environments for a clean, comparable HV-over-gen measurement
    fixed_envs = draw_fixed_eval_environments(cfg, seed)
    tracker = RobustHVTracker(cfg, fixed_envs, ref)

    rmp_history: list[float] = []
    rho_history: list[float] = []
    hv_history: list[float] = []

    for _g in range(cfg.max_gen):
        # 1. shared MC snapshots, one batch per regime
        envs = draw_generation_environments(cfg, gen)

        # 2. offspring via multifactorial mating (transfer- and intra-xover-tagged)
        offspring, off_skill, off_transfer, off_intra = generate_offspring(pop, skill, rmp.rmp, cfg, gen)

        # 3. fitness for the combined pool P u O (re-evaluate parents on this gen's Omega)
        genos_all = torch.cat([pop, offspring], dim=0)                 # (2N, D)
        skill_all = torch.cat([skill, off_skill], dim=0)               # (2N,)
        is_transfer_all = torch.cat(
            [torch.zeros(N, dtype=torch.bool, device=device), off_transfer], dim=0
        )
        F_all = evaluate_population(genos_all, skill_all, envs, cfg)   # (2N, 2)
        # intra-task crossover offspring are the apples-to-apples baseline for the
        # relative RMP signal (parents in second half)
        base_mask = torch.cat(
            [torch.zeros(N, dtype=torch.bool, device=device), off_intra], dim=0
        )

        # 4. per-regime NSGA-II environmental selection
        quotas = _regime_quotas(N, T)
        sel_global, front_rank_full = _select_per_regime(F_all, skill_all, quotas, cfg)
        sel_mask = torch.zeros(2 * N, dtype=torch.bool, device=device)
        sel_mask[sel_global] = True

        # 5. adaptive RMP from Pareto-survival of transfer vs intra-task offspring
        n_transfer_total = int(is_transfer_all.sum().item())
        n_base_total = int(base_mask.sum().item())
        n_success = _count_group_survivors(sel_mask, front_rank_full, is_transfer_all, F_all, skill_all, cfg)
        n_base_success = _count_group_survivors(sel_mask, front_rank_full, base_mask, F_all, skill_all, cfg)
        rmp.update(n_success, n_transfer_total, n_base_success, n_base_total)

        # 6. advance population; update archive with rank-1 per regime
        pop = genos_all[sel_global]
        skill = skill_all[sel_global]
        F_next = F_all[sel_global]
        skill_sel = skill_all[sel_global]

        gen_front_genos = []
        for t in range(T):
            mask_t = skill_sel == t
            if not bool(mask_t.any()):
                continue
            rank_t = fast_nondominated_sort(F_next[mask_t])
            front_genos = pop[mask_t][rank_t == 0]
            archive.update(front_genos, torch.full((front_genos.shape[0],), t, device=device))
            gen_front_genos.append(front_genos)
        # cumulative-best robust HV on the FIXED eval set (clean, comparable curve)
        union = torch.cat(gen_front_genos, dim=0) if gen_front_genos else pop[:0]
        tracker.update(union)
        hv_history.append(tracker.history[-1])
        rmp_history.append(rmp.rmp)
        rho_history.append(rmp.rho_history[-1])

        # progress log: every log_every gens (plus first and last)
        if cfg.log_every and ((_g + 1) % cfg.log_every == 0 or _g == 0 or _g == cfg.max_gen - 1):
            elapsed = time.perf_counter() - t_start
            rate = (_g + 1) / max(1e-9, elapsed)
            eta = (cfg.max_gen - (_g + 1)) / max(1e-9, rate)
            log.info(
                f"{tag} gen {_g + 1:>4d}/{cfg.max_gen}  HV={hv_history[-1]:.4g}  "
                f"RMP={rmp.rmp:.3f}  transfer={n_success}/{n_transfer_total}  "
                f"base={n_base_success}/{n_base_total}  "
                f"({rate:.2f} gen/s, ETA {eta:5.0f}s)"
            )

    if cfg.log_every:
        log.info(f"{tag} done in {time.perf_counter() - t_start:.1f}s  "
                 f"(final HV={hv_history[-1]:.4g}, archive={len(archive)})")

    # ----- final per-regime fronts on a fresh evaluation snapshot -----
    eval_envs = draw_generation_environments(cfg, gen)
    per_regime_front: dict = {}
    for t in range(T):
        mask_t = skill == t
        genos_t = pop[mask_t]
        if genos_t.shape[0] == 0:
            per_regime_front[t] = (torch.empty(0, 2, device=device), genos_t)
            continue
        Ft = evaluate_single_regime(genos_t, eval_envs[t], cfg)
        rank_t = fast_nondominated_sort(Ft)
        per_regime_front[t] = (Ft[rank_t == 0], genos_t[rank_t == 0])

    return MFEAResult(
        cfg=cfg, seed=seed,
        rmp_history=rmp_history, rho_history=rho_history, hv_history=hv_history,
        final_pop=pop, final_skill=skill,
        per_regime_front=per_regime_front, archive=archive,
    )
