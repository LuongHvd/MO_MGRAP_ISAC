"""Variation operators + multifactorial mating.  [spec Sec 5; INHERITED + transfer tag]

Crossover (SBX or linear-arithmetic) and Gaussian mutation are the standard MFEA
operators.  The only addition for the multi-objective multitask extension is the
**TRANSFER tag** (spec Sec 3.1 step 2): offspring produced by inter-task crossover
(``tau_a != tau_b``) are flagged so the adaptive-RMP logic can later measure their
Pareto-survival.
"""

from __future__ import annotations

import torch

from .config import Config
from .genotype import clip_genotype


def sbx(p1: torch.Tensor, p2: torch.Tensor, eta: float, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulated Binary Crossover, element-wise.  ``p1``,``p2``: ``(P, D)``."""
    u = torch.rand(p1.shape, generator=generator, device=p1.device)
    exp = 1.0 / (eta + 1.0)
    beta = torch.where(u <= 0.5, (2.0 * u) ** exp, (1.0 / (2.0 * (1.0 - u))) ** exp)
    c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
    c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
    return c1, c2


def linear_arithmetic(p1: torch.Tensor, p2: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear-arithmetic crossover with element-wise ``alpha ~ U(0,1)^D``."""
    alpha = torch.rand(p1.shape, generator=generator, device=p1.device)
    c1 = alpha * p1 + (1.0 - alpha) * p2
    c2 = alpha * p2 + (1.0 - alpha) * p1
    return c1, c2


def gaussian_mutate(p: torch.Tensor, sigma: float, generator: torch.Generator) -> torch.Tensor:
    """Gaussian mutation ``c = p + sigma * eps``, ``eps ~ N(0, I_D)``."""
    eps = torch.randn(p.shape, generator=generator, device=p.device)
    return p + sigma * eps


def generate_offspring(
    pop: torch.Tensor,
    skill: torch.Tensor,
    rmp: float,
    cfg: Config,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multifactorial mating (spec Sec 3.1 step 2).

    Returns ``(offspring (N,D), offspring_skill (N,), transfer_mask (N,) bool,
    intra_xover_mask (N,) bool)``.  Produces ``N//2`` parent pairs -> ``N`` children.
    """
    device = pop.device
    N, D = pop.shape
    n_pairs = N // 2

    # distinct random parents per pair
    perm = torch.randperm(N, generator=generator, device=device)
    a_idx = perm[0:2 * n_pairs:2]
    b_idx = perm[1:2 * n_pairs:2]
    pa, pb = pop[a_idx], pop[b_idx]                         # (n_pairs, D)
    sa, sb = skill[a_idx], skill[b_idx]                     # (n_pairs,)

    # decide crossover vs mutation per pair: same task, or RMP draw succeeds.
    # With allow_transfer=False (no-transfer baseline) only same-task pairs cross over.
    same_task = sa == sb
    rdraw = torch.rand(n_pairs, generator=generator, device=device)
    if cfg.allow_transfer:
        do_xover = same_task | (rdraw < rmp)               # (n_pairs,)
    else:
        do_xover = same_task

    # crossover candidates
    if cfg.crossover_kind == "sbx":
        cx1, cx2 = sbx(pa, pb, cfg.sbx_eta, generator)
    else:
        cx1, cx2 = linear_arithmetic(pa, pb, generator)
    # mutation candidates
    mu1 = gaussian_mutate(pa, cfg.mutation_sigma, generator)
    mu2 = gaussian_mutate(pb, cfg.mutation_sigma, generator)

    xo = do_xover.unsqueeze(-1)
    c1 = torch.where(xo, cx1, mu1)
    c2 = torch.where(xo, cx2, mu2)

    # skill factors: crossover children pick randomly from {tau_a, tau_b};
    # mutation children inherit their single parent's skill factor.
    pick1 = torch.rand(n_pairs, generator=generator, device=device) < 0.5
    pick2 = torch.rand(n_pairs, generator=generator, device=device) < 0.5
    sc1 = torch.where(do_xover, torch.where(pick1, sa, sb), sa)
    sc2 = torch.where(do_xover, torch.where(pick2, sa, sb), sb)

    # offspring origin tags: inter-task crossover (TRANSFER) vs intra-task crossover.
    # The intra-task-crossover group is the apples-to-apples baseline the relative
    # RMP signal compares transfer offspring against (both are crossover; they
    # differ only in same-task vs cross-task parents -- mutation children excluded).
    transfer = do_xover & (~same_task)
    intra_xover = do_xover & same_task

    offspring = clip_genotype(torch.cat([c1, c2], dim=0))
    off_skill = torch.cat([sc1, sc2], dim=0)
    off_transfer = torch.cat([transfer, transfer], dim=0)
    off_intra_xover = torch.cat([intra_xover, intra_xover], dim=0)
    return offspring, off_skill, off_transfer, off_intra_xover
