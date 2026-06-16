"""Objective assembly: ``[F_com, F_sen]`` per individual per regime.
[spec Sec 2.5, 4.2; CHANGED -> now a 2-vector]

Three aggregation layers, kept distinct (spec Sec 3.4):

1. ``min over users`` / ``min over targets``  -> inside each objective (fairness)
2. ``E over Omega`` (mean over S MC samples)   -> within a regime
3. ``min over regimes``                        -> only for the unified robust front (archive.py)

Objectives (both maximised):
    F_com = E_Omega[ min_k R_k ]      (expected worst-user rate, bps/Hz)
    F_sen = E_Omega[ min_q B_q ]      (expected worst-target illumination)
``F_sen`` is reported in dB for scale balance with ``F_com``.
"""

from __future__ import annotations

import torch

from .channels import Environment, build_comm_channels, sample_environment, sample_environments
from .comm import bd_precoding, evm_sinr_rate
from .config import Config
from .genotype import decode
from .sensing import beampattern, effective_steering

_DB_EPS = 1e-12


def draw_generation_environments(cfg: Config, generator: torch.Generator, S: int | None = None) -> list[Environment]:
    """Draw the per-regime MC snapshots ``Omega^(t)_{1:S}`` for a generation.

    With ``cfg.shared_geometry`` (default), every regime reuses the SAME geometry
    and fading draws and differs only in its LoS/NLoS K-factor weighting (applied
    at channel-construction time) -> strongly correlated regimes -> real synergy.
    Otherwise each regime draws an independent environment.
    """
    S = cfg.mc_samples if S is None else S
    return sample_environments(cfg.regimes, S, generator, cfg, cfg.shared_geometry)


_FIXED_EVAL_OFFSET = 246810


def draw_fixed_eval_environments(cfg: Config, seed: int, S: int | None = None) -> list[Environment]:
    """Deterministic per-regime eval environments shared by every method for a seed.

    Used for the clean, comparable HV-over-generation measurement (RobustHVTracker):
    the same seed yields the same fixed environments for MO-MFEA and the pooled
    baseline, so their HV curves are measured on identical data.
    """
    g = torch.Generator(device=cfg.device)
    g.manual_seed(int(seed) + _FIXED_EVAL_OFFSET)
    return draw_generation_environments(cfg, g, S)


def evaluate_single_regime(genotypes: torch.Tensor, env: Environment, cfg: Config) -> torch.Tensor:
    """Evaluate every genotype on one regime's environment.  Returns ``F (N,2)``.

    Column 0 = ``F_com`` (bps/Hz), column 1 = ``F_sen`` (dB).  Both maximised.

    The genotype axis is chunked to ``cfg.eval_batch_size`` so the channel
    tensors (which scale as O(batch * S * L * M)) stay within GPU memory even
    when ``genotypes`` holds thousands of archived configs.
    """
    N = genotypes.shape[0]
    bs = max(1, cfg.eval_batch_size)
    if N <= bs:
        return _evaluate_chunk(genotypes, env, cfg)
    parts = [_evaluate_chunk(genotypes[i:i + bs], env, cfg) for i in range(0, N, bs)]
    return torch.cat(parts, dim=0)


def _evaluate_chunk(genotypes: torch.Tensor, env: Environment, cfg: Config) -> torch.Tensor:
    """Core single-regime evaluation for a memory-sized chunk of genotypes."""
    P_ant, theta = decode(genotypes, cfg)
    Hh = build_comm_channels(P_ant, theta, env, cfg)    # (N,S,K,M)
    W = bd_precoding(Hh, cfg)                           # (N,S,M,K)
    R = evm_sinr_rate(Hh, W, cfg)                       # (N,S,K)

    a_eff = effective_steering(P_ant, theta, cfg)       # (N,Q,M)
    B = beampattern(a_eff, W, cfg)                      # (N,S,Q)

    # layer 1 (fairness) then layer 2 (E over Omega)
    f_com = R.min(dim=2).values.mean(dim=1)             # (N,)
    f_sen_lin = B.min(dim=2).values.mean(dim=1)         # (N,)
    f_sen_db = 10.0 * torch.log10(f_sen_lin + _DB_EPS)
    return torch.stack([f_com, f_sen_db], dim=-1)       # (N,2)


def robust_evaluate(genotypes: torch.Tensor, envs: list[Environment], cfg: Config) -> torch.Tensor:
    """Worst-regime objective (``min over regimes``) under fixed environments.  ``(N,2)``.

    This is the unified-robust-front objective (spec Sec 3.4 layer 3) evaluated on
    a supplied (typically *fixed*) per-regime environment list.
    """
    per_regime = [evaluate_single_regime(genotypes, env, cfg) for env in envs]
    return torch.stack(per_regime, dim=0).min(dim=0).values


def evaluate_population(
    genotypes: torch.Tensor,
    skill: torch.Tensor,
    envs: list[Environment],
    cfg: Config,
) -> torch.Tensor:
    """Evaluate each individual *only* on its skill-factor regime (spec Sec 3.1 step 3).

    ``envs`` is the per-regime environment list for this generation.
    Returns ``F (N,2)``.
    """
    N = genotypes.shape[0]
    F = torch.empty(N, 2, device=genotypes.device, dtype=cfg.dtype_real)
    for t, env in enumerate(envs):
        sel = skill == t
        if sel.any():
            F[sel] = evaluate_single_regime(genotypes[sel], env, cfg).to(F.dtype)
    return F
