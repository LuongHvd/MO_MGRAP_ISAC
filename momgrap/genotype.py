"""Genotype decode + repair, integrated into the genotype->phenotype map.
[spec Sec 2.1, 6; INHERITED]

A genotype is ``x in [0,1]^D`` with ``D = 2M + L``:

* first ``2M`` entries -> MA in-cell offsets, decoded by a soft-clip ``tanh`` so
  every antenna stays inside its rectangular cell;
* last ``L`` entries  -> normalised RIS phases, decoded ``theta = 2*pi*(x mod 1)``.

Repair (repulsive-force projection for ``d_min`` violations + boundary clip onto
the panel cell) is applied to the *phenotype* (decoded positions) so the search
only ever sees feasible layouts.  Repair is **not** written back to the genotype.
"""

from __future__ import annotations

import torch

from .channels import bs_cell_centers
from .config import Config


def clip_genotype(genotype: torch.Tensor) -> torch.Tensor:
    """Keep the search inside the unit hypercube ``[0,1]^D`` (standard MFEA)."""
    return genotype.clamp(0.0, 1.0)


def repair_positions(P_raw: torch.Tensor, centers: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Repulsive-force projection + boundary clip.  ``P_raw``/return: ``(N,M,3)``.

    Repulsion (spec Sec 6):
    ``p_m += sum_{n: d<d_min} 0.5*(d_min - d_mn) * (p_m - p_n)/d_mn``, iterated I times.
    Motion stays planar (x stays 0) because all antennas share x=0.
    """
    d_min = cfg.d_min
    half_h = cfg.cell_extent_factor * cfg.wavelength / 2.0
    half_v = half_h
    P = P_raw.clone()
    eye = torch.eye(cfg.M, dtype=torch.bool, device=P.device)

    for _ in range(cfg.repair_iters):
        diff = P[:, :, None, :] - P[:, None, :, :]          # (N,M,M,3) = p_i - p_j
        dist = torch.linalg.norm(diff, dim=-1)              # (N,M,M)
        mask = (dist < d_min) & (~eye)                      # violating pairs
        safe = dist.clamp_min(1e-9)
        mag = 0.5 * (d_min - dist) * mask                   # (N,M,M)
        force = (mag / safe).unsqueeze(-1) * diff           # (N,M,M,3)
        P = P + force.sum(dim=2)                            # simultaneous update

        # boundary clip back onto each cell C_m (centred at its grid centre)
        y = P[..., 1].clamp(centers[:, 1] - half_h, centers[:, 1] + half_h)
        z = P[..., 2].clamp(centers[:, 2] - half_v, centers[:, 2] + half_v)
        P = torch.stack([torch.zeros_like(y), y, z], dim=-1)
    return P


def decode(genotype: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a batch of genotypes ``(N,D)`` into ``(P_ant (N,M,3), theta (N,L))``.

    The MA decode maps the unit-cube gene to ``[-1,1]`` and applies the soft-clip
    ``tanh`` about each cell centre; repair then enforces ``d_min`` + cell bounds.
    """
    device = genotype.device
    N = genotype.shape[0]
    M, L = cfg.M, cfg.L
    centers = bs_cell_centers(cfg).to(device=device, dtype=genotype.dtype)   # (M,3)
    half_h = cfg.cell_extent_factor * cfg.wavelength / 2.0
    half_v = half_h

    ma = genotype[:, : 2 * M].reshape(N, M, 2)
    r = 2.0 * ma - 1.0                                       # [-1,1]
    dy = half_h * torch.tanh(cfg.softclip_mu * r[..., 0])    # (N,M)
    dz = half_v * torch.tanh(cfg.softclip_mu * r[..., 1])
    y = centers[:, 1].unsqueeze(0) + dy
    z = centers[:, 2].unsqueeze(0) + dz
    P_raw = torch.stack([torch.zeros_like(y), y, z], dim=-1)  # (N,M,3)
    P = repair_positions(P_raw, centers, cfg)

    phase_hat = genotype[:, 2 * M:]                          # (N,L)
    theta = 2.0 * torch.pi * (phase_hat - torch.floor(phase_hat))
    return P, theta
