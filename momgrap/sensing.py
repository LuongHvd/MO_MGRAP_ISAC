"""Sensing model with the RIS reflected path.  [spec Sec 2.3, 4.2; NEW]

This is the load-bearing model addition: without the BS->RIS->target reflected
path the RIS phase ``theta`` would not enter the comm-sensing tradeoff and the
Pareto front would collapse (spec Risk 1).

Effective transmit array response toward target ``q``:

    a_eff(P, theta; phi_q) = a_dir(P, phi_q) + gamma_ris/L * (G_geo(P)^T Phi a_ris(phi_q))

``a_dir``/``a_ris`` are unit-magnitude steering vectors and ``G_geo`` is the
*geometric* (deterministic) BS->RIS response.  The reflected path is normalised
by ``L`` and scaled by ``gamma_ris`` (config) so a coherently-steered RIS makes a
contribution comparable to the direct path -> ``theta`` is a genuine tradeoff lever.

Beampattern toward target ``q`` rides on the comm beamformers (no dedicated
sensing beam, no power split):  ``B_q = a_eff^H R_s a_eff``, ``R_s = sum_k P_k w_k w_k^H``.
Randomness enters only through ``R_s`` (via the comm precoders); the geometry is
deterministic.
"""

from __future__ import annotations

import torch

from .channels import build_G_geo, ris_element_positions, steering, target_directions
from .config import Config


def effective_steering(P_ant: torch.Tensor, theta: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Effective transmit array response ``a_eff``, ``(N,Q,M)`` complex (deterministic)."""
    device = P_ant.device
    k0 = cfg.k0
    tdir = target_directions(cfg).to(device=device, dtype=P_ant.dtype)        # (Q,3)
    ris_pos = ris_element_positions(cfg).to(device=device, dtype=P_ant.dtype)  # (L,3)
    N, M = P_ant.shape[0], cfg.M
    Q, L = cfg.Q, cfg.L

    # direct BS->target steering: (N,Q,M)
    a_dir = steering(P_ant[:, None, :, :], tdir[None, :, :], k0)               # (N,Q,M)

    # RIS->target steering: (Q,L)
    a_ris = steering(ris_pos[None, :, :].expand(Q, L, 3), tdir, k0)            # (Q,L)

    # geometric BS->RIS response + RIS phase
    G_geo = build_G_geo(P_ant, cfg)                                           # (N,L,M)
    phi = torch.polar(torch.ones_like(theta), theta)                          # (N,L)
    g_phi = G_geo * phi[:, :, None]                                           # (N,L,M)

    # reflected[n,q,m] = sum_l a_ris[q,l] * (G_geo Phi)[n,l,m]
    reflected = torch.einsum("ql,nlm->nqm", a_ris, g_phi)                     # (N,Q,M)
    a_eff = a_dir + (cfg.gamma_ris / L) * reflected
    return a_eff


def beampattern(a_eff: torch.Tensor, W: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Beampattern gain ``B_{n,s,q}``, ``(N,S,Q)`` real, >= 0.

    ``B_q = a_eff^H R_s a_eff = sum_k P_k |a_eff^H w_k|^2`` (avoids forming R_s).
    """
    p_k = cfg.p_per_user
    # a_eff^H w_k : (N,Q,M) x (N,S,M,K) -> (N,S,Q,K)
    proj = torch.einsum("nqm,nsmk->nsqk", a_eff.conj(), W)
    B = p_k * (proj.abs() ** 2).sum(dim=-1)                                   # (N,S,Q)
    return B
