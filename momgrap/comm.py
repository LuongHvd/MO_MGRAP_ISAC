"""Communication: BD precoding + EVM-aware SINR + rate.  [spec Sec 2.2; INHERITED]

Operates on the aggregated channel rows ``Hh = h_k^H`` of shape ``(N,S,K,M)``
(complex).  Block-diagonalisation (BD) precoding nulls inter-user interference
via a batched SVD null-space projection; the only residual impairment in the
SINR is receiver noise plus transmit EVM distortion.
"""

from __future__ import annotations

import torch

from .config import Config


def bd_precoding(Hh: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Block-diagonalisation precoders ``W``, ``(N,S,M,K)`` complex, unit-norm columns.

    For each user ``k`` the precoder lies in the null space of the other users'
    channels (so ``h_j^H w_k = 0`` for ``j != k``) and is aligned with ``h_k``:
    ``w_k = (V_null V_null^H h_k) / || . ||``.
    """
    N, S, K, M = Hh.shape
    h = Hh.conj()                                   # h_k (column) = conj(h_k^H), (N,S,K,M)
    W = torch.empty(N, S, M, K, dtype=Hh.dtype, device=Hh.device)

    idx_all = torch.arange(K, device=Hh.device)
    for k in range(K):
        others = idx_all[idx_all != k]
        H_minus = Hh.index_select(2, others)        # (N,S,K-1,M)
        # full_matrices=True -> Vh is (N,S,M,M); null space = last M-(K-1) cols of V.
        _, _, Vh = torch.linalg.svd(H_minus, full_matrices=True)
        V = Vh.mH                                   # (N,S,M,M)
        V_null = V[..., :, (K - 1):]                # (N,S,M, M-K+1)
        h_k = h[:, :, k, :]                         # (N,S,M)
        # project h_k onto the null space
        coeff = torch.einsum("nsmr,nsm->nsr", V_null.conj(), h_k)   # (N,S,R)
        w = torch.einsum("nsmr,nsr->nsm", V_null, coeff)            # (N,S,M)
        norm = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-12)
        W[:, :, :, k] = w / norm
    return W


def evm_sinr_rate(Hh: torch.Tensor, W: torch.Tensor, cfg: Config) -> torch.Tensor:
    """EVM-aware per-user rate ``R_{n,s,k}``, ``(N,S,K)`` real (bps/Hz).

    ``SINR_k = (1-k^2) P_k |h_k^H w_k|^2 / (sigma^2 + k^2 sum_m |h_{k,m}|^2 E|x_m|^2)``
    with ``E|x_m|^2 = sum_j P_j |w_{j,m}|^2`` and uniform ``P_k = P_max/K``.
    """
    kappa2 = cfg.evm_kappa ** 2
    p_k = cfg.p_per_user
    sigma2 = cfg.noise_power_watt

    # desired signal: h_k^H w_k  (Hh already is h^H)
    hkwk = torch.einsum("nskm,nsmk->nsk", Hh, W)             # (N,S,K)
    signal = (1.0 - kappa2) * p_k * (hkwk.abs() ** 2)        # (N,S,K)

    # transmit per-antenna power E|x_m|^2 = sum_j P_j |w_{j,m}|^2
    ex_m = p_k * (W.abs() ** 2).sum(dim=-1)                  # (N,S,M)
    # EVM distortion seen by user k: k^2 * sum_m |h_{k,m}|^2 E|x_m|^2
    h_abs2 = Hh.abs() ** 2                                   # (N,S,K,M)
    evm = kappa2 * torch.einsum("nskm,nsm->nsk", h_abs2, ex_m)  # (N,S,K)

    sinr = signal / (sigma2 + evm)
    rate = torch.log2(1.0 + sinr)
    return rate
