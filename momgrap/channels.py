"""Batched channel construction per regime.  [spec Sec 2.2, 4.1; INHERITED + small NEW]

Everything is GPU-native and batched over individuals ``N`` and MC realisations
``S``.  The random environment ``Omega`` for a regime/generation is captured once
in an :class:`Environment` object (user positions + multipath draws) and *shared*
across all individuals so that environmental selection compares them fairly.

Conventions (kept internally consistent, see spec Risk 2):

* Far-field array response toward unit direction ``u`` for element positions ``p``
  is ``a[i] = exp(+j * k0 * <u, p_i>)``.
* BS sits at the origin, panel in the y-z plane, broadside ``+x``.
* ``h_k^H = d_k^H + g_k^H Phi G`` is the aggregated channel (a 1xM row); we store
  the row ``Hh`` directly.  ``Phi = diag(exp(j*theta))``.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch

from .config import Config, RegimeSpec


# --------------------------------------------------------------------------- #
# Geometry helpers                                                            #
# --------------------------------------------------------------------------- #
def _planar_grid(rows: int, cols: int, spacing: float, plane: str) -> torch.Tensor:
    """Return ``(rows*cols, 3)`` element offsets on a centred planar grid.

    ``plane='yz'`` lays elements in the y-z plane (x=0); ``plane='xz'`` in x-z.
    """
    a = (torch.arange(rows, dtype=torch.float64) - (rows - 1) / 2.0) * spacing
    b = (torch.arange(cols, dtype=torch.float64) - (cols - 1) / 2.0) * spacing
    g1, g2 = torch.meshgrid(a, b, indexing="ij")
    g1 = g1.reshape(-1)
    g2 = g2.reshape(-1)
    zeros = torch.zeros_like(g1)
    if plane == "yz":
        return torch.stack([zeros, g1, g2], dim=-1)
    if plane == "xz":
        return torch.stack([g1, zeros, g2], dim=-1)
    raise ValueError(plane)


def bs_cell_centers(cfg: Config) -> torch.Tensor:
    """Grid centres ``c_m`` of the BS movable cells, ``(M, 3)`` (x=0, y-z plane)."""
    spacing = cfg.cell_spacing_factor * cfg.wavelength
    return _planar_grid(cfg.bs_grid_rows, cfg.bs_grid_cols, spacing, "yz")


def ris_element_positions(cfg: Config) -> torch.Tensor:
    """RIS element positions ``r_l``, ``(L, 3)``, centred at ``cfg.ris_position``."""
    spacing = cfg.ris_spacing_factor * cfg.wavelength
    local = _planar_grid(cfg.ris_grid_rows, cfg.ris_grid_cols, spacing, "xz")
    p_ris = torch.tensor(cfg.ris_position, dtype=torch.float64)
    return local + p_ris


def _unit_from_az_el(az: torch.Tensor, el: torch.Tensor) -> torch.Tensor:
    """Unit direction vector from azimuth/elevation (radians).  Shape ``(..., 3)``."""
    cx = torch.cos(el) * torch.cos(az)
    cy = torch.cos(el) * torch.sin(az)
    cz = torch.sin(el)
    return torch.stack([cx, cy, cz], dim=-1)


def target_directions(cfg: Config) -> torch.Tensor:
    """The Q sensing-target unit directions, ``(Q, 3)``.

    The cluster centre is offset from the user-sector centroid by ``delta_phi``
    (spec Sec 2.4): this angular separation is the knob that controls Pareto
    front width.
    """
    center = math.radians(cfg.user_sector_center_deg + cfg.target_sep_deg)
    if cfg.Q == 1:
        az = torch.tensor([center], dtype=torch.float64)
    else:
        half = math.radians(cfg.target_cluster_width_deg) / 2.0
        az = torch.linspace(center - half, center + half, cfg.Q, dtype=torch.float64)
    el = torch.zeros_like(az)
    return _unit_from_az_el(az, el)


def steering(positions: torch.Tensor, directions: torch.Tensor, k0: float) -> torch.Tensor:
    """Array response ``exp(j k0 <u, p>)``.

    ``positions``: ``(..., P, 3)``  element coordinates.
    ``directions``: ``(..., 3)``    unit direction(s); broadcast against positions.
    Returns complex tensor ``(..., P)``.
    """
    phase = k0 * torch.einsum("...d,...pd->...p", directions, positions)
    return torch.polar(torch.ones_like(phase), phase)


def _pathloss_amp(distance: torch.Tensor, cfg: Config) -> torch.Tensor:
    """sqrt(linear path loss) as an amplitude scaling for distance ``d`` (m)."""
    pl_db = cfg.pl0_db + 10.0 * cfg.pl_exponent * torch.log10(distance.clamp_min(1.0))
    return torch.sqrt(10.0 ** (-pl_db / 10.0))


def _cn(shape, generator: torch.Generator, device, dtype_c) -> torch.Tensor:
    """Standard complex normal CN(0,1) draws."""
    dtype_r = torch.float32 if dtype_c == torch.complex64 else torch.float64
    re = torch.randn(shape, generator=generator, device=device, dtype=dtype_r)
    im = torch.randn(shape, generator=generator, device=device, dtype=dtype_r)
    return torch.complex(re, im) / math.sqrt(2.0)


# --------------------------------------------------------------------------- #
# Environment (the shared random snapshot Omega for one regime/generation)     #
# --------------------------------------------------------------------------- #
@dataclass
class Environment:
    """A shared MC snapshot ``Omega^(t)_{1:S}`` for a single regime.

    Holds every random draw so that channel construction is *deterministic*
    given (Environment, MA positions, RIS phases).  All tensors live on
    ``cfg.device``.
    """

    regime: RegimeSpec
    S: int
    user_pos: torch.Tensor          # (S, K, 3)
    user_dir: torch.Tensor          # (S, K, 3) unit BS->user
    user_amp: torch.Tensor          # (S, K) sqrt path loss BS->user
    scat_dir_bu: torch.Tensor       # (S, K, P, 3) NLoS arrival dirs (BS->user)
    scat_gain_bu: torch.Tensor      # (S, K, P) complex
    G_nlos: torch.Tensor            # (S, L, M) complex iid
    g_los_dir: torch.Tensor         # (S, K, 3) unit RIS->user
    g_amp: torch.Tensor             # (S, K) sqrt path loss RIS->user
    g_nlos: torch.Tensor            # (S, K, L) complex iid


def sample_environment(regime: RegimeSpec, S: int, generator: torch.Generator, cfg: Config) -> Environment:
    """Draw the shared environment ``Omega`` for one regime and generation."""
    device = cfg.device
    dtype_c = cfg.dtype_complex
    K, L, M, P = cfg.K, cfg.L, cfg.M, cfg.n_scatter_paths

    # ----- user positions: uniform in sector, radius [r_min, r_max] -----
    az_c = math.radians(cfg.user_sector_center_deg)
    az_w = math.radians(cfg.user_sector_width_deg)
    az = az_c + (torch.rand(S, K, generator=generator, device=device) - 0.5) * az_w
    r = cfg.user_radius_min + torch.rand(S, K, generator=generator, device=device) * (
        cfg.user_radius_max - cfg.user_radius_min
    )
    el = torch.zeros(S, K, device=device)
    user_dir = _unit_from_az_el(az, el)                     # (S,K,3)
    user_pos = user_dir * r.unsqueeze(-1)                   # (S,K,3)
    user_amp = _pathloss_amp(r, cfg)                        # (S,K)

    # ----- NLoS scatter directions/gains for the BS->user link -----
    scat_w = math.radians(cfg.scatter_sector_deg)
    scat_az = az_c + (torch.rand(S, K, P, generator=generator, device=device) - 0.5) * scat_w
    scat_el = (torch.rand(S, K, P, generator=generator, device=device) - 0.5) * math.radians(20.0)
    scat_dir_bu = _unit_from_az_el(scat_az, scat_el)        # (S,K,P,3)
    scat_gain_bu = _cn((S, K, P), generator, device, dtype_c)

    # ----- BS->RIS NLoS component (iid), RIS->user link -----
    G_nlos = _cn((S, L, M), generator, device, dtype_c)

    p_ris = torch.tensor(cfg.ris_position, dtype=torch.float64, device=device)
    ris_to_user = user_pos.to(torch.float64) - p_ris        # (S,K,3)
    d_ru = torch.linalg.norm(ris_to_user, dim=-1)           # (S,K)
    g_los_dir = (ris_to_user / d_ru.clamp_min(1e-6).unsqueeze(-1)).to(user_dir.dtype)
    g_amp = _pathloss_amp(d_ru, cfg).to(user_amp.dtype)
    g_nlos = _cn((S, K, L), generator, device, dtype_c)

    return Environment(
        regime=regime, S=S,
        user_pos=user_pos, user_dir=user_dir, user_amp=user_amp,
        scat_dir_bu=scat_dir_bu, scat_gain_bu=scat_gain_bu,
        G_nlos=G_nlos, g_los_dir=g_los_dir, g_amp=g_amp, g_nlos=g_nlos,
    )


def sample_environments(regimes: list[RegimeSpec], S: int, generator: torch.Generator,
                        cfg: Config, shared_geometry: bool) -> list[Environment]:
    """Sample one Environment per regime for a generation.

    ``shared_geometry=False``: each regime is fully independent.

    ``shared_geometry=True``: all regimes share the same *geometry* (user/target
    positions and directions, scatter directions, path losses) but each redraws
    its own *fading gains* (NLoS path gains, BS-RIS and RIS-user NLoS matrices).
    This yields "synergy early, divergence late": the coarse optimum (where to
    steer) is common across regimes so good geometries transfer immediately, while
    the fine fading-specific tuning diverges -> the optimal transfer rate is high
    early and decays, which is exactly what the adaptive RMP can exploit and a
    fixed RMP cannot.
    """
    if not shared_geometry:
        return [sample_environment(reg, S, generator, cfg) for reg in regimes]

    base = sample_environment(regimes[0], S, generator, cfg)
    device, dtype_c = cfg.device, cfg.dtype_complex
    K, L, M, P = cfg.K, cfg.L, cfg.M, cfg.n_scatter_paths
    envs = []
    for reg in regimes:
        envs.append(dataclasses.replace(
            base,
            regime=reg,
            scat_gain_bu=_cn((S, K, P), generator, device, dtype_c),
            G_nlos=_cn((S, L, M), generator, device, dtype_c),
            g_nlos=_cn((S, K, L), generator, device, dtype_c),
        ))
    return envs


# --------------------------------------------------------------------------- #
# Channel construction                                                         #
# --------------------------------------------------------------------------- #
def _direct_channel(P_ant: torch.Tensor, env: Environment, cfg: Config) -> torch.Tensor:
    """BS->user direct channel ``d`` (field-response model), ``(N,S,K,M)`` complex.

    ``d_k[m] = sqrt(beta) * [ w_los * a_los[m] + w_nlos * (1/sqrt(P)) sum_p alpha_p a_p[m] ]``
    Positions ``P_ant`` enter through every steering term -> MA-position aware.
    """
    k0 = cfg.k0
    w_los, w_nlos = env.regime.los_nlos_weights()
    N = P_ant.shape[0]
    # LoS specular term: a_los[n,s,k,m]
    a_los = steering(P_ant[:, None, None, :, :], env.user_dir[None, :, :, :], k0)  # (N,S,K,M)
    # NLoS multipath term: sum over P paths
    a_scat = steering(
        P_ant[:, None, None, None, :, :],            # (N,1,1,1,M,3)
        env.scat_dir_bu[None, :, :, :, :],            # (1,S,K,P,3)
        k0,
    )                                                 # (N,S,K,P,M)
    alpha = env.scat_gain_bu[None, :, :, :, None]     # (1,S,K,P,1)
    a_nlos = (alpha * a_scat).sum(dim=3) / math.sqrt(cfg.n_scatter_paths)  # (N,S,K,M)
    amp = env.user_amp[None, :, :, None]              # (1,S,K,1)
    d = amp * (w_los * a_los + w_nlos * a_nlos)
    return d


def _bs_ris_channel(P_ant: torch.Tensor, env: Environment, cfg: Config) -> torch.Tensor:
    """BS->RIS faded channel ``G`` (comm path), ``(N,S,L,M)`` complex.

    Separable LoS (position-dependent via the BS array response) plus an iid
    NLoS matrix, mixed by the regime K-factor.
    """
    k0 = cfg.k0
    w_los, w_nlos = env.regime.los_nlos_weights()
    device = P_ant.device
    p_ris = torch.tensor(cfg.ris_position, dtype=torch.float64, device=device)
    ris_pos = ris_element_positions(cfg).to(device)                       # (L,3)

    # Geometric directions between the (fixed) RIS centre and the BS origin.
    u_bs_to_ris = (p_ris / torch.linalg.norm(p_ris)).to(P_ant.dtype)      # BS sees RIS
    u_ris_to_bs = (-p_ris / torch.linalg.norm(p_ris)).to(P_ant.dtype)     # RIS sees BS

    a_bs = steering(P_ant, u_bs_to_ris.expand(P_ant.shape[0], 3), k0)     # (N,M)
    a_ris = steering(ris_pos.to(P_ant.dtype), u_ris_to_bs, k0)            # (L,)
    G_los = a_ris[None, :, None] * a_bs[:, None, :]                       # (N,L,M)

    # BS-RIS path loss (fixed distance, infrastructure link).
    d_br = torch.linalg.norm(p_ris).to(torch.float64)
    amp_br = _pathloss_amp(d_br, cfg).to(P_ant.real.dtype)

    G = amp_br * (w_los * G_los[:, None, :, :] + w_nlos * env.G_nlos[None, :, :, :])
    return G                                                              # (N,S,L,M)


def _ris_user_channel(env: Environment, cfg: Config) -> torch.Tensor:
    """RIS->user channel ``g``, ``(S,K,L)`` complex (independent of MA positions)."""
    k0 = cfg.k0
    w_los, w_nlos = env.regime.los_nlos_weights()
    device = env.user_pos.device
    ris_pos = ris_element_positions(cfg).to(device).to(env.user_dir.dtype)  # (L,3)
    g_los = steering(ris_pos, env.g_los_dir, k0)                            # (S,K,L)
    amp = env.g_amp[:, :, None]                                             # (S,K,1)
    g = amp * (w_los * g_los + w_nlos * env.g_nlos)
    return g


def build_comm_channels(
    P_ant: torch.Tensor, theta: torch.Tensor, env: Environment, cfg: Config
) -> torch.Tensor:
    """Aggregated channel row ``Hh = h_k^H``, ``(N,S,K,M)`` complex.

    ``h_k^H = d_k^H + g_k^H Phi G`` with ``Phi = diag(exp(j theta))``.
    """
    d = _direct_channel(P_ant, env, cfg)            # (N,S,K,M)  (this is d_k^H, a row)
    G = _bs_ris_channel(P_ant, env, cfg)            # (N,S,L,M)
    g = _ris_user_channel(env, cfg)                 # (S,K,L)

    phi = torch.polar(torch.ones_like(theta), theta)            # (N,L)
    # g_k^H Phi : (S,K,L) * (N,L) -> (N,S,K,L)
    gphi = g[None, :, :, :] * phi[:, None, None, :]
    # (g_k^H Phi) G : sum over L -> (N,S,K,M)
    reflected = torch.einsum("nskl,nslm->nskm", gphi, G)
    Hh = d + reflected
    return Hh


def build_G_geo(P_ant: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Geometric (LoS) BS->RIS array response ``G_geo``, ``(N,L,M)`` complex.

    Deterministic (no ``Omega`` axis), unit-magnitude entries -- used by the
    sensing reflected path (spec Sec 2.3).  Same separable convention as the
    comm ``G`` LoS term so the two stay consistent (spec Risk 2).
    """
    k0 = cfg.k0
    device = P_ant.device
    p_ris = torch.tensor(cfg.ris_position, dtype=torch.float64, device=device)
    ris_pos = ris_element_positions(cfg).to(device).to(P_ant.dtype)
    u_bs_to_ris = (p_ris / torch.linalg.norm(p_ris)).to(P_ant.dtype)
    u_ris_to_bs = (-p_ris / torch.linalg.norm(p_ris)).to(P_ant.dtype)

    a_bs = steering(P_ant, u_bs_to_ris.expand(P_ant.shape[0], 3), k0)   # (N,M)
    a_ris = steering(ris_pos, u_ris_to_bs, k0)                          # (L,)
    G_geo = a_ris[None, :, None] * a_bs[:, None, :]                     # (N,L,M)
    return G_geo
