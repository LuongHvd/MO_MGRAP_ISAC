"""Configuration / hyperparameters.  [spec Sec 7]

A single dataclass holds every system parameter, decoder/repair knob, evolution
hyperparameter, RMP setting and output option.  Provenance tags from the spec
([INHERITED] / [CHANGED] / [NEW]) are noted inline so it is clear which knobs are
genuinely new for the multi-objective multitask extension.

All physical quantities are in SI base units (metres, hertz, watts) unless a name
ends in ``_db`` / ``_dbm``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

# Speed of light (m/s)
C_LIGHT = 299_792_458.0
# Thermal noise power spectral density at room temperature, dBm/Hz (-174 dBm/Hz)
N0_DBM_PER_HZ = -174.0


@dataclass
class RegimeSpec:
    """One propagation regime = one *task* on the multitask axis.

    A regime is fully described by its Rician K-factor (in dB):

    * ``+inf``  -> pure line-of-sight        (LoS task)
    * ``-inf``  -> pure NLoS / Rayleigh      (Rayleigh task)
    * finite    -> Rician with that K-factor (Rician task)
    """

    name: str
    k_factor_db: float

    def los_nlos_weights(self) -> tuple[float, float]:
        """Return ``(los_weight, nlos_weight)`` with ``los^2 + nlos^2 == 1``."""
        if math.isinf(self.k_factor_db):
            return (1.0, 0.0) if self.k_factor_db > 0 else (0.0, 1.0)
        k_lin = 10.0 ** (self.k_factor_db / 10.0)
        los = math.sqrt(k_lin / (k_lin + 1.0))
        nlos = math.sqrt(1.0 / (k_lin + 1.0))
        return los, nlos


def default_regimes() -> list[RegimeSpec]:
    """The T=3 regimes used in the paper (spec Sec 1, Sec 7)."""
    return [
        RegimeSpec("LoS", float("inf")),
        RegimeSpec("Rayleigh", float("-inf")),
        RegimeSpec("Rician", 6.0),
    ]


@dataclass
class Config:
    """Master configuration.  Construct via :func:`default_config` / :func:`smoke_config`."""

    # ----- System geometry (spec Sec 2.1) [INHERITED] -----
    M: int = 16                       # movable antennas at BS
    L: int = 64                       # RIS elements
    K: int = 4                        # single-antenna users
    Q: int = 3                        # sensing targets [NEW]
    carrier_hz: float = 28e9          # carrier frequency

    # BS panel geometry: M antennas on a planar (y-z) grid, broadside = +x.
    bs_grid_rows: int = 4             # rows x cols must equal M
    bs_grid_cols: int = 4
    cell_spacing_factor: float = 1.0  # nominal centre spacing = factor * lambda
    cell_extent_factor: float = 0.5   # cell full extent (L_h=L_v) = factor * lambda

    # RIS geometry: L elements on a planar (x-z) grid at a fixed location.
    ris_grid_rows: int = 8
    ris_grid_cols: int = 8
    ris_spacing_factor: float = 0.5   # RIS element spacing = factor * lambda
    ris_position: tuple[float, float, float] = (30.0, 40.0, 10.0)  # p_ris (m)

    # Users: uniform in an angular sector at radius [r_min, r_max].
    user_radius_min: float = 50.0
    user_radius_max: float = 200.0
    user_sector_center_deg: float = 0.0    # azimuth centroid of the user sector
    user_sector_width_deg: float = 60.0    # angular width of the user sector

    # Targets: Q directions, a cluster offset from the user centroid by delta_phi.
    #          delta_phi is a FIRST-CLASS knob (spec Sec 2.4) [NEW].
    target_sep_deg: float = 20.0           # delta_phi: target-cluster offset
    target_cluster_width_deg: float = 20.0  # angular spread of the Q targets

    # ----- Power / noise / impairment (spec Sec 2.2, 7) -----
    p_max_dbm: float = 30.0           # total BS transmit power budget (dBm)
    bandwidth_hz: float = 180e3       # AWGN bandwidth
    noise_figure_db: float = 9.0      # receiver noise figure
    evm_kappa: float = 0.05           # EVM impairment coefficient kappa

    # Path-loss model: PL(d)[dB] = pl0_db + 10 * n * log10(d).
    pl0_db: float = 61.4              # FSPL at 1 m for 28 GHz (~20log10(4*pi/lambda))
    pl_exponent: float = 2.2

    # NLoS multipath (field-response) model: each non-specular link is a sum of
    # n_scatter_paths plane waves with random AoD inside a scattering sector and
    # CN(0,1) gains. This makes MA positions matter even in the Rayleigh regime
    # (position enters every path's steering term), instead of degenerating to a
    # position-independent per-antenna fading.
    n_scatter_paths: int = 4
    scatter_sector_deg: float = 120.0  # angular spread of NLoS arrival directions

    # Reflected-sensing-path strength (spec Sec 2.3 / Risk 1) [NEW].
    # Scales the BS->RIS->target path relative to the direct BS->target path so
    # the RIS phase materially participates in the sensing tradeoff (prevents
    # Pareto-front collapse). The reflected path is normalised by L internally,
    # so gamma_ris ~ 1 makes a coherently-steered reflected path comparable in
    # magnitude to the unit-magnitude direct steering entries.
    gamma_ris: float = 1.0

    # ----- Decoder / repair (spec Sec 2.1, 6) [INHERITED] -----
    softclip_mu: float = 2.0          # soft-clip sharpness mu
    repair_iters: int = 3             # repulsive-force repair iterations I
    d_min_factor: float = 0.5         # d_min = factor * lambda

    # ----- Regimes (multitask axis) -----
    regimes: list[RegimeSpec] = field(default_factory=default_regimes)
    # If True, all regimes in a generation SHARE the same geometry + fading draws
    # (same users/targets/scatterers, same random fading) and differ ONLY in the
    # LoS/NLoS K-factor weighting. This models "one physical scene, different
    # propagation conditions" and creates genuine cross-regime synergy: a good
    # MA+RIS geometry transfers strongly (so adaptive transfer can pay off), while
    # fading-specific fine-tuning still diverges late (so RMP should decay).
    shared_geometry: bool = True

    # ----- Evolution (spec Sec 7) [INHERITED] -----
    pop_size: int = 200               # N
    max_gen: int = 300                # G
    mutation_sigma: float = 0.1       # Gaussian mutation sigma
    sbx_eta: float = 15.0             # SBX distribution index
    crossover_kind: str = "sbx"       # {"sbx", "linear"}
    n_seeds: int = 20                 # independent runs
    mc_samples: int = 32              # S: MC environment realisations per generation [tunable]

    # ----- RMP (form [INHERITED], success signal [NEW]) -----
    rmp_init: float = 0.3
    rmp_min: float = 0.05
    rmp_max: float = 0.5
    rmp_eta: float = 0.05             # learning rate eta
    rmp_rho_target: float = 0.4       # target survival rate (absolute mode only)
    rmp_beta_ema: float = 0.1         # EMA smoothing beta
    # Success-signal mode for the RMP controller:
    #   "relative" (default) -- compare the rank-1 survival rate of TRANSFER
    #       offspring against that of INTRA-task offspring (self-calibrating; RMP
    #       rises iff inter-task mating yields elites at least as often as staying
    #       in-task). Robust to the absolute base rate, so RMP does not collapse.
    #   "absolute" -- spec-literal: compare survival rate against a fixed rho_target.
    rmp_signal: str = "relative"      # {"relative", "absolute"}
    survival_mode: str = "rank1"      # {"rank1", "hv_contrib"} [NEW], default rank1
    fixed_rmp: float | None = None    # if set, RMP held constant (fixed-RMP baseline)
    allow_transfer: bool = True       # if False -> no inter-task mating (no-transfer baseline)

    # ----- Output (spec Sec 3.4, 7) [NEW] -----
    cross_regime_agg: str = "min"     # worst-regime aggregation for the unified front
    hv_ref_com: float = 0.0           # HV reference (nadir) point, comm objective
    hv_ref_sen_db: float = -30.0      # HV reference point, sensing objective (dB)

    # ----- Runtime -----
    seed: int = 0
    device_str: str = "auto"          # {"auto", "cpu", "cuda"}
    dtype_real: torch.dtype = torch.float32
    # Max individuals evaluated at once. Channel tensors scale as O(batch*S*L*M);
    # the genotype axis is chunked to this size so large evaluations (e.g. the
    # archive's unified-front build over thousands of configs) stay within GPU RAM.
    eval_batch_size: int = 256
    archive_max_size: int = 20000     # cap on stored rank-1 individuals
    log_every: int = 25               # log progress every N generations (0 = silent)

    # ---------------------------------------------------------------- derived
    @property
    def D(self) -> int:
        """Genotype dimensionality: 2M (MA offsets) + L (RIS phases)."""
        return 2 * self.M + self.L

    @property
    def T(self) -> int:
        """Number of regimes / tasks."""
        return len(self.regimes)

    @property
    def wavelength(self) -> float:
        return C_LIGHT / self.carrier_hz

    @property
    def k0(self) -> float:
        """Wavenumber 2*pi/lambda."""
        return 2.0 * math.pi / self.wavelength

    @property
    def d_min(self) -> float:
        return self.d_min_factor * self.wavelength

    @property
    def dtype_complex(self) -> torch.dtype:
        return torch.complex64 if self.dtype_real == torch.float32 else torch.complex128

    @property
    def p_max_watt(self) -> float:
        return 10.0 ** (self.p_max_dbm / 10.0) / 1000.0

    @property
    def p_per_user(self) -> float:
        """Uniform power allocation P_k = P_max / K."""
        return self.p_max_watt / self.K

    @property
    def noise_power_watt(self) -> float:
        """sigma^2 from thermal density + bandwidth + noise figure."""
        sigma_dbm = N0_DBM_PER_HZ + 10.0 * math.log10(self.bandwidth_hz) + self.noise_figure_db
        return 10.0 ** (sigma_dbm / 10.0) / 1000.0

    @property
    def device(self) -> torch.device:
        if self.device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_str)

    def validate(self) -> None:
        assert self.bs_grid_rows * self.bs_grid_cols == self.M, "BS grid must contain M antennas"
        assert self.ris_grid_rows * self.ris_grid_cols == self.L, "RIS grid must contain L elements"
        assert self.crossover_kind in ("sbx", "linear")
        assert self.survival_mode in ("rank1", "hv_contrib")
        assert self.rmp_signal in ("relative", "absolute")
        assert self.cross_regime_agg == "min", "spec uses worst-regime (min) aggregation"


def default_config(**overrides) -> Config:
    """Full-scale configuration matching the spec defaults (Sec 7)."""
    cfg = Config(**overrides)
    cfg.validate()
    return cfg


def smoke_config(**overrides) -> Config:
    """Small configuration for a fast CPU smoke test (spec Sec 11 build order).

    Reduces population, generations and MC samples so the end-to-end pipeline
    runs in a few seconds while still exercising every code path.
    """
    base = dict(
        pop_size=24,
        max_gen=8,
        mc_samples=4,
        n_seeds=1,
    )
    base.update(overrides)
    cfg = Config(**base)
    cfg.validate()
    return cfg
