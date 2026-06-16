"""Adaptive RMP driven by Pareto-survival of transfer offspring.
[spec Sec 3.2; form INHERITED, success signal NEW -- the paper's headline]

The success *signal* is the contribution:

* ``rank1`` (default): a transfer offspring is successful if, after environmental
  selection, it sits in its regime's rank-1 (non-dominated) front.
* ``hv_contrib`` (refined): successful if it makes a positive hypervolume
  contribution to its regime's rank-1 front (it expands the front).

The RMP then follows the inherited EMA controller:

    rho_ema <- (1-beta) rho_ema + beta * s_rate
    RMP     <- clip(RMP + eta (rho_ema - rho_target), RMP_min, RMP_max)

Interpretation: RMP rises when transfer offspring keep landing on each other's
fronts (synergistic regimes) and falls when they get dominated (divergent regimes,
suppress negative transfer).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .config import Config


def rank1_survivor_count(is_transfer: torch.Tensor, front_rank: torch.Tensor) -> int:
    """# transfer offspring that survived into a regime's rank-1 front (front index 0)."""
    return int(((front_rank == 0) & is_transfer).sum().item())


def hv_survivor_count(is_transfer: torch.Tensor, hv_contrib: torch.Tensor) -> int:
    """# transfer offspring with a positive HV contribution to their regime's front."""
    return int(((hv_contrib > 0.0) & is_transfer).sum().item())


@dataclass
class AdaptiveRMP:
    """EMA controller for the random-mating probability (single global RMP).

    ``relative`` mode (default) steers RMP by the gap between the transfer-offspring
    survival rate and the intra-task-offspring survival rate (a self-calibrating
    baseline), so the controller is insensitive to the absolute base rate and does
    not collapse to ``RMP_min``.  ``absolute`` mode reproduces the spec-literal form
    against a fixed ``rho_target``.
    """

    cfg: Config
    rmp: float = field(init=False)
    rho_ema: float = field(init=False)        # EMA of transfer-offspring survival rate
    base_ema: float = field(init=False)       # EMA of intra-task-offspring survival rate
    history: list[float] = field(default_factory=list)        # RMP per generation
    rho_history: list[float] = field(default_factory=list)    # transfer survival rate
    base_history: list[float] = field(default_factory=list)   # baseline (target) per gen

    def __post_init__(self) -> None:
        if self.cfg.fixed_rmp is not None:
            self.rmp = float(self.cfg.fixed_rmp)
        else:
            self.rmp = float(self.cfg.rmp_init)
        # init both EMAs equal so the initial relative gap is zero (RMP holds)
        self.rho_ema = float(self.cfg.rmp_rho_target)
        self.base_ema = float(self.cfg.rmp_rho_target)

    def update(self, n_success: int, n_transfer_total: int,
               n_base_success: int = 0, n_base_total: int = 0) -> float:
        """Update the EMA survival rates and (unless fixed) the RMP.  Returns the new RMP."""
        beta = self.cfg.rmp_beta_ema
        s_rate = n_success / max(1, n_transfer_total)
        self.rho_ema = (1.0 - beta) * self.rho_ema + beta * s_rate

        if self.cfg.rmp_signal == "relative":
            b_rate = n_base_success / max(1, n_base_total)
            self.base_ema = (1.0 - beta) * self.base_ema + beta * b_rate
            target = self.base_ema
        else:
            target = self.cfg.rmp_rho_target

        if self.cfg.fixed_rmp is None:
            new = self.rmp + self.cfg.rmp_eta * (self.rho_ema - target)
            self.rmp = float(min(self.cfg.rmp_max, max(self.cfg.rmp_min, new)))
        self.history.append(self.rmp)
        self.rho_history.append(s_rate)
        self.base_history.append(target)
        return self.rmp
