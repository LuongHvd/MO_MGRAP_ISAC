"""External archive + unified robust front.  [spec Sec 3.4; NEW -- the deployable product]

The archive collects every rank-1 individual across regimes and generations
(genotype + which regime it was rank-1 in).  The deployable product is the
**unified robust front**: re-evaluate every archived config on *all* regimes,
aggregate cross-regime by the worst regime (``min over t``), then take the
non-dominated set.

    F_com^robust(x) = min_t F_com^(t)(x)
    F_sen^robust(x) = min_t F_sen^(t)(x)

This is the one-time offline design; online operation only picks an operating
point on this front (zero re-optimisation).
"""

from __future__ import annotations

import torch

from .config import Config
from .nsga import fast_nondominated_sort
from .objectives import draw_generation_environments, evaluate_single_regime


class Archive:
    """Stores rank-1 genotypes and the regime they were rank-1 in."""

    def __init__(self, cfg: Config, max_size: int | None = None):
        self.cfg = cfg
        self.max_size = cfg.archive_max_size if max_size is None else max_size
        self.genotypes = torch.empty(0, cfg.D, device=cfg.device, dtype=cfg.dtype_real)
        self.regime = torch.empty(0, dtype=torch.long, device=cfg.device)

    def update(self, genotypes: torch.Tensor, regime_idx: torch.Tensor) -> None:
        """Append a batch of rank-1 individuals (genotypes + their regime index)."""
        self.genotypes = torch.cat([self.genotypes, genotypes.to(self.genotypes)], dim=0)
        self.regime = torch.cat([self.regime, regime_idx.to(self.regime)], dim=0)
        if self.genotypes.shape[0] > self.max_size:
            self.genotypes = self.genotypes[-self.max_size:]
            self.regime = self.regime[-self.max_size:]

    def __len__(self) -> int:
        return self.genotypes.shape[0]

    def robust_objectives(self, generator: torch.Generator, S_eval: int | None = None) -> torch.Tensor:
        """Re-evaluate every archived config on all regimes; aggregate by worst regime.

        Returns ``(A, 2)`` robust objective values (``min over regimes``).
        """
        A = len(self)
        if A == 0:
            return torch.empty(0, 2, device=self.cfg.device, dtype=self.cfg.dtype_real)
        envs = draw_generation_environments(self.cfg, generator, S_eval)
        per_regime = []
        for env in envs:
            per_regime.append(evaluate_single_regime(self.genotypes, env, self.cfg))  # (A,2)
        stacked = torch.stack(per_regime, dim=0)        # (T, A, 2)
        # spec Sec 3.4 layer 3: worst-regime guarantee = min over regimes
        robust = stacked.min(dim=0).values              # (A, 2)
        return robust

    def unified_robust_front(
        self, generator: torch.Generator, S_eval: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The non-dominated set of robust objectives.  Returns ``(front (P,2), genos (P,D))``."""
        robust = self.robust_objectives(generator, S_eval)
        if robust.shape[0] == 0:
            return robust, self.genotypes.clone()
        rank = fast_nondominated_sort(robust)
        keep = rank == 0
        return robust[keep], self.genotypes[keep]
