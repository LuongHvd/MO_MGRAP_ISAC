"""Metrics: hypervolume, HV-contribution, knee point, front width.  [spec Sec 9; NEW]

All objectives are *maximised*; the reference (nadir) point must be fixed and
identical across every method/run or HV numbers are not comparable (spec Risk 5).
"""

from __future__ import annotations

import torch

from .config import Config
from .nsga import fast_nondominated_sort


def non_dominated(points: torch.Tensor) -> torch.Tensor:
    """Return the non-dominated subset of ``points (P,m)`` (maximisation)."""
    if points.shape[0] == 0:
        return points
    rank = fast_nondominated_sort(points)
    return points[rank == 0]


def hypervolume_2d(points: torch.Tensor, ref: torch.Tensor) -> float:
    """2D hypervolume dominated by ``points`` above reference ``ref`` (maximisation).

    ``points (P,2)``, ``ref (2,)``.  Points must dominate ``ref`` to contribute.
    """
    if points.shape[0] == 0:
        return 0.0
    # keep only points that strictly dominate the reference, then their front
    keep = (points[:, 0] > ref[0]) & (points[:, 1] > ref[1])
    pts = points[keep]
    if pts.shape[0] == 0:
        return 0.0
    pts = non_dominated(pts)
    # sort by f0 descending (ties: f1 descending) -> f1 ascending on the front
    order = torch.argsort(pts[:, 0], descending=True, stable=True)
    pts = pts[order]
    hv = 0.0
    prev_f1 = float(ref[1])
    # f1 ascends across the front; accumulate stacked rectangles
    f1_sorted = pts[:, 1]
    f0_sorted = pts[:, 0]
    order2 = torch.argsort(f1_sorted, stable=True)
    f0_sorted = f0_sorted[order2]
    f1_sorted = f1_sorted[order2]
    for i in range(pts.shape[0]):
        width = float(f0_sorted[i]) - float(ref[0])
        height = float(f1_sorted[i]) - prev_f1
        if width > 0 and height > 0:
            hv += width * height
        prev_f1 = max(prev_f1, float(f1_sorted[i]))
    return hv


def hv_contribution(points: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    r"""Per-point HV contribution ``HV(all) - HV(all\{i})``, ``(P,)``.

    Positive contribution => the point expands the front (used by the
    ``hv_contrib`` survival mode, spec Sec 3.2).
    """
    P = points.shape[0]
    contrib = torch.zeros(P, dtype=torch.float64, device=points.device)
    if P == 0:
        return contrib
    total = hypervolume_2d(points, ref)
    idx_all = torch.arange(P, device=points.device)
    for i in range(P):
        rest = points[idx_all != i]
        contrib[i] = total - hypervolume_2d(rest, ref)
    return contrib


def knee_point(front: torch.Tensor) -> tuple[int, torch.Tensor]:
    """Knee of a 2D front: point of maximum distance from the extremes' chord.

    Returns ``(index_into_front, values (2,))``.  ``front`` should be the
    non-dominated set.
    """
    P = front.shape[0]
    if P == 0:
        raise ValueError("empty front")
    if P <= 2:
        return 0, front[0]
    # normalise objectives to [0,1]
    mn = front.min(dim=0).values
    mx = front.max(dim=0).values
    span = (mx - mn).clamp_min(1e-12)
    norm = (front - mn) / span
    # chord between the two extreme points (max f0 and max f1)
    a = norm[torch.argmax(norm[:, 0])]
    b = norm[torch.argmax(norm[:, 1])]
    ab = b - a
    ab_len = torch.linalg.norm(ab).clamp_min(1e-12)
    # perpendicular distance of each point to line a-b
    ap = norm - a
    cross = torch.abs(ap[:, 0] * ab[1] - ap[:, 1] * ab[0]) / ab_len
    idx = int(torch.argmax(cross).item())
    return idx, front[idx]


def front_width(front: torch.Tensor) -> dict:
    """Spread of a front: range of each objective (spec Sec 9, front-width-vs-delta_phi)."""
    if front.shape[0] == 0:
        return {"com_range": 0.0, "sen_range": 0.0, "n_points": 0}
    rng = (front.max(dim=0).values - front.min(dim=0).values)
    return {
        "com_range": float(rng[0]),
        "sen_range": float(rng[1]),
        "n_points": int(front.shape[0]),
    }


def hv_reference(cfg: Config) -> torch.Tensor:
    """Fixed HV reference (nadir) point ``[F_com_ref, F_sen_ref]`` from config."""
    return torch.tensor([cfg.hv_ref_com, cfg.hv_ref_sen_db], dtype=torch.float64)


class RobustHVTracker:
    """Cumulative-best hypervolume of the unified robust front on a FIXED eval set.

    Every generation the current front's genotypes are re-evaluated on the same
    fixed per-regime environments (robust = ``min over regimes``); the points are
    merged with all previously-seen non-dominated points so the reported HV is
    monotone non-decreasing (best-so-far).  Measuring all methods this way makes
    the HV-over-generation curves clean and directly comparable (spec Risk 5).
    """

    def __init__(self, cfg: Config, fixed_envs, ref: torch.Tensor):
        self.cfg = cfg
        self.envs = fixed_envs
        self.ref = ref
        self.best_pts = torch.empty(0, 2, device=ref.device, dtype=torch.float64)
        self.history: list[float] = []

    def update(self, front_genos: torch.Tensor) -> float:
        from .objectives import robust_evaluate

        if front_genos.shape[0] > 0:
            pts = robust_evaluate(front_genos, self.envs, self.cfg).double().to(self.ref.device)
            allpts = torch.cat([self.best_pts, pts], dim=0)
            self.best_pts = non_dominated(allpts)
        hv = hypervolume_2d(self.best_pts, self.ref) if self.best_pts.shape[0] else 0.0
        self.history.append(hv)
        return hv
