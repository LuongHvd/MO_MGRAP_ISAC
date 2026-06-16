"""Tensorized non-dominated sort + crowding distance.  [spec Sec 4.3; NEW EC kernel]

Both objectives are *maximised*.  Dominance: ``i`` dominates ``j`` iff ``i`` is
>= ``j`` on every objective and strictly > on at least one.

For a per-regime pool of a few hundred individuals the ``P x P`` dominance tensor
is trivial on a modern GPU (spec Sec 4.3 -- keep it simple, do not over-optimise).
Cite: Liang et al., "Bridging EMO and GPU acceleration via tensorization".

Convention: front index ``0`` is the non-dominated front (the spec's "rank-1").
"""

from __future__ import annotations

import torch


def dominance_matrix(F: torch.Tensor) -> torch.Tensor:
    """Boolean ``Dom`` with ``Dom[i,j] = (i dominates j)``.  ``F``: ``(P, m)`` -> ``(P, P)``."""
    Fi = F[:, None, :]
    Fj = F[None, :, :]
    ge = (Fi >= Fj).all(dim=-1)
    gt = (Fi > Fj).any(dim=-1)
    return ge & gt


def fast_nondominated_sort(F: torch.Tensor) -> torch.Tensor:
    """Front index per individual (0 = non-dominated front).  ``F (P,m)`` -> ``rank (P,)`` long.

    Iteratively peels fronts: front-0 has domination count 0; remove it, decrement
    the counts of everything it dominated, repeat.  A handful of vectorised steps.
    """
    P = F.shape[0]
    device = F.device
    if P == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    Dom = dominance_matrix(F)
    n_dom = Dom.sum(dim=0)                      # how many dominate j
    rank = torch.full((P,), -1, dtype=torch.long, device=device)
    assigned = torch.zeros(P, dtype=torch.bool, device=device)

    f = 0
    while not bool(assigned.all()):
        front = (n_dom == 0) & (~assigned)
        if not bool(front.any()):
            # safety net (e.g. duplicate points / numerical ties): dump the rest
            rank[~assigned] = f
            break
        rank[front] = f
        assigned |= front
        # each surviving j loses the front members that dominated it
        n_dom = n_dom - Dom[front].sum(dim=0)
        f += 1
    return rank


def crowding_distance(F: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
    """Crowding distance per individual, computed within each front.  -> ``(P,)`` float.

    Boundary points of a front get ``+inf``; interior points sum normalised
    neighbour gaps over objectives.
    """
    P, m = F.shape
    cd = torch.zeros(P, dtype=F.dtype, device=F.device)
    for f in torch.unique(rank).tolist():
        idx = torch.nonzero(rank == f, as_tuple=False).flatten()
        if idx.numel() <= 2:
            cd[idx] = float("inf")
            continue
        for obj in range(m):
            vals = F[idx, obj]
            order = torch.argsort(vals)
            s_idx = idx[order]
            v_sorted = vals[order]
            span = (v_sorted[-1] - v_sorted[0]).clamp_min(1e-12)
            cd[s_idx[0]] = float("inf")
            cd[s_idx[-1]] = float("inf")
            cd[s_idx[1:-1]] = cd[s_idx[1:-1]] + (v_sorted[2:] - v_sorted[:-2]) / span
    return cd


def environmental_selection(F: torch.Tensor, n_select: int) -> torch.Tensor:
    """NSGA-II survival: pick ``n_select`` by (rank asc, crowding desc).  -> indices ``(n_select,)``."""
    P = F.shape[0]
    n_select = min(n_select, P)
    rank = fast_nondominated_sort(F)
    cd = crowding_distance(F, rank)
    # two-stage stable sort: crowding desc, then rank asc (stable keeps crowding order)
    order_cd = torch.argsort(cd, descending=True, stable=True)
    order_rank = torch.argsort(rank[order_cd], stable=True)
    final = order_cd[order_rank]
    return final[:n_select]
