"""Add more seeds to an existing experiment pickle WITHOUT recomputing old ones.

Runs only the new seeds (current_count .. --to-1) for every method already present
and appends their HV/RMP histories to the pickle, then you re-make the table/figures.
More seeds = more data = tighter CIs (the legitimate way to sharpen significance).

What it updates:  per-seed arrays -> Fig 2 (mean+/-std), Table HV column, paired CI.
What it does NOT change:  Fig 1 / Fig 3 / IGD / knee (those are seed-0 representatives).

Budget MUST match the original run; ``default_config()`` already uses N=200, G=300,
S=32. The new seeds are independent (each seeds its own fixed eval set), so this is
genuinely more data, not a re-roll of existing seeds.

Usage
-----
    python add_seeds.py --pkl results/experiment.pkl --to 25 --eval-batch 128
    python make_table.py        --pkl results/experiment.pkl
    python make_paper_figures.py --pkl results/experiment.pkl --outdir figures_paper
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from momgrap import default_config
from momgrap import baselines as B
from momgrap.experiments import load, save
from momgrap.logging_utils import setup_logging

_RUNNERS = {
    "adaptive": B.run_adaptive,
    "fixed_rmp": B.run_fixed_rmp,
    "no_transfer": B.run_no_transfer,
    "pooled": B.run_pooled_single_task,
    "mo_de": B.run_mo_de,
    "mopso": B.run_mopso,
}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Append more seeds to an experiment pickle")
    p.add_argument("--pkl", default="results/experiment.pkl")
    p.add_argument("--to", type=int, required=True, help="target TOTAL number of seeds (e.g. 25)")
    p.add_argument("--gens", type=int, default=None)
    p.add_argument("--pop", type=int, default=None)
    p.add_argument("--mc", type=int, default=None)
    p.add_argument("--eval-batch", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    log = setup_logging(level=logging.WARNING if args.quiet else logging.INFO)
    data = load(args.pkl)
    seeds = list(data.seeds)
    n0 = len(seeds)
    new_seeds = list(range(n0, args.to))
    if not new_seeds:
        log.info(f"Already have {n0} seeds (>= --to {args.to}); nothing to do.")
        return

    cfg = default_config()
    over = {}
    for k, v in (("max_gen", args.gens), ("pop_size", args.pop), ("mc_samples", args.mc),
                 ("eval_batch_size", args.eval_batch), ("log_every", args.log_every)):
        if v is not None:
            over[k] = v
    cfg = dataclasses.replace(cfg, **over)

    G = int(np.asarray(data.hv_gen["adaptive"]).shape[1])
    if G != cfg.max_gen:
        log.info(f"WARNING: stored G={G} but cfg G={cfg.max_gen}. Pass --gens {G} to match the "
                 f"original budget, or the new seeds will be inconsistent.")
    methods = [m for m in data.hv_gen if m in _RUNNERS and np.asarray(data.hv_gen[m]).size]
    log.info(f"Have {n0} seeds; adding {len(new_seeds)} -> total {args.to}. "
             f"Methods: {methods}  (N={cfg.pop_size} G={cfg.max_gen} S={cfg.mc_samples})")

    new_hv = {m: [] for m in methods}
    new_rmp, new_rho = [], []
    for sd in new_seeds:
        log.info(f"=== new seed {sd} ===")
        for m in methods:
            res = _RUNNERS[m](cfg, sd, tag=f"[{m} s{sd}]")
            new_hv[m].append(list(res.hv_history))
            if m == "adaptive":
                new_rmp.append(list(res.rmp_history))
                new_rho.append(list(res.rho_history))
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # append rows
    for m in methods:
        data.hv_gen[m] = np.vstack([np.asarray(data.hv_gen[m]), np.asarray(new_hv[m])])
    if new_rmp and getattr(data, "rmp_gen", None) is not None and np.asarray(data.rmp_gen).size:
        data.rmp_gen = np.vstack([np.asarray(data.rmp_gen), np.asarray(new_rmp)])
        data.rho_gen = np.vstack([np.asarray(data.rho_gen), np.asarray(new_rho)])
    data.seeds = seeds + new_seeds

    save(data, args.pkl)
    log.info(f"Now {len(data.seeds)} seeds. Final HV (mean+/-std):")
    for m in methods:
        col = np.asarray(data.hv_gen[m])[:, -1]
        log.info(f"  {m:12s}: {col.mean():.2f} +/- {col.std():.2f}")
    log.info(f"Saved -> {args.pkl}. Re-run make_table.py / make_paper_figures.py "
             f"(Fig 2, Table HV & paired CI update; Fig 1/3/IGD/knee stay seed-0).")


if __name__ == "__main__":
    main()
