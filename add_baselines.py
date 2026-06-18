"""Augment an existing experiment pickle with the MO-DE and MOPSO baselines
*without* re-running the methods already stored.

Why: ``make_table.py`` / ``make_paper_figures.py`` only re-plot a saved pickle — they
do not compute new methods. A pickle produced before MO-DE/MOPSO existed has no rows
for them, so the table is short and Fig 2 has too few curves. This script runs only
MO-DE + MOPSO (the missing ones) over the SAME seeds / budget / fixed eval set as the
stored run, injects their HV history / robust front / knee into the pickle, and
re-saves. Then re-generate the table and figures.

Budget must match the original run. ``default_config()`` already uses N=200, G=300,
S=32 (the standard full-run values); pass ``--gens/--pop/--mc`` only if your run used
different ones. (``eval_batch`` does not affect results, only memory.)

Usage
-----
    python add_baselines.py --pkl results/experiment.pkl --eval-batch 128
    python make_table.py        --pkl results/experiment.pkl
    python make_paper_figures.py --pkl results/experiment.pkl --outdir figures_paper
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from momgrap import default_config
from momgrap import baselines as B
from momgrap.experiments import _eval_generator, _unified_front_np, load, save
from momgrap.logging_utils import setup_logging
from momgrap.metrics import knee_point


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Add MO-DE + MOPSO to an existing experiment pickle")
    p.add_argument("--pkl", default="results/experiment.pkl")
    p.add_argument("--gens", type=int, default=None)
    p.add_argument("--pop", type=int, default=None)
    p.add_argument("--mc", type=int, default=None)
    p.add_argument("--eval-batch", type=int, default=None)
    p.add_argument("--s-eval", type=int, default=None, help="eval samples for the robust front (default mc*2)")
    p.add_argument("--force", action="store_true", help="recompute even if already present")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    log = setup_logging(level=logging.WARNING if args.quiet else logging.INFO)
    data = load(args.pkl)
    seeds = list(data.seeds)
    s0 = seeds[0]

    cfg = default_config()
    over = {}
    for k, v in (("max_gen", args.gens), ("pop_size", args.pop), ("mc_samples", args.mc),
                 ("eval_batch_size", args.eval_batch)):
        if v is not None:
            over[k] = v
    cfg = dataclasses.replace(cfg, **over)
    s_eval = args.s_eval if args.s_eval is not None else cfg.mc_samples * 2

    log.info(f"Augmenting {args.pkl}: seeds={seeds} N={cfg.pop_size} G={cfg.max_gen} "
             f"S={cfg.mc_samples} s_eval={s_eval}")
    if abs(float(data.hv_gen['adaptive'].shape[1]) - cfg.max_gen) > 0:
        log.info(f"WARNING: stored runs have G={data.hv_gen['adaptive'].shape[1]} but cfg G={cfg.max_gen}; "
                 f"pass --gens {data.hv_gen['adaptive'].shape[1]} to match the original budget.")

    runners = {"mo_de": B.run_mo_de, "mopso": B.run_mopso}
    for name, fn in runners.items():
        if name in data.hv_gen and not args.force:
            log.info(f"{name}: already present (use --force to recompute) — skipping")
            continue
        hvs, seed0 = [], None
        for sd in seeds:
            res = fn(cfg, sd, tag=f"[{name} s{sd}]")
            hvs.append(res.hv_history)
            if sd == s0:
                seed0 = res
        data.hv_gen[name] = np.array(hvs)
        front = _unified_front_np(seed0, cfg, s0, s_eval)
        data.unified_fronts[name] = front
        if front.shape[0] >= 1:
            _, kv = knee_point(torch.tensor(front))
            data.knee[name] = (float(kv[0]), float(kv[1]))
        log.info(f"{name}: final HV (mean over seeds) = {data.hv_gen[name][:, -1].mean():.4g}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    path = save(data, args.pkl)
    log.info(f"Saved augmented data -> {path}")
    log.info("Now run:  python make_table.py --pkl %s   and   "
             "python make_paper_figures.py --pkl %s --outdir figures_paper" % (args.pkl, args.pkl))


if __name__ == "__main__":
    main()
