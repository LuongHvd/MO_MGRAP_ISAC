"""Render Fig 1 (unified robust fronts) from a CHOSEN seed.

The experiment pickle stores only seed-0 fronts (to save memory), so plotting Fig 1
for another seed requires recomputing that seed's per-method unified fronts. This
runs the table methods for one seed, rebuilds their fronts + USPA/random markers,
and renders Fig 1 — without touching the stored pickle or the all-seed quantities
(Table, Fig 2, paired CI stay as they are).

Fig 1 is a *qualitative* figure (front shape + dominance over USPA/random); state
the chosen seed in the caption ("representative run, seed N"). Budget must match the
run (default_config = N200/G300/S32).

Usage
-----
    python make_fig1_seed.py --pkl results/experiment.pkl --seed 1 --eval-batch 128 --outdir figures_paper
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import logging
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from momgrap import default_config
from momgrap import baselines as B
from momgrap.experiments import _eval_generator, _front_to_np, _unified_front_np, load
from momgrap.logging_utils import setup_logging
from momgrap.metrics import knee_point
from momgrap.plots import fig1_unified_front

import torch

_RUNNERS = {
    "adaptive": B.run_adaptive,
    "no_transfer": B.run_no_transfer,
    "mo_de": B.run_mo_de,
    "mopso": B.run_mopso,
    "pooled": B.run_pooled_single_task,
}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Render Fig 1 from a chosen seed")
    p.add_argument("--pkl", default="results/experiment.pkl")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--outdir", default="figures_paper")
    p.add_argument("--gens", type=int, default=None)
    p.add_argument("--pop", type=int, default=None)
    p.add_argument("--mc", type=int, default=None)
    p.add_argument("--eval-batch", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    log = setup_logging(level=logging.WARNING if args.quiet else logging.INFO)
    data = load(args.pkl)

    cfg = default_config()
    over = {}
    for k, v in (("max_gen", args.gens), ("pop_size", args.pop), ("mc_samples", args.mc),
                 ("eval_batch_size", args.eval_batch)):
        if v is not None:
            over[k] = v
    cfg = dataclasses.replace(cfg, **over)
    s = args.seed
    s_eval = cfg.mc_samples * 2

    log.info(f"Recomputing seed-{s} fronts (N={cfg.pop_size} G={cfg.max_gen} S={cfg.mc_samples}) ...")
    unified = {}
    knee = {}
    for m, fn in _RUNNERS.items():
        res = fn(cfg, s, tag=f"[{m} s{s}]")
        front = _unified_front_np(res, cfg, s, s_eval)
        unified[m] = front
        if front.shape[0] >= 1:
            _, kv = knee_point(torch.tensor(front))
            knee[m] = (float(kv[0]), float(kv[1]))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ref = B.reference_layouts(cfg, s)

    # shallow copy of data with seed-s fronts/markers; keep the delta_phi inset as-is
    d2 = copy.copy(data)
    d2.unified_fronts = unified
    d2.reference = {"uspa": _front_to_np(ref["uspa"]), "random": _front_to_np(ref["random"])}
    d2.knee = knee

    path = fig1_unified_front(d2, args.outdir)
    log.info(f"Wrote {path}  (seed {s}). State 'representative run, seed {s}' in the caption.")
    log.info("Knee points (seed %d): %s" % (s, {m: tuple(round(x, 2) for x in kv) for m, kv in knee.items()}))


if __name__ == "__main__":
    main()
