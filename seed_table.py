"""Print a single-seed Table-I-style breakdown to check a chosen seed against the
aggregate table.

For the given seed it shows, per method: HV at that seed (from the stored per-seed
HV), the all-seed HV mean +/- std (for comparison), and the F_com/F_sen knee + IGD
computed from that seed's recomputed fronts. Lets you verify the seed used for Fig 1
is representative (its numbers track the aggregate Table I).

Usage
-----
    python seed_table.py --pkl results/experiment.pkl --seed 6 --eval-batch 128
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import types

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from momgrap import default_config
from momgrap import baselines as B
from momgrap.experiments import _unified_front_np, load
from momgrap.logging_utils import setup_logging
from momgrap.metrics import knee_point
from momgrap.tables import _ROWS, compute_igd

_RUNNERS = {
    "adaptive": B.run_adaptive,
    "no_transfer": B.run_no_transfer,
    "mo_de": B.run_mo_de,
    "mopso": B.run_mopso,
    "pooled": B.run_pooled_single_task,
}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Single-seed Table-I-style breakdown")
    p.add_argument("--pkl", default="results/experiment.pkl")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default=None, help="optional .txt output")
    p.add_argument("--gens", type=int, default=None)
    p.add_argument("--pop", type=int, default=None)
    p.add_argument("--mc", type=int, default=None)
    p.add_argument("--eval-batch", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    log = setup_logging(level=logging.WARNING if args.quiet else logging.INFO)
    data = load(args.pkl)
    seeds = list(data.seeds)
    if args.seed not in seeds:
        raise SystemExit(f"seed {args.seed} not in run seeds {seeds}")
    idx = seeds.index(args.seed)

    cfg = default_config()
    over = {}
    for k, v in (("max_gen", args.gens), ("pop_size", args.pop), ("mc_samples", args.mc),
                 ("eval_batch_size", args.eval_batch)):
        if v is not None:
            over[k] = v
    cfg = dataclasses.replace(cfg, **over)
    s_eval = cfg.mc_samples * 2

    log.info(f"Recomputing seed-{args.seed} fronts (knee/IGD) ...")
    unified = {}
    for key, _ in _ROWS:
        if key not in _RUNNERS:
            continue
        res = _RUNNERS[key](cfg, args.seed, tag=f"[{key} s{args.seed}]")
        unified[key] = _unified_front_np(res, cfg, args.seed, s_eval)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    igd = compute_igd(types.SimpleNamespace(unified_fronts=unified),
                      [k for k, _ in _ROWS if k in unified])

    # ---- build table ----
    rows = []
    for key, lbl in _ROWS:
        if key not in data.hv_gen:
            continue
        col = np.asarray(data.hv_gen[key])[:, -1]
        hv_seed = float(col[idx])
        hv_mean, hv_std = float(col.mean()), float(col.std())
        front = unified.get(key)
        if front is not None and front.shape[0] >= 1:
            _, kv = knee_point(torch.tensor(front))
            kc, ks = float(kv[0]), float(kv[1])
        else:
            kc = ks = float("nan")
        rows.append((lbl, hv_seed, hv_mean, hv_std, kc, ks, igd.get(key, float("nan"))))

    head = (f"{'Method':22s} | {'HV@s'+str(args.seed):>8s} | {f'HV mean({len(seeds)})':>12s} | "
            f"{'Fcom knee':>9s} | {'Fsen knee':>9s} | {'IGD@s'+str(args.seed):>9s}")
    lines = [f"Single-seed breakdown (seed {args.seed}) vs aggregate HV — compare to Table I:",
             head, "-" * len(head)]
    for lbl, hv_s, hv_m, hv_sd, kc, ks, ig in rows:
        lines.append(f"{lbl:22s} | {hv_s:8.2f} | {hv_m:7.2f}±{hv_sd:<4.1f} | "
                     f"{kc:9.2f} | {ks:9.2f} | {ig:9.3f}")
    text = "\n".join(lines)
    print(text)

    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
