"""Run the adaptive-RMP diagnostic protocol (Diagnostic_Protocol.md).

Sweeps the inter-task conflict knob delta_task, runs adaptive + a fixed-RMP grid
(oracle) + a naïve single-fixed baseline on PAIRED seeds, reads the pre-registered
decision tree (gate -> outcome A/B/C), and writes D-Fig 1/2/3.

Examples
--------
Quick plumbing check (CPU, ~minutes)::

    python run_diagnostic.py --quick

Full diagnostic (GPU)::

    python run_diagnostic.py --seeds 10 --gens 300 --pop 200 --mc 32 \
        --deltas 0 15 30 45 60 --full-grid 0 30 60 --eval-batch 128 --log-every 25
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pickle

from momgrap import default_config, smoke_config
from momgrap.diagnostic import read_decision_tree, run_diagnostic
from momgrap.logging_utils import setup_logging
from momgrap.plots import make_diagnostic_figures


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MO-MGRAP-ISAC adaptive-RMP diagnostic")
    p.add_argument("--quick", action="store_true", help="tiny fast run (plumbing check)")
    p.add_argument("--seeds", type=int, default=None, help="paired seed count")
    p.add_argument("--gens", type=int, default=None)
    p.add_argument("--pop", type=int, default=None)
    p.add_argument("--mc", type=int, default=None)
    p.add_argument("--eval-batch", type=int, default=None)
    p.add_argument("--deltas", type=float, nargs="*", default=[0.0, 15.0, 30.0, 45.0, 60.0],
                   help="delta_task values (deg) — the conflict sweep")
    p.add_argument("--grid", type=float, nargs="*", default=[0.05, 0.15, 0.3, 0.5, 0.9],
                   help="fixed-RMP grid for the oracle")
    p.add_argument("--full-grid", type=float, nargs="*", default=[0.0, 30.0, 60.0],
                   help="delta_task values where the full fixed grid runs")
    p.add_argument("--single-fixed", type=float, default=0.3, help="naïve fixed-RMP baseline")
    p.add_argument("--no-subdiag", action="store_true", help="skip rank1-vs-hv_contrib sub-diagnostic")
    p.add_argument("--outdir", default="figures_diag")
    p.add_argument("--save", default="results/diagnostic.pkl")
    p.add_argument("--log-file", default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    log = setup_logging(level=logging.WARNING if args.quiet else logging.INFO, log_file=args.log_file)

    cfg = smoke_config() if args.quick else default_config()
    over = {}
    for k, v in (("max_gen", args.gens), ("pop_size", args.pop), ("mc_samples", args.mc),
                 ("eval_batch_size", args.eval_batch), ("log_every", args.log_every)):
        if v is not None:
            over[k] = v
    cfg = dataclasses.replace(cfg, **over)

    if args.quick:
        deltas = [0.0, 30.0]
        grid = [0.05, 0.3, 0.9]
        full_grid = [0.0, 30.0]
        seeds = list(range(args.seeds if args.seeds is not None else 2))
    else:
        deltas, grid, full_grid = args.deltas, args.grid, args.full_grid
        seeds = list(range(args.seeds if args.seeds is not None else cfg.n_seeds))

    log.info(f"Diagnostic: device={cfg.device} N={cfg.pop_size} G={cfg.max_gen} S={cfg.mc_samples} "
             f"seeds={len(seeds)} deltas={deltas} full_grid={full_grid} single_fixed={args.single_fixed}")

    data = run_diagnostic(
        cfg, deltas=deltas, grid_values=grid, full_grid_deltas=full_grid,
        single_fixed=args.single_fixed, seeds=seeds, sub_diagnostic=not args.no_subdiag,
    )

    log.info("================ DECISION TREE (protocol §7) ================")
    verdict = read_decision_tree(data)
    log.info(f"================ VERDICT: outcome = {verdict.get('outcome')} ================")

    import os
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    with open(args.save, "wb") as f:
        pickle.dump({"data": data, "verdict": verdict}, f)
    log.info(f"Saved diagnostic data -> {args.save}")

    figs = make_diagnostic_figures(data, args.outdir)
    log.info("Diagnostic figures: " + ", ".join(figs))


if __name__ == "__main__":
    main()
