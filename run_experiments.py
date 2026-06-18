"""Entry point: run the MO-MGRAP-ISAC experiment matrix and produce the 3 figures.

Examples
--------
Quick sanity (small, ~1 min on CPU)::

    python run_experiments.py --quick

Full paper run (heavy; minutes-to-hours on CPU, fast on a CUDA GPU)::

    python run_experiments.py --seeds 20 --gens 300 --pop 200 --mc 32

Reduced but realistic run::

    python run_experiments.py --seeds 5 --gens 100 --pop 120 --mc 16
"""

from __future__ import annotations

import argparse
import dataclasses
import logging

from momgrap import default_config, smoke_config
from momgrap.experiments import run_all, save
from momgrap.logging_utils import setup_logging
from momgrap.plots import make_all_figures


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MO-MGRAP-ISAC experiment runner")
    p.add_argument("--quick", action="store_true", help="small fast run (smoke-sized)")
    p.add_argument("--seeds", type=int, default=None, help="number of random seeds")
    p.add_argument("--gens", type=int, default=None, help="generations G")
    p.add_argument("--pop", type=int, default=None, help="population size N")
    p.add_argument("--mc", type=int, default=None, help="MC samples per generation S")
    p.add_argument("--no-online", action="store_true", help="skip the expensive online reference")
    p.add_argument("--n-real-offline", type=int, default=200, help="realisations for Fig 3 CDF")
    p.add_argument("--n-real-online", type=int, default=20, help="realisations for online band")
    p.add_argument("--survival", choices=["rank1", "hv_contrib"], default="rank1")
    p.add_argument("--rmp-signal", choices=["relative", "absolute"], default=None,
                   help="RMP success signal (default relative: self-calibrating, no floor collapse)")
    p.add_argument("--eval-batch", type=int, default=None,
                   help="genotypes evaluated at once (lower if GPU OOM; default 256)")
    p.add_argument("--archive-max", type=int, default=None,
                   help="cap on stored rank-1 individuals (lower to speed up the unified-front build)")
    p.add_argument("--deltas", type=float, nargs="*", default=[0.0, 15.0, 30.0, 45.0, 60.0],
                   help="delta_phi values (deg) for the front-width sweep")
    p.add_argument("--rmp-sweep", type=float, nargs="*", default=[0.0, 0.1, 0.3, 0.5],
                   help="fixed-RMP values to sweep (the inverted-U; adaptive auto-tunes to it)")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--save", default="results/experiment.pkl")
    p.add_argument("--log-file", default=None, help="also write logs to this file")
    p.add_argument("--log-every", type=int, default=None,
                   help="log progress every N generations (0 = silent; default 25)")
    p.add_argument("--quiet", action="store_true", help="only warnings/errors")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    log = setup_logging(level=logging.WARNING if args.quiet else logging.INFO,
                        log_file=args.log_file)
    cfg = smoke_config() if args.quick else default_config()
    over = {}
    if args.gens is not None:
        over["max_gen"] = args.gens
    if args.pop is not None:
        over["pop_size"] = args.pop
    if args.mc is not None:
        over["mc_samples"] = args.mc
    if args.eval_batch is not None:
        over["eval_batch_size"] = args.eval_batch
    if args.archive_max is not None:
        over["archive_max_size"] = args.archive_max
    if args.log_every is not None:
        over["log_every"] = args.log_every
    if args.rmp_signal is not None:
        over["rmp_signal"] = args.rmp_signal
    over["survival_mode"] = args.survival
    cfg = dataclasses.replace(cfg, **over)

    n_seeds = args.seeds if args.seeds is not None else (2 if args.quick else cfg.n_seeds)
    seeds = list(range(n_seeds))

    log.info(f"Running: device={cfg.device} N={cfg.pop_size} G={cfg.max_gen} "
             f"S={cfg.mc_samples} T={cfg.T} seeds={n_seeds} survival={cfg.survival_mode}")

    data = run_all(
        cfg,
        seeds=seeds,
        do_online=not args.no_online,
        n_real_offline=args.n_real_offline,
        n_real_online=args.n_real_online,
        delta_phis=args.deltas,
        rmp_sweep_values=args.rmp_sweep,
    )

    log.info("Final HV (mean over seeds):")
    for m, hv in data.hv_gen.items():
        log.info(f"  {m:12s}: {hv.mean(axis=0)[-1]:.4g}")
    log.info("Knee-point physical values (F_com bps/Hz, F_sen dB):")
    for m, kv in data.knee.items():
        log.info(f"  {m:12s}: ({kv[0]:.3f}, {kv[1]:.2f})")
    log.info("Front width vs delta_phi:")
    for d in sorted(data.delta_sweep):
        v = data.delta_sweep[d]
        log.info(f"  delta_phi={d:5.1f} deg -> com_range={v['com_range']:.3f}  "
                 f"sen_range={v['sen_range']:.2f} dB  HV={v['hv']:.3g}")

    path = save(data, args.save)
    log.info(f"Saved experiment data -> {path}")
    figs = make_all_figures(data, args.outdir)
    log.info("Figures: " + ", ".join(figs))

    # Table I (LaTeX) — method comparison on the robust front
    from momgrap.tables import save_table1, table1_latex
    log.info("Table I (LaTeX):\n" + table1_latex(data))
    tpath = save_table1(data, "results/table1.tex")
    log.info(f"Saved Table I -> {tpath}")


if __name__ == "__main__":
    main()
