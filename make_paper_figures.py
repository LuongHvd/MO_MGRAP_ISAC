"""Generate the three paper figures (framing C) from a saved experiment.

  Fig 1 — unified robust front of the proposed framework vs USPA/random + Δφ inset
  Fig 2 — HV over generations: multitask (adaptive) vs pooled single-task
  Fig 3 — HV-CDF over realisations: offline unified front vs online reference

These read a pickle produced by ``run_experiments.py`` (default
``results/experiment.pkl``). No re-optimisation — just re-plots, in seconds.

Usage
-----
    python make_paper_figures.py                                  # results/experiment.pkl -> figures_paper/
    python make_paper_figures.py --pkl results/experiment.pkl --outdir figures_paper

If you do not have ``results/experiment.pkl`` yet, produce it first with::

    python run_experiments.py --seeds 20 --gens 300 --pop 200 --mc 32 \
        --eval-batch 128 --n-real-online 25 --log-every 25 --log-file results/run.log
"""

from __future__ import annotations

import argparse
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from momgrap.experiments import load
from momgrap.plots import fig1_unified_front, fig2_hv_over_gen, fig3_hv_cdf


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Generate the 3 paper figures (framing C)")
    p.add_argument("--pkl", default="results/experiment.pkl", help="experiment data pickle")
    p.add_argument("--outdir", default="figures_paper", help="output directory")
    args = p.parse_args(argv)

    data = load(args.pkl)
    paths = [
        fig1_unified_front(data, args.outdir),   # money figure: front vs references
        fig2_hv_over_gen(data, args.outdir),     # multitask >> pooled (+ RMP inset)
        fig3_hv_cdf(data, args.outdir),          # robustness offline vs online
    ]
    print("Paper figures written:")
    for x in paths:
        print(f"  {x}")


if __name__ == "__main__":
    main()
