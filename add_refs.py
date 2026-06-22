"""Augment an existing experiment pickle with the extra reference curves:

  * Fig 3: HV-over-realisation CDFs for the fixed **USPA** and **random** layouts
           (cheap — single configs over the same realisations as the offline curve).
  * Fig 1: the **real-time** reachable envelope (union of per-realisation online
           fronts). This recomputes the online reference to obtain its fronts
           (the per-realisation optimisations) — the only non-cheap part.

Re-plotting alone can't add these (they were never stored), so this computes and
injects them into the pickle, then you re-plot. Budget must match the original run;
``default_config()`` already uses N=200, G=300, S=32.

Usage
-----
    python add_refs.py --pkl results/experiment.pkl --eval-batch 128
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
from momgrap.experiments import _front_to_np, load, save
from momgrap.logging_utils import setup_logging


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Add USPA/random (Fig 3) + real-time (Fig 1) references")
    p.add_argument("--pkl", default="results/experiment.pkl")
    p.add_argument("--gens", type=int, default=None)
    p.add_argument("--pop", type=int, default=None)
    p.add_argument("--mc", type=int, default=None)
    p.add_argument("--eval-batch", type=int, default=None)
    p.add_argument("--n-real-online", type=int, default=None, help="realisations for the real-time envelope")
    p.add_argument("--no-realtime", action="store_true", help="skip the (costly) real-time front")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    log = setup_logging(level=logging.WARNING if args.quiet else logging.INFO)
    data = load(args.pkl)
    s0 = list(data.seeds)[0]

    cfg = default_config()
    over = {}
    for k, v in (("max_gen", args.gens), ("pop_size", args.pop), ("mc_samples", args.mc),
                 ("eval_batch_size", args.eval_batch)):
        if v is not None:
            over[k] = v
    cfg = dataclasses.replace(cfg, **over)

    n_real_off = int(np.asarray(data.fig3["offline_hv"]).size) or 200
    n_real_on = args.n_real_online or (int(np.asarray(data.fig3.get("online_hv", [])).size) or 25)
    log.info(f"Augmenting {args.pkl}: seed0={s0} N={cfg.pop_size} G={cfg.max_gen} S={cfg.mc_samples} "
             f"offline_real={n_real_off} online_real={n_real_on}")

    # ---- Fig 3: USPA + random fixed-layout HV-over-realisations (cheap) ----
    ref_hv = B.ref_layout_hv_over_realisations(cfg, s0, n_real_off)
    data.fig3["uspa_hv"] = ref_hv["uspa_hv"]
    data.fig3["random_hv"] = ref_hv["random_hv"]
    log.info(f"USPA HV median={np.median(ref_hv['uspa_hv']):.4g}  "
             f"random HV median={np.median(ref_hv['random_hv']):.4g}")

    # ---- Fig 1: real-time reachable envelope (recompute online to get fronts) ----
    if not args.no_realtime:
        online_hv, realtime_front = B.online_reference_hv(cfg, s0, n_real_on, collect_fronts=True)
        data.fig3["online_hv"] = online_hv.cpu().numpy()
        data.reference["realtime_front"] = _front_to_np(realtime_front)
        log.info(f"real-time envelope: {data.reference['realtime_front'].shape[0]} points")

    save(data, args.pkl)
    log.info(f"Saved -> {args.pkl}.  Now: python make_paper_figures.py --pkl {args.pkl} --outdir figures_paper")


if __name__ == "__main__":
    main()
