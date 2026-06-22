"""Per-seed breakdown of the run (for inspection / understanding variance).

Reads an experiment pickle and reports the final robust HV of every method for
EACH seed, plus per-seed paired differences vs the proposed method (RAMP) and a
summary (mean / std / min / max). Writes a CSV and prints a readable table.

This is a transparency/diagnostic tool: it shows ALL seeds. The paper must report
all seeds (mean +/- std and the paired CI) — not a hand-picked subset.

Usage
-----
    python per_seed_report.py --pkl results/experiment.pkl --out results/per_seed_hv.csv
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from momgrap.experiments import load

# data key -> short column label, in a sensible order
_COLS = [
    ("adaptive", "RAMP"),
    ("fixed_rmp", "fixedRMP"),
    ("no_transfer", "no_transf"),
    ("mo_de", "MO-DE"),
    ("mopso", "MOPSO"),
    ("pooled", "pooled"),
]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Per-seed final-HV breakdown")
    p.add_argument("--pkl", default="results/experiment.pkl")
    p.add_argument("--out", default="results/per_seed_hv.csv")
    p.add_argument("--proposed", default="adaptive")
    args = p.parse_args(argv)

    data = load(args.pkl)
    seeds = list(data.seeds)
    cols = [(k, lbl) for k, lbl in _COLS if k in data.hv_gen and np.asarray(data.hv_gen[k]).size]
    # final robust HV per seed for each present method
    finals = {k: np.asarray(data.hv_gen[k])[:, -1] for k, _ in cols}
    n = len(seeds)

    # ---------- console table ----------
    labels = [lbl for _, lbl in cols]
    header = "seed  | " + " | ".join(f"{l:>9s}" for l in labels)
    print(header)
    print("-" * len(header))
    for i, sd in enumerate(seeds):
        row = " | ".join(f"{finals[k][i]:9.2f}" for k, _ in cols)
        print(f"{sd:>4d}  | {row}")
    print("-" * len(header))
    for stat, fn in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)):
        row = " | ".join(f"{fn(finals[k]):9.2f}" for k, _ in cols)
        print(f"{stat:>4s}  | {row}")

    # ---------- per-seed paired diff vs proposed ----------
    if args.proposed in finals:
        base = finals[args.proposed]
        others = [(k, lbl) for k, lbl in cols if k != args.proposed]
        print("\nPer-seed paired difference (RAMP - method)  [>0 = RAMP better that seed]:")
        head2 = "seed  | " + " | ".join(f"{lbl:>9s}" for _, lbl in others)
        print(head2)
        print("-" * len(head2))
        for i, sd in enumerate(seeds):
            row = " | ".join(f"{base[i] - finals[k][i]:9.2f}" for k, _ in others)
            print(f"{sd:>4d}  | {row}")
        print("-" * len(head2))
        winrow = " | ".join(f"{int((base - finals[k] > 0).sum()):>4d}/{n:<4d}" for k, _ in others)
        print(f"win   | {winrow}    (# seeds RAMP > method)")

    # ---------- CSV ----------
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed"] + labels)
        for i, sd in enumerate(seeds):
            w.writerow([sd] + [f"{finals[k][i]:.4f}" for k, _ in cols])
        w.writerow(["mean"] + [f"{np.mean(finals[k]):.4f}" for k, _ in cols])
        w.writerow(["std"] + [f"{np.std(finals[k]):.4f}" for k, _ in cols])
    print(f"\nSaved CSV -> {args.out}")
    print("Note: report ALL seeds in the paper (mean +/- std + paired CI), not a subset.")


if __name__ == "__main__":
    main()
