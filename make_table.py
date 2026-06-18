"""Emit Table I (LaTeX) from a saved experiment.

Reads a pickle produced by ``run_experiments.py`` (default ``results/experiment.pkl``),
computes robust-HV / knee / IGD per method, and writes a ready-to-paste IEEE LaTeX
table to ``results/table1.tex`` (also prints it).

Usage
-----
    python make_table.py
    python make_table.py --pkl results/experiment.pkl --out results/table1.tex --no-igd
"""

from __future__ import annotations

import argparse
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from momgrap.experiments import load
from momgrap.tables import save_table1, table1_latex


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Emit Table I (LaTeX)")
    p.add_argument("--pkl", default="results/experiment.pkl")
    p.add_argument("--out", default="results/table1.tex")
    p.add_argument("--no-igd", action="store_true", help="omit the optional IGD column")
    p.add_argument("--no-no-transfer", action="store_true", help="omit the optional no-transfer row")
    args = p.parse_args(argv)

    data = load(args.pkl)
    tex = table1_latex(data, with_igd=not args.no_igd, include_no_transfer=not args.no_no_transfer)
    print(tex)
    path = save_table1(data, args.out, with_igd=not args.no_igd,
                       include_no_transfer=not args.no_no_transfer)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
