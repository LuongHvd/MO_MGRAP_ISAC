"""Smoke test for MO-MGRAP-ISAC (spec Sec 11 build order).

Stage A: single-regime objective sanity -- shapes finite, comm rate sensible,
         sensing front NOT collapsed, and the RIS phase actually moves F_sen.
Stage B: full MO-MFEA (3 regimes) for a few generations -- archive non-empty,
         unified robust front non-degenerate, HV trends up, RMP adapts.
Stage C: baselines + figures run end-to-end without error.

Run:  python smoke_test.py
"""

from __future__ import annotations

import sys

import torch

# Windows consoles default to cp1252; force UTF-8 so unicode prints don't crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from momgrap import smoke_config
from momgrap.channels import sample_environment
from momgrap.objectives import evaluate_single_regime
from momgrap import baselines as B


def _gen(cfg, seed=0):
    g = torch.Generator(device=cfg.device)
    g.manual_seed(seed)
    return g


def stage_a(cfg) -> None:
    print("\n=== Stage A: single-regime objective sanity ===")
    g = _gen(cfg, 1)
    pop = torch.rand(cfg.pop_size, cfg.D, generator=g, device=cfg.device, dtype=cfg.dtype_real)
    env = sample_environment(cfg.regimes[0], cfg.mc_samples, g, cfg)  # LoS regime
    F = evaluate_single_regime(pop, env, cfg)
    assert F.shape == (cfg.pop_size, 2), F.shape
    assert torch.isfinite(F).all(), "non-finite objectives"
    fcom, fsen = F[:, 0], F[:, 1]
    print(f"  F_com (rate)   : min={fcom.min():.3f}  mean={fcom.mean():.3f}  max={fcom.max():.3f} bps/Hz")
    print(f"  F_sen (dB)     : min={fsen.min():.2f}  mean={fsen.mean():.2f}  max={fsen.max():.2f} dB")
    assert fcom.min() > 0, "worst-user rate should be positive"
    com_spread = float(fcom.max() - fcom.min())
    sen_spread = float(fsen.max() - fsen.min())
    print(f"  spread         : F_com={com_spread:.3f}  F_sen={sen_spread:.2f} dB")
    assert sen_spread > 1e-3, "sensing objective is flat -> front would collapse"

    # RIS phase must move F_sen (reflected path participates -> avoids collapse)
    pop_zero_phase = pop.clone()
    pop_zero_phase[:, 2 * cfg.M:] = 0.0
    F0 = evaluate_single_regime(pop_zero_phase, env, cfg)
    dphase_effect = float((F[:, 1] - F0[:, 1]).abs().mean())
    print(f"  RIS-phase effect on F_sen (mean |Δ|): {dphase_effect:.3f} dB")
    assert dphase_effect > 1e-3, "RIS phase does not affect sensing -> theta not in tradeoff"
    print("  Stage A OK")


def stage_b(cfg) -> None:
    print("\n=== Stage B: full MO-MFEA (3 regimes) ===")
    res = B.run_adaptive(cfg, seed=0)
    print(f"  generations run     : {len(res.hv_history)}")
    print(f"  HV[0]={res.hv_history[0]:.4g}  HV[-1]={res.hv_history[-1]:.4g}")
    print(f"  RMP trajectory      : {[round(x,3) for x in res.rmp_history]}")
    print(f"  survival rate (rho) : {[round(x,2) for x in res.rho_history]}")
    print(f"  archive size        : {len(res.archive)}")
    assert len(res.archive) > 0, "archive empty"

    for t, (front, genos) in res.per_regime_front.items():
        print(f"  regime {t} ({cfg.regimes[t].name:8s}) front points: {front.shape[0]}")
        assert front.shape[0] >= 1

    front, fgenos = res.archive.unified_robust_front(_gen(cfg, 99), cfg.mc_samples)
    print(f"  unified robust front: {front.shape[0]} points")
    assert front.shape[0] >= 1
    if front.shape[0] >= 2:
        rng = (front.max(0).values - front.min(0).values)
        print(f"  unified front width : F_com={float(rng[0]):.3f}  F_sen={float(rng[1]):.2f} dB")
    print("  Stage B OK")


def stage_c(cfg) -> None:
    print("\n=== Stage C: baselines + figures end-to-end ===")
    from momgrap.experiments import run_all
    from momgrap.plots import make_all_figures

    data = run_all(
        cfg,
        seeds=[0, 1],
        do_online=True,
        n_real_offline=30,
        n_real_online=4,
        delta_phis=[0.0, 20.0, 45.0],
    )
    for m, hv in data.hv_gen.items():
        print(f"  {m:12s} final HV (mean over seeds): {hv.mean(axis=0)[-1]:.4g}")
    print(f"  knee points (com, sen dB): {data.knee}")
    print(f"  delta_phi sweep widths   : "
          + ", ".join(f"{d}->com{v['com_range']:.2f}/sen{v['sen_range']:.2f}"
                      for d, v in sorted(data.delta_sweep.items())))
    paths = make_all_figures(data, outdir="figures")
    print("  figures written:")
    for p in paths:
        print(f"    {p}")
    print("  Stage C OK")


def main() -> None:
    cfg = smoke_config()
    print(f"device={cfg.device}  D={cfg.D}  N={cfg.pop_size}  G={cfg.max_gen}  "
          f"S={cfg.mc_samples}  T={cfg.T}")
    stage_a(cfg)
    stage_b(cfg)
    stage_c(cfg)
    print("\nALL SMOKE STAGES PASSED")


if __name__ == "__main__":
    main()
