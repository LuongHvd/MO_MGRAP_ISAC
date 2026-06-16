# MO-MGRAP-ISAC

GPU-accelerated **multi-objective multitask** pre-optimization for
**movable-antenna-and-RIS-aided ISAC**, implemented from scratch in PyTorch from
[`MO_MGRAP_ISAC_spec.md`](MO_MGRAP_ISAC_spec.md).

The framework evolves a **single shared genotype** (MA in-cell offsets + RIS
phases) across two orthogonal axes:

* **Multitask axis** = propagation regime (`LoS / Rayleigh / Rician`) — knowledge
  transfer happens *between* regimes.
* **Multi-objective axis** = `(communication fairness, sensing fairness)` — a
  Pareto front lives *inside* each regime.

The headline contribution is an **adaptive RMP driven by Pareto-survival of
transfer offspring** (see `rmp.py`).

---

## Install & run

```bash
pip install -r requirements.txt          # torch, numpy, matplotlib

python smoke_test.py                      # fast end-to-end sanity (CPU, seconds)
python run_experiments.py --quick         # small experiment + 3 figures
python run_experiments.py --seeds 20 --gens 300 --pop 200 --mc 32   # full paper run
```

The code is **device-agnostic**: it uses CUDA when available, otherwise CPU.
At full scale (`N=200, S=32`) one generation is ~0.7 s on CPU; a CUDA GPU is much
faster (the spec targets an RTX-class card).

Figures are written to `figures/`, raw data to `results/experiment.pkl`.

### Progress logging

Long runs print timestamped progress per generation (HV, RMP, survival rate ρ,
transfer success ratio, gen/s, ETA) plus per-step messages:

```
20:39:14 | [adaptive s1] gen   25/300  HV=312.4  RMP=0.318  rho=0.42  transfer=18/41  (4.1 gen/s, ETA  67s)
```

Flags: `--log-every N` (cadence, `0` = silent; default 25), `--log-file PATH`
(also write to file), `--quiet` (warnings only). The cadence is also exposed as
`Config.log_every` for programmatic use; `momgrap.logging_utils.setup_logging`
configures it when calling `run_all` from your own script.

---

## Module map (spec Sec 11)

| Module | Role | Spec tag |
|---|---|---|
| `config.py` | all hyperparameters, system/regime/target params | Sec 7 |
| `channels.py` | geometry, steering, per-regime channel construction, `G_geo` | Sec 2.2 / 4.1 |
| `comm.py` | BD precoding (batched SVD), EVM-aware SINR, rate | Sec 2.2 |
| `genotype.py` | soft-clip MA decode, RIS-phase decode, repulsive-force repair | Sec 2.1 / 6 |
| `operators.py` | SBX / linear-arith crossover, Gaussian mutation, multifactorial mating + **transfer tag** | Sec 5 |
| `sensing.py` | `a_eff` with RIS reflected path, beampattern `B_q` | Sec 2.3 / 4.2 |
| `objectives.py` | assemble `[F_com, F_sen]` per individual per regime, MC aggregation | Sec 2.5 |
| `nsga.py` | tensorized non-dominated sort + crowding distance | Sec 4.3 |
| `rmp.py` | **Pareto-survival** tracking (`rank1` / `hv_contrib`) + EMA RMP controller | Sec 3.2 |
| `mfea.py` | main MO-MFEA loop; per-regime NSGA-II selection | Sec 3.1 |
| `archive.py` | rank-1 archive; **unified robust front** (min over regimes) | Sec 3.4 |
| `metrics.py` | hypervolume, HV-contribution, knee point, front width | Sec 9 |
| `baselines.py` | no-transfer, fixed-RMP, pooled single-task, USPA/random, online ref | Sec 8 |
| `experiments.py` | run matrix, seeds, persistence | Sec 10/11 |
| `plots.py` | the 3 figures | Sec 10 |

Build/verification order is the spec's recommendation: single-regime evaluation
sanity → full MO + multitask (`smoke_test.py` stages A → B → C).

---

## The three aggregation layers (kept distinct — spec Sec 3.4)

1. `min over users` / `min over targets` — inside each objective (fairness).
2. `E over Omega` (mean over `S` MC samples) — within a regime (`objectives.py`).
3. `min over regimes` — only for the unified robust front (`archive.py`).

---

## Engineering decisions (where the spec left choices open)

The spec assumed most components were `[INHERITED]` from an existing MGRAP
codebase. That codebase was **not** available, so **everything here is written
from scratch**. Decisions made to keep the implementation faithful, internally
consistent and runnable:

1. **Regimes parameterised by a single Rician K-factor.** `LoS = K→∞`,
   `Rayleigh = K=0`, `Rician = finite K (6 dB default)`. One clean knob, uniform
   channel code (`config.RegimeSpec.los_nlos_weights`).

2. **NLoS uses a field-response multipath model** (`channels._direct_channel`):
   each non-specular link is a sum of `n_scatter_paths` plane waves with random
   AoD and CN(0,1) gains. This makes **MA positions matter in every regime**
   (position enters each path's steering term) instead of degenerating to a
   position-independent per-antenna fading.

3. **Reflected-sensing-path normalisation (`gamma_ris`, addresses Risk 1).**
   `a_eff = a_dir + (gamma_ris / L) · G_geoᵀ Φ a_ris`. The `1/L` keeps a
   coherently-steered RIS path comparable in magnitude to the unit-magnitude
   direct steering, so the RIS phase `θ` is a genuine tradeoff lever and the
   Pareto front does not collapse. `gamma_ris` is the documented knob to
   strengthen RIS participation. The smoke test verifies `θ` actually moves
   `F_sen`, and the `Δφ` sweep verifies front width is controlled by user–target
   angular overlap (the spec's Sec 2.4 finding).

4. **Steering convention is consistent by construction** (spec Risk 2): the comm
   `G` LoS term and the sensing `G_geo` use the *same* separable far-field
   convention `exp(+j k0 ⟨u, p⟩)`, so beampattern signs match the comm model.

5. **Both parents and offspring are re-evaluated each generation** on that
   generation's shared `Omega`, so environmental selection compares `P ∪ O`
   fairly under identical noise.

6. **Soft-clip decode** maps the unit-cube gene to `[-1,1]` then applies
   `tanh(μ··)` about the cell centre, so the cell is spanned symmetrically.

7. **Fixed HV reference point** (`config.hv_ref_*`) is identical across every
   method/run (spec Risk 5), and `F_sen` is reported in **dB** for scale balance
   with `F_com` (bps/Hz).

### Measurement & RMP refinements (post-first-run tuning)

Two changes after the first full run, both to make the results support the paper's
thesis rather than to flatter it:

8. **HV-over-generation is the cumulative-best robust HV on a FIXED eval set**
   (`metrics.RobustHVTracker`). Each generation the current front is re-evaluated
   on the same per-seed fixed environments (robust = `min over regimes`) and merged
   into a running non-dominated set, so the curve is monotone and *identically
   measured* for every method (MO-MFEA and pooled). This replaced a noisy per-gen
   HV on the training environment that made Fig 2 unreadable.

9. **The RMP success signal is RELATIVE by default** (`rmp_signal="relative"`).
   The first run exposed that the natural rank-1 survival rate of transfer
   offspring (~0.16) sits far below the fixed `rho_target=0.4`, so the absolute
   controller drove RMP straight to `RMP_min` and the adaptive method degenerated
   to near-no-transfer. The relative signal compares the rank-1 survival rate of
   **transfer** offspring against that of **intra-task crossover** offspring (the
   apples-to-apples baseline; mutation children excluded). RMP rises iff inter-task
   mating produces elites at least as often as staying in-task — self-calibrating,
   collapse-free, and it makes RMP *adapt to the problem* (e.g. it settles higher
   when targets overlap users and decays when regimes diverge under large `Δφ`).
   `rmp_signal="absolute"` reproduces the spec-literal fixed-target form.

### Regime synergy + the honest adaptive-RMP claim

10. **Shared-geometry synergy** (`shared_geometry=True`, default). All regimes in a
    generation share the same scene — user/target/scatterer *directions* and path
    losses — but each redraws its own *fading gains*. So the coarse optimum (where
    to steer the MA+RIS) is common across regimes (good geometries transfer
    immediately → synergy), while the fading-specific fine tuning diverges. Without
    this, the three regimes have largely independent optima and inter-task transfer
    is roughly neutral, so adaptive RMP has nothing to exploit.

11. **What the fixed-RMP sweep (Fig 4) actually shows — and the honest claim.**
    A 20-seed sweep shows transfer is beneficial (HV rises with RMP, worst at
    RMP→0 = no-transfer) — so multitask transfer helps. The adaptive RMP beats
    no-transfer and low fixed RMP, **but at long horizons its Pareto-survival signal
    decays the RMP and it sits *below* the best-tuned fixed RMP rather than matching
    it** (rank-1 survival under-rewards transfer once each regime converges, even
    though sustained transfer still improves the robust front). Therefore the
    **defensible claims are**: multitask **>** pooled, transfer helps, every front
    dominates USPA/random, and the RMP **adapts to the problem** (the Δφ sweep moves
    it — higher when regimes are synergistic, lower when they diverge). We do **not**
    claim adaptive strictly beats the best-tuned fixed RMP. A principled fix (define
    survival by contribution to the robust min-over-regimes front instead of
    single-regime rank-1) is the natural future step but needs full-scale validation.

### Diagnostic protocol — does adaptive RMP actually help?

Because adaptive RMP showed no clear win over fixed on the standard fading regimes,
a **pre-registered diagnostic** (run via `run_diagnostic.py`, implementing
`Diagnostic_Protocol.md`) tests *whether there is any task conflict for adaptation
to exploit* before judging the mechanism. Key pieces:

* **`delta_task` knob** (`Config.delta_task_deg`): task `t` serves a sector centred
  at `t·delta_task`. `0` = all tasks share the sector (synergy anchor, reproduces
  the standard behaviour); large = sectors separate, so a single frozen RIS phase
  `θ` cannot serve all tasks → durable conflict. (Distinct from `delta_phi`, the
  within-task front-width knob, held fixed across the sweep.) Secondary amplifier:
  `direct_atten_db` (RIS-reliance).
* **Paired evaluation** (`Config.paired_envs`): per-generation MC snapshots are
  seeded by `(seed, generation)` only — independent of the evolution RNG — so every
  method at a seed sees identical environments and HV is compared *per-seed* with a
  tight paired CI (fixes the overlapping-band problem in the plain HV-vs-gen plot).
* **Oracle + decision tree** (`momgrap/diagnostic.py`): at each `delta_task` it finds
  the best fixed RMP (`rmp*`), then applies the protocol §7 gate (does `rmp*` shift
  with conflict, with a significant penalty?) and maps to **Outcome A** (adaptive
  tracks oracle and beats naïve fixed → robustness-mechanism paper), **B** (matches
  best single fixed without tuning), or **C** (no help → drop adaptive). The gate
  watches the *oracle* (method-independent), so knob-turning cannot p-hack the
  method into winning.
* **D-Fig 1/2/3**: HV-vs-`delta_task` (adaptive/oracle/naïve), `rmp*`-vs-`delta_task`
  with adaptive's converged RMP overlaid, and the paired `adaptive−fixed` difference
  with CI. Run: `python run_diagnostic.py --quick` (plumbing) or full on GPU.

### Effect sizes, noise, and what to actually run

Be realistic about magnitudes (these are stochastic EC results):

* **Robust at any scale:** multitask (adaptive/fixed) **> pooled** (~7 HV, consistent),
  and every optimized front dominates USPA/random. These need no large seed count.
* **Real but small (~1–2 HV):** transfer **>** no-transfer, and the fixed-RMP
  inverted-U with adaptive near its top. At a few seeds / small `N,G` these sit
  **inside the seed-to-seed noise** and can even flip. They only resolve with tight
  error bars at the **full scale** (`--seeds 20 --gens 300 --pop 200 --mc 32`).

So the headline plots (Fig 2 separation, Fig 4 inverted-U) should be generated from
a full-scale run; the moderate/quick runs are for plumbing and sanity, not for the
final claims. Honest framing of the adaptive contribution: *auto-tunes to the RMP
sweet spot without tuning, beats no-transfer and mis-set fixed RMP* — not *strictly
dominates the single best-tuned fixed value*.

### What is genuinely new (small, as the spec intended)
`sensing.py` (RIS path), `rmp.py` (Pareto-survival success signal), the tensorized
non-dominated sort in `nsga.py`, and the unified-front archive in `archive.py`.

---

## Baselines → figures

* **Fig 1** — unified robust front (`F_com` vs `F_sen`): adaptive · no-transfer ·
  fixed-RMP · USPA/random markers; inset = front width vs `Δφ`.
* **Fig 2** — HV over generations: adaptive · fixed-RMP · pooled-single-task;
  inset = RMP-vs-generation (mean ± std).
* **Fig 3** — HV-CDF over realisations of the offline unified front, with the
  online-reference HV overlaid as a summary band.
* **Fig 4** — HV vs fixed RMP (the inverted-U) with adaptive drawn as a horizontal
  line: evidence that adaptive auto-tunes to the RMP sweet spot. Produced when
  `--rmp-sweep` values are given.

Configuration covers the baselines: `fixed_rmp` set → fixed-RMP MO-MFEA;
`allow_transfer=False` → no-transfer; `run_pooled_single_task` → pooled strawman.

---

## Out of scope (spec Sec 13)
CVaR objectives, scaling-vs-`M` study, per-task-pair RMP, CRB sensing metric and
dedicated sensing beamforming/power-split are intentionally **not** implemented.
