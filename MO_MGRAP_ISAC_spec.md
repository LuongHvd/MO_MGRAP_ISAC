# Implementation Spec — MO-MGRAP-ISAC
### GPU-Accelerated Multi-Objective Multitask Pre-Optimization for Movable-Antenna-and-RIS-aided ISAC

> **Purpose of this document.** This is an engineering spec to be handed to a coding agent (Claude Code). It defines the system model, the optimization algorithm, the GPU evaluation pipeline, the experiment matrix, and a suggested module layout. It is written so that an implementer who has the existing **MGRAP** conference codebase (the GECCO '26 paper) can reuse most of it and add only what is new.
>
> **Read this first — what is the paper.** We extend the conference MGRAP framework (single-task-axis, scalar utilities, evolves *one* robust MA+RIS configuration) into a **multi-objective multitask** framework for ISAC. The headline contribution is an **adaptive RMP driven by Pareto-survival of transfer offspring**. The target venue is a **4-page letter** (WCL/COMML style), so the spec is scoped tightly: one model addition (RIS-in-sensing), one algorithmic novelty (Pareto-RMP), one new GPU kernel (tensorized non-dominated sort), and a 3-figure experiment plan.

---

## 0. Provenance tags — read these before coding

Every component below is tagged so you know how much to write:

- **[INHERITED]** — exists in the MGRAP codebase. Port it; do not redesign. If the MGRAP code is available, reuse the functions directly.
- **[CHANGED]** — an MGRAP component that must be modified for the multi-objective setting.
- **[NEW]** — does not exist in MGRAP; implement from this spec.

The whole point of the spec is that **[INHERITED]** dominates. The genuinely new code is small: the sensing path, the Pareto-survival RMP logic, the tensorized non-dominated sort, and the unified-front archive.

---

## 1. Two orthogonal axes (the core design decision)

The framework optimizes a **single shared genotype** across two independent axes. Keep these conceptually separate everywhere in the code — they are the spine of the paper.

- **Multitask axis = propagation regime.** `T = 3` regimes: `LoS`, `Rayleigh`, `Rician`. Each regime is a "task." Knowledge transfer happens *between regimes*.
- **Multi-objective axis = (communication fairness, sensing fairness).** Each task is a **2-objective** problem. The Pareto front lives *inside each regime*.

So a task `t` is: `maximize [ F_com^(t)(x), F_sen^(t)(x) ]`, where the only difference between tasks is the channel model used to draw environment realizations. This separation is what makes the extension non-trivial (transfer on one axis, Pareto on the other) rather than a relabeling of the conference paper.

> **Naming collision warning.** The conference paper used `Q` for Monte-Carlo realizations. The ISAC draft uses `Q` for sensing targets. In this spec:
> - **`Q`** = number of sensing targets (directions `φ_q`).
> - **`S`** = number of Monte-Carlo environment realizations per generation (the conference paper's `Q`).
> Do not conflate them.

---

## 2. System model

### 2.1 Geometry and genotype  **[INHERITED]**

- BS with `M = 16` movable antennas. Planar motion (`x_m = 0`); each antenna moves inside a rectangular cell with extents `(L_h, L_v)`, centered at grid center `(c_{y,m}, c_{z,m})`.
- RIS with `L = 64` elements at a fixed position `p_ris`, unit-modulus phase shifts `θ_l ∈ [0, 2π)`.
- `K = 4` single-antenna users, uniform in a sector, radius `50–200 m`.
- `Q = 3` sensing targets at directions `φ_q` (**NEW — see 2.4 for placement, it is a first-class design knob**).
- Carrier `28 GHz`. Min antenna spacing `d_min = λ/2`.

Genotype `x ∈ [0,1]^D`, `D = 2M + L`:
- First `2M` entries → MA in-cell offsets `(Δy_m, Δz_m)`, decoded by soft-clip:
  `y_m = c_{y,m} + (L_h/2)·tanh(μ·Δy_m)`, `z_m = c_{z,m} + (L_v/2)·tanh(μ·Δz_m)`.
- Last `L` entries → normalized phases, decoded `θ_l = 2π·(θ̂_l mod 1)`.

Decode + repair (Sec 6) are integrated into the genotype→phenotype map, so the search only ever sees feasible layouts.

### 2.2 Communication channel + precoding + rate  **[INHERITED]**

Port directly from MGRAP. Summary for reference:

- Aggregated channel for user `k`: `h_k^H(P,Φ) = d_k^H(P) + g_k^H Φ G(P)`, with `d_k ∈ C^{1×M}` (BS→UE), `G ∈ C^{L×M}` (BS→RIS), `g_k ∈ C^{1×L}` (RIS→UE). All depend on the random environment `Ω`.
- BD precoding `w_k(P,Φ)` via null-space projection (batched SVD), unit-norm. Uniform power `P_k = P_max/K`.
- EVM impairment model, `κ = 0.05`. EVM-aware SINR:
  `SINR_k = (1−κ²)P_k|h_k^H w_k|² / ( σ² + κ² Σ_m |h_{k,m}|² E[|x_m|²] )`, with `E[|x_m|²] = Σ_k P_k |w_{k,m}|²`.
- Rate `R_k = log2(1 + SINR_k)`.

### 2.3 Sensing model with RIS path  **[NEW — most important model change]**

This is the only model addition, and it is load-bearing: **without the RIS reflected path, the RIS phase does not participate in the communication–sensing tradeoff and the Pareto front collapses.** The reflected path puts `θ` into the tradeoff.

Define the **effective transmit array response toward target `q`** as the sum of a direct BS→target path and a BS→RIS→target reflected path:

```
a_eff(P, θ; φ_q) = a_dir(P, φ_q) + G_geo(P)^T · Φ · a_ris(φ_q)        ∈ C^{M×1}
```

- `a_dir(P, φ_q)`: direct array steering, entry `m` = `exp( j (2π/λ) u^T(φ_q) p_m )`. Depends on MA positions. **(geometric, deterministic)**
- `a_ris(φ_q)`: RIS array response toward target direction `φ_q`, `∈ C^{L×1}`. **(geometric, deterministic)**
- `G_geo(P)`: the **geometric (LoS)** BS→RIS array response, `∈ C^{L×M}`. Use the deterministic BS→RIS geometry here, **not** the faded `G(P;Ω)` from the comm model. Rationale: BS↔RIS is a fixed infrastructure link, typically LoS-dominated; this keeps the reflected sensing path deterministic so the optimizer can steer it, and removes spurious `Ω`-noise from the sensing geometry. *(If you prefer to reuse the faded `G`, that is a documented alternative — but the geometric version is cleaner for the letter.)*
- `Φ = diag(e^{jθ_1}, …, e^{jθ_L})`.

> **Convention check — do this against the existing comm code.** The conjugate/transpose convention for `G` and the steering vectors must match how MGRAP defines `G` (BS→RIS, `C^{L×M}`) and the array manifold. The expression above assumes `G` maps the `M` antenna signals to the `L` RIS elements, so `G_geo^T` is `M×L` and `a_ris` is `L×1` → `M×1`. Verify the sign of the phase exponents matches the comm model before trusting beampattern numbers.

Transmit covariance (rides on the comm beamformers — there is **no** dedicated sensing beam and **no** power split, by design):

```
R_s(x; Ω) = Σ_k P_k · w_k(x; Ω) w_k^H(x; Ω)        ∈ C^{M×M}
```

Beampattern gain toward target `q`:

```
B_q(x; Ω) = a_eff^H(P,θ;φ_q) · R_s(x;Ω) · a_eff(P,θ;φ_q)        (scalar, ≥ 0; report in dB)
```

Randomness in `B_q` enters **only** through `R_s` (i.e. through the comm beamformers, which depend on `Ω`). The geometry (`a_dir`, `a_ris`, `G_geo`) is deterministic. This is the intended structure: *sensing rides on communication beamforming + RIS geometry.*

### 2.4 Target placement — a first-class knob, not a detail  **[NEW]**

Because geometry is the *only* tradeoff lever, the existence and width of the Pareto front depend on the **angular overlap between users and targets**:
- Targets in the **same angular sector** as users → RIS/array cannot illuminate both → **wide, meaningful front**.
- Targets in an **empty sector** → both served easily → **front collapses to a point**.

Therefore expose target directions as a configurable parameter with a **separation control** `Δφ` (angular offset of the target cluster relative to the user sector centroid). The sweep over `Δφ` produces a reported finding (Sec 9): *front width vs angular separation*. Default for the main figures: choose `Δφ` so users and targets **partially overlap** (front visibly non-degenerate).

### 2.5 Objectives (per task)  **[CHANGED — now a 2-vector]**

Both maximized:

```
F_com(x) = E_Ω[ min_k  R_k(x; Ω) ]      # expected worst-user rate  (user fairness)
F_sen(x) = E_Ω[ min_q  B_q(x; Ω) ]      # expected worst-target illumination (target fairness)
```

Symmetric "worst-case fairness" on both axes — keep this symmetry, it is the paper's clean story (fairness across users ↔ fairness across targets).

Monte-Carlo approximation with a shared per-regime batch `Ω^(t)_{1:S}`:

```
F̂_com^(t)(x) = (1/S) Σ_s min_k R_k(x; Ω^(t)_s)
F̂_sen^(t)(x) = (1/S) Σ_s min_q B_q(x; Ω^(t)_s)
```

Each task `t`: `max_x [ F̂_com^(t)(x), F̂_sen^(t)(x) ]`. Tasks differ only in the channel model behind `Ω^(t)`.

---

## 3. Algorithm — MO-MFEA with Pareto-transfer RMP

The orchestration generalizes MGRAP's Algorithm 1. **Three things change vs the conference version; everything else is ported.**

| Block | Status | Note |
|---|---|---|
| Unified genotype `[0,1]^D`, decode, init | [INHERITED] | identical |
| Repair (repulsive force + boundary clip) | [INHERITED] | identical |
| Multifactorial mating (SBX + linear-arithmetic xover, Gaussian mutation, skill factors) | [INHERITED] | identical operators; **add transfer-offspring tagging** |
| Fitness = `[F_com, F_sen]` under the offspring's regime | [CHANGED] | vector, not scalar; evaluate each individual only on its skill-factor regime |
| **Environmental selection** | [CHANGED] | per-regime NSGA-II (non-dom sort + crowding) replaces "keep N/3 best per task" |
| **Skill-factor assignment** | [CHANGED] | regime in which the individual attains its best (lowest) non-domination rank |
| **Adaptive RMP** | [CHANGED→NEW logic] | EMA form is inherited; the *success signal* is redefined by Pareto-survival |
| Unified robust front + archive | [NEW] | the deployable product |

### 3.1 Per-generation loop

```
Input: PopSize N, MaxGen G, regimes {0,1,2},
       RMP params (η, ρ_target, RMP_min, RMP_max, β_EMA),
       MC samples S, targets {φ_q}
Init:  Population P ~ U([0,1]^D); skill factors τ assigned round-robin / by init eval; RMP ← 0.3
Repair P.

for gen = 1..G:
    # 1. Shared MC snapshots, one batch PER regime (regimes have different channel models)
    for t in regimes: draw Ω^(t)_{1:S}

    # 2. Offspring (multifactorial mating) — INHERITED operators
    O ← ∅
    for i = 1..N/2:
        pick parents p_a, p_b
        if τ_a == τ_b or rand() < RMP:
            c1, c2 ← Crossover(p_a, p_b)            # SBX or linear-arithmetic
            τ_c1, τ_c2 ← random from {τ_a, τ_b}
            if τ_a != τ_b: tag c1, c2 as TRANSFER offspring   # <-- NEW tag
        else:
            c1 ← Mutate(p_a); c2 ← Mutate(p_b)      # Gaussian σ=0.1
            τ_c1 ← τ_a; τ_c2 ← τ_b
        O ← O ∪ {Repair(c1), Repair(c2)}

    # 3. Fitness — CHANGED to vector; evaluate each c only on regime τ_c
    for c in O:  f(c) = [F̂_com^(τ_c)(c), F̂_sen^(τ_c)(c)]   # via GPU pipeline, Sec 4

    # 4. Environmental selection — CHANGED: per-regime NSGA-II
    for t in regimes:
        pool_t = {individuals in P∪O with τ == t}
        fronts  = fast_nondominated_sort(pool_t.objectives)   # NEW tensorized kernel
        crowd   = crowding_distance(fronts)
        P_next_t = select top (N/T) by (rank, crowding)
    P_next = ∪_t P_next_t

    # 5. Adaptive RMP — NEW success signal (Pareto-survival)
    survivors = TRANSFER offspring that landed in rank-1 of their regime in P_next   # default
    s_rate = |survivors| / max(1, |TRANSFER offspring|)
    ρ_EMA ← (1-β_EMA)·ρ_EMA + β_EMA·s_rate
    RMP   ← clip(RMP + η·(ρ_EMA − ρ_target), RMP_min, RMP_max)

    # 6. Update archive (Sec 3.4)
    archive.update(rank-1 individuals per regime)
    P ← P_next

return per-regime fronts, archive
```

### 3.2 Pareto-survival — the headline definition  **[NEW]**

In the scalar MGRAP, a transfer offspring "survives" if it beats the selection cutoff — unambiguous. In multi-objective there is no single cutoff, and **defining survival is exactly the contribution.** Implement both; default to the first.

- **Rank-1 survival (default, cheap, robust).** A transfer offspring is *successful* if, after environmental selection in its assigned regime, it sits in **front rank-1** (the non-dominated front) of that regime. `s_rate = #successful / #transfer_offspring`.
- **Hypervolume-contribution survival (refined option).** A transfer offspring is *successful* if it makes a **positive hypervolume contribution** to its regime's rank-1 front (it expands the front). Stronger signal; needs a per-offspring HV-contribution computation. Gate behind a config flag.

Interpretation to put in the paper:
- `RMP ↑` when transfer offspring keep landing on each other's fronts → regimes are **synergistic** (a geometry good for the comm–sensing tradeoff under LoS is also good under Rician).
- `RMP ↓` when they get dominated → regimes **diverge**; suppress inter-task mating to avoid negative transfer.

Use a **single global RMP** (as in MGRAP). Per-task-pair RMP is a possible extension but is scope creep for a 4-page letter — note it, don't build it.

### 3.3 Skill-factor assignment  **[CHANGED]**

An individual's skill factor = the regime in which it attains its **best (lowest) non-domination rank**, tie-broken by crowding. New offspring keep the skill factor assigned at birth for that generation's evaluation; reassignment (if used) happens at selection time. Keep it simple: birth-assigned skill factor + per-regime pools is sufficient and matches the loop above.

### 3.4 Two-tier output + archive  **[NEW]**

Maintain an **external archive** of all rank-1 individuals across regimes and generations (genotypes + which regime they were rank-1 in). Two products:

1. **Per-regime fronts** (evidence the multitask machinery works): the final rank-1 set within each regime. Compare against independent NSGA-II per regime.
2. **Unified robust front** (the deployable product): take all archived configs, **re-evaluate each on all `T` regimes**, aggregate cross-regime by **worst-regime (min over regimes)**:
   ```
   F_com^robust(x) = min_t  F̂_com^(t)(x)
   F_sen^robust(x) = min_t  F̂_sen^(t)(x)
   ```
   then take the non-dominated set of `{(F_com^robust, F_sen^robust)}`. This is the one-time offline design; online operation only picks an operating point on this front (zero re-optimization).

> **Three aggregation layers — keep them distinct in code:**
> 1. `min over users` / `min over targets` — inside each objective (fairness).
> 2. `E over Ω` (mean over `S` MC samples) — within a regime.
> 3. `min over regimes` — only for the unified robust front.
> We keep `E[·]` for layer 2 (CVaR is future work). Layer 3 is `min` (worst-regime guarantee).

---

## 4. GPU evaluation pipeline

Population-level, GPU-native, batched over individuals `N` and MC samples `S`. Almost entirely inherited; one new kernel.

### 4.1 Inherited tensor pipeline  **[INHERITED]**

Port from MGRAP. Shapes for reference:
- `P ∈ R^{N×M×3}` (MA coords after decode+repair), user locations `U ∈ R^{S×K×3}`.
- Distance tensor `D ∈ R^{N×S×K×M}` by broadcasting; path-loss + phase → direct channels `d^H ∈ C^{N×S×K×M}`, BS–RIS `G ∈ C^{N×S×L×M}`.
- Aggregated channel `H_{n,s} ∈ C^{K×M}` per (n,s).
- Batched BD precoding via `torch.linalg.svd` → `w_{n,s,k} ∈ C^{M×1}`.
- EVM-aware SINR/rate → `R_{n,s,k}` (real, `∈ R^{N×S×K}`).

### 4.2 Sensing tensors  **[NEW]**

- `a_dir ∈ C^{N×Q×M}` from MA coords + target directions.
- `a_ris ∈ C^{N×Q×M}` `= einsum(G_geo^T, Φ, a_ris_steer)`; `G_geo ∈ C^{N×L×M}` (geometric, depends on MA coords; **no S axis** — deterministic), `Φ ∈ C^{N×L×L}` diag, `a_ris_steer ∈ C^{Q×L}`.
- `a_eff = a_dir + a_ris ∈ C^{N×Q×M}`.
- `R_s ∈ C^{N×S×M×M}` `= Σ_k P_k w_{n,s,k} w_{n,s,k}^H`.
- Beampattern `B_{n,s,q} = a_eff_{n,q}^H R_s_{n,s} a_eff_{n,q}` (real `≥ 0`), `∈ R^{N×S×Q}`. Implement with batched `einsum`/`matmul`; broadcast `a_eff` over the `S` axis.
- Objective vectors: `F_com_n = mean_s min_k R_{n,s,k}`, `F_sen_n = mean_s min_q B_{n,s,q}` (convert `F_sen` to dB for plotting/scale balance). Output `F ∈ R^{N×2}`.

### 4.3 Tensorized non-dominated sort  **[NEW — the new EC kernel]**

Operate on a per-regime objective matrix `F ∈ R^{P_t × 2}` (`P_t` = pool size, a few hundred). Cite the tensorization line (Liang et al., "Bridging EMO and GPU acceleration via tensorization") as the basis.

- Pairwise dominance: boolean tensor `Dom ∈ {0,1}^{P_t×P_t}`, `Dom[i,j] = i dominates j` (both objectives ≥, at least one >). Fully vectorized for 2 objectives.
- Domination counts `n_dom[j] = Σ_i Dom[i,j]`; peel fronts iteratively (front-1 = `n_dom==0`, remove, recompute) — a handful of vectorized iterations.
- **Crowding distance** per front: sort by each objective, sum normalized neighbor gaps, boundary points → `∞`. Vectorizable per front.

For `P_t ~ few hundred`, the `P_t×P_t` tensor is trivial on the RTX-class GPU. Keep it simple; do not over-optimize.

---

## 5. Operators  **[INHERITED]**

Port from MGRAP:
- **Crossover:** Simulated Binary Crossover (SBX) and Linear-Arithmetic Crossover `c1 = α∘p_a + (1−α)∘p_b`, `c2 = α∘p_b + (1−α)∘p_a`, `α ~ U(0,1)^D` element-wise. Offspring skill factors random from `{τ_a, τ_b}`.
- **Mutation:** Gaussian, `c = p + σ·ε`, `ε ~ N(0, I_D)`, `σ = 0.1`. Skill factor inherited.

The only addition is the **TRANSFER tag** (Sec 3.1 step 2) on offspring produced by inter-task crossover.

---

## 6. Repair  **[INHERITED]**

Port from MGRAP. Integrated into genotype→phenotype map:
- **Repulsive-force projection** for `d_min` violations: `p_m^new = p_m^old + ½(d_min − d_mn)·(p_m − p_n)/d_mn`, symmetric on `p_n`; iterate `I = 3`.
- **Boundary clip** onto the feasible BS panel cell `C_m`.

---

## 7. Configuration / hyperparameters

Provide a single `config` with defaults below. Mark which are inherited vs new.

**System (inherited unless noted):**
- `M = 16`, `L = 64`, `K = 4`, carrier `28 GHz`, `d_min = λ/2`.
- `Q = 3` targets **(NEW)**; target directions parameterized by separation `Δφ` **(NEW)**.
- Users uniform in sector, radius `50–200 m`.
- AWGN: `180 kHz` bandwidth, `9 dB` noise figure → `σ²` from thermal density + BW + NF.
- EVM `κ = 0.05`. Power uniform `P_k = P_max/K`.
- Regimes `T = 3`: LoS, Rayleigh, Rician (Rician K-factor configurable).

**Decoder / repair (inherited):**
- soft-clip sharpness `μ`, cell extents `(L_h, L_v)`, repair iters `I = 3`.

**Evolution (inherited):**
- `N = 200`, `G = 300`, mutation `σ = 0.1`, runs `= 20` seeds.
- MC samples per generation `S` **(set explicitly — try `S = 32`; larger = lower variance, more compute; flag as tunable)**.

**RMP (form inherited, success signal NEW):**
- init `RMP = 0.3`, `RMP_min = 0.05`, `RMP_max = 0.5` (to reproduce the conference behavior: peak ≈ 0.4 early, decay to ≈ 0.05).
- `η ≈ 0.05`, `ρ_target ∈ [0.3, 0.5]`, EMA `β ≈ 0.1`.
- `survival_mode ∈ {rank1, hv_contrib}` **(NEW)**, default `rank1`.

**Output (new):**
- `cross_regime_agg = min` (worst-regime) for the unified front.
- HV reference (nadir) point: fixed, slightly worse than the worst objective values observed across all methods, **identical across all comparisons** (HV is only comparable with a fixed reference).

---

## 8. Baselines (drive the 3 figures)

1. **No-transfer** — NSGA-II run **independently per regime** (no inter-task mating; equivalently `T` separate single-task MO runs). Isolates the value of transfer. → Fig 1.
2. **Fixed-RMP MO-MFEA** — same framework, `RMP` held constant (e.g. 0.3). Isolates the value of *adaptive* RMP (ablation). → Figs 1 & 2.
3. **Pooled single-task** — a **single** MO problem where each evaluation draws from a *mixture of all regimes* (regimes pooled into one task), single-population NSGA-II. This is the "is multitask just data augmentation?" strawman that the method must beat. **Folded as a curve in the HV-vs-generation plot (Fig 2)** — do not give it its own figure.
4. **Reference layouts** — Fixed USPA (uniform square planar array, no MA optimization) + random layout. → Fig 1 reference markers.
5. **Online reference** — per-realization MO optimization (yields a *front per realization*); the upper bound for Fig 3. **Expensive** → compute on only `~20–30` realizations and overlay as a summary band.

---

## 9. Metrics

- **Hypervolume (HV)** — primary, native to MO. Fixed reference point (Sec 7). Report HV-over-generation and HV-over-realization.
- **Knee-point physical values** — at the knee of the unified robust front, report `(min-rate [bps/Hz], min-beampattern [dB])` offline vs online. **Required** so the comms audience has real units, not just an abstract volume. Put this in text + caption.
- **Front width vs angular separation** `Δφ` — the finding from Sec 2.4. Sweep `Δφ`, measure front spread (range of `F_com` and `F_sen` across the front, or HV). Show that front width is controlled by user–target overlap → demonstrates the tradeoff is real and not a modeling artifact.
- IGD — optional, only if a reference front is constructed. HV is enough for a letter.

---

## 10. The three figures (this is the entire results section)

- **Fig 1 — Unified robust front (money figure).** `F_com` vs `F_sen`. Curves: MO-MGRAP (adaptive) · no-transfer (independent NSGA-II) · fixed-RMP · reference markers (USPA, random). Optional second panel or inset: **front width vs `Δφ`**.
- **Fig 2 — HV over generations (algorithm proof).** Curves: MO-MGRAP (adaptive) · fixed-RMP · pooled-single-task (the augmentation ablation). **Inset: RMP-vs-generation (mean ± std)** — reproduces the conference's permissive-early-then-decay behavior. Do not spend a separate figure on RMP.
- **Fig 3 — HV-CDF over realizations (robustness).** CDF of the offline unified front's HV over `200` stochastic realizations; overlay the **online-reference HV** as a summary band over `~20–30` realizations. This single figure carries both the *spread/predictability* story and the *recovery-vs-online* story without collapsing the front to a point. Knee-point numbers (Sec 9) in caption.

---

## 11. Suggested module layout

```
config.py        # all hyperparameters, system params, regime defs, target placement (Δφ)
channels.py      # batched channel construction per regime (LoS/Rayleigh/Rician); G_geo for sensing [INHERITED + small NEW]
comm.py          # BD precoding, EVM SINR, rates [INHERITED]
sensing.py       # a_eff with RIS path, beampattern B_q, F_sen [NEW]
objectives.py    # assemble [F_com, F_sen] per individual per regime; MC aggregation [CHANGED]
genotype.py      # decode (soft-clip MA + phase), repair (repulsive + clip) [INHERITED]
operators.py     # SBX, linear-arith xover, Gaussian mutation, multifactorial mating + transfer tag [INHERITED + tag]
nsga.py          # tensorized non-dominated sort + crowding distance [NEW]
rmp.py           # Pareto-survival tracking (rank1 / hv_contrib), EMA update [NEW logic]
mfea.py          # main MO-MFEA loop; generalizes Algorithm 1 [CHANGED]
archive.py       # store rank-1 individuals; build unified robust front (min over regimes) [NEW]
baselines.py     # no-transfer, fixed-RMP, pooled-single-task, USPA/random, online reference [mix]
metrics.py       # hypervolume, knee point, front width vs Δφ, (IGD optional) [NEW]
experiments.py   # run matrix, seeds, persistence
plots.py         # the 3 figures
```

Build order recommendation: `config → channels/comm (port) → genotype/operators/repair (port) → sensing (new) → objectives → nsga (new) → rmp (new) → mfea (assemble) → archive → metrics → baselines → experiments → plots`. Get a **single regime, single objective** smoke test passing (reproduce a conference number) before turning on multi-objective + multitask.

---

## 12. Risks & things to verify (do not skip)

1. **Front collapse (highest risk).** With no power split and no dedicated sensing beam, the *only* tradeoff lever is geometry (MA + RIS phase). If the front degenerates to a point, the whole multi-objective story dies. **Mitigation is target placement (Sec 2.4):** verify early that with partial user–target angular overlap the front is visibly non-degenerate. If it still collapses, the documented fallback is to add a single scalar power-split variable as a backstop — but try geometry-only first.
2. **Sensing convention.** Verify `a_eff`'s conjugate/transpose convention against the existing comm `G` definition before trusting beampattern values (Sec 2.3).
3. **Multitask ≠ augmentation.** The pooled-single-task baseline (Sec 8.3 / Fig 2) is the reviewer's obvious objection. The method must beat it on unified HV and convergence speed, or the multitask framing is unjustified.
4. **Novelty crispness.** The delta vs MFEA-II (adaptive RMP but scalar) and MO-MFEA (multi-objective but fixed RMP) is the **Pareto-survival success signal**. Keep this one sentence sharp.
5. **HV comparability.** Fixed reference point across every method and run, or HV numbers are meaningless.
6. **Compute.** Batched SVD over `(N×S)` for BD is the heaviest op; the sensing `R_s` (`N×S×M×M`) and non-dom sort add to it. `M=16` on an RTX 5070 Ti should be fine; the online-reference for Fig 3 is the expensive part — keep it to `~20–30` realizations.
7. **Citability gate (not code, but blocks submission).** The conference paper must be citable (at minimum an arXiv preprint) when the letter is submitted, because the letter compresses the inherited pipeline into a citation. If it isn't available, the letter must be made more self-contained.

---

## 13. Out of scope for this letter (explicit cuts)

- CVaR risk-averse objectives → keep `E[·]`; one future-work sentence.
- Scaling vs number of antennas `M` (conference Figs 6/7) → cut, or one sentence + small inset at most.
- Per-task-pair RMP → single global RMP only.
- CRB / FIM-based sensing metric → use min-beampattern; CRB is a future, fuller-paper direction.
- Dedicated sensing beamforming / power split → geometry-only (unless the front-collapse fallback in Risk 1 forces a single scalar power-split).
