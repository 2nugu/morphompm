# Phase 3 — Morphogenesis validation + differentiable inverse (design)

> **NOTE — literature values are UNVERIFIED.** Paper names below are pointers to
> check against the primary source. Any specific numeric value (moduli, wavelengths,
> growth strains, ranges) must be read from the paper itself before use — do NOT treat
> numbers stated here as authoritative; a secondary search can be inaccurate.


**Goal.** Turn the verified differentiable core into a *scientific result*: reproduce
published morphogenesis (forward) and then do what the source papers could NOT — the
**differentiable inverse** (observed morphology → inferred material/growth law) —
validated against ALREADY-PUBLISHED data (no own experiments; lit-confirmed same-study
calibration+validation datasets exist). This is where the empty-axis novelty
(growth-in-MPM + differentiable inverse) becomes a claim, not a capability.

**Gating principles (carry forward):** calibration inputs ≠ validation targets;
pre-register predictions (git-timestamp) before comparing; a dynamic validation is
only trusted after a **relaxation-convergence** check; reproduction (forward-match)
and inverse are BOTH first-class, analyzed together.

## The ladder (low code-burden → high value; climb only when each rung's gate is green)

### Rung 1 — Bilayer bending vs Timoshenko  [tighten what exists]
- **Scenario:** differential-growth bilayer (have it, T8). **Fix first:** symmetric
  layers (even count; audit B3) + run to **relaxation convergence** (audit: 0.60@4k→
  0.76@12k; not converged) → get the true ratio, expect →1 in the thin-beam limit.
- **Validates (forward):** bending curvature κ vs analytic κ=1.5·ε/h (independent).
- **Inverse (novel):** observed κ → infer growth mismatch ε (or modulus ratio).
  Gauss-Newton on a validated forward. Achievable at ~current scale.
- **Triggers:** relaxation-convergence gate; symmetric-bilayer scenario in **numpy**
  (first numpy morphogenesis forward — currently C++-only). This is the 2nd growth
  consumer that JUSTIFIES the deferred `GrowthDriver`/`LayeredGrowth` seam.

### Rung 2 — Constrained swelling wrinkles vs Guvendiren 2009  [needs BC + resolution]
- **Data (same-study):** Guvendiren 2009 — reports gel stiffness + a measured
  swelling-wrinkle wavelength (values pending primary-source check). Calibrate on
  stiffness; PREDICT wavelength; validate against the measured wavelength.
- **Triggers:** constrained/substrate **boundary condition** in the transfer (gel
  bonded to rigid base — new); wrinkle instability needs **resolution → vectorization/
  scale**; Biot analytic cross-check for the wavelength.
- **Caveat (audit/lit):** λ ∝ (E-ratio)^{1/3} — a WEAK test; use multiple observables.

### Rung 3 — Gut looping vs Savin & Tabin 2011  [gold standard; scale]
- **Data (same-study, reported as tabulated):** Savin & Tabin 2011 — measured tissue
  moduli, differential growth strain, and emergent loop geometry (wavelength, loop
  count, radius). Specific numbers PENDING primary-source verification — do not hardcode
  from memory.
- **Forward:** reproduce the looping (composite tube+mesentery, differential growth).
- **Inverse (the headline):** observed (λ, n, R) → infer modulus ratio E_m/E_t, scored
  vs the paper's own reported measurements — what the original paper did NOT do. Use
  wavelength AND loop-count AND radius together (a single wavelength is under-determined).
- **Triggers:** 1D rod-sheet composite geometry; `GrowthDriver` field g(x); **scale**
  (vectorization / production substrate — the decision point PIPELINE §Substrate defers).

## Extensibility this phase legitimately unlocks (now with real 2nd consumers)
- **GrowthDriver seam** (`IsotropicGrowth` → `LayeredGrowth` → `FieldGrowth g(x)`):
  build at Rung 1 (LayeredGrowth) / Rung 3 (FieldGrowth) — NOT before (was correctly
  deferred; Rung 1 is the 2nd consumer).
- **Boundary conditions** in transfer (substrate/wall Dirichlet) — Rung 2.
- **Scale**: vectorize numpy (batched adjoints, numpy stays reference oracle via the
  parity gate) or port to production substrate — forced by Rung 2/3 resolution.
- **StateCotangent**: only if a state field is added (diffusion concentration) — later.

## Sequencing / first concrete step
1. Rung-1 numpy bilayer forward (symmetric) + relaxation-convergence gate → tighten
   the Timoshenko match. **This is the next code step** (low burden, current scale,
   builds the numpy morphogenesis forward + LayeredGrowth seam).
2. Rung-1 differentiable inverse (κ → ε) — the first inverse on a non-trivial,
   validated morphology.
3. Pre-register Rung-2/3 predictions; then climb as scale/BC land.

## Honest risks
- Scale cliff: Rungs 2–3 need resolution the pure-numpy loops can't reach → the
  deferred substrate decision becomes forcing. Benchmark numpy limits at Rung 1.
- Inverse on an under-relaxed / biased forward yields confident-wrong params → the
  relaxation-convergence gate is a PREREQUISITE for the inverse (Rung 1 order matters).
- Literature digitization (Guvendiren λ from a plot) is lossy + must be version-tracked
  (a reproducible extraction protocol before use).
