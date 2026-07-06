# Roadmap & open problems

This is the outward-facing map of where **morphompm** is, where it's going, and —
honestly — what is *not* solved yet. If you're looking for something to work on,
the [Where to help](#where-to-help) section maps directly to labelled GitHub issues.

> **One-line status.** The differentiable core is *verified* along three independent
> axes and fully reproducible. It is **not yet validated against experimental data**.
> Treat it as a verified *method*, not a validated *result*.

## Where we are

**Done & verified** (see [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md)):

- Forward MLS-MPM with morphoelastic growth (`F = Fe·Fg`); pluggable constitutive
  models (neo-Hookean, Hencky, a rate-dependent Herschel–Bulkley bioink) at a single
  seam (`python/morphompm/constitutive.py`).
- Hand-written, finite-difference-gated adjoints for every piece — constitutive VJP
  (incl. the SVD adjoint), one MPM step (incl. particle advection), full trajectory.
- Differentiable inverse demo: recover a growth rate from an observed final shape
  (Gauss–Newton), sharing the exact code path as the verification gate.
- Three verification axes (FD-gradient consistency, forward-physics oracle, confined-
  swell analytic oracle) + numpy↔C++ parity, all reproducible from one command
  (`python scripts/reproduce.py`).

**Current frontier:** turning the verified core into a *scientific result* —
reproduce published morphogenesis (forward) and then do what the source papers did
not: the **differentiable inverse** (observed morphology → inferred growth/material
law), validated against already-published data. The plan is a three-rung ladder
(full detail in [`docs/PHASE3_morphogenesis.md`](docs/PHASE3_morphogenesis.md)):

| Rung | Forward validation | Inverse (the novel part) | What it needs |
|---|---|---|---|
| **1 — Bilayer bending** | curvature κ vs Timoshenko `κ = 1.5·ε/h` | observed κ → infer growth mismatch ε | symmetric-bilayer scenario in numpy; run-to-relaxation gate |
| **2 — Constrained swelling wrinkles** | wavelength vs a bonded-substrate gel dataset | measured λ → infer modulus ratio | a substrate boundary condition; higher resolution |
| **3 — Gut looping** (gold standard) | loop geometry vs published tabulated data | (λ, loop count, radius) → infer modulus ratio | 1D rod–sheet composite; a growth *field* `g(x)`; scale |

## Open problems & unknowns

These are real and unsolved. They are stated plainly on purpose — each is an
invitation, not a hidden weakness.

1. **Zero experimental / real-data validation.** Everything runs on *simulated*
   observations today. The whole scientific claim hinges on matching published
   morphogenesis data, and that has not been done yet. This is the single biggest
   open item.
2. **The bilayer bending offset is a numerical-convergence artifact, not physics.**
   T8 reproduces the *linear-in-mismatch* scaling but sits at ~0.60× of Timoshenko at
   4000 steps (→0.76× at 12000). Diagnosed as under-relaxation, not a constitutive
   error — but it has not been run to full relaxation convergence, so the true ratio
   is unconfirmed.
3. **Order-of-operations hazard in the inverse.** Running the inverse on an
   under-relaxed / biased forward yields confident-but-wrong parameters. A
   relaxation-convergence gate is a *prerequisite* for trusting any inverse result.
4. **Scale cliff.** The pure-numpy loops cannot reach the resolution Rungs 2–3 need.
   At that point the deferred "production substrate vs vectorize numpy" decision
   ([`docs/PIPELINE.md`](docs/PIPELINE.md) §Substrate) becomes forcing.
5. **Literature digitization is lossy.** Extracting a wavelength from a published plot
   needs a reproducible, version-tracked protocol before any number is used.

## Where to help

Mapped to GitHub issues by label — [good first issue], [help wanted], [open-problem].

**Good first issues** (small, well-scoped, low context):

- Add a new constitutive model at the `constitutive.py` seam — implement `stress` +
  `stress_vjp`, and it inherits the FD gate for free.
- Add a round-trip correctness test for `load_curve` / the VTK export in `io.py`.
- Add a symmetric-layer variant of the bilayer scenario (even layer count).

**Help wanted** (larger, more context):

- Run T8 bilayer bending to relaxation convergence and add the convergence check as an
  assertable gate (unblocks open problem #2 and the Rung-1 inverse).
- Build the first *numpy* morphogenesis forward (bilayer) + the `LayeredGrowth` seam
  (the second growth consumer that justifies the seam).
- Vectorize / scale the numpy substrate so Rungs 2–3 are reachable (keeping numpy as
  the reference oracle via the parity gate).

**Open problems** (research-scale; a paper's worth):

- Validate the forward + inverse against a published morphogenesis dataset (bonded-gel
  swelling wrinkles, or gut looping). This is the north-star result.

## Contributing

New models, tests, scenarios, and dataset validations are all welcome — see the
issues above. The design discipline is *deliverable-first*: build the minimum a
verifiable result needs, extend a seam only when a second consumer appears, and gate
every claim. Please keep new gradient code finite-difference-gated.
