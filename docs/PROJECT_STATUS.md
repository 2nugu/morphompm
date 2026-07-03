# PROJECT STATUS

## Overview
**morphompm** — a differentiable morphoelastic growth-MPM for soft/living matter;
verified differentiable core → infer material & growth laws from observed
deformation. Forked from Basements 2026-06-28 (see `../README.md`). Architecture:
[`PIPELINE.md`](./PIPELINE.md).

## Current Phase
**Phase 2: PIVOT to morphoelastic morphogenesis (2026-07-03).** North star moved
from bioink-EXTRUSION → **differential-growth morphogenesis** (bending, swelling
wrinkles, gut looping) — where published same-study calibration+validation data
exists (Savin&Tabin 2011, Guvendiren 2009) and the current core predicts it with
LOW code burden (no nozzle/free-surface). Aligns with neighbor lab's cell-culture/
scaffold axis. HB(bioink) preserved as a dormant module for a future extrusion track.

First quantitative morphogenesis validation DONE: **T8 bilayer bending curvature vs
Timoshenko** — reproduces linear-in-mismatch scaling (1.86≈2.0) and correct order
of magnitude, systematic ~0.58× vs idealized 1D thin-beam (3D/finite-thickness/
Poisson; analytic assumptions unmet). Honest: scaling+magnitude, not tight — tight
match needs a convergence study (thinner + plane-strain + finer → scale/vectorization).

**Phase 1 (differentiable core) — COMPLETE:** constitutive(elastic+bioink) → transfer
(advect adjoint) → trajectory → inverse design; 3 verify axes (FD, forward-physics,
confined-swell analytic). `python -m morphompm.verify` → ALL PASS.

## Phase Checklist
- [x] [2] Constitutive seam: neo-Hookean + Hencky, forward + **manual VJP**
      (incl. hand-derived SVD adjoint), FD-gated (rel.err ≤ 1.5e-9).
- [x] [3] Single-step transfer: P2G/grid/G2P forward + manual VJP; **assembly
      gate** (composed adjoint vs FD, rel.err 4.5e-10).
- [x] Forward oracle: C++ MLS-MPM growth (T1–T7, 18 checks); Taichi parity
      (det F = 3.3750 exact).
- [ ] [0]/[1] Config + State dataclasses; re-home modules into `python/morphompm/`.
- [x] [4] Trajectory rollout + adjoint (compose per-step VJPs), FD-gated
      (3-step, momentum path active, rel.err 3.7e-10; advect=False).
- [x] [6] Inverse-design demo: recover growth rate from observed shape
      (Gauss-Newton, g_true=1.4 recovered in 6 iters, err 1.4e-13).
- [x] [3]/[4] advect adjoint (∂weights/∂x): full MPM adjoint, moving particles;
      advect=True trajectory gradient vs FD rel.err 3.1e-10.
- [x] [2] Herschel-Bulkley (bioink): rate-dependent yield-stress fluid model
      (Papanastasiou-reg.), stress(F,Fg,C) + VJP incl C_bar, FD-gated (9.4e-10);
      plugged into the SAME transfer/integrate with ZERO core change (HB trajectory
      gradient vs FD 2.3e-10). Elastic gates unchanged (refactor behavior-preserving).
- [~] Validation: swelling-hydrogel pre-registration written (`docs/PREREG_hydrogel.md`);
      P2 (constrained-swell wrinkle wavelength) sim readout + experiment pending.
- [ ] (deferred) production substrate; reaction-diffusion; neural constitutive; UI.

## Completed Milestones
| Milestone | Output | Date |
|-----------|--------|------|
| C++ growth-MPM forward (iso/aniso/differential), 18 checks | `tests/test_growth.cpp` | 2026-06-28 |
| Differential-growth residual-stress + bend figure | `outputs/figures/differential_growth.*` | 2026-06-28 |
| Self-contained (math headers vendored) | `include/basements/core/math/` | 2026-06-28 |
| Taichi forward parity + constitutive autodiff | `python/growth_mpm.py` | 2026-06-29 |
| Constitutive seam + manual VJPs (incl. SVD adjoint), FD-gated | `constitutive.py` | 2026-07-01 |
| Transfer seam + composed assembly gate | `transfer.py` | 2026-07-01 |
| Pipeline architecture formalized | `docs/PIPELINE.md` | 2026-07-01 |
| Advect adjoint (full MPM, moving particles) + Herschel-Bulkley bioink model | integrate/constitutive | 2026-07-01 |
| Logic audit: caught HB pressure-sign bug (FD-invisible); forward-physics guard added | verify [1p] | 2026-07-01 |
| Strong forward oracle (confined-swell, modulus-dependent, teeth-verified) — 3 independent verify axes | constitutive | 2026-07-03 |

## Active Decisions & Rationale
- **Reproduction is first-class, not throwaway sanity.** (i) Reproducing established
  results (Timoshenko/Savin) with the new differentiable method IS a contribution and
  the substrate for the inverse; (ii) reproducibility (deterministic, versioned,
  one-command) is a research pillar. Both preserved & analyzed alongside the novel
  inverse. Infra: `scripts/reproduce.py` (one command, env-stamped), git-versioned
  (first commit 675b030), determinism = single-thread/no-RNG → bit-identical.
- **FD gates verify gradient-consistency, NOT forward correctness.** A wrong
  forward + matching adjoint passes every FD gate — the HB pressure-sign bug
  (2026-07-01 logic audit) did exactly that. Guard: `forward_physics_gate` (free-swell
  det F→g³ vs C++/analytic oracle) is now a permanent `verify` stage [1p].
- **Modular manual adjoints, NOT monolithic global AD.** Stock Taichi 1.7.4
  autodiff segfaults on matrix-MPM adjoints; per-module hand-written VJPs (FD-gated)
  are robust and how production diff-sim works. User insight ("build pieces with
  connection points").
- **Substrate roles fixed** (PIPELINE §Substrate): C++=frozen oracle, numpy=
  differentiable reference/spec, production substrate deferred.
- **AI is not the differentiator; the differentiable physics is.** Neural/learned
  constitutive is a future module that plugs into the [2] seam, gated on real data.
- **UI/broad-compat deferred to last** (Basements died from premature breadth);
  compatibility via standard formats, not built integrations.

## Open Questions
- [ ] SVD adjoint degeneracy clamp (isotropic growth s_i≈s_j) — error band TBD.
- [ ] Trajectory adjoint memory strategy (checkpointing) at scale.
- [ ] Herschel-Bulkley VJP difficulty (yield-stress nonsmoothness) — gauge before committing.
- [ ] Real-data access shape (rheometry vs full-field DIC) from neighbor lab.

## Key Outputs Registry
| Type | File | Description |
|------|------|-------------|
| Oracle | `include/morphompm/growth_solver.h` + `tests/test_growth.cpp` | C++ forward, 18 checks |
| Module | `python/morphompm/constitutive.py` | stress + VJP (neo-Hookean, Hencky) |
| Module | `python/morphompm/transfer.py` | single-step MPM + VJP, assembly gate |
| Figure | `outputs/figures/differential_growth.png` | residual-stress bending |

## Docs Registry
| File | Purpose | Created | Updated |
|------|---------|---------|---------|
| `docs/PROJECT_STATUS.md` | This — phase + decisions | 2026-07-01 | 2026-07-01 |
| `docs/PIPELINE.md` | End-to-end architecture | 2026-07-01 | 2026-07-01 |
| `README.md` | Deliverable-first principle + stack | 2026-06-28 | 2026-06-28 |
