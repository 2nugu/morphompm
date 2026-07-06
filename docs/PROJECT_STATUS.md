# PROJECT STATUS

## Overview
**morphompm** — a differentiable morphoelastic growth-MPM for soft/living matter;
verified differentiable core → infer material & growth laws from observed
deformation. Architecture: [`PIPELINE.md`](./PIPELINE.md).

## Public release & pause checkpoint (2026-07-06)

**Status: PUBLISHED + PAUSED (resume-ready).** Public at
https://github.com/2nugu/morphompm (MIT); archived on Zenodo —
DOI [10.5281/zenodo.21215516](https://doi.org/10.5281/zenodo.21215516).

Done this session: renamed the vendored math core to `morphompm::math` (sole-
authored framing, no external-provenance language); public packaging
(`pyproject.toml`, LICENSE, CITATION.cff, README with a "what you can use it for"
section); fixed clipped/overlapping figure titles; **clean-clone build verified
GREEN** from a fresh GitHub clone (`pip install -e .` + `python -m morphompm.verify`
ALL GATES PASS + C++ `test_growth` ALL CHECKS PASSED); added [`../ROADMAP.md`](../ROADMAP.md)
(status + honest open problems + where-to-help) and **7 labelled contributor
issues** (#1–#7: good-first-issue / help-wanted / open-problem).

**Resume here (when reopening):**
- Traffic/discoverability (owner action): call-for-contributors post, arXiv /
  Papers-with-Code entry (drafts can be assisted). Repos are conversion-ready;
  only traffic remains.
- Decide the contributor-contact channel (GitHub Discussions on? email public?)
  so ROADMAP's "open a discussion first" actually works.
- Cross-verify the file pointers in issues #1–#7 resolve (first-impression defence).
- Next code step (Rung 1): symmetric numpy bilayer forward + relaxation-convergence
  gate → then the κ→ε differentiable inverse (see `PHASE3_morphogenesis.md`).

## Current Phase
**Phase 2: PIVOT to morphoelastic morphogenesis (2026-07-03).** North star moved
from bioink-EXTRUSION → **differential-growth morphogenesis** (bending, swelling
wrinkles, gut looping) — where published same-study calibration+validation data
exists (Savin&Tabin 2011, Guvendiren 2009) and the current core predicts it with
LOW code burden (no nozzle/free-surface). Aligns with neighbor lab's cell-culture/
scaffold axis. HB(bioink) preserved as a dormant module for a future extrusion track.

First quantitative morphogenesis validation DONE: **T8 bilayer bending curvature vs
Timoshenko** — reproduces linear-in-mismatch scaling (1.86≈2.0), correct order of
magnitude, ratio 0.60 @ 4000 steps. Cause of the offset DIAGNOSED (2026-07-04 multi-
agent logic audit) as **UNDER-RELAXATION** — NOT the earlier (retracted) "3D/finite-
thickness/Poisson" guess: transverse growth ruled out (axial-only=isotropic=0.60),
grid resolution minor (h/dx 2→6: 0.60→0.63), relaxation confirmed (0.60@4k→0.76@12k,
converging toward Timoshenko). A numerical-convergence artifact, not physics; the
constitutive law itself is exact (confined-swell analytic oracle). Not yet run to
full relaxation convergence.

**Phase 1 (differentiable core) — COMPLETE:** constitutive(elastic+bioink) → transfer
(advect adjoint) → trajectory → inverse design; 3 verify axes (FD, forward-physics,
confined-swell analytic). `python -m morphompm.verify` → ALL PASS.

## Phase Checklist
- [x] [2] Constitutive seam: neo-Hookean + Hencky, forward + **manual VJP**
      (incl. hand-derived SVD adjoint), FD-gated (rel.err ≤ 1.5e-9).
- [x] [3] Single-step transfer: P2G/grid/G2P forward + manual VJP; **assembly
      gate** (composed adjoint vs FD, rel.err 4.5e-10).
- [x] Forward oracle: C++ MLS-MPM growth (T1–T8, incl. bilayer-vs-Timoshenko + degeneracy); Taichi parity
      (det F = 3.3750 exact).
- [x] [0]/[1] Config + State dataclasses; modules re-homed into `python/morphompm/` (done).
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
| C++ growth-MPM forward (iso/aniso/differential), T1–T8 | `tests/test_growth.cpp` | 2026-06-28 |
| Differential-growth residual-stress + bend figure | `outputs/figures/differential_growth.*` | 2026-06-28 |
| Self-contained linear-algebra core (Vec3/Matrix3/SVD) | `include/morphompm/math/` | 2026-06-28 |
| Taichi forward parity + constitutive autodiff | `python/experiments/growth_mpm.py` | 2026-06-29 |
| Constitutive seam + manual VJPs (incl. SVD adjoint), FD-gated | `constitutive.py` | 2026-07-01 |
| Transfer seam + composed assembly gate | `transfer.py` | 2026-07-01 |
| Pipeline architecture formalized | `docs/PIPELINE.md` | 2026-07-01 |
| Advect adjoint (full MPM, moving particles) + Herschel-Bulkley bioink model | integrate/constitutive | 2026-07-01 |
| Logic audit: caught HB pressure-sign bug (FD-invisible); forward-physics guard added | verify [1p] | 2026-07-01 |
| Strong forward oracle (confined-swell, modulus-dependent, teeth-verified) — 3 independent verify axes | constitutive | 2026-07-03 |
| Public release: GitHub + Zenodo DOI, clean-clone build verified, ROADMAP + 7 labelled issues | repo + `ROADMAP.md` | 2026-07-06 |

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
- **UI/broad-compat deferred to last** (an earlier over-scoped effort stalled from premature breadth);
  compatibility via standard formats, not built integrations.

## Open Questions
- [x] SVD adjoint degeneracy: RESOLVED (2026-07-04) — Daleckii-Krein form, degeneracy-safe (VJP vs FD 5e-10 at repeated SVs); replaced the wrong clamp-to-zero.
- [ ] Trajectory adjoint memory strategy (checkpointing) at scale.
- [ ] Herschel-Bulkley VJP difficulty (yield-stress nonsmoothness) — gauge before committing.
- [ ] Real-data access shape (rheometry vs full-field DIC) from neighbor lab.

## Key Outputs Registry
| Type | File | Description |
|------|------|-------------|
| Oracle | `include/morphompm/growth_solver.h` + `tests/test_growth.cpp` | C++ forward oracle, T1–T8 |
| Module | `python/morphompm/constitutive.py` | stress + VJP (neo-Hookean, Hencky) |
| Module | `python/morphompm/transfer.py` | single-step MPM + VJP, assembly gate |
| Figure | `outputs/figures/differential_growth.png` | residual-stress bending |
| Release | https://github.com/2nugu/morphompm · DOI 10.5281/zenodo.21215516 | public repo + Zenodo archive |

## Docs Registry
| File | Purpose | Created | Updated |
|------|---------|---------|---------|
| `docs/PROJECT_STATUS.md` | This — phase + decisions | 2026-07-01 | 2026-07-06 |
| `docs/PIPELINE.md` | End-to-end architecture | 2026-07-01 | 2026-07-01 |
| `ROADMAP.md` | Outward-facing status + open problems + where-to-help | 2026-07-06 | 2026-07-06 |
| `README.md` | Deliverable-first principle + stack | 2026-06-28 | 2026-07-06 |
