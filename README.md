# morphompm — Differentiable Morphoelastic MPM for Soft / Living Matter

**Working title.** Forked from `G:/CPP/Basements` (2026-06-28). This is *not*
a rescue of the stalled agricultural-robotics engine; it is a **new, deliberately
small** project that reuses Basements' verified material-agnostic numerical core
and builds toward a single novel deliverable.

## Governing principle: deliverable-first, not capability-first

The Basements project stalled twice from the same disease: building *capabilities*
(galaxies → agriculture → biofilm → "cube of everything") before a single validated
result existed. The cure is to fix **one result that cannot be produced without this
tool**, then build only the minimum slice of each layer that the result requires.

Novelty lives in the **deliverable**, not in the stack. Every component below already
exists in the literature (MPM, Drucker-Prager, differentiable MPM, morphoelastic
growth F=Fe·Fg, reaction-diffusion, printability windows). Assembling them is *not*
novelty by itself — paper #1 was killed for exactly this. The defensible contribution
is a *computed result the assembled parts produce that nobody has produced before*.

## The stack (build bottom-up; motivate top-down)

```
[5] printability window + Pareto + optimal control   (application/output)
[4] adaptive-sparse sampling -> surrogate operator    (NOT a dense "cube")
[3] slow time axis: growth (F_g) + reaction-diffusion (active matter)
[2] constitutive plugin interface (granular -> bioink -> viscoelastic gel)
[1] differentiable MPM core (Fe, Hencky stress, autodiff)   [REUSED from Basements]
[0] single trajectory, analytically verified                 <-- CURRENT STEP
```

### Discipline gates (each layer is GATED on the one below being validated)
- **[0] now:** isotropic morphoelastic growth `F_g = g·I` inserted into the Hencky
  stress loop via `Fe = F·F_g^{-1}`. Verified by (a) analytic constitutive stress and
  (b) free-swelling equilibrium (`det F → g^3`, residual elastic strain → 0).
- One constitutive model / one field at a time. Operator-split the slow (growth /
  diffusion) clock from the fast (mechanics) clock — never one `dt` for both.
- A feasibility map / surrogate is built on a *validated* single trajectory, never before.

## Reused from Basements (the real salvage)
- `basements::math` — Vec3 (AVX2), Matrix3, **SVD** (constitutive substrate).
- MLS-MPM transfer + Hencky-Kirchhoff stress formulation (Hu 2018 / Klar 2016).

## Shed (dead weight under this pivot)
- Agricultural framing, terramechanics/Bekker, `ros2_bridge`, myCobot, editor,
  rover/tine scenarios.

## Open novelty question (under test)
A background literature check is verifying whether the intersection
*(open · differentiable · growth + diffusion · bioprinting-validated MPM)* is
genuinely unoccupied. **No commitment beyond [0] until that returns.**

## Build
```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
./build/Release/test_growth.exe
```
