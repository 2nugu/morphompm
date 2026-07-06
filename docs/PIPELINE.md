# morphompm — Pipeline Architecture

The whole pipeline, designed end-to-end. Work slots into this frame; no ad-hoc
accretion. Bricks are implemented into designated stages, each with a verification
gate. Future stages are **interface-defined here but not coded** until reached.

## North star (bounded)
A **differentiable morphoelastic growth-MPM** for soft / living matter, whose
verified differentiable core lets one **infer material & growth laws from
observed deformation** (physics as the inductive bias). First empty-axis claim:
*growth (F_g) inside MPM* + *differentiable inverse of it*. Everything below
exists to serve that one deliverable. Anything not required by it is OUT (see §7).

## Data flow (stages)

```
[0 Config]  material θ, constitutive choice, growth law, scenario, integration
     │                                (config.py — dataclass, JSON/HDF5 ser.)
[1 State]   ParticleState{x,v,C,F,Fg[,c,S]} + Grid            (state.py)
     │
[2 Constitutive]  τ = stress(F,Fg,θ)  + VJP        ✅ DONE  (constitutive.py)
     │   pluggable: neo-Hookean✅  Hencky✅  Herschel-Bulkley✅  [viscoelastic]○
[3 Transfer]  P2G → grid → G2P  + VJP  (material-agnostic, calls [2])
     │   single-step + advect adjoint ✅ DONE (transfer.py); trajectory ✅ (integrate.py)
[4 Integrate]  operator split: fast mech clock ⟂ slow growth/diffusion clock
     │   trajectory forward + adjoint (compose per-step [2]∘[3] VJPs)   ✅ DONE
[5 Drivers]  growth law g(t)/g(x,t)  [reaction-diffusion field]○
     │
[6 Diff/Inverse/Learn]  compose adjoints → ∂loss/∂θ; inverse design;
     │                  [neural constitutive plugged into 2]○
[7 Verify]  (cross-cutting) per-module FD gate + assembly gate + C++ oracle
     │       parity + determinism                                (verify.py)
[8 I/O + Viz]  standard formats only (npy/JSON now; HDF5/VTK hooks); figures
     │
[9 UI]  headless library API now; thin GUI ONLY if/when users exist   ○ (last)
```
✅ built & FD-gated  ○ designed, not yet coded

## Module interfaces (the seams — stable contracts)

```
# [2] constitutive
stress(F, Fg, θ)              -> tau (3x3 sym)
stress_vjp(F, Fg, θ, tau_bar) -> (F_bar, Fg_bar)      # FD-gated

# [3] transfer  (takes a constitutive callback)
step(state, θ)                -> state'                # one MLS-MPM step
step_vjp(state, θ, state'_bar) -> (state_bar, θ_bar)  # FD-gated (assembly gate)

# [4] integrate
rollout(state0, θ, schedule)  -> trajectory / final    # operator split
rollout_vjp(...)              -> gradients             # compose [3] VJPs
```

## Substrate strategy (resolves the C++/Taichi/numpy sprawl)
- **C++** (`include/`, `tests/test_growth`): forward-physics **oracle**, FROZEN.
  Reference for parity; not extended.
- **numpy** (`python/morphompm/`): the **differentiable reference** — correctness
  first, hand-written FD-gated adjoints. This is the SPEC. Slow; not for scale.
- **Taichi**: forward parity ✅; its autodiff SEGFAULTS on matrix-MPM adjoints
  (2026-06-29) → NOT used for gradients. Candidate production forward.
- **Production differentiable substrate (scale): DEFERRED.** Port the verified
  numpy adjoints to Taichi (custom svd_grad) or JAX only when scale demands it.
  numpy stays the oracle. Decide at [4]-trajectory scale, not before.

## Verification spine (every stage has a gate)
- [2],[3] each: analytic forward check + VJP vs finite difference (< 1e-5).
- [3]/[4]: end-to-end **assembly gate** (composed adjoint vs FD) — catches
  correct-module / wrong-wiring bugs.
- Forward parity: numpy↔C++ (shared-Hencky scenario, gate [P]) — det F match.
- Determinism: identical runs bit-identical (numpy gate [D] + C++ T6).
- SVD degeneracy (s_i≈s_j, isotropic growth): Daleckii-Krein divided-difference form (degeneracy-safe; VJP vs FD 5e-10 at repeated SVs). [was a wrong clamp-to-0; fixed 2026-07-04]

## Validation spine (epistemology — avoids the retrodiction trap)
- **Calibration inputs ≠ validation targets.** Measure intrinsic params
  (rheometry); PREDICT emergent behavior; validate against DIFFERENT observations.
- **Pre-register predictions** (git-timestamp) before experiments.
- Anchor: swelling-hydrogel (free-swell det F=g³; constrained → Biot buckling
  wavelength) — cheap, self-doable, physical analog of the growth kinematics.
- Report predictions as UQ bands (propagate calibration covariance), not points.

## Build order (implement INTO the frame, one gate at a time)
1. ✅ [2] constitutive (neo-Hookean, Hencky+manual SVD adjoint).
2. ✅ [3] single-step transfer + assembly gate.
3. ✅ [1]+[0] State/Config dataclasses; modules re-homed into package.
4. ✅ [4] trajectory rollout + adjoint (compose per-step VJPs); FD-gated.
5. ✅ [6] inverse design demo (recover growth rate from observed shape) — DONE.
6. ✅ [2] Herschel-Bulkley (bioink) constitutive + VJP — DONE (plugged in, zero core change).
7. → validation: hydrogel pre-registered predict/verify.
8. → (only if scale) production substrate port.

## Design note — bioink (Herschel-Bulkley) needs a rate-carrying seam
The current [2] seam is `stress(F, Fg)` — pure **hyperelastic** (stress from
deformation). Bioinks are **yield-stress / rate-dependent** fluids:
`τ = -p(J)·I + 2·η(γ̇)·D`, with apparent viscosity `η = τ_y/γ̇ + K·γ̇^{n-1}` and
strain-rate `D = sym(∇v) ≈ sym(C)`. So the seam must be extended to
`stress(F, Fg, C)` (C = APIC affine = velocity gradient) and its VJP must return
a `C_bar` stress-path contribution (transfer's C currently only feeds APIC
momentum). Decision: extend the seam signature (C optional; elastic models ignore
it). A rate-INDEPENDENT first step (von Mises / Bingham yield **return-map** on the
Hencky strain, seam-compatible `stress(F,Fg)` + F-plastic-update) can precede the
full rate-dependent HB. Either way the return-map/rate VJP has a yield kink
(nonsmooth) — FD-gate away from the kink; clamp/​smooth at it. **DONE 2026-07-01** (HerschelBulkley: rate-dependent form, C_bar path, FD-gated).

## Explicitly OUT / deferred (anti-scope-creep — the premature-breadth lesson)
- UI/GUI, broad app features → §9, last, only with users.
- Broad compatibility → achieved by STANDARD formats, not by building integrations.
- Granular/Drucker-Prager, GPU, ROS2, URDF, editor → not this project.
- Reaction-diffusion, neural constitutive, cell biology → designed seams, coded
  only after [4]+[6] work and real data exists.
- Rule: a seam is built when a **second** consumer appears, not preemptively.
```
