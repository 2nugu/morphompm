# Pre-registration — Hydrogel swelling as the growth-kinematics anchor

**Purpose.** Start the validation spine with a cheap, self-doable, PHYSICAL
anchor for the morphoelastic growth kinematics — and do it as a **prediction
registered before measurement** (git-timestamp this file), so it can never
degrade into retrodiction (fitting the sim to data already seen).

A swelling hydrogel *is* isotropic morphoelastic growth: solvent uptake expands
the stress-free reference (Fg = g·I). This validates the growth math (not cell
biology) at ~$10 / hours cost.

## Calibration inputs vs validation targets — kept strictly separate

**Calibrate (measured, fed INTO the sim):**
- C1. Free-swelling volumetric ratio Λ = V_swollen / V_dry (caliper/photo + balance)
  → sets the growth stretch g = Λ^(1/3).
- C2. Shear modulus (or E, ν) of the swollen gel (indentation / simple compression).

**Predict (sim output, computed & registered BEFORE the validation experiment):**
- P1. Free-swelling: an unconstrained gel cube expands by Λ with **zero residual
  stress** (sim: mean det F = g³, ‖ε_e‖→0 — already verified numerically).
- P2. **Constrained swelling (gel bonded to a rigid substrate) buckles into surface
  wrinkles of wavelength λ.** The sim predicts λ from (g, modulus) — a quantity
  NOT used in calibration. Analytic cross-check: Biot surface-instability
  wavelength for a swelling film on substrate.

**Validate (measured AFTER, compared to registered P1/P2):**
- V1. Measured swollen volume ratio vs P1 (sanity; Λ is calibration so this is weak).
- V2. **Measured wrinkle wavelength λ_exp vs predicted λ_sim** — the real test.
  Calibration (Λ, modulus) ≠ validation target (λ). Not circular.

## Cardinal rule (anti-retrodiction)
The wrinkle wavelength λ is **never** used to calibrate the sim. It is predicted
from independently-measured (Λ, modulus), registered here, then measured. If
λ_sim ∉ [λ_exp ± uncertainty], the growth model is falsified in this regime.

## Prediction protocol (fill BEFORE the experiment)
1. Measure C1, C2 on the specific gel batch; record with timestamp.
2. Run sim (constrained-swell scenario) → record P1, P2 as UQ bands (propagate
   C1/C2 measurement uncertainty), commit this file.
3. Run the constrained-swell experiment; photograph wrinkles; measure λ_exp.
4. Compare V2 to the committed P2. Report agreement or falsification verbatim.

## Status
- Sim capability for P1: **ready** (free-swell det F = g³ verified).
- Sim capability for P2: needs constrained-swell scenario + wrinkle-wavelength
  readout + Biot analytic cross-check (scheduled).
- Experiment: pending gel batch + substrate (self-doable or neighbor lab).

## Scope honesty
This validates the **growth kinematics** (Fg volumetric + residual-stress buckling)
against a deterministic physical analog. It does NOT validate cell biology, rate-
dependent bioink rheology, or printing — those are separate, later pre-registrations.
