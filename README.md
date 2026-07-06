# morphompm

**A differentiable morphoelastic growth-MPM for soft / living matter.**

[![DOI](https://zenodo.org/badge/1290685748.svg)](https://doi.org/10.5281/zenodo.21215516)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

A small, heavily-verified [Material Point Method](https://en.wikipedia.org/wiki/Material_point_method)
solver for *morphoelastic growth* (`F = Fe·Fg`) with hand-written, finite-difference-gated
adjoints — so you can **infer material and growth laws from an observed shape** by
differentiating through the simulation.

> **Status: research work-in-progress.** The differentiable core is verified along three
> independent axes and is fully reproducible. It has **not** yet been validated against
> experimental data — that (against *published* morphogenesis datasets) is the next phase
> (`docs/PHASE3_morphogenesis.md`). Treat this as a verified method, not a validated result.

<p align="center">
  <img src="outputs/figures/inverse_recovery.png" width="46%" alt="inverse growth-rate recovery">
  <img src="outputs/figures/differential_growth.png" width="46%" alt="differential-growth bending">
</p>

## What you can use it for

Concretely, the pipeline lets you:

- **Infer an unknown growth law from a shape.** Given an observed final geometry, recover
  the growth parameter that produced it by differentiating through the simulation
  (`morphompm.diff.infer_growth`) — the headline use case.
- **Predict morphogenesis forward.** Prescribe a differential-growth field and see the
  resulting bending, swelling, wrinkling, and residual stress (e.g. the bilayer strip that
  curls, validated against the Timoshenko analytic solution).
- **Fit soft-material / bioink parameters.** Because the constitutive map is a pluggable
  seam with a hand-written adjoint, you get gradients of any deformation w.r.t. its material
  parameters — usable for calibration or gradient-based design.
- **Drop in your own constitutive model** at the `constitutive.py` seam and get its VJP
  gated against finite differences, then reuse the same transfer/trajectory/inverse machinery
  unchanged.
- **Use it as a verified reference.** A small, deterministic, three-axis-verified
  growth-MPM with cross-checked C++/numpy implementations — a teaching or benchmarking
  substrate for differentiable MPM and hand-written matrix-kernel adjoints.

> **Scope today:** all of the above run on *simulated* observations. Inference against
> real experimental data is the next phase (`docs/PHASE3_morphogenesis.md`).

## How it works

- **Forward:** MLS-MPM with morphoelastic growth. Constitutive models are pluggable
  (neo-Hookean, Hencky, a rate-dependent Herschel-Bulkley bioink) and injected into a
  material-agnostic transfer.
- **Adjoint:** every gradient (constitutive VJP, one MPM step incl. particle advection,
  full trajectory) is **hand-written and gated against finite differences** — not left to
  a framework's autodiff (which we found segfaults on these matrix-heavy kernels).
- **Inverse:** compose the module adjoints → recover a growth parameter from an observed
  final shape by gradient descent (the intended contribution).

## Install

```bash
pip install -e .            # installs the `morphompm` package (numpy, matplotlib)
```

The C++ forward oracle (optional, used for cross-checking) builds with CMake:

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64   # or your generator
cmake --build build --config Release
```

## Quickstart

```bash
python -m morphompm.verify     # run every verification gate (~seconds)
python scripts/reproduce.py    # one command: verify + C++ oracle + parity + regenerate figures/dashboard
```

## Verification (three independent axes + cross-implementation parity)

| Axis | What it guards | Typical result |
|------|----------------|----------------|
| FD-gradient gates | adjoint ↔ forward consistency | rel.err ≤ 1e-9 |
| forward-physics guard | forward *correctness* (free-swell `det F → g³`) | matches analytic |
| confined-swell analytic oracle | constitutive coefficients (catches μ↔λ swaps) | exact |
| numpy ↔ C++ parity | the two implementations agree | `max|ΔF|` ≈ 2e-5 |

FD gates verify *consistency*, not *correctness* — so the forward and analytic oracles are
separate axes (this split has caught real bugs). Runs are deterministic (single-thread).

## Repository layout

```
python/morphompm/     the package — config, state, constitutive, transfer, integrate, diff, io, verify
python/experiments/   Taichi ports (superseded; kept for the record — see banners)
include/morphompm/     C++ forward-oracle solver + self-contained linear-algebra core (math/)
tests/                 C++ oracle tests (analytic + Timoshenko bending)
scripts/               reproduce, parity, results (figures), make_dashboard, run_scene
docs/                  PROJECT_STATUS, PIPELINE (architecture), PHASE3 (roadmap), PREREG
outputs/figures/       preserved result figures (.png + .csv + _desc.md)
```

## Design notes

- **PIPELINE.md** — end-to-end architecture, module seams, substrate strategy, verification/validation spines.
- **PROJECT_STATUS.md** — current phase, decisions, honest open items.
- **PHASE3_morphogenesis.md** — the next stage: validate + invert against published morphogenesis data.
- Guiding principle: *deliverable-first* — build the minimum that a verifiable result needs;
  extend a seam only when a second consumer appears; keep every claim gated.

> Any literature values referenced in the docs are pointers to check against the primary
> source, not verified numbers.

## Roadmap & contributing

See **[`ROADMAP.md`](ROADMAP.md)** for the plan, the honest list of open problems (what
is *not* solved yet), and a [where-to-help](ROADMAP.md#where-to-help) guide mapped to
labelled issues (`good first issue`, `help wanted`, `open-problem`). Contributions —
new constitutive models at the seam, tests, scenarios, or validation against published
data — are welcome. 🤝

## Citation & license

See `CITATION.cff`. Licensed under the MIT License (`LICENSE`).
