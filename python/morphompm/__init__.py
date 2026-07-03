"""morphompm — differentiable morphoelastic growth-MPM for soft/living matter.

Package stages (see ../../docs/PIPELINE.md):
  config       [0] material / sim parameters (single source of constants)
  state        [1] particle data model (ParticleState)
  constitutive [2] pluggable material models: stress + manual VJP
  transfer     [3] one MLS-MPM step + manual VJP (model-injected, growth-agnostic)
  integrate    [4] trajectory rollout + adjoint (compose per-step VJPs)
  diff         [6] inverse: recover growth from observed shape
  verify       [7] verification harness (runs all gates)
"""
from . import config, state, constitutive, transfer, integrate, diff, io  # noqa: F401

__all__ = ["config", "state", "constitutive", "transfer", "integrate", "diff", "io"]
