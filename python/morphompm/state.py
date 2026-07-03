"""[1] State — particle data model (the backbone the transfer/integrator read).

Arrays, not per-particle objects, so it serializes to standard formats (npy/HDF5)
and ports to array frameworks later. Grid is allocated by the transfer stage.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class ParticleState:
    x: np.ndarray    # (n, 3) position
    v: np.ndarray    # (n, 3) velocity
    C: np.ndarray    # (n, 3, 3) APIC affine
    F: np.ndarray    # (n, 3, 3) total deformation gradient
    Fg: np.ndarray   # (n, 3, 3) growth tensor

    @property
    def n(self) -> int:
        return self.x.shape[0]

    @staticmethod
    def rest(x0: np.ndarray) -> "ParticleState":
        """Particles at rest: v=C=0, F=Fg=I."""
        n = x0.shape[0]
        I = np.tile(np.eye(3), (n, 1, 1))
        return ParticleState(x=x0.copy(),
                             v=np.zeros((n, 3)),
                             C=np.zeros((n, 3, 3)),
                             F=I.copy(),
                             Fg=I.copy())
