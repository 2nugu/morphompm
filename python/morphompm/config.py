"""[0] Config — material + discretization. Single source of physical constants
(no more module-level globals scattered across stages)."""
from dataclasses import dataclass, field


@dataclass
class Material:
    E: float = 1.0e4          # Young's modulus [Pa]
    nu: float = 0.3           # Poisson ratio
    density: float = 1000.0   # [kg/m^3]

    @property
    def mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self) -> float:
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


@dataclass
class SimConfig:
    N: int = 8                 # dense grid resolution per axis
    dx: float = 0.1
    dt: float = 1.0e-3
    damping: float = 0.05      # quasi-static relaxation per step
    sp: float = 0.02           # particle spacing (sets mass/volume)
    material: Material = field(default_factory=Material)

    @property
    def inv_dx(self) -> float:
        return 1.0 / self.dx

    @property
    def pmass(self) -> float:
        return self.material.density * self.sp ** 3

    @property
    def vol0(self) -> float:
        return self.sp ** 3    # = pmass / density
