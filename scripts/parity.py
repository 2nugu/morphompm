"""numpy ↔ C++ trajectory PARITY gate — enforces the "numpy is the SPEC, C++ is the
oracle" contract that was previously only asserted in docs.

The C++ oracle (test_growth) dumps a small Hencky free-swell reference to
outputs/parity_ref.csv; here we run the IDENTICAL scenario in numpy (same grid,
particles, Hencky law, growth, steps) and compare the deformation gradient F.
Tolerance reflects C++ float32 vs numpy float64 accumulated over 30 steps.

    python scripts/parity.py      # requires outputs/parity_ref.csv (run C++ first)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import numpy as np
from morphompm.config import SimConfig
from morphompm.constitutive import Hencky
from morphompm.integrate import rollout, iso_growth_state

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "outputs", "parity_ref.csv")


def main():
    print("== numpy↔C++ parity: identical Hencky free-swell, compare F ==")
    if not os.path.exists(REF):
        print("    [skip] outputs/parity_ref.csv missing — run the C++ oracle first")
        print("           (build + ./build/Release/test_growth.exe, or scripts/reproduce.py)")
        return 0
    ref = np.genfromtxt(REF, delimiter=",", names=True)
    F_cpp = np.stack([np.array([[r[f"F{i}{j}"] for j in range(3)] for i in range(3)])
                      for r in ref])

    # IDENTICAL scenario in numpy (must match dump_parity_reference in test_growth.cpp)
    cfg = SimConfig(N=12, dx=0.05, dt=1.0e-3, damping=0.05, sp=0.03)
    model = Hencky(cfg.material)
    coords = np.arange(0.27, 0.33 + 1e-9, 0.03)
    pts = np.array([(x, y, z) for x in coords for y in coords for z in coords])
    st, _ = rollout(iso_growth_state(pts, 1.05), model, cfg, 30, advect=True)
    F_np = st.F

    assert F_np.shape[0] == F_cpp.shape[0], f"particle count mismatch {F_np.shape[0]} vs {F_cpp.shape[0]}"
    maxdF = float(np.max(np.abs(F_np - F_cpp)))
    det_np = float(np.mean([np.linalg.det(F_np[p]) for p in range(len(F_np))]))
    det_cpp = float(np.mean([np.linalg.det(F_cpp[p]) for p in range(len(F_cpp))]))
    reldet = abs(det_np - det_cpp) / (abs(det_cpp) + 1e-30)
    print(f"    {len(F_np)} particles; max|F_np - F_cpp| = {maxdF:.2e}  "
          f"mean detF: numpy {det_np:.5f} vs C++ {det_cpp:.5f} (rel {reldet:.2e})")
    ok = maxdF < 1e-2 and reldet < 1e-3          # float32(C++) vs float64(numpy), 30 steps
    print(f"    [{'ok' if ok else 'FAIL'}] numpy forward matches the C++ oracle (spec↔oracle contract)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
