"""[8] I/O — canonical state/config <-> STANDARD formats.

Interoperability by standards, not N x M integrations: every format is an adapter
to/from ONE canonical form (ParticleState / SimConfig). Extensible — a new format
is one save_/load_ branch. Formats with a real consumer now:
    .npz   lossless canonical (numpy)            [load + save]
    .csv   human-readable / spreadsheet interop  [load + save]
    .json  config (material + discretization)    [load + save]
    .vtk   ParaView / VisIt visualization        [save; point fields det F, det Fg]
Literature curves (rheometry / measured wavelength) load via load_curve().
ply/obj/hdf5 etc. plug in here when an actual input needs them.
"""
import csv
import json
import os
import numpy as np

from .state import ParticleState
from .config import SimConfig, Material

_CCOLS = [f"C{i}{j}" for i in range(3) for j in range(3)]
_FCOLS = [f"F{i}{j}" for i in range(3) for j in range(3)]
_GCOLS = [f"Fg{i}{j}" for i in range(3) for j in range(3)]
# CSV is LOSSLESS (carries APIC affine C too) so a mid-trajectory checkpoint
# survives round-trip; .npz remains the compact binary canonical form.
_CSV_HEADER = ["x", "y", "z", "vx", "vy", "vz"] + _CCOLS + _FCOLS + _GCOLS


# ── ParticleState <-> file (dispatch on extension) ───────────────────────────
def save_state(state, path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        np.savez(path, x=state.x, v=state.v, C=state.C, F=state.F, Fg=state.Fg)
    elif ext == ".csv":
        _save_csv(state, path)
    elif ext == ".vtk":
        _save_vtk(state, path)
    else:
        raise ValueError(f"unsupported save format '{ext}' (use .npz/.csv/.vtk)")


def load_state(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        d = np.load(path)
        return ParticleState(d["x"], d["v"], d["C"], d["F"], d["Fg"])
    if ext == ".csv":
        return _load_csv(path)
    raise ValueError(f"unsupported load format '{ext}' (use .npz/.csv)")


def _save_csv(state, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(_CSV_HEADER)
        for p in range(state.n):
            row = list(state.x[p]) + list(state.v[p]) + list(state.C[p].ravel()) \
                + list(state.F[p].ravel()) + list(state.Fg[p].ravel())
            w.writerow([f"{v:.9g}" for v in row])


def _load_csv(path):
    xs, vs, Cs, Fs, Fgs = [], [], [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            xs.append([float(r[c]) for c in ("x", "y", "z")])
            vs.append([float(r[c]) for c in ("vx", "vy", "vz")])
            Cs.append([float(r[c]) for c in _CCOLS])
            Fs.append([float(r[c]) for c in _FCOLS])
            Fgs.append([float(r[c]) for c in _GCOLS])
    n = len(xs)
    return ParticleState(np.array(xs), np.array(vs),
                         np.array(Cs).reshape(n, 3, 3),
                         np.array(Fs).reshape(n, 3, 3),
                         np.array(Fgs).reshape(n, 3, 3))


def _save_vtk(state, path):
    """Legacy-ASCII VTK POLYDATA — points + scalar fields (ParaView-ready)."""
    n = state.n
    detF = np.array([np.linalg.det(state.F[p]) for p in range(n)])
    detFg = np.array([np.linalg.det(state.Fg[p]) for p in range(n)])
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nmorphompm particles\nASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n")
        for p in range(n):
            f.write(f"{state.x[p,0]:.6g} {state.x[p,1]:.6g} {state.x[p,2]:.6g}\n")
        f.write(f"VERTICES {n} {2*n}\n")
        for p in range(n):
            f.write(f"1 {p}\n")
        f.write(f"POINT_DATA {n}\n")
        for name, arr in (("detF", detF), ("detFg", detFg)):
            f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
            for v in arr:
                f.write(f"{v:.6g}\n")


# ── SimConfig <-> json ───────────────────────────────────────────────────────
def save_config(cfg, path):
    with open(path, "w") as f:
        json.dump({"N": cfg.N, "dx": cfg.dx, "dt": cfg.dt, "damping": cfg.damping,
                   "sp": cfg.sp, "material": {"E": cfg.material.E, "nu": cfg.material.nu,
                                              "density": cfg.material.density}}, f, indent=2)


def load_config(path):
    with open(path) as f:
        d = json.load(f)
    m = d.get("material", {})
    return SimConfig(N=d["N"], dx=d["dx"], dt=d["dt"], damping=d["damping"], sp=d["sp"],
                     material=Material(E=m.get("E", 1e4), nu=m.get("nu", 0.3),
                                       density=m.get("density", 1000.0)))


# ── literature/experimental curves (calibration + validation data) ───────────
def load_curve(path, xcol=0, ycol=1):
    """Two-column numeric CSV (e.g. digitized rheometry or measured wavelength)."""
    rows = []
    with open(path) as f:
        for r in csv.reader(f):
            try:
                rows.append((float(r[xcol]), float(r[ycol])))
            except (ValueError, IndexError):
                continue                                   # skip header/blank
    a = np.array(rows)
    return a[:, 0], a[:, 1]


# ── round-trip correctness gate ──────────────────────────────────────────────
def main():
    print("== I/O round-trip: canonical <-> standard formats ==")
    rng = np.random.default_rng(0)
    n = 20
    st = ParticleState(rng.standard_normal((n, 3)), rng.standard_normal((n, 3)),
                       rng.standard_normal((n, 3, 3)),
                       np.tile(np.eye(3), (n, 1, 1)) + 0.05 * rng.standard_normal((n, 3, 3)),
                       np.tile(1.2 * np.eye(3), (n, 1, 1)))
    import tempfile
    fails = 0
    d = tempfile.mkdtemp()
    for ext, atol in ((".npz", 0.0), (".csv", 1e-6)):
        p = os.path.join(d, "s" + ext)
        save_state(st, p); back = load_state(p)
        ok = all(np.allclose(getattr(st, a), getattr(back, a), atol=atol)
                 for a in ("x", "v", "C", "F", "Fg"))          # ALL fields (C included)
        print(f"    {ext}: round-trip x/v/C/F/Fg {'exact' if atol==0 else f'<{atol:g}'}  [{'ok' if ok else 'FAIL'}]")
        fails += not ok
    # config round-trip
    cfg = SimConfig(N=12, dx=0.05); save_config(cfg, os.path.join(d, "c.json"))
    c2 = load_config(os.path.join(d, "c.json"))
    okc = (c2.N == cfg.N and abs(c2.material.mu - cfg.material.mu) < 1e-9)
    print(f"    .json: config round-trip  [{'ok' if okc else 'FAIL'}]")
    fails += not okc
    # vtk writes (ParaView-readable) — smoke
    save_state(st, os.path.join(d, "s.vtk"))
    okv = os.path.getsize(os.path.join(d, "s.vtk")) > 0
    print(f"    .vtk: ParaView export written  [{'ok' if okv else 'FAIL'}]")
    fails += not okv
    print("\nALL CHECKS PASSED" if fails == 0 else f"\n{fails} CHECK(S) FAILED")
    return fails


if __name__ == "__main__":
    raise SystemExit(0 if main() == 0 else 1)
