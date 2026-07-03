"""Logic audit — beyond 'FD gates pass'.

FD gates only prove adjoint-consistency with the forward. They do NOT prove the
forward is physically correct, nor cover the hard regimes. This script checks:
  L1  numpy neo-Hookean free-swell forward vs C++ oracle (det F -> g^3).
  L2  HB free-swell forward: does it expand & equilibrate (pressure SIGN right)?
  L3  HB VJP FD near the yield regime (γ̇ -> 0) — the defining bioink behavior.
  L4  Hencky SVD adjoint FD near degenerate singular values (isotropic-ish).
"""
import numpy as np
from morphompm.config import SimConfig
from morphompm.constitutive import NeoHookean, HerschelBulkley, Hencky, _frob
from morphompm.state import ParticleState
from morphompm.integrate import rollout


def blob(center, half, sp):
    c = np.arange(center - half, center + half + 1e-9, sp)
    return np.array([(x, y, z) for x in c for y in c for z in c])


def free_swell_detF(model, cfg, g, n_steps):
    pts = blob(0.40, 0.03, 0.03)               # 27-particle cube
    st = ParticleState.rest(pts)
    st.Fg[:] = g * np.eye(3)
    st, _ = rollout(st, model, cfg, n_steps, advect=True)
    return float(np.mean([np.linalg.det(st.F[p]) for p in range(st.n)])), st.n


def vjp_rel_err(model, F, Fg, C, rng, eps=1e-7):
    tb = rng.standard_normal((3, 3)); tb = 0.5 * (tb + tb.T)
    Fb, Fgb, Cb = model.stress_vjp(F, Fg, C, tb)
    dF = rng.standard_normal((3, 3)); dFg = rng.standard_normal((3, 3))
    dC = rng.standard_normal((3, 3)) if C is not None else None
    Cp = C + eps * dC if C is not None else None
    Cm = C - eps * dC if C is not None else None
    fd = (_frob(tb, model.stress(F + eps * dF, Fg + eps * dFg, Cp))
          - _frob(tb, model.stress(F - eps * dF, Fg - eps * dFg, Cm))) / (2 * eps)
    an = _frob(Fb, dF) + _frob(Fgb, dFg) + (_frob(Cb, dC) if C is not None else 0.0)
    return abs(an - fd) / (abs(fd) + 1e-12)


if __name__ == "__main__":
    cfg = SimConfig(N=16, dx=0.05, dt=1.0e-3, damping=0.05, sp=0.03)
    g = 1.5
    rng = np.random.default_rng(1)
    print("== LOGIC AUDIT (forward physics + uncovered regimes) ==\n")

    # L1 — numpy forward vs C++ oracle
    d_nh, n = free_swell_detF(NeoHookean(cfg.material), cfg, g, 400)
    print(f"L1 neo-Hookean free-swell: mean det F = {d_nh:.4f}  (C++ oracle / g^3 = {g**3:.4f}, {n} particles)")
    print(f"   -> numpy FORWARD {'MATCHES' if abs(d_nh - g**3) < 0.1 * g**3 else 'DIVERGES FROM'} the verified physics\n")

    # L2 — HB free-swell (pressure sign / stability)
    d_hb, _ = free_swell_detF(HerschelBulkley(), cfg, g, 400)
    finite = np.isfinite(d_hb)
    print(f"L2 HB free-swell: mean det F = {d_hb:.4f}  (expect -> g^3 = {g**3:.4f} if pressure sign correct)")
    exp = finite and abs(d_hb - g**3) < 0.15 * g**3
    print(f"   -> HB forward {'expands & equilibrates (sign OK)' if exp else 'WRONG (sign/stability BUG)'}\n")

    # L3 — HB VJP near yield regime (γ̇ -> 0)
    F = np.diag([1.05, 0.98, 1.02]); Fg = np.diag([1.2, 1.0, 0.9])
    hb = HerschelBulkley()
    for scaleC in [1e-1, 1e-3, 1e-5]:
        errs = [vjp_rel_err(hb, F, Fg, scaleC * rng.standard_normal((3, 3)), rng) for _ in range(5)]
        print(f"L3 HB VJP FD, |C|~{scaleC:.0e}: worst rel.err = {max(errs):.2e}")
    print()

    # L4 — Hencky SVD adjoint near degeneracy (equal singular values)
    hk = Hencky(cfg.material)
    for d in [1e-1, 1e-3, 1e-6]:
        F = np.diag([1.0 + d, 1.0, 1.0 - d]); Fg = np.eye(3)     # s spread ~ 2d
        errs = [vjp_rel_err(hk, F, Fg, None, rng) for _ in range(5)]
        print(f"L4 Hencky SVD VJP FD, singular-value spread ~{2*d:.0e}: worst rel.err = {max(errs):.2e}")
