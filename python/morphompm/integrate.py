"""[4] Integrate — trajectory rollout + adjoint, composing per-step VJPs.

rollout(state0, model, cfg, n_steps, advect)      -> final state, caches
rollout_vjp(caches, final_bars..., advect)        -> state cotangents (incl. Fg_bar)

Growth stays a parameter of the caller: build state with Fg = g·I, then
iso_growth_grad(Fg_bar) = Σ tr(Fg_bar) gives d/dg (the trivial isotropic driver;
a field g(x) or learned law would replace only this line).
"""
import numpy as np
from .state import ParticleState
from .transfer import step, step_vjp


def rollout(state0, model, cfg, n_steps, advect=False):
    st = state0
    caches = []
    for _ in range(n_steps):
        st, cache = step(st, model, cfg, advect=advect)
        caches.append(cache)
    return st, caches


def rollout_vjp(caches, xf_b, vf_b, Cf_b, Ff_b, Fgf_b, advect=False):
    xb, vb, Cb, Fb, Fgb = xf_b, vf_b, Cf_b, Ff_b, Fgf_b
    for cache in reversed(caches):
        xb, vb, Cb, Fb, Fgb = step_vjp(cache, xb, vb, Cb, Fb, Fgb, advect=advect)
    return xb, vb, Cb, Fb, Fgb


def iso_growth_state(pts, g):
    st = ParticleState.rest(pts)
    st.Fg[:] = g * np.eye(3)
    return st


def iso_growth_grad(Fg_b):
    return float(sum(np.trace(Fg_b[p]) for p in range(len(Fg_b))))


def loss_detF(F):
    return float(sum(np.linalg.det(F[p]) for p in range(len(F))))


def dloss_dg(pts, model, cfg, g, n_steps, advect):
    """d(Σ det F_final)/dg via composed trajectory adjoint."""
    st, caches = rollout(iso_growth_state(pts, g), model, cfg, n_steps, advect=advect)
    n = st.n
    Ff_b = np.stack([np.linalg.det(st.F[p]) * np.linalg.inv(st.F[p]).T for p in range(n)])
    z3, z33 = np.zeros((n, 3)), np.zeros((n, 3, 3))
    _, _, _, _, Fg_b = rollout_vjp(caches, z3, z3, z33, Ff_b, z33, advect=advect)
    return loss_detF(st.F), iso_growth_grad(Fg_b)


def _gate(pts, model, cfg, g0, n_steps, advect):
    _, dg = dloss_dg(pts, model, cfg, g0, n_steps, advect)
    eps = 1e-6
    fp, _ = dloss_dg(pts, model, cfg, g0 + eps, n_steps, advect)
    fm, _ = dloss_dg(pts, model, cfg, g0 - eps, n_steps, advect)
    fd = (fp - fm) / (2 * eps)
    rel = abs(dg - fd) / (abs(fd) + 1e-30)
    tag = "advect=True (moving)" if advect else "advect=False (fixed)"
    print(f"    [{tag}] {n_steps} steps: adjoint={dg:.6e}  FD={fd:.6e}  rel.err={rel:.2e}")
    return rel < 1e-5


def forward_physics_gate():
    """[1p] FORWARD-physics guard (NOT gradient): free isotropic swelling must
    expand toward det F = g^3 (C++/analytic oracle). FD gates can't catch a wrong
    forward — this can (it caught the HB pressure-sign bug, 2026-07-01)."""
    from .config import SimConfig
    from .constitutive import NeoHookean, Hencky, HerschelBulkley
    cfg = SimConfig(N=12, dx=0.05, dt=1.0e-3, damping=0.05, sp=0.03)
    g, n_steps = 1.5, 150
    c = np.arange(0.30 - 0.015, 0.30 + 0.015 + 1e-9, 0.03)
    pts = np.array([(x, y, z) for x in c for y in c for z in c])   # 8-particle cube
    print("== [1p] forward physics: free-swell det F -> g^3 (guards forward correctness) ==")
    ok = True
    # Hencky included: it is the law the C++ oracle runs (numpy↔C++ law consistency).
    for name, model in [("NeoHookean", NeoHookean(cfg.material)),
                        ("Hencky", Hencky(cfg.material)),
                        ("HerschelBulkley", HerschelBulkley())]:
        st = iso_growth_state(pts, g)
        st, _ = rollout(st, model, cfg, n_steps, advect=True)
        d = float(np.mean([np.linalg.det(st.F[p]) for p in range(st.n)]))
        good = np.isfinite(d) and 1.5 < d < 2.0 * g ** 3     # expanding, not collapsed/diverged
        print(f"    {name}: mean det F = {d:.4f} (toward g^3={g**3:.3f})  [{'ok' if good else 'FAIL'}]")
        ok &= good
    return 0 if ok else 1


def determinism_gate():
    """Reproducibility pillar: identical numpy runs must be bit-identical (single
    thread, no RNG in the physics). Guards the determinism claim in Python."""
    from .config import SimConfig
    from .constitutive import NeoHookean
    cfg = SimConfig()
    model = NeoHookean(cfg.material)
    pts = np.array([[0.40, 0.40, 0.40], [0.42, 0.40, 0.41], [0.40, 0.43, 0.40],
                    [0.41, 0.40, 0.44], [0.44, 0.44, 0.43]])

    def run():
        st, _ = rollout(iso_growth_state(pts, 1.3), model, cfg, 3, advect=True)
        return st.F.copy()

    a, b = run(), run()
    d = float(np.max(np.abs(a - b)))
    print("== [D] determinism: identical numpy runs bit-identical ==")
    print(f"    max|F_run1 - F_run2| = {d:.1e}  [{'ok' if d == 0.0 else 'FAIL'}]")
    return 0 if d == 0.0 else 1


def main():
    from .config import SimConfig
    from .constitutive import NeoHookean, HerschelBulkley
    cfg = SimConfig()
    pts = np.array([[0.40, 0.40, 0.40], [0.42, 0.40, 0.41], [0.40, 0.43, 0.40],
                    [0.41, 0.40, 0.44], [0.44, 0.44, 0.43]])
    print("== integrate: multi-step trajectory assembly gate ==")
    nh = NeoHookean(cfg.material)
    ok = _gate(pts, nh, cfg, 1.30, 3, advect=False) & _gate(pts, nh, cfg, 1.30, 3, advect=True)
    # bioink model plugged into the SAME transfer/integrate (no core change);
    # after step 1 C≠0 → activates the C_bar path through the transfer adjoint.
    print("    -- HerschelBulkley (bioink) through the same pipeline --")
    hb = HerschelBulkley()
    ok &= _gate(pts, hb, cfg, 1.30, 3, advect=False)
    print(f"    [{'ok' if ok else 'FAIL'}] trajectory gradient matches FD (elastic + bioink)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
