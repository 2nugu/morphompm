# EXPERIMENTAL / SUPERSEDED — not part of the morphompm package. Taichi autodiff
# segfaults on matrix-MPM adjoints (see docs/PIPELINE.md); superseded by the numpy
# package (morphompm.integrate/diff). Kept for the record; do NOT build on this.

"""morphompm — Taichi port of the morphoelastic growth MLS-MPM (v1 forward),
plus the differentiability the C++ path would have to hand-roll.

Two checks:
  (A) forward parity — free isotropic swelling reaches det(F) -> g^3, matching
      the verified C++ oracle (3.3750 for g=1.5).
  (B) differentiability — autodiff gradient of the growth-aware Hencky energy
      w.r.t. the growth stretch g, cross-checked against finite difference.
      This is the SVD-through-growth gradient the C++ route would hand-code.

Weight / APIC / growth-stress conventions are identical to the C++ kernel, so
(A) is a genuine cross-implementation parity test, not a re-derivation.
"""
import numpy as np
import taichi as ti

ti.init(arch=ti.cpu, default_fp=ti.f64)

# ── params (match C++ oracle exactly) ────────────────────────────────────────
N, dx = 24, 0.05
inv_dx = 1.0 / dx
E, nu = 1.0e4, 0.3
mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))
damping, dt = 0.05, 1.0e-3
density = 1000.0

# ── seed centered cube blob (same as C++ seed_blob) ──────────────────────────
center, half, sp = 0.6, 0.10, 0.025
coords = np.arange(center - half, center + half + 1e-9, sp)
pts = np.array([(a, b, c) for a in coords for b in coords for c in coords], dtype=np.float64)
n = len(pts)
pmass = density * sp ** 3
vol0 = pmass / density

x = ti.Vector.field(3, ti.f64, n)
v = ti.Vector.field(3, ti.f64, n)
F = ti.Matrix.field(3, 3, ti.f64, n)
C = ti.Matrix.field(3, 3, ti.f64, n)
grid_v = ti.Vector.field(3, ti.f64, (N, N, N))
grid_m = ti.field(ti.f64, (N, N, N))
gg = ti.field(ti.f64, ())          # global isotropic growth stretch


@ti.kernel
def reset():
    for p in x:
        v[p] = ti.Vector([0.0, 0.0, 0.0])
        F[p] = ti.Matrix.identity(ti.f64, 3)
        C[p] = ti.Matrix.zero(ti.f64, 3, 3)


@ti.func
def hencky_tau(Fp, g):
    Fe = Fp * (1.0 / g)                      # isotropic Fg = g·I  ->  Fe = F/g
    U, sig, V = ti.svd(Fe)
    e0 = ti.log(ti.max(sig[0, 0], 1e-9))
    e1 = ti.log(ti.max(sig[1, 1], 1e-9))
    e2 = ti.log(ti.max(sig[2, 2], 1e-9))
    tr = e0 + e1 + e2
    Td = ti.Matrix([[2 * mu * e0 + lam * tr, 0.0, 0.0],
                    [0.0, 2 * mu * e1 + lam * tr, 0.0],
                    [0.0, 0.0, 2 * mu * e2 + lam * tr]])
    return U @ Td @ U.transpose()


@ti.kernel
def clear_grid():
    for I in ti.grouped(grid_m):
        grid_m[I] = 0.0
        grid_v[I] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def p2g():
    g = gg[None]
    volg = vol0 * g * g * g
    for p in x:
        Xp = x[p] * inv_dx
        base = ti.floor(Xp - 0.5).cast(ti.i32)
        fx = Xp - base.cast(ti.f64)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        sk = hencky_tau(F[p], g) * (-4.0 * inv_dx * inv_dx * volg)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offs = ti.Vector([i, j, k])
            dpos = (offs.cast(ti.f64) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            idx = base + offs
            grid_m[idx] += weight * pmass
            # APIC momentum + folded internal-force impulse (force·dt).
            grid_v[idx] += weight * (pmass * (v[p] + C[p] @ dpos) + (sk @ dpos) * dt)


@ti.kernel
def grid_update():
    for I in ti.grouped(grid_m):
        if grid_m[I] > 1e-12:
            vel = grid_v[I] / grid_m[I]      # no gravity (isolate growth)
            grid_v[I] = vel * (1.0 - damping)


@ti.kernel
def g2p():
    for p in x:
        Xp = x[p] * inv_dx
        base = ti.floor(Xp - 0.5).cast(ti.i32)
        fx = Xp - base.cast(ti.f64)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        newv = ti.Vector.zero(ti.f64, 3)
        newC = ti.Matrix.zero(ti.f64, 3, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offs = ti.Vector([i, j, k])
            dpos = (offs.cast(ti.f64) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            gv = grid_v[base + offs]
            newv += weight * gv
            newC += 4.0 * inv_dx * inv_dx * weight * gv.outer_product(dpos)
        v[p] = newv
        C[p] = newC
        x[p] += dt * newv
        F[p] = (ti.Matrix.identity(ti.f64, 3) + dt * newC) @ F[p]


def substep():
    clear_grid(); p2g(); grid_update(); g2p()


# ── (A) forward parity vs C++ oracle ─────────────────────────────────────────
def check_forward():
    reset()
    x.from_numpy(pts)
    gg[None] = 1.0
    K, M, g_target = 10, 250, 1.5
    per = g_target ** (1.0 / K)
    for _ in range(K):
        gg[None] *= per
        for _ in range(M):
            substep()
    dets = np.linalg.det(F.to_numpy())
    mean_det = float(dets.mean())
    expected = g_target ** 3
    print(f"[A] forward parity (free swelling)")
    print(f"    {n} particles; mean det(F) = {mean_det:.4f}  "
          f"(C++ oracle = 3.3750, g^3 = {expected:.4f})")
    ok = abs(mean_det - expected) < 0.08 * expected
    print(f"    [{'ok' if ok else 'FAIL'}] Taichi forward matches C++ oracle within 8%")
    return ok


# ── (B) differentiability: dΨ/dg, autodiff vs finite difference ──────────────
# NOTE: stock Taichi autodiff does NOT support ti.svd (confirmed: MakeAdjoint
# "Not supported"). The Hencky/Drucker-Prager path therefore needs a CUSTOM SVD
# adjoint (Jiang et al.) — a known, scoped next step. Here we demonstrate that
# GROWTH is differentiable using an SVD-free neo-Hookean energy (det + trace
# only), which is the standard differentiable-MPM workaround.
#   Ψ_neo(Fe) = ½μ(I1 − 3) − μ·ln J + ½λ(ln J)²,   Fe = F·Fg⁻¹,  Fg = g·I.
g_ad = ti.field(ti.f64, (), needs_grad=True)
psi = ti.field(ti.f64, (), needs_grad=True)


@ti.kernel
def energy():
    # Single trivial loop: Taichi autodiff forbids mixing loop / non-loop
    # statements at kernel top level.
    for _ in range(1):
        Fp = ti.Matrix([[1.10, 0.02, 0.0],
                        [0.0, 1.00, 0.03],
                        [0.01, 0.0, 0.95]])      # fixed deformed state
        Fe = Fp * (1.0 / g_ad[None])             # growth enters here
        J = Fe.determinant()
        I1 = (Fe.transpose() @ Fe).trace()
        lnJ = ti.log(J)
        psi[None] = 0.5 * mu * (I1 - 3.0) - mu * lnJ + 0.5 * lam * lnJ * lnJ


def check_grad():
    print(f"[B] differentiability (dΨ/dg, SVD-free neo-Hookean growth energy)")
    g0 = 1.30
    g_ad[None] = g0
    with ti.ad.Tape(psi):
        energy()
    grad_ad = g_ad.grad[None]

    def L(gv):
        g_ad[None] = gv
        energy()
        return psi[None]

    eps = 1e-6
    grad_fd = (L(g0 + eps) - L(g0 - eps)) / (2 * eps)
    rel = abs(grad_ad - grad_fd) / (abs(grad_fd) + 1e-30)
    print(f"    autodiff dΨ/dg = {grad_ad:.6f}")
    print(f"    finite-diff   = {grad_fd:.6f}   (rel.err = {rel:.2e})")
    ok = rel < 1e-4
    print(f"    [{'ok' if ok else 'FAIL'}] autodiff gradient matches FD (growth is differentiable; SVD-free)")
    return ok


if __name__ == "__main__":
    print("== morphompm Taichi: forward parity + differentiability ==\n")
    a = check_forward()
    print()
    b = check_grad()
    print()
    print("ALL CHECKS PASSED" if (a and b) else "SOME CHECKS FAILED")
    raise SystemExit(0 if (a and b) else 1)
