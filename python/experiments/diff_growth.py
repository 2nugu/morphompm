# EXPERIMENTAL / SUPERSEDED — not part of the morphompm package. Taichi autodiff
# segfaults on matrix-MPM adjoints (see docs/PIPELINE.md); superseded by the numpy
# package (morphompm.integrate/diff). Kept for the record; do NOT build on this.

"""morphompm — differentiable trajectory: gradient through the full multi-step
growth-MPM sim, and inverse recovery of the growth rate from observed shape.

This is the real "differentiable growth-MPM" milestone: a gradient that flows
through every P2G/grid/G2P step (DiffTaichi time-indexed fields), not just the
constitutive map. Constitutive model = SVD-free neo-Hookean (Taichi autodiff
does not support ti.svd; the Hencky/DP path needs a custom SVD adjoint, deferred).

  (A) trajectory gradient: d(mean det F_final)/dg, autodiff vs finite difference.
  (B) inverse design: recover g_true from a target shape by gradient descent.
"""
import numpy as np
import taichi as ti

ti.init(arch=ti.cpu, default_fp=ti.f64)

# ── params ───────────────────────────────────────────────────────────────────
steps = 40
N, dx = 20, 0.05
inv_dx = 1.0 / dx
E, nu = 1.0e4, 0.3
mu0 = E / (2 * (1 + nu))
lam0 = E * nu / ((1 + nu) * (1 - 2 * nu))
dt, damping = 1.0e-3, 0.05
density = 1000.0

center, half, sp = 0.5, 0.05, 0.025
coords = np.arange(center - half, center + half + 1e-9, sp)
pts = np.array([(a, b, c) for a in coords for b in coords for c in coords], dtype=np.float64)
n = len(pts)
pmass = density * sp ** 3
vol0 = pmass / density

# time-indexed fields (DiffTaichi idiom) — every step writes a fresh slot.
x = ti.Vector.field(3, ti.f64, (steps, n), needs_grad=True)
v = ti.Vector.field(3, ti.f64, (steps, n), needs_grad=True)
C = ti.Matrix.field(3, 3, ti.f64, (steps, n), needs_grad=True)
F = ti.Matrix.field(3, 3, ti.f64, (steps, n), needs_grad=True)
grid_v = ti.Vector.field(3, ti.f64, (steps, N, N, N), needs_grad=True)
grid_m = ti.field(ti.f64, (steps, N, N, N), needs_grad=True)
g = ti.field(ti.f64, (), needs_grad=True)
loss = ti.field(ti.f64, (), needs_grad=True)

x0 = ti.Vector.field(3, ti.f64, n)   # initial positions (constant)


@ti.kernel
def set_initial():
    for p in range(n):
        x[0, p] = x0[p]
        v[0, p] = ti.Vector([0.0, 0.0, 0.0])
        C[0, p] = ti.Matrix.zero(ti.f64, 3, 3)
        F[0, p] = ti.Matrix.identity(ti.f64, 3)


@ti.kernel
def clear_grid():
    for f, i, j, k in grid_m:
        grid_m[f, i, j, k] = 0.0
        grid_v[f, i, j, k] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def p2g(f: ti.i32):
    for p in range(n):
        gv = g[None]
        volg = vol0 * gv * gv * gv
        Xp = x[f, p] * inv_dx
        base = ti.floor(Xp - 0.5).cast(ti.i32)
        fx = Xp - base.cast(ti.f64)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        Fe = F[f, p] * (1.0 / gv)                       # growth enters: Fe = F/g
        J = Fe.determinant()
        I3 = ti.Matrix.identity(ti.f64, 3)
        tau = mu0 * (Fe @ Fe.transpose() - I3) + lam0 * ti.log(J) * I3   # neo-Hookean Kirchhoff
        sk = tau * (-4.0 * inv_dx * inv_dx * volg)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offs = ti.Vector([i, j, k])
            dpos = (offs.cast(ti.f64) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            idx = base + offs
            grid_m[f, idx[0], idx[1], idx[2]] += weight * pmass
            contrib = weight * (pmass * (v[f, p] + C[f, p] @ dpos) + (sk @ dpos) * dt)
            grid_v[f, idx[0], idx[1], idx[2]] += contrib


@ti.kernel
def grid_op(f: ti.i32):
    for i, j, k in ti.ndrange(N, N, N):
        m = grid_m[f, i, j, k]
        if m > 1e-12:
            grid_v[f, i, j, k] = grid_v[f, i, j, k] * ((1.0 - damping) / m)


@ti.kernel
def g2p(f: ti.i32):
    for p in range(n):
        Xp = x[f, p] * inv_dx
        base = ti.floor(Xp - 0.5).cast(ti.i32)
        fx = Xp - base.cast(ti.f64)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        newv = ti.Vector.zero(ti.f64, 3)
        newC = ti.Matrix.zero(ti.f64, 3, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offs = ti.Vector([i, j, k])
            dpos = (offs.cast(ti.f64) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            idx = base + offs
            gvel = grid_v[f, idx[0], idx[1], idx[2]]
            newv += weight * gvel
            newC += 4.0 * inv_dx * inv_dx * weight * gvel.outer_product(dpos)
        v[f + 1, p] = newv
        C[f + 1, p] = newC
        x[f + 1, p] = x[f, p] + dt * newv
        F[f + 1, p] = (ti.Matrix.identity(ti.f64, 3) + dt * newC) @ F[f, p]


@ti.kernel
def compute_loss():
    for p in range(n):
        loss[None] += (1.0 / n) * F[steps - 1, p].determinant()   # mean det F (final)


def forward():
    clear_grid()
    set_initial()
    for f in range(steps - 1):
        p2g(f); grid_op(f); g2p(f)


def mean_detF(gv):
    g[None] = gv
    loss[None] = 0.0
    forward()
    compute_loss()
    return loss[None]


if __name__ == "__main__":
    x0.from_numpy(pts)
    print("== morphompm: differentiable trajectory (neo-Hookean growth-MPM) ==\n")
    print(f"{n} particles, {steps} steps")

    # (A) trajectory gradient: autodiff vs finite difference.
    g0 = 1.30
    g[None] = g0
    loss[None] = 0.0
    with ti.ad.Tape(loss):
        forward()
        compute_loss()
    grad_ad = g.grad[None]
    eps = 1e-6
    grad_fd = (mean_detF(g0 + eps) - mean_detF(g0 - eps)) / (2 * eps)
    rel = abs(grad_ad - grad_fd) / (abs(grad_fd) + 1e-30)
    print(f"\n[A] trajectory gradient d(mean det F_final)/dg")
    print(f"    autodiff = {grad_ad:.6f}   FD = {grad_fd:.6f}   rel.err = {rel:.2e}")
    okA = rel < 1e-4
    print(f"    [{'ok' if okA else 'FAIL'}] gradient flows correctly through the full sim")

    # (B) inverse design: recover g_true from a target shape.
    g_true = 1.35
    target = mean_detF(g_true)
    print(f"\n[B] inverse design: recover g_true = {g_true} from target mean det F = {target:.5f}")
    gk = 1.10
    lr = 0.5
    for it in range(20):
        g[None] = gk
        loss[None] = 0.0
        with ti.ad.Tape(loss):
            forward()
            compute_loss()
        resid = loss[None] - target
        dJ = 2.0 * resid * g.grad[None]        # J = (loss - target)^2
        gk -= lr * dJ
        if it % 8 == 0 or it == 19:
            print(f"    it {it:2d}: g = {gk:.5f}  (residual {resid:+.3e})")
    err = abs(gk - g_true)
    print(f"    recovered g = {gk:.5f}  (true {g_true}, err {err:.2e})")
    okB = err < 1e-3
    print(f"    [{'ok' if okB else 'FAIL'}] gradient descent recovers the growth rate")

    print("\n" + ("ALL CHECKS PASSED" if (okA and okB) else "SOME CHECKS FAILED"))
    raise SystemExit(0 if (okA and okB) else 1)
