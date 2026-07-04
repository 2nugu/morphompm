# EXPERIMENTAL / SUPERSEDED — not part of the morphompm package. Taichi autodiff
# segfaults on matrix-MPM adjoints (see docs/PIPELINE.md); superseded by the numpy
# package (morphompm.integrate/diff). Kept for the record; do NOT build on this.

"""morphompm — differentiable trajectory in 2D (proof that the differentiable
growth + inverse-design machinery works).

3D matrix-heavy MPM autodiff SEGFAULTS in stock Taichi 1.7.4 (matrix-op adjoint
codegen). 2D is the DiffTaichi-proven regime (small, stable IR). We demonstrate
here that the FULL trajectory is differentiable and that the growth rate can be
recovered by gradient descent from an observed shape. The 3D forward is verified
separately (test_growth: C++/Taichi parity); 3D autodiff needs a scalar-decomposed
rewrite or a framework with robust 3D-matrix AD.

  (A) trajectory gradient d(mean det F_final)/dg: autodiff vs finite difference.
  (B) inverse design: recover g_true from a target shape via gradient descent.
"""
import numpy as np
import taichi as ti

ti.init(arch=ti.cpu, default_fp=ti.f32)

steps = 48
N, dx = 24, 0.05
inv_dx = 1.0 / dx
E, nu = 1.0e4, 0.3
mu0 = E / (2 * (1 + nu))
lam0 = E * nu / ((1 + nu) * (1 - 2 * nu))
dt, damping = 1.0e-3, 0.05
density = 1000.0

center, half, sp = 0.5, 0.06, 0.02
coords = np.arange(center - half, center + half + 1e-9, sp)
pts = np.array([(a, b) for a in coords for b in coords], dtype=np.float32)
n = len(pts)
pmass = density * sp * sp
vol0 = pmass / density

x = ti.Vector.field(2, ti.f32, (steps, n), needs_grad=True)
v = ti.Vector.field(2, ti.f32, (steps, n), needs_grad=True)
C = ti.Matrix.field(2, 2, ti.f32, (steps, n), needs_grad=True)
F = ti.Matrix.field(2, 2, ti.f32, (steps, n), needs_grad=True)
grid_v = ti.Vector.field(2, ti.f32, (steps, N, N), needs_grad=True)
grid_m = ti.field(ti.f32, (steps, N, N), needs_grad=True)
g = ti.field(ti.f32, (), needs_grad=True)
loss = ti.field(ti.f32, (), needs_grad=True)
x0 = ti.Vector.field(2, ti.f32, n)


@ti.kernel
def set_initial():
    for p in range(n):
        x[0, p] = x0[p]
        v[0, p] = ti.Vector([0.0, 0.0])
        C[0, p] = ti.Matrix.zero(ti.f32, 2, 2)
        F[0, p] = ti.Matrix.identity(ti.f32, 2)


@ti.kernel
def clear_grid():
    for f, i, j in grid_m:
        grid_m[f, i, j] = 0.0
        grid_v[f, i, j] = ti.Vector([0.0, 0.0])


@ti.kernel
def p2g(f: ti.i32):
    for p in range(n):
        gv = g[None]
        volg = vol0 * gv * gv
        Xp = x[f, p] * inv_dx
        base = ti.floor(Xp - 0.5).cast(ti.i32)
        fx = Xp - base.cast(ti.f32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        Fe = F[f, p] * (1.0 / gv)
        J = Fe.determinant()
        I2 = ti.Matrix.identity(ti.f32, 2)
        tau = mu0 * (Fe @ Fe.transpose() - I2) + lam0 * ti.log(J) * I2
        sk = tau * (-4.0 * inv_dx * inv_dx * volg)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offs = ti.Vector([i, j])
            dpos = (offs.cast(ti.f32) - fx) * dx
            weight = w[i][0] * w[j][1]
            idx = base + offs
            grid_m[f, idx[0], idx[1]] += weight * pmass
            grid_v[f, idx[0], idx[1]] += weight * (pmass * (v[f, p] + C[f, p] @ dpos) + (sk @ dpos) * dt)


@ti.kernel
def grid_op(f: ti.i32):
    for i, j in ti.ndrange(N, N):
        m = grid_m[f, i, j]
        if m > 1e-12:
            grid_v[f, i, j] = grid_v[f, i, j] * ((1.0 - damping) / m)


@ti.kernel
def g2p(f: ti.i32):
    for p in range(n):
        Xp = x[f, p] * inv_dx
        base = ti.floor(Xp - 0.5).cast(ti.i32)
        fx = Xp - base.cast(ti.f32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        newv = ti.Vector.zero(ti.f32, 2)
        newC = ti.Matrix.zero(ti.f32, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offs = ti.Vector([i, j])
            dpos = (offs.cast(ti.f32) - fx) * dx
            weight = w[i][0] * w[j][1]
            idx = base + offs
            gvel = grid_v[f, idx[0], idx[1]]
            newv += weight * gvel
            newC += 4.0 * inv_dx * inv_dx * weight * gvel.outer_product(dpos)
        v[f + 1, p] = newv
        C[f + 1, p] = newC
        x[f + 1, p] = x[f, p] + dt * newv
        F[f + 1, p] = (ti.Matrix.identity(ti.f32, 2) + dt * newC) @ F[f, p]


@ti.kernel
def compute_loss():
    for p in range(n):
        loss[None] += (1.0 / n) * F[steps - 1, p].determinant()


def forward():
    clear_grid(); set_initial()
    for f in range(steps - 1):
        p2g(f); grid_op(f); g2p(f)


def mean_detF(gv):
    g[None] = gv; loss[None] = 0.0; forward(); compute_loss()
    return loss[None]


if __name__ == "__main__":
    x0.from_numpy(pts)
    print(f"== morphompm 2D differentiable trajectory ==  {n} particles, {steps} steps", flush=True)

    g0 = 1.30
    g[None] = g0; loss[None] = 0.0
    with ti.ad.Tape(loss):
        forward(); compute_loss()
    grad_ad = g.grad[None]
    eps = 1e-3
    grad_fd = (mean_detF(g0 + eps) - mean_detF(g0 - eps)) / (2 * eps)
    rel = abs(grad_ad - grad_fd) / (abs(grad_fd) + 1e-30)
    print(f"\n[A] trajectory gradient d(mean det F)/dg", flush=True)
    print(f"    autodiff = {grad_ad:.5f}   FD = {grad_fd:.5f}   rel.err = {rel:.2e}", flush=True)
    okA = rel < 5e-3
    print(f"    [{'ok' if okA else 'FAIL'}] gradient flows through the full sim", flush=True)

    g_true = 1.35
    target = mean_detF(g_true)
    print(f"\n[B] inverse design: recover g_true = {g_true} (target mean det F = {target:.5f})", flush=True)
    gk, lr = 1.10, 0.5
    for it in range(30):
        g[None] = gk; loss[None] = 0.0
        with ti.ad.Tape(loss):
            forward(); compute_loss()
        resid = loss[None] - target
        gk -= lr * 2.0 * resid * g.grad[None]
        if it % 6 == 0 or it == 29:
            print(f"    it {it:2d}: g = {gk:.5f}  (residual {resid:+.3e})", flush=True)
    err = abs(gk - g_true)
    print(f"    recovered g = {gk:.5f}  (true {g_true}, err {err:.2e})", flush=True)
    okB = err < 5e-3
    print(f"    [{'ok' if okB else 'FAIL'}] gradient descent recovers the growth rate", flush=True)

    print("\n" + ("ALL CHECKS PASSED" if (okA and okB) else "SOME CHECKS FAILED"), flush=True)
