"""[3] Transfer seam — one general MLS-MPM step: forward + MANUAL adjoint (VJP).

    step(state, model, cfg, advect)          -> state', cache
    step_vjp(cache, state'_bars, advect)     -> state_bars   (x,v,C,F,Fg cotangents + g via caller)

Material-agnostic: the constitutive `model` is injected ([2] seam). Growth-agnostic:
operates on per-particle state.Fg and returns Fg_bar (the growth parameterization —
scalar g, field g(x), … — lives in the caller / [5] driver). Rate models supply a
C_bar that is wired into C_bar here. Full MPM adjoint incl. particle advection
(∂weights/∂x) when advect=True. Every path FD-gated.
"""
import numpy as np
from .state import ParticleState

_STEN = [(i, j, k) for i in range(3) for j in range(3) for k in range(3)]


def _wts(xp, inv_dx):
    X = xp * inv_dx
    base = np.floor(X - 0.5).astype(int)
    fx = X - base
    w = np.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2])
    dw = np.stack([-(1.5 - fx), -2.0 * (fx - 1.0), (fx - 0.5)])
    return base, fx, w, dw


def _dWdx(i, j, k, w, dw, inv_dx):
    return np.array([dw[i, 0] * w[j, 1] * w[k, 2],
                     w[i, 0] * dw[j, 1] * w[k, 2],
                     w[i, 0] * w[j, 1] * dw[k, 2]]) * inv_dx


def step(state, model, cfg, advect=True):
    x, v, C, F, Fg = state.x, state.v, state.C, state.F, state.Fg
    n = state.n
    N, inv_dx, dx, dt = cfg.N, cfg.inv_dx, cfg.dx, cfg.dt
    s, pmass, vol0 = 1.0 - cfg.damping, cfg.pmass, cfg.vol0
    k4 = 4.0 * inv_dx * inv_dx

    gm = np.zeros((N, N, N)); gmom = np.zeros((N, N, N, 3))
    pc = []
    for p in range(n):
        tau = model.stress(F[p], Fg[p], C[p])
        c = -k4 * vol0 * np.linalg.det(Fg[p])           # per-particle grown volume
        sk = c * tau
        base, fx, w, dw = _wts(x[p], inv_dx)
        for (i, j, k) in _STEN:
            wt = w[i, 0] * w[j, 1] * w[k, 2]
            dpos = (np.array([i, j, k]) - fx) * dx
            ni, nj, nk = base + [i, j, k]
            gm[ni, nj, nk] += wt * pmass
            gmom[ni, nj, nk] += wt * (pmass * (v[p] + C[p] @ dpos) + (sk @ dpos) * dt)
        pc.append((base, fx, w, dw, tau, c, sk))

    gv = np.zeros((N, N, N, 3))
    for idx in np.argwhere(gm > 1e-12):
        ni, nj, nk = idx
        gv[ni, nj, nk] = s * gmom[ni, nj, nk] / gm[ni, nj, nk]

    xn = x.copy(); vn = np.zeros((n, 3)); Cn = np.zeros((n, 3, 3)); Fn = np.zeros((n, 3, 3))
    for p in range(n):
        base, fx, w, dw, tau, c, sk = pc[p]
        nv = np.zeros(3); nC = np.zeros((3, 3))
        for (i, j, k) in _STEN:
            wt = w[i, 0] * w[j, 1] * w[k, 2]
            dpos = (np.array([i, j, k]) - fx) * dx
            ni, nj, nk = base + [i, j, k]
            vi = gv[ni, nj, nk]
            nv += wt * vi
            nC += k4 * wt * np.outer(vi, dpos)
        vn[p] = nv; Cn[p] = nC
        if advect:
            xn[p] = x[p] + dt * nv
        Fn[p] = (np.eye(3) + dt * nC) @ F[p]
    new_state = ParticleState(xn, vn, Cn, Fn, Fg.copy())
    cache = (state, cfg, model, gm, gv, pc, Cn)
    return new_state, cache


def step_vjp(cache, xn_b, vn_b, Cn_b, Fn_b, Fgn_b, advect=True):
    state, cfg, model, gm, gv, pc, Cn = cache
    x, v, C, F, Fg = state.x, state.v, state.C, state.F, state.Fg
    n = state.n
    N, inv_dx, dx, dt = cfg.N, cfg.inv_dx, cfg.dx, cfg.dt
    s, pmass, vol0 = 1.0 - cfg.damping, cfg.pmass, cfg.vol0
    k4 = 4.0 * inv_dx * inv_dx

    F_b = np.zeros((n, 3, 3)); v_b = np.zeros((n, 3)); C_b = np.zeros((n, 3, 3))
    Fg_b = Fgn_b.copy()                                  # Fg passes through unchanged
    x_b = np.zeros((n, 3))
    gv_b = np.zeros((N, N, N, 3))

    # --- g2p adjoint ---
    for p in range(n):
        base, fx, w, dw, tau, c, sk = pc[p]
        nv_b = vn_b[p] + (dt * xn_b[p] if advect else 0.0)
        if advect:
            x_b[p] += xn_b[p]
        Cn_b_p = Cn_b[p] + dt * (Fn_b[p] @ F[p].T)
        F_b[p] += (np.eye(3) + dt * Cn[p]).T @ Fn_b[p]
        for (i, j, k) in _STEN:
            wt = w[i, 0] * w[j, 1] * w[k, 2]
            dpos = (np.array([i, j, k]) - fx) * dx
            ni, nj, nk = base + [i, j, k]
            vi = gv[ni, nj, nk]
            gv_b[ni, nj, nk] += wt * nv_b
            gv_b[ni, nj, nk] += k4 * wt * (Cn_b_p @ dpos)
            if advect:
                dWdx = _dWdx(i, j, k, w, dw, inv_dx)
                x_b[p] += (nv_b @ vi) * dWdx
                x_b[p] += k4 * float(np.tensordot(Cn_b_p, np.outer(vi, dpos))) * dWdx
                x_b[p] += -k4 * wt * (Cn_b_p.T @ vi)

    # --- grid adjoint: v_i = s gmom_i/m_i ---
    gmom_b = np.zeros((N, N, N, 3)); gm_b = np.zeros((N, N, N))
    for idx in np.argwhere(gm > 1e-12):
        ni, nj, nk = idx
        gmom_b[ni, nj, nk] = s * gv_b[ni, nj, nk] / gm[ni, nj, nk]
        if advect:
            gm_b[ni, nj, nk] = -float(gv_b[ni, nj, nk] @ gv[ni, nj, nk]) / gm[ni, nj, nk]

    # --- p2g adjoint + constitutive VJP + volume path + advect x-deps ---
    for p in range(n):
        base, fx, w, dw, tau, c, sk = pc[p]
        sk_b = np.zeros((3, 3))
        for (i, j, k) in _STEN:
            wt = w[i, 0] * w[j, 1] * w[k, 2]
            dpos = (np.array([i, j, k]) - fx) * dx
            ni, nj, nk = base + [i, j, k]
            mb = gmom_b[ni, nj, nk]
            v_b[p] += wt * pmass * mb
            C_b[p] += wt * pmass * np.outer(mb, dpos)
            sk_b += wt * dt * np.outer(mb, dpos)
            if advect:
                dWdx = _dWdx(i, j, k, w, dw, inv_dx)
                Q2 = pmass * (v[p] + C[p] @ dpos) + (sk @ dpos) * dt
                x_b[p] += (mb @ Q2) * dWdx
                x_b[p] += gm_b[ni, nj, nk] * pmass * dWdx
                x_b[p] += -(pmass * (C[p].T @ mb) + dt * (sk.T @ mb)) * wt
        # sk = c(Fg)·tau(F,Fg,C)
        tau_b = c * sk_b
        Fp_b, Fg_bar_stress, C_bar = model.stress_vjp(F[p], Fg[p], C[p], tau_b)
        F_b[p] += Fp_b
        C_b[p] += C_bar                                  # rate-model stress path (0 for elastic)
        # Fg cotangent: via stress + via grown volume c = -k4·vol0·det(Fg)
        Fg_inv_p = np.linalg.inv(Fg[p])
        Fg_b[p] += Fg_bar_stress
        Fg_b[p] += float(np.sum(sk_b * tau)) * (-k4 * vol0) * np.linalg.det(Fg[p]) * Fg_inv_p.T

    return x_b, v_b, C_b, F_b, Fg_b


# ── single-step assembly gate (from rest) ────────────────────────────────────
def main():
    from .config import Material, SimConfig
    from .constitutive import NeoHookean
    cfg = SimConfig()
    model = NeoHookean(cfg.material)
    pts = np.array([[0.40, 0.40, 0.40], [0.42, 0.40, 0.41], [0.40, 0.43, 0.40],
                    [0.41, 0.40, 0.44], [0.44, 0.44, 0.43]])
    n = len(pts)

    def make_state(g):
        st = ParticleState.rest(pts)
        st.Fg[:] = g * np.eye(3)
        return st

    def loss_of(g):
        (ns), _ = step(make_state(g), model, cfg, advect=False)
        return float(sum(np.linalg.det(ns.F[p]) for p in range(n)))

    g0 = 1.30
    ns, cache = step(make_state(g0), model, cfg, advect=False)
    Fn_b = np.stack([np.linalg.det(ns.F[p]) * np.linalg.inv(ns.F[p]).T for p in range(n)])
    z3, z33 = np.zeros((n, 3)), np.zeros((n, 3, 3))
    _, _, _, _, Fg_b = step_vjp(cache, z3, z3, z33, Fn_b, z33, advect=False)
    dg = float(sum(np.trace(Fg_b[p]) for p in range(n)))     # Fg = g·I  → dg = Σ tr(Fg_b)
    eps = 1e-6
    fd = (loss_of(g0 + eps) - loss_of(g0 - eps)) / (2 * eps)
    rel = abs(dg - fd) / (abs(fd) + 1e-30)
    print(f"== transfer: single-step assembly gate ==")
    print(f"    adjoint={dg:.6e}  FD={fd:.6e}  rel.err={rel:.2e}")
    ok = rel < 1e-5
    print(f"    [{'ok' if ok else 'FAIL'}] single-step transfer o constitutive matches FD")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
