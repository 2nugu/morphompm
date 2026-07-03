"""[2] Constitutive seam — pluggable material models: forward stress + MANUAL VJP.

Uniform interface (so [3] transfer is model-agnostic and new materials plug in):
    model.stress(F, Fg, C=None)               -> tau  (Kirchhoff, 3x3 sym)
    model.stress_vjp(F, Fg, C, tau_bar)       -> (F_bar, Fg_bar, C_bar)

C (velocity-gradient / APIC affine) is accepted for RATE-dependent models
(bioink / Herschel-Bulkley); rate-independent elastic models ignore it and
return C_bar = 0. Hand-written adjoints isolate the hard nonlinear part (esp. the
SVD adjoint) behind this seam; every VJP is gated against finite differences.

Morphoelastic kinematics: Fe = F·Fg^{-1}.  Frobenius <A,B> = Σ A∘B.
"""
import numpy as np

I3 = np.eye(3)
_DEGEN = 1.0e-9            # SVD singular-value degeneracy clamp (isotropic states)


def _frob(A, B):
    return float(np.sum(A * B))


# ── elastic stress kernels (functional impl) ─────────────────────────────────
def _nh_stress(F, Fg, mu, lam):
    Fe = F @ np.linalg.inv(Fg)
    J = np.linalg.det(Fe)
    return mu * (Fe @ Fe.T - I3) + lam * np.log(J) * I3


def _nh_vjp(F, Fg, mu, lam, tau_bar):
    Fg_inv = np.linalg.inv(Fg)
    Fe = F @ Fg_inv
    Fe_invT = np.linalg.inv(Fe).T
    tb = 0.5 * (tau_bar + tau_bar.T)
    Fe_bar = 2.0 * mu * (tb @ Fe) + lam * np.trace(tb) * Fe_invT
    F_bar = Fe_bar @ Fg_inv.T
    Fg_bar = -Fg_inv.T @ F.T @ Fe_bar @ Fg_inv.T
    return F_bar, Fg_bar


def _hencky_stress(F, Fg, mu, lam):
    Fe = F @ np.linalg.inv(Fg)
    U, s, _ = np.linalg.svd(Fe)
    S = np.sum(np.log(s))
    h = 2.0 * mu * np.log(s) + lam * S
    return U @ np.diag(h) @ U.T


def _hencky_vjp(F, Fg, mu, lam, tau_bar):
    Fg_inv = np.linalg.inv(Fg)
    Fe = F @ Fg_inv
    U, s, Vh = np.linalg.svd(Fe)
    tb = 0.5 * (tau_bar + tau_bar.T)
    S = np.sum(np.log(s))
    h = 2.0 * mu * np.log(s) + lam * S
    U_bar = 2.0 * tb @ U @ np.diag(h)
    P = U.T @ tb @ U
    s_bar = (2.0 * mu * np.diag(P) + lam * np.trace(P)) / s
    Fmat = np.zeros((3, 3))                     # 1/(s_j^2 - s_i^2), clamped at degeneracy
    for i in range(3):
        for j in range(3):
            if i != j:
                d = s[j] ** 2 - s[i] ** 2
                Fmat[i, j] = 1.0 / d if abs(d) > _DEGEN else 0.0
    UtUb = U.T @ U_bar
    inner = np.diag(s_bar) + (Fmat * (UtUb - UtUb.T)) @ np.diag(s)
    Fe_bar = U @ inner @ Vh
    F_bar = Fe_bar @ Fg_inv.T
    Fg_bar = -Fg_inv.T @ F.T @ Fe_bar @ Fg_inv.T
    return F_bar, Fg_bar


# ── model objects (the seam) ─────────────────────────────────────────────────
class _ElasticModel:
    """Rate-independent base: wraps an (stress, vjp) kernel; C ignored, C_bar=0."""
    _stress = None
    _vjp = None

    def __init__(self, material):
        self.mu = material.mu
        self.lam = material.lam

    def stress(self, F, Fg, C=None):
        return type(self)._stress(F, Fg, self.mu, self.lam)

    def stress_vjp(self, F, Fg, C, tau_bar):
        F_bar, Fg_bar = type(self)._vjp(F, Fg, self.mu, self.lam, tau_bar)
        return F_bar, Fg_bar, np.zeros((3, 3))


class NeoHookean(_ElasticModel):
    _stress = staticmethod(_nh_stress)
    _vjp = staticmethod(_nh_vjp)


class Hencky(_ElasticModel):
    _stress = staticmethod(_hencky_stress)
    _vjp = staticmethod(_hencky_vjp)


class HerschelBulkley:
    """Rate-dependent yield-stress fluid (bioink), Papanastasiou-regularized.
        τ = -p(Je)·I + 2·η(γ̇)·D ,  D = sym(C),  Je = det F / det Fg
        η = K·γ̇^{n-1} + τ_y·(1-e^{-m γ̇})/γ̇ ,  p = κ·(Je-1)
    Uses C (velocity gradient) → exercises the C_bar seam path. NOT elastic:
    own params (κ,K,n,τ_y,m), so it does not inherit _ElasticModel (O1)."""

    def __init__(self, kappa=1.0e4, K=1.0, n=0.5, tau_y=50.0, m=100.0, eps=1e-4):
        self.kappa, self.K, self.n, self.tau_y, self.m, self.eps = kappa, K, n, tau_y, m, eps

    def _rate(self, C):
        D = 0.5 * (C + C.T)
        gdot = np.sqrt(2.0 * np.sum(D * D) + self.eps ** 2)   # regularized shear rate
        return D, gdot

    def _eta(self, gdot):
        return self.K * gdot ** (self.n - 1) + self.tau_y * (1 - np.exp(-self.m * gdot)) / gdot

    def stress(self, F, Fg, C):
        Je = np.linalg.det(F) / np.linalg.det(Fg)
        D, gdot = self._rate(C)
        # volumetric Kirchhoff stress κ·ln(Je)·I (NEGATIVE at Je<1 → drives
        # expansion, matching neo-Hookean's λ·ln(Je) sign; a -κ(Je-1) form has
        # the OPPOSITE sign and collapses the material — caught by logic L2).
        return self.kappa * np.log(Je) * I3 + 2.0 * self._eta(gdot) * D

    def stress_vjp(self, F, Fg, C, tau_bar):
        tb = 0.5 * (tau_bar + tau_bar.T)
        trtb = np.trace(tb)
        # volumetric path: τ_vol = κ·ln(Je)·I,  d ln Je = <F^-T,dF> - <Fg^-T,dFg>
        F_bar = self.kappa * trtb * np.linalg.inv(F).T
        Fg_bar = -self.kappa * trtb * np.linalg.inv(Fg).T
        # viscous path: τ_visc = 2η(γ̇)D, D=sym(C), γ̇=√(2 D:D + eps²)
        D, gdot = self._rate(C)
        K, n, ty, m = self.K, self.n, self.tau_y, self.m
        deta = (K * (n - 1) * gdot ** (n - 2)
                + ty * ((m * np.exp(-m * gdot)) * gdot - (1 - np.exp(-m * gdot))) / gdot ** 2)
        tbD = float(np.sum(tb * D))
        C_bar = 2.0 * self._eta(gdot) * tb + (4.0 * tbD * deta / gdot) * D
        return F_bar, Fg_bar, C_bar


# ── STRONG forward oracle (catches coefficient/transcription bugs) ───────────
# Confined uniaxial swelling: F = diag(1,1,λ*), Fg = g·I, free in z (τ_zz=0).
# Independently (from Ψ, not the model code) solve μ(λ²/g²-1)+L·ln(λ/g³)=0 for λ*,
# giving the closed form  τ_xx = μ(1-λ*²)/g²  (nonzero, modulus-DEPENDENT — unlike
# free-swell, so it detects wrong/swapped μ,λ that all FD/zero-state gates miss).
def confined_swell_neohookean(mat, g=1.5):
    mu, L = mat.mu, mat.lam

    def f(lz):
        return mu * (lz ** 2 / g ** 2 - 1.0) + L * np.log(lz / g ** 3)

    lo, hi = 1e-3, 20.0                      # f(lo)<0, f(hi)>0 → bisection
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    lz = 0.5 * (lo + hi)
    return lz, mu * (1.0 - lz ** 2) / g ** 2


def _confined_swell_check(mat):
    lz, txx_exp = confined_swell_neohookean(mat, g=1.5)
    F = np.diag([1.0, 1.0, lz]); Fg = 1.5 * np.eye(3)
    tau = NeoHookean(mat).stress(F, Fg)
    ok = abs(tau[2, 2]) < 1e-6 and abs(tau[0, 0] - txx_exp) < 1e-6 * abs(txx_exp) + 1e-6
    print(f"[confined-swell analytic oracle]  λ*={lz:.4f}  τ_zz={tau[2,2]:.2e} (→0)  "
          f"τ_xx={tau[0,0]:.3f} vs analytic {txx_exp:.3f}  [{'ok' if ok else 'FAIL'}]")
    return 0 if ok else 1


# ── verification gate ────────────────────────────────────────────────────────
def _vjp_rel_err(model, F, Fg, rng, C=None, eps=1e-6):
    tau_bar = rng.standard_normal((3, 3)); tau_bar = 0.5 * (tau_bar + tau_bar.T)
    F_bar, Fg_bar, C_bar = model.stress_vjp(F, Fg, C, tau_bar)
    dF = rng.standard_normal((3, 3)); dFg = rng.standard_normal((3, 3))
    dC = rng.standard_normal((3, 3)) if C is not None else None
    Cp = (C + eps * dC) if C is not None else None
    Cm = (C - eps * dC) if C is not None else None
    Lp = _frob(tau_bar, model.stress(F + eps * dF, Fg + eps * dFg, Cp))
    Lm = _frob(tau_bar, model.stress(F - eps * dF, Fg - eps * dFg, Cm))
    fd = (Lp - Lm) / (2 * eps)
    analytic = _frob(F_bar, dF) + _frob(Fg_bar, dFg) + (_frob(C_bar, dC) if C is not None else 0.0)
    return abs(analytic - fd) / (abs(fd) + 1e-12)


def main():
    from .config import Material
    mat = Material()
    rng = np.random.default_rng(0)
    print("== constitutive seam: forward + manual VJP (vs FD) ==\n")
    states = [
        (np.diag([1.20, 1.00, 0.85]) + 0.03 * np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
         np.diag([1.30, 1.10, 0.90])),
        (np.diag([1.05, 0.95, 1.15]) + 0.02 * rng.standard_normal((3, 3)),
         np.diag([1.20, 0.85, 1.05])),
        (I3 + 0.05 * rng.standard_normal((3, 3)), np.diag([1.4, 1.0, 0.8])),
    ]
    fails = 0
    # rate-independent elastic models (C ignored)
    for name, Model in [("NeoHookean", NeoHookean), ("Hencky (manual SVD adjoint)", Hencky)]:
        model = Model(mat)
        Fg0 = np.diag([1.3, 1.1, 0.9])
        z = np.max(np.abs(model.stress(Fg0.copy(), Fg0.copy())))
        ok_zero = z < 1e-8
        worst = max(_vjp_rel_err(model, F, Fg, rng) for F, Fg in states for _ in range(4))
        print(f"[{name}]  F=Fg->||tau||={z:.1e} [{'ok' if ok_zero else 'FAIL'}]"
              f"   VJP vs FD worst={worst:.2e} [{'ok' if worst < 1e-5 else 'FAIL'}]")
        fails += (not ok_zero) + (worst >= 1e-5)
    # rate-dependent bioink (needs C; exercises C_bar path)
    hb = HerschelBulkley()
    z = np.max(np.abs(hb.stress(np.eye(3), np.eye(3), np.zeros((3, 3)))))   # rest, Je=1 -> 0
    ok_zero = z < 1e-8
    worst = max(_vjp_rel_err(hb, F, Fg, rng, C=0.5 * rng.standard_normal((3, 3)))
                for F, Fg in states for _ in range(4))
    print(f"[HerschelBulkley (bioink, rate-dep)]  rest->||tau||={z:.1e} [{'ok' if ok_zero else 'FAIL'}]"
          f"   VJP(F,Fg,C) vs FD worst={worst:.2e} [{'ok' if worst < 1e-5 else 'FAIL'}]")
    fails += (not ok_zero) + (worst >= 1e-5)
    # strong forward oracle: nonzero modulus-dependent stress (catches μ,λ bugs)
    fails += _confined_swell_check(mat)
    print("\nALL CHECKS PASSED" if fails == 0 else f"\n{fails} CHECK(S) FAILED")
    return fails


if __name__ == "__main__":
    raise SystemExit(0 if main() == 0 else 1)
