#pragma once
//
// morphompm — minimal MLS-MPM with morphoelastic growth.
//
// v1: forward growth physics. Self-contained dense grid. Reuses only
// morphompm::math (Vec3 / Matrix3 / SVD) — the verified material-agnostic core.
//
// Morphoelastic kinematics (Rodriguez 1994):  F = Fe · Fg.
//   Elastic stress depends ONLY on Fe = F · Fg^{-1}.
//   Growth Fg is a per-particle tensor (default I). Isotropic Fg = g·I is the
//   special case; anisotropic / differential growth (the morphogenesis regime,
//   producing residual stress and bending) is the general case.
//   Stress: SVD(Fe) → ε_e = log Σ_e → Hencky-Kirchhoff τ → MLS-MPM force.
//
// Growth touches ONLY the constitutive map (Fe = F·Fg^{-1}); the P2G/G2P
// transfer pipeline is unchanged.

#include <vector>
#include <cmath>
#include <algorithm>

#include "morphompm/math/matrix3.h"
#include "morphompm/math/svd.h"

namespace morphompm {

using morphompm::math::Vec3;
using morphompm::math::Matrix3;
using morphompm::math::SVD;

struct Particle {
    Vec3    x;                               // position
    Vec3    v;                               // velocity
    Matrix3 F  = Matrix3::identity();        // TOTAL deformation gradient
    Matrix3 C  = Matrix3::zero();            // APIC affine momentum
    Matrix3 Fg = Matrix3::identity();        // growth tensor (Fg = g·I if isotropic)
    float   mass = 1.0f;
    float   vol0 = 1.0f;                     // initial reference volume V_p^0
};

class GrowthSolver {
public:
    // Dense grid covering [0, N·dx]^3, node-centered quadratic B-spline MLS-MPM.
    GrowthSolver(int N, float dx)
        : N_(N), dx_(dx), inv_dx_(1.0f / dx),
          mass_((size_t)N * N * N, 0.0f),
          vel_((size_t)N * N * N, Vec3()),
          force_((size_t)N * N * N, Vec3()) {}

    void set_material(float E, float nu) { E_ = E; nu_ = nu; }
    void set_gravity(const Vec3& gv)     { gravity_ = gv; }
    void set_damping(float d)            { damping_ = d; }  // per-step relaxation

    void add_particle(const Vec3& x, float mass, float density) {
        Particle p;
        p.x = x; p.mass = mass;
        p.vol0 = (density > 1e-12f) ? (mass / density) : 1.0f;
        particles_.push_back(p);
    }

    std::vector<Particle>&       particles()       { return particles_; }
    const std::vector<Particle>& particles() const { return particles_; }

    // Prescribed growth (researcher-supplied law; the slow biological clock).
    // Isotropic: scale every particle's growth tensor uniformly.
    void grow_all_isotropic(float factor) {
        for (auto& p : particles_) p.Fg = p.Fg * factor;
    }
    // Anisotropic: per-axis principal growth stretches (applied multiplicatively).
    void grow_all_anisotropic(const Vec3& fac) {
        const Matrix3 G = Matrix3::scale(fac);
        for (auto& p : particles_) p.Fg = G * p.Fg;
    }

    float mu_lame()     const { return E_ / (2.0f * (1.0f + nu_)); }
    float lambda_lame() const { return E_ * nu_ / ((1.0f + nu_) * (1.0f - 2.0f * nu_)); }

    // Elastic Hencky-Kirchhoff stress τ from Fe = F·Fg^{-1} (general Fg).
    // Optionally returns ‖ε_e‖ — elastic log-strain norm; the residual-stress
    // diagnostic (→ 0 at a stress-free / compatible-growth equilibrium,
    // stays > 0 when growth is incompatible → residual stress / bending).
    Matrix3 kirchhoff_stress(const Matrix3& F, const Matrix3& Fg,
                             float* eps_e_norm = nullptr) const {
        const Matrix3 Fe = F * Fg.inversed();
        Matrix3 U, V; Vec3 s;
        SVD::compute(Fe, U, s, V);
        s.x = std::max(s.x, 1e-6f);
        s.y = std::max(s.y, 1e-6f);
        s.z = std::max(s.z, 1e-6f);

        const Vec3 e(std::log(s.x), std::log(s.y), std::log(s.z));   // elastic log-strain
        const float tr = e.x + e.y + e.z;
        const float mu_ = mu_lame(), lam_ = lambda_lame();
        const Vec3 td(2.0f * mu_ * e.x + lam_ * tr,
                      2.0f * mu_ * e.y + lam_ * tr,
                      2.0f * mu_ * e.z + lam_ * tr);
        if (eps_e_norm) *eps_e_norm = std::sqrt(e.x * e.x + e.y * e.y + e.z * e.z);
        return U * Matrix3::scale(td) * U.transposed();
    }

    void step(float dt) {
        clear_grid();
        for (auto& p : particles_) p2g(p);
        update_grid(dt);
        for (auto& p : particles_) g2p(p, dt);
    }

private:
    int   N_;
    float dx_, inv_dx_;
    float E_ = 1.0e4f, nu_ = 0.3f;
    Vec3  gravity_ = Vec3(0, 0, 0);
    float damping_ = 0.0f;

    std::vector<Particle> particles_;
    std::vector<float>    mass_;
    std::vector<Vec3>     vel_;     // momentum during P2G, velocity after update
    std::vector<Vec3>     force_;

    inline int  idx(int i, int j, int k) const { return (i * N_ + j) * N_ + k; }
    inline bool in_range(int i, int j, int k) const {
        return i >= 0 && i < N_ && j >= 0 && j < N_ && k >= 0 && k < N_;
    }
    void clear_grid() {
        std::fill(mass_.begin(), mass_.end(), 0.0f);
        std::fill(vel_.begin(),  vel_.end(),  Vec3());
        std::fill(force_.begin(),force_.end(),Vec3());
    }

    // Quadratic B-spline weights; standard node-centered MLS-MPM.
    void weights(const Vec3& x, int base[3],
                 float wx[3], float wy[3], float wz[3]) const {
        const float fx = x.x * inv_dx_ - 0.5f;
        const float fy = x.y * inv_dx_ - 0.5f;
        const float fz = x.z * inv_dx_ - 0.5f;
        base[0] = (int)std::floor(fx);
        base[1] = (int)std::floor(fy);
        base[2] = (int)std::floor(fz);
        const float ax = fx - base[0], ay = fy - base[1], az = fz - base[2];
        wx[0] = 0.5f*(1-ax)*(1-ax); wx[1] = 0.75f-(ax-0.5f)*(ax-0.5f); wx[2] = 0.5f*ax*ax;
        wy[0] = 0.5f*(1-ay)*(1-ay); wy[1] = 0.75f-(ay-0.5f)*(ay-0.5f); wy[2] = 0.5f*ay*ay;
        wz[0] = 0.5f*(1-az)*(1-az); wz[1] = 0.75f-(az-0.5f)*(az-0.5f); wz[2] = 0.5f*az*az;
    }

    void p2g(Particle& p) {
        int base[3]; float wx[3], wy[3], wz[3];
        weights(p.x, base, wx, wy, wz);

        const Matrix3 tau = kirchhoff_stress(p.F, p.Fg);
        // Grown reference volume V_p^g = V_p^0 · det(Fg).
        // (Equilibrium is independent of this choice; it sets transient stiffness.)
        const float volg = p.vol0 * p.Fg.determinant();
        const float force_factor = -4.0f * inv_dx_ * inv_dx_ * volg;
        const Matrix3 stress_kernel = tau * force_factor;

        const float fx = p.x.x * inv_dx_ - 0.5f;
        const float fy = p.x.y * inv_dx_ - 0.5f;
        const float fz = p.x.z * inv_dx_ - 0.5f;

        for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) {
            const float w = wx[i] * wy[j] * wz[k];
            if (w == 0.0f) continue;
            const int ni = base[0]+i, nj = base[1]+j, nk = base[2]+k;
            if (!in_range(ni, nj, nk)) continue;

            const Vec3 dist(((float)ni - fx - 0.5f) * dx_,
                            ((float)nj - fy - 0.5f) * dx_,
                            ((float)nk - fz - 0.5f) * dx_);
            const Vec3 affine = p.C * dist;
            const int id = idx(ni, nj, nk);
            const float mc = w * p.mass;
            mass_[id]  += mc;
            vel_[id]   += (p.v + affine) * mc;          // momentum accumulation
            force_[id] += (stress_kernel * dist) * w;   // internal stress reaction
        }
    }

    void update_grid(float dt) {
        for (size_t id = 0; id < mass_.size(); ++id) {
            if (mass_[id] <= 1e-9f) continue;
            Vec3 v = (vel_[id] + force_[id] * dt) / mass_[id];
            v += gravity_ * dt;
            v *= (1.0f - damping_);                      // quasi-static relaxation
            vel_[id] = v;
        }
    }

    void g2p(Particle& p, float dt) {
        int base[3]; float wx[3], wy[3], wz[3];
        weights(p.x, base, wx, wy, wz);

        const float fx = p.x.x * inv_dx_ - 0.5f;
        const float fy = p.x.y * inv_dx_ - 0.5f;
        const float fz = p.x.z * inv_dx_ - 0.5f;

        p.v = Vec3();
        p.C = Matrix3::zero();

        for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) {
            const float w = wx[i] * wy[j] * wz[k];
            if (w == 0.0f) continue;
            const int ni = base[0]+i, nj = base[1]+j, nk = base[2]+k;
            if (!in_range(ni, nj, nk)) continue;

            const Vec3 dist(((float)ni - fx - 0.5f) * dx_,
                            ((float)nj - fy - 0.5f) * dx_,
                            ((float)nk - fz - 0.5f) * dx_);
            const Vec3 gv = vel_[idx(ni, nj, nk)] * w;
            p.v += gv;
            p.C  = p.C + Matrix3::outer_product(gv, dist) * (4.0f * inv_dx_ * inv_dx_);
        }

        p.x += p.v * dt;
        // Total deformation gradient update (no plasticity in the growth foundation).
        p.F = (Matrix3::identity() + p.C * dt) * p.F;
    }
};

} // namespace morphompm
