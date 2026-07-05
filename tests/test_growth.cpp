// v1 verification for morphoelastic growth in MLS-MPM.
//
//   T1  — constitutive (isotropic):   τ(Fe) closed form.
//   T1b — constitutive (anisotropic): per-axis closed form.
//   T2  — dynamics (free swell):      det(F) → g^3, residual strain → 0.
//   T3  — dynamics (differential):    incompatible growth → residual stress + bend.
//   T4  — objectivity:                τ(R·F) = R·τ(F)·Rᵀ (frame-indifference).
//   T5  — dynamics (aniso free):      uniform anisotropic growth → stress-free.
//   T6  — determinism:                identical runs → bit-identical.
//   T7  — quantitative:               bend increases monotonically with mismatch.
//
// No framework — plain asserts, nonzero exit on failure. T3 dumps a CSV for
// visualization (scripts/plot_growth.py).

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <filesystem>
#include <fstream>

#include "morphompm/growth_solver.h"

using morphompm::GrowthSolver;
using morphompm::Particle;
using basements::math::Vec3;
using basements::math::Matrix3;

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    std::printf("  [FAIL] %s\n", msg); ++g_fail; } \
    else { std::printf("  [ ok ] %s\n", msg); } } while (0)

static bool approx(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

static float max_abs_diff(const Matrix3& A, const Matrix3& B) {
    float m = 0.0f;
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
        m = std::max(m, std::fabs(A.m[r][c] - B.m[r][c]));
    return m;
}

// Centered cube blob; returns particle count.
static int seed_blob(GrowthSolver& s, float center, float half, float sp, float density) {
    const float pmass = density * sp * sp * sp;
    int n = 0;
    for (float x = center - half; x <= center + half + 1e-6f; x += sp)
    for (float y = center - half; y <= center + half + 1e-6f; y += sp)
    for (float z = center - half; z <= center + half + 1e-6f; z += sp) {
        s.add_particle(Vec3(x, y, z), pmass, density); ++n;
    }
    return n;
}

// ── T1: isotropic constitutive ───────────────────────────────────────────────
static void test_constitutive_isotropic() {
    std::printf("[T1] isotropic constitutive (Fe = F·Fg^-1, Fg = g·I)\n");
    GrowthSolver s(8, 0.1f);
    s.set_material(1.0e4f, 0.3f);
    const float mu = s.mu_lame(), lam = s.lambda_lame(), g = 1.5f;
    {
        Matrix3 F = Matrix3::scale(Vec3(g,g,g)), Fg = Matrix3::scale(Vec3(g,g,g));
        float en = -1.0f; Matrix3 tau = s.kirchhoff_stress(F, Fg, &en);
        CHECK(max_abs_diff(tau, Matrix3::zero()) < 1e-2f, "F = Fg gives zero stress (Fe = I)");
        CHECK(approx(en, 0.0f, 1e-5f), "F = Fg gives zero elastic strain");
    }
    {
        Matrix3 F = Matrix3::identity(), Fg = Matrix3::scale(Vec3(g,g,g));
        Matrix3 tau = s.kirchhoff_stress(F, Fg);
        const float expected = -(2.0f*mu + 3.0f*lam) * std::log(g);
        CHECK(approx(tau.m[0][0], expected, std::fabs(expected)*1e-3f + 1e-2f), "diag τ = −(2μ+3λ)ln g");
        CHECK(approx(tau.m[1][2], 0.0f, 1e-2f), "off-diag = 0 (isotropic)");
        std::printf("       (τ_diag = %.3f Pa, expected %.3f Pa)\n", tau.m[0][0], expected);
    }
}

// ── T1b: anisotropic constitutive ────────────────────────────────────────────
static void test_constitutive_anisotropic() {
    std::printf("[T1b] anisotropic constitutive (Fg = diag(g1,g2,g3))\n");
    GrowthSolver s(8, 0.1f);
    s.set_material(1.0e4f, 0.3f);
    const float mu = s.mu_lame(), lam = s.lambda_lame();
    const float g1 = 1.2f, g2 = 1.0f, g3 = 0.8f;
    Matrix3 tau = s.kirchhoff_stress(Matrix3::identity(), Matrix3::scale(Vec3(g1,g2,g3)));
    const Vec3 e(-std::log(g1), -std::log(g2), -std::log(g3));
    const float tr = e.x + e.y + e.z;
    CHECK(approx(tau.m[0][0], 2*mu*e.x + lam*tr, 1.0f), "τ_xx = 2μ ε_x + λ trε");
    CHECK(approx(tau.m[1][1], 2*mu*e.y + lam*tr, 1.0f), "τ_yy = 2μ ε_y + λ trε");
    CHECK(approx(tau.m[2][2], 2*mu*e.z + lam*tr, 1.0f), "τ_zz = 2μ ε_z + λ trε");
    CHECK(approx(tau.m[0][1], 0.0f, 1.0f), "off-diag = 0 (axis-aligned growth)");
    std::printf("       (τ = [%.1f, %.1f, %.1f] Pa)\n", tau.m[0][0], tau.m[1][1], tau.m[2][2]);
}

// ── T2: free isotropic swelling ──────────────────────────────────────────────
static void test_free_swelling() {
    std::printf("[T2] free swelling (det F -> g^3, residual strain -> 0)\n");
    GrowthSolver s(24, 0.05f);
    s.set_material(1.0e4f, 0.3f); s.set_gravity(Vec3(0,0,0)); s.set_damping(0.05f);
    const int np = seed_blob(s, 0.6f, 0.10f, 0.025f, 1000.0f);
    const float g_target = 1.5f; const int K = 10, M = 250;
    const float dt = 1.0e-3f, per = std::pow(g_target, 1.0f/K);
    for (int kk = 0; kk < K; ++kk) { s.grow_all_isotropic(per); for (int m = 0; m < M; ++m) s.step(dt); }

    double mean_detF = 0.0; float max_eps = 0.0f;
    for (const auto& p : s.particles()) {
        mean_detF += p.F.determinant();
        float en = 0.0f; s.kirchhoff_stress(p.F, p.Fg, &en); max_eps = std::max(max_eps, en);
    }
    mean_detF /= (double)np;
    const double expected = std::pow((double)g_target, 3.0);
    std::printf("       %d particles; mean det(F) = %.4f (expect %.4f); max ||eps_e|| = %.4f\n",
                np, mean_detF, expected, max_eps);
    CHECK(approx((float)mean_detF, (float)expected, 0.08f*(float)expected), "mean det(F) within 8% of g^3");
    CHECK(max_eps < 0.05f, "compatible growth -> near stress-free");
}

// Bilayer strip: bottom grows g_b, top grows g_t. Returns mean ||eps_e|| and bend.
static void run_bilayer(float g_b, float g_t, bool dump, double& mean_eps, float& bend) {
    GrowthSolver s(28, 0.05f);
    s.set_material(1.0e4f, 0.3f); s.set_gravity(Vec3(0,0,0)); s.set_damping(0.05f);
    const float cx = 0.7f, cy = 0.7f, cz = 0.7f, sp = 0.025f, density = 1000.0f;
    const float hx = 0.20f, hy = 0.05f, hz = 0.05f;
    const float pmass = density * sp * sp * sp;
    for (float x = cx-hx; x <= cx+hx+1e-6f; x += sp)
    for (float y = cy-hy; y <= cy+hy+1e-6f; y += sp)
    for (float z = cz-hz; z <= cz+hz+1e-6f; z += sp)
        s.add_particle(Vec3(x,y,z), pmass, density);
    for (auto& p : s.particles())
        p.Fg = Matrix3::scale(Vec3(1,1,1) * ((p.x.y >= cy) ? g_t : g_b));

    const float dt = 1.0e-3f;
    for (int m = 0; m < 4000; ++m) s.step(dt);

    const int np = (int)s.particles().size();
    mean_eps = 0.0; float xlo = 1e9f, xhi = -1e9f;
    for (const auto& p : s.particles()) {
        float en = 0.0f; s.kirchhoff_stress(p.F, p.Fg, &en); mean_eps += en;
        xlo = std::min(xlo, p.x.x); xhi = std::max(xhi, p.x.x);
    }
    mean_eps /= (double)np;
    double sy[3] = {0,0,0}; int cnt[3] = {0,0,0};
    const float bw = (xhi - xlo) / 3.0f + 1e-9f;
    for (const auto& p : s.particles()) { int b = std::min(2, (int)((p.x.x - xlo)/bw)); sy[b] += p.x.y; ++cnt[b]; }
    const float ml = (float)(sy[0]/cnt[0]), mc = (float)(sy[1]/cnt[1]), mr = (float)(sy[2]/cnt[2]);
    bend = std::fabs(mc - 0.5f*(ml + mr));

    if (dump) {
        std::filesystem::create_directories("outputs/figures");
        std::ofstream f("outputs/figures/differential_growth.csv");
        f << "x,y,z,eps_e,layer\n";
        for (const auto& p : s.particles()) {
            float en = 0.0f; s.kirchhoff_stress(p.F, p.Fg, &en);
            f << p.x.x << "," << p.x.y << "," << p.x.z << "," << en << ","
              << ((p.x.y >= cy) ? "top" : "bot") << "\n";
        }
        std::printf("       midline mean-y [L,C,R] = [%.4f, %.4f, %.4f]\n", ml, mc, mr);
        std::printf("       wrote outputs/figures/differential_growth.csv\n");
    }
}

// ── T3: differential (incompatible) growth -> residual stress + bend ─────────
static void test_differential_growth() {
    std::printf("[T3] differential growth (bilayer) -> residual stress + bend\n");
    double mean_eps; float bend;
    run_bilayer(1.10f, 1.50f, /*dump=*/true, mean_eps, bend);
    std::printf("       mean ||eps_e|| = %.4f, bend = %.4f m\n", mean_eps, bend);
    CHECK(mean_eps > 0.02f, "incompatible growth sustains residual stress (>> free-swell floor)");
    CHECK(bend > 0.010f, "initially-flat strip acquires curvature (differential-growth bending)");
}

// ── T4: objectivity / frame-indifference ─────────────────────────────────────
// Superposed rigid rotation: τ(R·F) must equal R·τ(F)·Rᵀ.
static void test_objectivity() {
    std::printf("[T4] objectivity: τ(R·F) = R·τ(F)·Rᵀ\n");
    GrowthSolver s(8, 0.1f);
    s.set_material(1.0e4f, 0.3f);
    Matrix3 F0 = Matrix3::from_columns(Vec3(1.2f,0.10f,0.0f), Vec3(0.0f,0.9f,0.05f), Vec3(0.03f,0.0f,1.1f));
    Matrix3 Fg = Matrix3::scale(Vec3(1.3f, 1.0f, 0.8f));
    Matrix3 tau0 = s.kirchhoff_stress(F0, Fg);
    Matrix3 R    = Matrix3::from_axis_angle(Vec3(0.3f,0.7f,0.6f).normalized(), 0.9f);
    Matrix3 tau1 = s.kirchhoff_stress(R * F0, Fg);
    Matrix3 expected = R * tau0 * R.transposed();
    const float d = max_abs_diff(tau1, expected);
    float scale = 0.0f;
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) scale = std::max(scale, std::fabs(tau0.m[r][c]));
    std::printf("       max |τ(RF) − RτRᵀ| = %.4g Pa  (stress scale %.4g Pa)\n", d, scale);
    CHECK(d < scale * 1e-3f + 1e-2f, "stress is frame-indifferent under rigid rotation");
}

// ── T5: uniform anisotropic free growth -> stress-free, det F -> det Fg ──────
static void test_free_growth_anisotropic() {
    std::printf("[T5] uniform anisotropic free growth -> stress-free\n");
    GrowthSolver s(24, 0.05f);
    s.set_material(1.0e4f, 0.3f); s.set_gravity(Vec3(0,0,0)); s.set_damping(0.05f);
    const int np = seed_blob(s, 0.6f, 0.10f, 0.025f, 1000.0f);
    const Vec3 fac(1.3f, 0.8f, 1.1f); const int K = 10, M = 250; const float dt = 1.0e-3f;
    const Vec3 per(std::pow(fac.x,1.0f/K), std::pow(fac.y,1.0f/K), std::pow(fac.z,1.0f/K));
    for (int kk = 0; kk < K; ++kk) { s.grow_all_anisotropic(per); for (int m = 0; m < M; ++m) s.step(dt); }

    double mean_detF = 0.0; float max_eps = 0.0f;
    for (const auto& p : s.particles()) {
        mean_detF += p.F.determinant();
        float en = 0.0f; s.kirchhoff_stress(p.F, p.Fg, &en); max_eps = std::max(max_eps, en);
    }
    mean_detF /= (double)np;
    const double expected = (double)fac.x * fac.y * fac.z;   // det Fg
    std::printf("       %d particles; mean det F = %.4f (expect det Fg = %.4f); max ||eps_e|| = %.4f\n",
                np, mean_detF, expected, max_eps);
    CHECK(approx((float)mean_detF, (float)expected, 0.08f*(float)expected), "det F -> det Fg (anisotropic)");
    CHECK(max_eps < 0.06f, "uniform anisotropic growth is compatible -> stress-free");
}

// ── T6: determinism ──────────────────────────────────────────────────────────
static std::vector<Particle> run_iso_swell() {
    GrowthSolver s(24, 0.05f);
    s.set_material(1.0e4f, 0.3f); s.set_gravity(Vec3(0,0,0)); s.set_damping(0.05f);
    seed_blob(s, 0.6f, 0.10f, 0.025f, 1000.0f);
    const int K = 10, M = 250; const float dt = 1.0e-3f, per = std::pow(1.5f, 1.0f/K);
    for (int kk = 0; kk < K; ++kk) { s.grow_all_isotropic(per); for (int m = 0; m < M; ++m) s.step(dt); }
    return s.particles();
}
static void test_determinism() {
    std::printf("[T6] determinism (identical runs -> bit-identical)\n");
    auto a = run_iso_swell(), b = run_iso_swell();
    float maxd = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) maxd = std::max(maxd, (a[i].x - b[i].x).length());
    std::printf("       max position diff over %zu particles = %.3g m\n", a.size(), maxd);
    CHECK(maxd == 0.0f, "two identical runs produce bit-identical results");
}

// ── T7: bend increases monotonically with growth mismatch ────────────────────
static void test_bend_monotonic() {
    std::printf("[T7] bend monotonic in growth mismatch\n");
    double me; float bend_small, bend_large;
    run_bilayer(1.10f, 1.20f, false, me, bend_small);
    run_bilayer(1.10f, 1.50f, false, me, bend_large);
    std::printf("       bend(Δg=0.10) = %.4f m  <  bend(Δg=0.40) = %.4f m\n", bend_small, bend_large);
    CHECK(bend_large > bend_small, "larger growth mismatch -> larger curvature");
    CHECK(bend_small > 0.0f, "small mismatch still bends");
}

// ── T8: bilayer bending curvature vs Timoshenko analytic (STRONG oracle) ─────
// Thin bilayer (h/L≈0.1), small growth mismatch → small curvature (linear regime).
// Independent analytic (equal thickness & modulus): κ = 1.5·ε_m / h,  ε_m = g_t-g_b.
// Catches coefficient bugs free-swell can't (nonzero, modulus-coupled morphology).
static float bilayer_curvature(float g_b, float g_t, bool axial = false, int res = 1,
                               int steps = 4000) {
    GrowthSolver s(48 * res, 0.025f / res);                 // domain 1.2 fixed; h/dx = 2·res
    s.set_material(1.0e4f, 0.3f); s.set_gravity(Vec3(0,0,0)); s.set_damping(0.05f);
    const float cx = 0.6f, cy = 0.6f, cz = 0.6f, sp = 0.0125f / res, density = 1000.0f;
    const float hx = 0.25f, hy = 0.025f, hz = 0.0125f;      // L=0.5, h=0.05, h/L=0.1
    const float pmass = density * sp * sp * sp;
    for (float x = cx-hx; x <= cx+hx+1e-6f; x += sp)
    for (float y = cy-hy; y <= cy+hy+1e-6f; y += sp)
    for (float z = cz-hz; z <= cz+hz+1e-6f; z += sp)
        s.add_particle(Vec3(x,y,z), pmass, density);
    for (auto& p : s.particles()) {
        const float gg = (p.x.y >= cy) ? g_t : g_b;
        // axial: growth only along beam axis x (Timoshenko's axial eigenstrain);
        // isotropic: also transverse (y,z) — tests whether transverse growth is
        // the source of the ~0.58 deviation.
        p.Fg = axial ? Matrix3::scale(Vec3(gg, 1.0f, 1.0f))
                     : Matrix3::scale(Vec3(gg, gg, gg));
    }
    const float dt = 1.0e-3f;
    for (int m = 0; m < steps; ++m) s.step(dt);

    // centerline: mean y per x-bin, then least-squares quadratic (x centered).
    const int NB = 9;
    double sxb[NB] = {0}, syb[NB] = {0}; int cnt[NB] = {0};
    float xlo = 1e9f, xhi = -1e9f;
    for (const auto& p : s.particles()) { xlo = std::min(xlo, p.x.x); xhi = std::max(xhi, p.x.x); }
    const float bw = (xhi - xlo) / NB + 1e-9f;
    for (const auto& p : s.particles()) {
        int b = std::min(NB-1, std::max(0, (int)((p.x.x - xlo)/bw)));
        sxb[b] += p.x.x; syb[b] += p.x.y; ++cnt[b];
    }
    double xm = 0; int used = 0;
    for (int b = 0; b < NB; ++b) if (cnt[b]) { xm += sxb[b]/cnt[b]; ++used; }
    xm /= used;
    // normal equations for y = a·xc^2 + b·xc + c   (xc = x - xm)
    double A[3][3] = {{0}}, r[3] = {0};
    for (int b = 0; b < NB; ++b) if (cnt[b]) {
        double xc = sxb[b]/cnt[b] - xm, y = syb[b]/cnt[b];
        double p2 = xc*xc;
        A[0][0]+=p2*p2; A[0][1]+=p2*xc; A[0][2]+=p2;
        A[1][1]+=p2;    A[1][2]+=xc;    A[2][2]+=1;
        r[0]+=p2*y; r[1]+=xc*y; r[2]+=y;
    }
    A[1][0]=A[0][1]; A[2][0]=A[0][2]; A[2][1]=A[1][2];
    Matrix3 M = Matrix3::from_rows(Vec3((float)A[0][0],(float)A[0][1],(float)A[0][2]),
                                   Vec3((float)A[1][0],(float)A[1][1],(float)A[1][2]),
                                   Vec3((float)A[2][0],(float)A[2][1],(float)A[2][2]));
    Vec3 abc = M.inversed() * Vec3((float)r[0], (float)r[1], (float)r[2]);
    return std::fabs(2.0f * abc.x);                          // κ ≈ 2a (small slope)
}

static void test_bilayer_timoshenko() {
    std::printf("[T8] bilayer bending curvature vs Timoshenko analytic\n");
    const float h = 0.05f;                                   // total thickness
    const float e1 = 0.02f, e2 = 0.04f;
    const float k1 = bilayer_curvature(1.0f, 1.0f + e1);
    const float k2 = bilayer_curvature(1.0f, 1.0f + e2);
    const float kT1 = 1.5f * e1 / h, kT2 = 1.5f * e2 / h;    // κ = 1.5 ε_m / h
    std::printf("       ISOTROPIC growth:\n");
    std::printf("         ε=%.2f: κ_sim=%.3f  κ_T=%.3f  ratio=%.2f\n", e1, k1, kT1, k1/kT1);
    std::printf("         ε=%.2f: κ_sim=%.3f  κ_T=%.3f  ratio=%.2f\n", e2, k2, kT2, k2/kT2);
    std::printf("         linear scaling κ(2ε)/κ(ε) = %.2f (predicts 2.0)\n", k2/k1);
    // DIAGNOSTIC: axial-only growth isolates the transverse-growth effect on 0.58.
    const float ka1 = bilayer_curvature(1.0f, 1.0f + e1, /*axial=*/true);
    const float ka2 = bilayer_curvature(1.0f, 1.0f + e2, /*axial=*/true);
    std::printf("       AXIAL-ONLY growth (Fg=diag(g,1,1); Timoshenko's assumption):\n");
    std::printf("         ε=%.2f: κ_sim=%.3f  κ_T=%.3f  ratio=%.2f\n", e1, ka1, kT1, ka1/kT1);
    std::printf("         ε=%.2f: κ_sim=%.3f  κ_T=%.3f  ratio=%.2f\n", e2, ka2, kT2, ka2/kT2);
    std::printf("       => isotropic %.2f vs axial %.2f: transverse growth %s the deviation\n",
                k1/kT1, ka1/kT1, (ka1/kT1 > k1/kT1 + 0.1f) ? "explains part of" : "does NOT explain");
    // DIAGNOSTIC: grid-resolution convergence (h/dx = 2·res). If ratio -> 1 as
    // dx refines, the deviation is discretization (bending under-resolved).
    std::printf("       GRID CONVERGENCE (ε=%.2f, h/dx = 2·res):\n", e1);
    for (int res = 1; res <= 3; ++res) {
        float kr = bilayer_curvature(1.0f, 1.0f + e1, false, res);
        std::printf("         res=%d (h/dx=%d): κ_sim=%.3f  ratio=%.2f\n", res, 2*res, kr, kr/kT1);
    }
    // relaxation convergence: is 4000 steps enough for the slow bending mode?
    const float kr4 = bilayer_curvature(1.0f, 1.0f + e1, false, 1, 4000);
    const float kr12 = bilayer_curvature(1.0f, 1.0f + e1, false, 1, 12000);
    std::printf("       RELAXATION (ε=%.2f, res=1): 4000 steps ratio=%.2f, 12000 ratio=%.2f => %s\n",
                e1, kr4/kT1, kr12/kT1, (kr12/kT1 > kr4/kT1 + 0.05f) ? "UNDER-RELAXED" : "converged");
    CHECK(k1/kT1 > 0.4f && k1/kT1 < 1.7f, "curvature within ~Timoshenko (finite-thickness tol)");
    CHECK(k2/k1 > 1.6f && k2/k1 < 2.4f, "curvature scales ~linearly with growth mismatch");
}

// Dump a small Hencky free-swell reference for the numpy↔C++ parity gate
// (scripts/parity.py runs the IDENTICAL scenario in numpy and compares F).
// Scenario is fixed & tiny so numpy can match it exactly: N=12, dx=0.05,
// 27-particle cube, Fg=1.05·I, 30 steps, dt=1e-3, damping=0.05.
static void dump_parity_reference() {
    GrowthSolver s(12, 0.05f);
    s.set_material(1.0e4f, 0.3f); s.set_gravity(Vec3(0,0,0)); s.set_damping(0.05f);
    const float sp = 0.03f, density = 1000.0f, pmass = density * sp * sp * sp;
    for (float x = 0.27f; x <= 0.33f + 1e-6f; x += sp)
    for (float y = 0.27f; y <= 0.33f + 1e-6f; y += sp)
    for (float z = 0.27f; z <= 0.33f + 1e-6f; z += sp)
        s.add_particle(Vec3(x, y, z), pmass, density);
    for (auto& p : s.particles()) p.Fg = Matrix3::scale(Vec3(1.05f, 1.05f, 1.05f));
    for (int m = 0; m < 30; ++m) s.step(1.0e-3f);
    std::filesystem::create_directories("outputs");
    std::ofstream f("outputs/parity_ref.csv");
    f << "x,y,z,F00,F01,F02,F10,F11,F12,F20,F21,F22\n";
    for (const auto& p : s.particles()) {
        f << p.x.x << "," << p.x.y << "," << p.x.z;
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) f << "," << p.F.m[r][c];
        f << "\n";
    }
    std::printf("[parity] wrote outputs/parity_ref.csv (%d particles, Hencky free-swell)\n",
                (int)s.particles().size());
}

int main() {
    std::printf("== morphompm v1: morphoelastic growth in MLS-MPM ==\n\n");
    dump_parity_reference();
    test_constitutive_isotropic();   std::printf("\n");
    test_constitutive_anisotropic(); std::printf("\n");
    test_free_swelling();            std::printf("\n");
    test_differential_growth();      std::printf("\n");
    test_objectivity();              std::printf("\n");
    test_free_growth_anisotropic();  std::printf("\n");
    test_determinism();              std::printf("\n");
    test_bend_monotonic();           std::printf("\n");
    test_bilayer_timoshenko();       std::printf("\n");
    if (g_fail == 0) { std::printf("ALL CHECKS PASSED\n"); return 0; }
    std::printf("%d CHECK(S) FAILED\n", g_fail);
    return 1;
}
