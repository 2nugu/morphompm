"""[6] Diff / inverse — recover a growth parameter from an observed shape, using
the trajectory gradient ([4]). Gauss-Newton on g:
    loss(g)=Σ det F_final ;  observe target=loss(g_true);  g <- g - (loss-target)/(dloss/dg).
"""
import numpy as np
from .integrate import dloss_dg, rollout, iso_growth_state, loss_detF


def infer_growth(pts, model, cfg, target, n_steps, g_init, advect=False, iters=20, tol=1e-9):
    g = g_init
    for it in range(iters):
        loss, dg = dloss_dg(pts, model, cfg, g, n_steps, advect)
        resid = loss - target
        if abs(resid) < tol:
            break
        if abs(dg) < 1e-12 or not np.isfinite(dg):    # guard: dg crosses 0 (root ~g≈0.6)
            g += 1e-3 * (1.0 if resid < 0 else -1.0)   # nudge off the stationary point
            continue
        step = resid / dg
        g -= max(-0.5, min(0.5, step))                 # damp: cap |Δg| for robustness
    return g, it + 1


def main():
    from .config import SimConfig
    from .constitutive import NeoHookean
    cfg = SimConfig()
    model = NeoHookean(cfg.material)
    pts = np.array([[0.40, 0.40, 0.40], [0.42, 0.40, 0.41], [0.40, 0.43, 0.40],
                    [0.41, 0.40, 0.44], [0.44, 0.44, 0.43]])
    n_steps = 3
    g_true = 1.40
    st, _ = rollout(iso_growth_state(pts, g_true), model, cfg, n_steps)
    target = loss_detF(st.F)
    print("== diff: inverse growth-rate recovery ==")
    print(f"    observed shape from g_true={g_true} -> target Σdet(F)={target:.6f}")
    g_rec, iters = infer_growth(pts, model, cfg, target, n_steps, g_init=1.15)
    err = abs(g_rec - g_true)
    print(f"    recovered g={g_rec:.6f} in {iters} iters (err {err:.2e})")
    ok = err < 1e-4
    print(f"    [{'ok' if ok else 'FAIL'}] gradient-based inversion recovers the growth rate")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
