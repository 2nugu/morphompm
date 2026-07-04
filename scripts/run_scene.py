"""Demonstrate the load -> simulate -> export loop, and hand interactivity to a
mature tool. We do NOT build an interactive UX; we export a VTK time-sequence that
ParaView/VisIt animate — rotate, zoom, pick, color-by-field, time-scrub — for free.

    python scripts/run_scene.py
    # then open outputs/scene/frame_*.vtk in ParaView and press Play.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import numpy as np
from morphompm.config import SimConfig
from morphompm.constitutive import NeoHookean
from morphompm.integrate import rollout, iso_growth_state
from morphompm.io import save_state, save_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "outputs", "scene")
os.makedirs(SCENE, exist_ok=True)


def main():
    cfg = SimConfig(N=16, dx=0.05, dt=1.0e-3, damping=0.05, sp=0.03)
    model = NeoHookean(cfg.material)
    c = np.arange(0.40 - 0.03, 0.40 + 0.03 + 1e-9, 0.03)
    pts = np.array([(x, y, z) for x in c for y in c for z in c])   # 27-particle cube
    save_config(cfg, os.path.join(SCENE, "config.json"))

    st = iso_growth_state(pts, 1.0)
    g_target, frames, relax = 1.6, 12, 60
    save_state(st, os.path.join(SCENE, "frame_0000.vtk"))
    for k in range(1, frames + 1):
        g = 1.0 + (g_target - 1.0) * k / frames
        st.Fg[:] = g * np.eye(3)
        st, _ = rollout(st, model, cfg, relax, advect=True)
        save_state(st, os.path.join(SCENE, f"frame_{k:04d}.vtk"))
    save_state(st, os.path.join(SCENE, "final.npz"))              # lossless canonical
    d = np.mean([np.linalg.det(st.F[p]) for p in range(st.n)])
    print(f"simulated {st.n} particles, {frames} frames -> {SCENE}")
    print(f"final mean det F = {d:.3f} (g^3 = {g_target**3:.3f})")
    print("open outputs/scene/frame_*.vtk in ParaView, press Play  (rotate/zoom/pick/color free)")


if __name__ == "__main__":
    main()
