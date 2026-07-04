#!/usr/bin/env python3
"""One-command reproduction of morphompm's verified results.

Reproducibility is a FIRST-CLASS research element here — reproduction results
(forward validation vs analytic/literature oracles) are preserved and analyzed
ALONGSIDE the novel differentiable analyses, and are re-runnable by anyone:

    python scripts/reproduce.py

Runs (1) the Python 3-axis verification (deterministic: FD-gradient + forward-physics
+ confined-swell analytic), and — if the C++ oracle is built — (2) its analytic +
Timoshenko-bending tests and (3) regenerates the differential-growth figure.
Determinism: single-thread CPU, no RNG in the physics → bit-identical re-runs.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYDIR = os.path.join(ROOT, "python")


def _env_stamp():
    print("environment:")
    print(f"  python  = {sys.version.split()[0]}")
    try:
        import numpy
        print(f"  numpy   = {numpy.__version__}")
    except Exception:
        print("  numpy   = (missing)")
    try:
        h = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True)
        if h.returncode == 0:
            print(f"  git     = {h.stdout.strip()}")
    except Exception:
        pass


def run(cmd, cwd, label):
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd).returncode == 0


def main():
    print("############ morphompm reproduce ############")
    _env_stamp()
    ok = True

    ok &= run([sys.executable, "-m", "morphompm.verify"], PYDIR,
              "[1] Python 3-axis verify (FD-gradient + forward-physics + analytic)")

    exe = os.path.join(ROOT, "build", "Release", "test_growth.exe")
    if os.path.exists(exe):
        ok &= run([exe], ROOT, "[2] C++ oracle (T1-T8: analytic + Timoshenko bending)")
        run([sys.executable, os.path.join("scripts", "plot_growth.py")], ROOT,
            "[3] regenerate preserved figure (differential_growth.*)")
    else:
        print("\n[2] C++ oracle not built (skipped).")
        print("    build:  cmake -S . -B build -G \"Visual Studio 17 2022\" -A x64")
        print("            cmake --build build --config Release")

    # regenerate the presented artifacts from the source of truth (no hand-editing)
    ok &= run([sys.executable, os.path.join(ROOT, "scripts", "results.py")], PYDIR,
              "[4] regenerate result figures (inverse_recovery, bending_timoshenko)")
    ok &= run([sys.executable, os.path.join(ROOT, "scripts", "make_dashboard.py")], ROOT,
              "[5] regenerate dashboard (embeds current figures + live git hash)")

    print("\n############ REPRODUCE:", "ALL OK" if ok else "FAILURE", "############")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
