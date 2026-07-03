"""[7] Verification harness — runs every stage's gate as one command.

    python -m morphompm.verify

Each stage gate is FD-based (analytic adjoints checked against finite differences)
or oracle parity. A green run here is the pipeline's regression guard.
"""
from . import constitutive, transfer, integrate, diff


def run() -> int:
    print("################  morphompm pipeline verify  ################\n")
    print("### [2] constitutive: forward + VJP vs FD ###")
    c_fail = constitutive.main()

    print("\n### [3] transfer: single-step assembly gate ###")
    t_fail = transfer.main()

    print("\n### [1p] forward physics guard (det F -> g^3) ###")
    p_fail = integrate.forward_physics_gate()

    print("\n### [4] integrate: multi-step trajectory gate ###")
    i_fail = integrate.main()

    print("\n### [6] diff: inverse growth-rate recovery ###")
    d_fail = diff.main()

    ok = (c_fail == 0) and (t_fail == 0) and (p_fail == 0) and (i_fail == 0) and (d_fail == 0)
    print("\n############################################################")
    print("PIPELINE VERIFY:", "ALL GATES PASS" if ok else "FAILURE")
    print("############################################################")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
