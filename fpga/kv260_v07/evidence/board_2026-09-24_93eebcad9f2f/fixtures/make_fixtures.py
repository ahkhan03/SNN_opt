#!/usr/bin/env python3
"""Create the four v07 C-simulation problem bundles and cone sidecars."""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
NATIVE = ROOT / "fpga/kv260_v07/build/work/native-c4b/native_fixed_v07"
V06 = ROOT / "fpga/kv260_v06/evidence/board_2026-09-06_37f33fcfe2ca/fixtures/prob_6x18.bin"


# Match the standard MSRPDL1 bundle consumed by msrp_v05::load_problem.
def write_problem(
    path: Path, A, b, C, d, cns, scale, G, x0, *, k0, ctol, iters, cap, lower=None, upper=None
) -> None:
    A = np.asarray(A, float)
    b = np.asarray(b, float).reshape(-1)
    n = b.size
    C = np.asarray(C, float).reshape(-1, n)
    d = np.asarray(d, float).reshape(-1)
    m = C.shape[0]
    cns = np.asarray(cns, float).reshape(m)
    scale = np.asarray(scale, float).reshape(m)
    G = np.asarray(G, float).reshape(m, m)
    x0 = np.asarray(x0, float).reshape(n)
    with path.open("wb") as f:
        f.write(b"MSRPDL1\0")
        f.write(
            struct.pack(
                "<7I", 1, n, m, int(iters), int(cap), int(lower is not None), int(upper is not None)
            )
        )
        f.write(
            struct.pack(
                "<4d",
                float(k0),
                float(ctol),
                float(-1.0 if lower is None else lower),
                float(1.0 if upper is None else upper),
            )
        )
        for v in (A.ravel(), b, C.ravel(), d, cns, scale, G.ravel(), x0):
            f.write(np.asarray(v, dtype="<f8").tobytes())


def write_sidecar(path: Path, rows) -> None:
    path.write_text(
        "# kind offset length radius mu center\n"
        + "".join(
            f"{kind} {off} {length} {radius:.17g} {mu:.17g} {center:.17g}\n"
            for kind, off, length, radius, mu, center in rows
        ),
        encoding="utf-8",
    )


def run(name: str, sidecar: bool) -> None:
    problem = OUT / f"{name}.bin"
    expected = OUT / f"{name}.expected.bin"
    cmd = [str(NATIVE), str(problem), str(expected)]
    if sidecar:
        cmd += ["--cones", str(OUT / f"{name}.cones")]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


# (a) n=8 Euclidean ball anchor from run_cone_parity.sh.
n = 8
A = np.eye(n)
b = np.zeros(n)
C = np.zeros((1, n))
d = np.zeros(1)
write_problem(
    OUT / "a_ball_n8.bin",
    A,
    b,
    C,
    d,
    [1.0],
    [1.0],
    [[0.0]],
    np.r_[2.0, np.zeros(n - 1)],
    k0=0.03,
    ctol=1e-6,
    iters=3,
    cap=32,
)
write_sidecar(OUT / "a_ball_n8.cones", [("ball", 0, 8, 1.0, 1.0, 0.0)])
run("a_ball_n8", True)

# (b) q=1 friction anchor, seed 4903, mu=.4.
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from fpga.kv260_v07.src.kernel_model import friction_objective, k0_for  # noqa: E402

A, b, _ = friction_objective(4903, 1)
n = len(b)
C = np.zeros((1, n))
d = np.zeros(1)
write_problem(
    OUT / "b_friction_q1_seed4903.bin",
    A,
    b,
    C,
    d,
    [1.0],
    [1.0],
    [[0.0]],
    np.zeros(n),
    k0=k0_for(A),
    ctol=1e-6,
    iters=1024,
    cap=64,
)
write_sidecar(OUT / "b_friction_q1_seed4903.cones", [("scaled_soc", 0, 3, 0.0, 0.4, 0.0)])
run("b_friction_q1_seed4903", True)

# (c) mixed row + upper bound + ball + scaled SOC anchor.
n = 6
A = np.eye(n)
b = np.zeros(n)
C = np.array([[1.0, 0, 0, 0, 0, 0]])
d = np.array([-0.25])
write_problem(
    OUT / "c_mixed_anchor.bin",
    A,
    b,
    C,
    d,
    [1.0],
    [1.0],
    C @ C.T,
    np.array([2.0, 0, 0, -1, 2, 0.0]),
    k0=0.03,
    ctol=1e-6,
    iters=3,
    cap=32,
    upper=0.75,
)
write_sidecar(
    OUT / "c_mixed_anchor.cones",
    [("ball", 0, 3, 1.0, 1.0, 0.0), ("scaled_soc", 3, 3, 0.0, 0.4, 0.0)],
)
run("c_mixed_anchor", True)

# (d) Existing v06 board parity fixture, intentionally cones-off.
shutil.copyfile(V06, OUT / "d_cones_off_v06_parity.bin")
run("d_cones_off_v06_parity", False)
print("created", *sorted(p.name for p in OUT.iterdir()), sep="\n")
