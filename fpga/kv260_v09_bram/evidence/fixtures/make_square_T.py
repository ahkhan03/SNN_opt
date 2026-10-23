#!/usr/bin/env python3
"""Square alt-investment MSRPDL1 bundles for the v09b parity gate.

Mirrors analysis/1_scripts/22a_fpga_v08.py build_problems(): the same
Luxenberg-Boyd reduction, the same REF constants, k0 = 0.05 / lambda_max(A)
and row scales from the paper venv's snn_opt (box-in-sweep). K=2000 and
K=30000 share every array and the projection cap; only the iteration field
changes. projmax stays 10000, the per-iteration cap the paper ships.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

# The paper tree is not vendored. Point ALT_INVEST_PAPER_ROOT at the
# alt-investment paper root (the directory holding analysis/common/
# alt_invest_qp.py) and run this inside that paper's venv.
PAPER = Path(os.environ.get("ALT_INVEST_PAPER_ROOT", ""))
ANALYSIS = PAPER / "analysis"
if not (ANALYSIS / "common" / "alt_invest_qp.py").is_file():
    raise SystemExit("set ALT_INVEST_PAPER_ROOT to the alt-investment paper root")
sys.path.insert(0, str(ANALYSIS))

from common.alt_invest_qp import build_qp  # noqa: E402
from snn_opt import OptimizationProblem, SNNSolver, SolverConfig  # noqa: E402
import snn_opt  # noqa: E402

OUT = Path(__file__).resolve().parent
T_VALUES = (16, 32, 64, 80)
K_VALUES = (2000, 30000)
REF = dict(r=0.04, gamma_call=0.23, gamma_dist=0.15,
           c_max=4.0, u_max=10.0, n_des=15.0, lmbda=5.0)
K0_SCALE = 0.05
PROJMAX = 10000
CTOL = 1e-6


def solver_terms(A, b, C, d):
    prob = OptimizationProblem(A=A, b=b, C=C, d=d)
    cfg = SolverConfig(
        k0=None, k0_scale=K0_SCALE, max_iterations=30000,
        max_projection_iters=PROJMAX, projection_method="adaptive",
        lower_bound=0.0, upper_bound=REF["c_max"], backend="python",
        record_trajectory=False,
    )
    solver = SNNSolver(prob, cfg)
    return (float(solver._k0),
            np.asarray(solver._c_norms_sq, float).copy(),
            np.asarray(solver._row_scale, float).copy())


def write_bundle(path: Path, n, m, K, k0, A, b, C, d, cns, scale, G, x0):
    def f64(a):
        return np.ascontiguousarray(a, dtype="<f8").ravel().tobytes()

    blob = bytearray(b"MSRPDL1\0")
    blob += struct.pack("<7I", 1, n, m, K, PROJMAX, 1, 1)
    blob += struct.pack("<4d", k0, CTOL, 0.0, float(REF["c_max"]))
    for arr in (A, b, C, d, cns, scale, G, x0):
        blob += f64(arr)
    path.write_bytes(blob)
    return hashlib.sha256(blob).hexdigest(), len(blob)


def main() -> int:
    rows = []
    for T in T_VALUES:
        A, b, C, d, *_rest = build_qp(
            T, REF["gamma_call"], REF["gamma_dist"], REF["r"],
            REF["n_des"], REF["lmbda"], REF["c_max"], REF["u_max"])
        A, b, C, d = (np.asarray(a, float) for a in (A, b, C, d))
        n = int(A.shape[0])
        m = int(C.shape[0])
        if n != T or m != T:
            raise SystemExit(f"T={T} came out n={n} m={m}")
        k0, cns, scale = solver_terms(A, b, C, d)
        G = C @ C.T
        x0 = np.zeros(n)
        absmax = {
            name: float(np.max(np.abs(np.asarray(arr, float))))
            for name, arr in (
                ("A", A), ("b", b), ("C", C), ("d", d),
                ("cns", cns), ("row_scale", scale), ("G", G), ("x0", x0))
        }
        if any(v >= 128.0 for v in absmax.values()):
            raise SystemExit(f"T={T} leaves ap_fixed<32,8>: {absmax}")
        for K in K_VALUES:
            path = OUT / f"square_T{T}_K{K}.bin"
            digest, nbytes = write_bundle(
                path, n, m, K, k0, A, b, C, d, cns, scale, G, x0)
            rows.append({
                "file": path.name,
                "T": T,
                "n": n,
                "m": m,
                "K": K,
                "projmax": PROJMAX,
                "k0": k0,
                "ctol": CTOL,
                "lower": 0.0,
                "upper": float(REF["c_max"]),
                "sha256": digest,
                "nbytes": nbytes,
                "absmax": absmax,
            })
            print(f"wrote {path.name} k0={k0:.8g} sha256={digest} bytes={nbytes}")
    meta = {
        "provenance": (
            "Equivalent to 22a_fpga_v08.py build_problems() in "
            "2026-07-06_snn-alt-investment. Matrices from "
            "common.alt_invest_qp.build_qp. k0 and row_scale from the paper "
            "venv snn_opt SNNSolver (k0_scale=0.05, lower=0, upper=c_max=4). "
            "G = C @ C.T. x0 = 0. K=2000 and K=30000 differ only in the "
            "iteration header. projmax stays 10000."
        ),
        "paper_script": str(ANALYSIS / "1_scripts" / "22a_fpga_v08.py"),
        "snn_opt_file": snn_opt.__file__,
        "snn_opt_version": getattr(snn_opt, "__version__", "unknown"),
        "ref": REF,
        "bundles": rows,
    }
    (OUT / "provenance.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("snn_opt", meta["snn_opt_version"], meta["snn_opt_file"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
