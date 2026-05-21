"""Golden-parity harness: compiled C++ backend vs the Python lean solve path.

The C++ kernel (``snn_opt._kernel``, driven via ``backend='c'``) is a faithful
port of ``SNNSolver._solve_euler_lean``. This module asserts the two agree
across a battery of QPs and is the regression guard that keeps the native
kernel from silently drifting from the golden Python reference.

Numerical note: the C++ kernel uses an explicit-loop matvec while NumPy uses a
BLAS gemv; the two differ at the ULP level, so iterate trajectories drift by
~1e-12 over a run. Parity is therefore asserted with tolerances, not bitwise:
``final_x`` to 1e-7, ``converged`` exactly, and iteration/projection counts
within a small slack (they match exactly for well-conditioned problems).

Run directly for a detailed table (includes a wall-time speedup column)::

    python tests/test_c_backend_parity.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from snn_opt import OptimizationProblem, SNNSolver, SolverConfig  # noqa: E402

_kernel = pytest.importorskip(
    "snn_opt._kernel",
    reason="compiled C++ kernel not built (python setup.py build_ext --inplace)",
)


# --------------------------------------------------------------------------
# Problem battery
# --------------------------------------------------------------------------
def _random_qp(n, m, seed):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((n, n))
    A = Q @ Q.T + 0.1 * np.eye(n)
    b = rng.standard_normal(n)
    C = rng.standard_normal((m, n))
    d = -np.ones(m) - np.maximum(C @ np.zeros(n), 0)
    x0 = rng.standard_normal(n) * 0.1
    return dict(A=A, b=b, C=C, d=d, x0=x0, lower=None, upper=None)


def _box_qp(n, seed):
    """SVM-like: PSD Hessian, inactive linear row, box-clipped variables."""
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((n, n))
    A = Q @ Q.T + 0.1 * np.eye(n)
    b = rng.standard_normal(n)
    C = np.zeros((1, n))
    d = np.array([-1e9])
    x0 = np.full(n, 0.5)
    return dict(A=A, b=b, C=C, d=d, x0=x0, lower=0.0, upper=1.0)


def _unconstrained_qp(n, seed):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((n, n))
    A = Q @ Q.T + 0.5 * np.eye(n)
    b = rng.standard_normal(n)
    C = np.zeros((0, n))
    d = np.zeros(0)
    x0 = rng.standard_normal(n) * 0.1
    return dict(A=A, b=b, C=C, d=d, x0=x0, lower=None, upper=None)


def _well_conditioned_qp(n, seed):
    """Tight, well-conditioned problem that converges well before max_iter."""
    rng = np.random.default_rng(seed)
    A = np.eye(n) + 0.05 * rng.standard_normal((n, n))
    A = A @ A.T
    b = rng.standard_normal(n)
    C = rng.standard_normal((n // 2, n))
    d = -np.ones(n // 2) * 2.0
    x0 = np.zeros(n)
    return dict(A=A, b=b, C=C, d=d, x0=x0, lower=None, upper=None)


def _battery():
    cases = []
    for n, m, s in [(10, 5, 1), (50, 30, 7), (50, 30, 2),
                    (100, 60, 3), (200, 120, 4), (300, 150, 5)]:
        cases.append((f"random n={n} m={m} s{s}", _random_qp(n, m, s), 4000))
    for n, s in [(20, 11), (80, 12)]:
        cases.append((f"box n={n} s{s}", _box_qp(n, s), 4000))
    for n, s in [(30, 21), (120, 22)]:
        cases.append((f"unconstrained n={n} s{s}", _unconstrained_qp(n, s), 4000))
    for n, s in [(40, 31), (150, 32)]:
        cases.append((f"well-cond n={n} s{s}", _well_conditioned_qp(n, s), 6000))
    return cases


def _solve(p, backend, max_it):
    cfg = SolverConfig(max_iterations=max_it, backend=backend,
                       record_trajectory=False,
                       lower_bound=p["lower"], upper_bound=p["upper"])
    solver = SNNSolver(OptimizationProblem(A=p["A"], b=p["b"], C=p["C"], d=p["d"]),
                       cfg)
    t0 = time.perf_counter()
    res = solver.solve(p["x0"])
    return res, time.perf_counter() - t0


def _compare(p, max_it):
    py, t_py = _solve(p, "python", max_it)
    c, t_c = _solve(p, "c", max_it)
    dx = float(np.max(np.abs(py.final_x - c.final_x)))
    fscale = max(abs(py.final_objective), 1e-12)
    dobj = abs(py.final_objective - c.final_objective) / fscale
    return dict(py=py, c=c, t_py=t_py, t_c=t_c, dx=dx, dobj=dobj)


# --------------------------------------------------------------------------
# pytest cases
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name,p,max_it", _battery())
def test_c_backend_matches_python_lean(name, p, max_it):
    r = _compare(p, max_it)
    py, c = r["py"], r["c"]

    # solution agreement -- the primary correctness check
    assert r["dx"] < 1e-7, f"{name}: final_x diverged by {r['dx']:.2e}"
    assert r["dobj"] < 1e-9, f"{name}: objective diverged by {r['dobj']:.2e}"

    # convergence verdict must match exactly
    assert c.converged == py.converged, f"{name}: converged flag differs"

    # iteration / projection counts: exact for well-conditioned problems,
    # within a small slack where ULP-level matvec drift moves a borderline
    # convergence check or argmax tie.
    assert abs(c.iterations_used - py.iterations_used) <= 50, (
        f"{name}: iterations {py.iterations_used} (py) vs "
        f"{c.iterations_used} (c)")
    nproj_scale = max(py.n_projections, 1)
    assert abs(c.n_projections - py.n_projections) / nproj_scale < 0.01, (
        f"{name}: n_projections {py.n_projections} (py) vs "
        f"{c.n_projections} (c)")


# --------------------------------------------------------------------------
# Standalone detailed report
# --------------------------------------------------------------------------
def _main():
    print(f"{'case':<24}{'iters py/c':<16}{'nproj py/c':<20}"
          f"{'max|dx|':<11}{'|dobj|/f':<11}{'t_py(ms)':<10}"
          f"{'t_c(ms)':<10}{'speedup':<9}")
    print("-" * 110)
    all_ok = True
    speedups = []
    for name, p, max_it in _battery():
        r = _compare(p, max_it)
        py, c = r["py"], r["c"]
        speed = r["t_py"] / r["t_c"] if r["t_c"] > 0 else float("nan")
        speedups.append(speed)
        ok = (r["dx"] < 1e-7 and c.converged == py.converged
              and abs(c.iterations_used - py.iterations_used) <= 50)
        all_ok &= ok
        print(f"{name:<24}"
              f"{f'{py.iterations_used}/{c.iterations_used}':<16}"
              f"{f'{py.n_projections}/{c.n_projections}':<20}"
              f"{r['dx']:<11.2e}{r['dobj']:<11.2e}"
              f"{r['t_py'] * 1e3:<10.2f}{r['t_c'] * 1e3:<10.2f}"
              f"{speed:<9.1f}{'' if ok else '  ** FAIL **'}")
    print("-" * 110)
    print(f"median C-backend speedup vs Python lean: {np.median(speedups):.1f}x")
    print("ALL OK" if all_ok else "SOME CASES FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(_main())
