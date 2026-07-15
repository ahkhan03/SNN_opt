"""Tests for the transform axis (snn_opt.transforms).

A transform must produce a solution equivalent to the canonical solve. The
eigenbasis transform is exercised across every backend (it must compose with
all of them), plus its applicability guards (box constraints rejected, unknown
name rejected).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from snn_opt import (  # noqa: E402
    OptimizationProblem, SNNSolver, SolverConfig, EigenbasisTransform,
)

# 'c_openmp' only when the compiled kernel has OpenMP; 'c'/'c_serial' need the
# kernel at all. Build the backend list from what's available.
try:
    from snn_opt import _kernel
    _HAS_KERNEL = True
    _HAS_OMP = bool(getattr(_kernel, "HAS_OPENMP", False))
except Exception:
    _HAS_KERNEL = False
    _HAS_OMP = False

_BACKENDS = ["python"] + (["c", "c_serial"] if _HAS_KERNEL else []) \
    + (["c_openmp"] if _HAS_OMP else [])


def _qp(n, m, seed):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((n, n))
    A = Q @ Q.T + 0.1 * np.eye(n)            # symmetric PSD
    b = rng.standard_normal(n)
    C = rng.standard_normal((m, n))
    d = -np.ones(m)
    x0 = rng.standard_normal(n) * 0.1
    return OptimizationProblem(A=A, b=b, C=C, d=d), x0


def _solve(prob, x0, backend, transform=None, max_it=4000):
    cfg = SolverConfig(max_iterations=max_it, backend=backend,
                       transform=transform, record_trajectory=False)
    return SNNSolver(prob, cfg).solve(x0)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("n,m,seed", [(20, 10, 1), (80, 40, 3), (150, 70, 5)])
def test_eigenbasis_matches_canonical(backend, n, m, seed):
    """Eigenbasis solve == canonical solve, on every backend."""
    prob, x0 = _qp(n, m, seed)
    ref = _solve(prob, x0, "python", transform=None)
    got = _solve(prob, x0, backend, transform="eigenbasis")
    dx = float(np.max(np.abs(got.final_x - ref.final_x)))
    assert dx < 1e-7, f"{backend}: eigenbasis diverged from canonical by {dx:.2e}"
    fscale = max(abs(ref.final_objective), 1e-12)
    assert abs(got.final_objective - ref.final_objective) / fscale < 1e-9


def test_eigenbasis_instance_and_string_agree():
    prob, x0 = _qp(60, 30, 7)
    r_str = _solve(prob, x0, "python", transform="eigenbasis")
    r_obj = _solve(prob, x0, "python", transform=EigenbasisTransform())
    assert np.max(np.abs(r_str.final_x - r_obj.final_x)) == 0.0


def test_eigenbasis_supports_box_via_rotated_facet_rows():
    """v0.5.0: box bounds are folded into explicit rotated unit-norm rows
    (the box is not axis-aligned in the eigenbasis), and the solution must be
    jointly feasible against the ORIGINAL box + rows."""
    prob, x0 = _qp(20, 5, 2)
    cfg = SolverConfig(max_iterations=4000, transform="eigenbasis",
                       lower_bound=0.0, upper_bound=1.0)
    res = SNNSolver(prob, cfg).solve(x0)
    assert res.max_violation_box <= 1e-5
    assert res.max_distance_rows <= 1e-5
    assert np.all(res.final_x >= -1e-5) and np.all(res.final_x <= 1.0 + 1e-5)


def test_unknown_transform_raises():
    prob, x0 = _qp(20, 5, 2)
    cfg = SolverConfig(max_iterations=100, transform="does-not-exist")
    with pytest.raises(ValueError, match="unknown transform"):
        SNNSolver(prob, cfg).solve(x0)


def test_eigenbasis_unconstrained():
    """No constraints (m=0): transform must still round-trip correctly."""
    rng = np.random.default_rng(11)
    n = 40
    Q = rng.standard_normal((n, n))
    prob = OptimizationProblem(A=Q @ Q.T + 0.5 * np.eye(n),
                               b=rng.standard_normal(n),
                               C=np.zeros((0, n)), d=np.zeros(0))
    x0 = rng.standard_normal(n) * 0.1
    ref = _solve(prob, x0, "python", transform=None)
    got = _solve(prob, x0, "python", transform="eigenbasis")
    assert np.max(np.abs(got.final_x - ref.final_x)) < 1e-7
