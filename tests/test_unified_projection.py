"""Tests for the v0.5.0 unified projection (clip-after-project fix).

Covers the converged spec of 2026-07-15 (paper workspace
discussions/agent_discussion/6_converged_spec.md):
  - mixed box+row problems reach JOINT feasibility (the old terminal clip
    left an infeasible fixed point that undercut the true optimum);
  - native implicit facets are trajectory-identical to the materialized
    box-as-rows encoding (frozen candidate ordering);
  - selection is invariant to positive row rescaling (normalized distances);
  - budget exhaustion aborts with its own status;
  - box-only problems keep the exact vectorized projection;
  - zero-row screening; eigenbasis+box via rotated facet rows;
  - the NNLS stationarity certificate behaves (near 0 on a benign problem).
"""

import numpy as np
import pytest

from snn_opt import (ConvergenceConfig, OptimizationProblem, SNNSolver,
                     SolverConfig)


def _markowitz(n=40, seed=0):
    """Long-only budget-capped Markowitz: the defect's canonical shape."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n, max(2, n // 3)))
    A = F @ F.T / F.shape[1] + np.diag(0.1 + 0.2 * rng.random(n))
    b = -(0.05 + 0.15 * rng.random(n)) / 5.0
    C = np.ones((1, n))
    d = np.array([-1.0])
    return A, b, C, d


def _box_as_rows(A, b, C, d, lo, hi):
    n = A.shape[0]
    eye = np.eye(n)
    C2 = np.vstack([C, -eye, eye])
    d2 = np.concatenate([d, np.full(n, lo), np.full(n, -hi)])
    return A, b, C2, d2


def _solve(A, b, C, d, lo=None, hi=None, backend="python", tol=1e-6,
           max_iterations=6000, record=False, **cfg_over):
    prob = OptimizationProblem(A, b, C, d)
    cfg = SolverConfig(backend=backend, lower_bound=lo, upper_bound=hi,
                       constraint_tol=tol, max_iterations=max_iterations,
                       record_trajectory=record, record_spike_history=False,
                       convergence=ConvergenceConfig(require_feasibility=True),
                       **cfg_over)
    solver = SNNSolver(prob, cfg)
    return solver.solve(np.zeros(A.shape[0]))


class TestJointFeasibility:
    def test_markowitz_mixed_reaches_joint_feasibility(self):
        A, b, C, d = _markowitz()
        res = _solve(A, b, C, d, lo=0.0, hi=1.0)
        assert res.max_distance_rows <= 1e-5
        assert res.max_violation_box <= 1e-5
        assert not res.projection_budget_exhausted

    def test_no_undercutting_of_reference_optimum(self):
        cvxpy = pytest.importorskip("cvxpy")
        A, b, C, d = _markowitz()
        x = cvxpy.Variable(A.shape[0])
        prob = cvxpy.Problem(
            cvxpy.Minimize(0.5 * cvxpy.quad_form(x, A) + b @ x),
            [C @ x + d <= 0, x >= 0, x <= 1])
        prob.solve(solver=cvxpy.CLARABEL)
        res = _solve(A, b, C, d, lo=0.0, hi=1.0)
        # A feasible point can never beat f*; allow tolerance-level slack.
        assert res.final_objective >= prob.value - 1e-6

    def test_svm_dual_shape(self):
        rng = np.random.default_rng(1)
        n = 30
        X = rng.standard_normal((n, 8))
        y = np.sign(rng.standard_normal(n))
        Q = (y[:, None] * X) @ (y[:, None] * X).T + 1e-3 * np.eye(n)
        C = np.vstack([y, -y])   # equality via inequality pair
        d = np.zeros(2)
        res = _solve(Q, -np.ones(n), C, d, lo=0.0, hi=1.0,
                     max_iterations=20000)
        assert res.max_distance_rows <= 1e-5
        assert res.max_violation_box <= 1e-5


class TestFacetRowEquivalence:
    def test_native_facets_match_materialized_rows_trajectory(self):
        A, b, C, d = _markowitz(n=25)
        res_native = _solve(A, b, C, d, lo=0.0, hi=1.0, record=True,
                            max_iterations=800)
        A2, b2, C2, d2 = _box_as_rows(A, b, C, d, 0.0, 1.0)
        res_rows = _solve(A2, b2, C2, d2, record=True, max_iterations=800)
        k = min(len(res_native.X), len(res_rows.X))
        assert np.max(np.abs(res_native.X[:k] - res_rows.X[:k])) < 1e-12


class TestRowRescalingInvariance:
    def test_positive_rescaling_leaves_trajectory_unchanged(self):
        A, b, C, d = _markowitz(n=20)
        res1 = _solve(A, b, C, d, lo=0.0, hi=1.0, record=True,
                      max_iterations=500)
        res2 = _solve(A, b, 37.0 * C, 37.0 * d, lo=0.0, hi=1.0, record=True,
                      max_iterations=500)
        assert np.max(np.abs(res1.X - res2.X)) < 1e-9


class TestBudgetExhaustion:
    def test_tiny_cap_aborts_with_status(self):
        A, b, C, d = _markowitz()
        res = _solve(A, b, C, d, lo=0.0, hi=1.0, max_projection_iters=3)
        assert res.projection_budget_exhausted
        assert res.convergence_reason == "projection_budget_exhausted"
        assert not res.converged

    def test_exhaustion_is_not_reported_when_tolerance_reached(self):
        A, b, C, d = _markowitz()
        res = _solve(A, b, C, d, lo=0.0, hi=1.0)
        assert not res.projection_budget_exhausted


class TestBoxOnlyFastPath:
    def test_box_only_exact_projection(self):
        n = 15
        A = np.eye(n)
        b = -2.0 * np.ones(n)          # unconstrained optimum at 2 -> clip to 1
        C = np.zeros((0, n))
        d = np.zeros(0)
        res = _solve(A, b, C, d, lo=0.0, hi=1.0)
        assert np.allclose(res.final_x, np.ones(n), atol=1e-8)
        assert res.joint_feasible


class TestZeroRowScreening:
    def test_zero_row_positive_d_raises(self):
        n = 5
        prob = OptimizationProblem(np.eye(n), np.zeros(n),
                                   np.zeros((1, n)), np.array([1.0]))
        with pytest.raises(ValueError, match="infeasible"):
            SNNSolver(prob, SolverConfig())

    def test_zero_row_nonpositive_d_is_inert(self):
        n = 5
        A, b = np.eye(n), np.ones(n)
        C = np.vstack([np.zeros(n), np.ones(n)])
        d = np.array([-1.0, -10.0])
        res = _solve(A, b, C, d)
        assert res.max_distance_rows <= 1e-6


class TestFixedMethodGuard:
    def test_fixed_projection_with_box_raises(self):
        n = 4
        prob = OptimizationProblem(np.eye(n), np.zeros(n),
                                   np.ones((1, n)), np.array([-1.0]))
        cfg = SolverConfig(projection_method="fixed", lower_bound=0.0)
        with pytest.raises(ValueError, match="fixed"):
            SNNSolver(prob, cfg)


class TestEigenbasisWithBox:
    def test_transform_accepts_box_and_matches_canonical(self):
        A, b, C, d = _markowitz(n=20)
        res_canon = _solve(A, b, C, d, lo=0.0, hi=1.0, max_iterations=4000)
        res_eig = _solve(A, b, C, d, lo=0.0, hi=1.0, max_iterations=4000,
                         transform="eigenbasis")
        assert res_eig.max_distance_rows <= 1e-5
        assert res_eig.max_violation_box <= 1e-5
        assert abs(res_eig.final_objective - res_canon.final_objective) < 1e-4


class TestStationarityCertificate:
    def test_near_zero_on_benign_active_set(self):
        # min ||x - a||^2 s.t. x <= 0 (active box at optimum x* = 0 for a > 0)
        n = 6
        A = np.eye(n)
        b = -np.ones(n)     # optimum of unconstrained at 1, box hi=0 -> x*=0
        C = np.zeros((0, n))
        d = np.zeros(0)
        res = _solve(A, b, C, d, hi=0.0)
        assert res.stationarity_residual < 1e-6

    def test_reports_gradient_norm_when_nothing_active(self):
        n = 3
        A = np.eye(n)
        b = np.zeros(n)     # optimum at 0, constraint far away
        C = np.ones((1, n))
        d = np.array([-100.0])
        res = _solve(A, b, C, d)
        assert res.stationarity_residual < 1e-6  # grad ~ 0 at optimum
