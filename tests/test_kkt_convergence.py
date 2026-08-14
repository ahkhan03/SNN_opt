"""Tests for the v0.6.0 scale-invariant KKT convergence certificate.

Covers the residual mathematics (scale/permutation/duplication invariance,
LP and box edge cases, sign-constrained fit), the fail-closed numerical
paths, the criterion selector and deprecated-alias migration, and
python-vs-compiled backend policy parity under the chunked driver.
"""

import warnings

import numpy as np
import pytest

from snn_opt import OptimizationProblem, SNNSolver
from snn_opt.solver import ConvergenceConfig, SolverConfig

try:
    from snn_opt import _kernel  # noqa: F401
    HAVE_KERNEL = True
except ImportError:
    HAVE_KERNEL = False


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _random_qp(seed=7, n=20, m=12):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = M @ M.T + n * np.eye(n)
    b = rng.standard_normal(n) * 10.0
    C = rng.standard_normal((m, n))
    d = -rng.uniform(0.2, 1.0, size=m)
    return A, b, C, d


def _solve(A, b, C, d, backend="python", max_iterations=20000, **cfg):
    prob = OptimizationProblem(A=A, b=b, C=C, d=d)
    solver = SNNSolver(prob, SolverConfig(backend=backend,
                                          max_iterations=max_iterations,
                                          **cfg))
    return solver, solver.solve(np.zeros(A.shape[0]))


def _certificate(A, b, C, d, x, **cfg):
    prob = OptimizationProblem(A=A, b=b, C=C, d=d)
    solver = SNNSolver(prob, SolverConfig(**cfg))
    return solver._compute_kkt_certificate(np.asarray(x, dtype=float))


# ---------------------------------------------------------------------------
# Residual mathematics
# ---------------------------------------------------------------------------

def test_converges_at_natural_and_high_scale_with_identical_decision():
    """Correlated active normals: the flag must be scale-invariant."""
    A, b, C, d = _random_qp()
    results = {}
    for s in (1.0, 1e10):
        _, r = _solve(s * A, s * b, C, d)
        results[s] = r
        assert r.converged, f"scale {s}: expected convergence at the floor"
    r1, r2 = results[1.0], results[1e10]
    assert r1.iterations_used == r2.iterations_used
    # relative residual is exactly homogeneous under objective scaling
    rel1 = r1.kkt_residual / r1.kkt_scale
    rel2 = r2.kkt_residual / r2.kkt_scale
    assert rel1 == pytest.approx(rel2, rel=1e-9)


def test_interior_optimum_high_scale_converges():
    """Interior optimum at 1e10 scale: the absolute-tolerance defect case."""
    A, b, C, d = _random_qp()
    _, r = _solve(1e10 * A, 1e10 * b, C, d - 1e3)
    assert r.converged
    assert r.kkt_fit_status == "ok"


def test_lp_scale_invariance():
    """A=0 LP: automatic k0 falls back to a constant, so any k0-dependent
    criterion breaks under objective scaling; the certificate must not."""
    A = np.zeros((2, 2))
    b = np.array([-1.0, -1.0])
    C = np.vstack([np.eye(2), -np.eye(2)])
    d = np.array([-1.0, -1.0, 0.0, 0.0])  # 0 <= x <= 1, optimum (1, 1)
    opt = np.array([1.0, 1.0])
    interior = np.array([0.4, 0.5])
    for s in (1.0, 1e10):
        cert_opt = _certificate(A, s * b, C, d, opt)
        cert_int = _certificate(A, s * b, C, d, interior)
        assert cert_opt.passed, f"scale {s}: LP optimum must certify"
        assert not cert_int.passed, f"scale {s}: LP interior must not certify"


def test_zero_linear_term_absolute_floor():
    """b=0, SPD A, optimum at 0: gradient scale collapses; the absolute
    tolerance must let a machine-accurate point pass."""
    A, _, C, d = _random_qp()
    cert = _certificate(A, np.zeros(A.shape[0]), C, d - 1e3, np.zeros(A.shape[0]))
    assert cert.passed


def test_row_permutation_invariance():
    """Duplicate active/slack rows: r_kkt must not depend on row order (the
    pre-v0.6 eps-KKT diagnostic flipped 0.0 <-> 0.1 on this exact case)."""
    A = np.array([[1.0]])
    b = np.array([-1.0])
    x = np.array([0.0])  # true optimum of min 0.5 x^2 - x s.t. x <= 0
    residuals = []
    for order in ([0, 1], [1, 0]):
        C = np.array([[1.0], [1.0]])[order]
        d = np.array([0.0, -0.1])[order]
        residuals.append(_certificate(A, b, C, d, x).residual)
    assert residuals[0] == pytest.approx(residuals[1], abs=1e-12)
    assert residuals[0] == pytest.approx(0.0, abs=1e-9)


def test_duplicate_rows_cannot_game_complementarity():
    """Splitting a multiplier across k copies of a slack row must not shrink
    the residual (total complementarity |s|^T mu is split-invariant)."""
    A = np.array([[1.0]])
    b = np.array([-1.0])
    x = np.array([-0.1])  # non-optimal boundary-adjacent point
    base = _certificate(A, b, np.array([[1.0]]), np.array([0.0]), x)
    dup = _certificate(A, b, np.ones((6, 1)), np.zeros(6), x)
    assert not base.passed
    assert not dup.passed
    assert dup.residual >= 0.5 * base.residual


def test_slack_normal_cannot_fake_stationarity():
    """At x=-0.1 for min 0.5 x^2 - x s.t. x <= 0, the (slack) constraint
    normal can absorb the gradient in a stationarity-only fit; the
    complementarity row must keep the residual large."""
    cert = _certificate(np.array([[1.0]]), np.array([-1.0]),
                        np.array([[1.0]]), np.array([0.0]), np.array([-0.1]))
    assert not cert.passed
    assert cert.complementarity > cert.tolerance


def test_wrong_multiplier_sign_rejected():
    """Boundary point with a FEASIBLE descent direction: an unconstrained
    span projection would zero the residual; the sign-constrained fit
    must reject. min 0.5 x^2 + x s.t. x <= 0 at x=0: optimum is x=-1,
    descent -g = -1 points inside the feasible set."""
    cert = _certificate(np.array([[1.0]]), np.array([1.0]),
                        np.array([[1.0]]), np.array([0.0]), np.array([0.0]))
    assert not cert.passed
    assert cert.residual >= 0.9  # ~||g|| = 1: mu=0 is the best cone fit


def test_constraint_row_scaling_invariance():
    """Multiplying individual rows (c_i, d_i) by positive factors changes
    nothing geometrically; the normalized certificate must be unchanged."""
    A, b, C, d = _random_qp()
    _, r = _solve(A, b, C, d)
    x = r.final_x
    base = _certificate(A, b, C, d, x)
    factors = np.geomspace(1e-6, 1e6, C.shape[0])
    scaled = _certificate(A, b, C * factors[:, None], d * factors, x)
    assert scaled.residual == pytest.approx(base.residual, rel=1e-8)
    assert scaled.passed == base.passed


def test_box_only_certificate():
    """Box-active optimum: min 0.5||x||^2 - 2*1^T x with x <= 1 elementwise
    (via scalar bounds): optimum at x = 1 (upper bound active)."""
    n = 5
    A = np.eye(n)
    b = -2.0 * np.ones(n)
    prob = OptimizationProblem(A=A, b=b, C=np.zeros((0, n)), d=np.zeros(0))
    solver = SNNSolver(prob, SolverConfig(upper_bound=1.0, lower_bound=-5.0,
                                          max_iterations=5000))
    r = solver.solve(np.zeros(n))
    assert r.converged
    assert np.allclose(r.final_x, np.ones(n), atol=1e-6)
    cert = solver._compute_kkt_certificate(r.final_x)
    assert cert.passed


def test_no_facets_reduces_to_gradient_norm():
    A = np.eye(3)
    b = np.array([1.0, -2.0, 0.5])
    x = np.array([0.3, 0.3, 0.3])
    cert = _certificate(A, b, np.zeros((0, 3)), np.zeros(0), x)
    g = A @ x + b
    assert cert.residual == pytest.approx(np.linalg.norm(g), rel=1e-12)
    assert cert.complementarity == 0.0


def test_certificate_tracks_true_error_on_sweep():
    """Floor-quality solutions certify; clearly-unconverged ones must not.
    (Seeds chosen from the reproduction sweep: 1 converges to ~5e-7, 7
    stalls at ~1.5e-2 relative error.)"""
    A, b, C, d = _random_qp(seed=101)
    _, r = _solve(A, b, C, d)
    assert r.converged

    A, b, C, d = _random_qp(seed=107)
    _, r = _solve(A, b, C, d)
    assert not r.converged
    assert r.kkt_residual / r.kkt_scale > 1e-3


# ---------------------------------------------------------------------------
# Fail-closed numerical paths
# ---------------------------------------------------------------------------

def test_nnls_failure_fails_closed(monkeypatch):
    import scipy.optimize

    def boom(*a, **k):
        raise RuntimeError("forced NNLS failure")

    monkeypatch.setattr(scipy.optimize, "nnls", boom)
    A, b, C, d = _random_qp()
    _, r = _solve(A, b, C, d, max_iterations=500)
    assert not r.converged
    assert r.kkt_fit_status == "fit_failed"
    assert np.isnan(r.kkt_residual)


def test_non_finite_state_fails_closed():
    A, b, C, d = _random_qp()
    cert = _certificate(A, b, C, d, np.full(A.shape[0], np.nan))
    assert cert.fit_status == "non_finite"
    assert not cert.passed


def test_gate_uses_less_equal():
    from snn_opt.solver import KKTCertificate
    at = KKTCertificate(residual=1.0, stationarity=1.0, complementarity=0.0,
                        scale=1.0, tolerance=1.0, fit_status="ok")
    above = KKTCertificate(residual=1.0 + 1e-12, stationarity=1.0,
                           complementarity=0.0, scale=1.0, tolerance=1.0,
                           fit_status="ok")
    assert at.passed
    assert not above.passed


# ---------------------------------------------------------------------------
# Criterion selector and migration
# ---------------------------------------------------------------------------

def test_legacy_mode_reproduces_old_behavior():
    """legacy_projected_gradient: interior natural-scale converges (pgn -> 0),
    the scaled twin does NOT (the documented pre-v0.6 defect, preserved
    verbatim under the legacy selector)."""
    A, b, C, d = _random_qp()
    conv = dict(convergence=ConvergenceConfig(
        optimality_test="legacy_projected_gradient"))
    _, r_nat = _solve(A, b, C, d - 1e3, **conv)
    assert r_nat.converged
    conv = dict(convergence=ConvergenceConfig(
        optimality_test="legacy_projected_gradient"))
    _, r_scaled = _solve(1e10 * A, 1e10 * b, C, d - 1e3, **conv)
    assert not r_scaled.converged


def test_optimality_none_uses_cheap_criteria_only():
    A, b, C, d = _random_qp()
    _, r = _solve(A, b, C, d,
                  convergence=ConvergenceConfig(optimality_test="none"))
    assert r.converged  # plateau alone fires at the limit cycle
    assert "kkt" not in r.convergence_reason


def test_deprecated_aliases_map_and_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = ConvergenceConfig(proj_grad_tol=1e-5)
    assert cfg.optimality_test == "legacy_projected_gradient"
    assert cfg.proj_grad_tol == 1e-5
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = ConvergenceConfig(use_projected_gradient=False)
    assert cfg.optimality_test == "none"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_mixed_old_and_new_criterion_families_raise():
    with pytest.raises(ValueError):
        ConvergenceConfig(proj_grad_tol=1e-5, optimality_test="none")
    with pytest.raises(ValueError):
        ConvergenceConfig(use_projected_gradient=True, kkt_rel_tol=1e-5)
    with pytest.raises(ValueError):
        ConvergenceConfig(optimality_test="not_a_mode")


def test_result_reports_certificate_components():
    A, b, C, d = _random_qp()
    _, r = _solve(A, b, C, d)
    assert r.optimality_test == "kkt"
    assert r.kkt_fit_status == "ok"
    recombined = float(np.hypot(r.kkt_stationarity_residual,
                                r.kkt_complementarity_residual))
    assert r.kkt_residual == pytest.approx(recombined, rel=1e-12)
    assert r.kkt_tolerance == pytest.approx(1e-9 + 1e-4 * r.kkt_scale)
    assert "kkt(" in r.convergence_reason



def _reason_shape(reason):
    """Criteria structure of a convergence reason, numeric diagnostics
    stripped (backends may differ in the last float digits of the embedded
    plateau range while firing identical criteria)."""
    import re as _re
    return _re.sub(r"[-+0-9.eE]+", "#", reason)

# ---------------------------------------------------------------------------
# Backend policy parity (python vs compiled chunked driver)
# ---------------------------------------------------------------------------

pytestmark_kernel = pytest.mark.skipif(not HAVE_KERNEL,
                                       reason="compiled kernel unavailable")


@pytestmark_kernel
@pytest.mark.parametrize("scale", [1.0, 1e10])
def test_backend_flag_and_state_parity(scale):
    A, b, C, d = _random_qp()
    _, rp = _solve(scale * A, scale * b, C, d)
    _, rc = _solve(scale * A, scale * b, C, d, backend="c")
    assert rp.converged == rc.converged
    assert rp.iterations_used == rc.iterations_used
    assert _reason_shape(rp.convergence_reason) == _reason_shape(rc.convergence_reason)
    np.testing.assert_allclose(rp.final_x, rc.final_x, rtol=0, atol=1e-12)


@pytestmark_kernel
def test_backend_parity_non_converging_case():
    A, b, C, d = _random_qp(seed=107)
    _, rp = _solve(A, b, C, d)
    _, rc = _solve(A, b, C, d, backend="c")
    assert rp.converged == rc.converged == False  # noqa: E712
    assert rp.iterations_used == rc.iterations_used


@pytestmark_kernel
def test_chunked_observer_stream_matches_python():
    """Digest, counts, and candidate IDs must be identical between the
    python backend and the chunked native driver (absolute iteration
    tokens + seeded digest across chunks)."""
    A, b, C, d = _random_qp()
    obs = dict(observe_projection_events=True, record_trajectory=False)
    _, rp = _solve(A, b, C, d, **obs)
    _, rc = _solve(A, b, C, d, backend="c", **obs)
    assert rp.projection_event_digest == rc.projection_event_digest
    assert rp.explicit_row_events == rc.explicit_row_events
    assert rp.projection_first_candidate_id == rc.projection_first_candidate_id
    assert rp.projection_last_candidate_id == rc.projection_last_candidate_id
    np.testing.assert_array_equal(rp.explicit_row_event_counts,
                                  rc.explicit_row_event_counts)


@pytestmark_kernel
def test_chunked_budget_exhaustion_propagates():
    A, b, C, d = _random_qp()
    for be in ("python", "c"):
        _, r = _solve(A, b, C, d, backend=be, max_projection_iters=1)
        assert r.projection_budget_exhausted
        assert r.convergence_reason == "projection_budget_exhausted"
        assert not r.converged


@pytestmark_kernel
def test_chunked_respects_unaligned_min_iterations():
    """min_iterations not a multiple of check_every: first checkpoint must
    land on the same iteration in both backends."""
    A, b, C, d = _random_qp()
    conv = dict(min_iterations=77, check_every=30)
    _, rp = _solve(A, b, C, d, convergence=ConvergenceConfig(**conv))
    _, rc = _solve(A, b, C, d, backend="c",
                   convergence=ConvergenceConfig(**conv))
    assert rp.converged == rc.converged
    assert rp.iterations_used == rc.iterations_used


@pytestmark_kernel
def test_chunked_solution_stable_criterion_parity():
    A, b, C, d = _random_qp()
    conv = dict(use_solution_stable=True)
    _, rp = _solve(A, b, C, d, convergence=ConvergenceConfig(**conv))
    _, rc = _solve(A, b, C, d, backend="c",
                   convergence=ConvergenceConfig(**conv))
    assert rp.converged == rc.converged
    assert rp.iterations_used == rc.iterations_used
    assert (_reason_shape(rp.convergence_reason)
            == _reason_shape(rc.convergence_reason))


@pytestmark_kernel
def test_legacy_native_path_still_monolithic():
    """legacy mode on the compiled backend keeps the historical in-kernel
    checks and reason label."""
    A, b, C, d = _random_qp()
    _, rc = _solve(A, b, C, d - 1e3, backend="c",
                   convergence=ConvergenceConfig(
                       optimality_test="legacy_projected_gradient"))
    assert rc.converged
    assert rc.convergence_reason == "converged(c-backend)"
