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
    """b=0, SPD A, optimum at 0: the gradient scale is 0 and the residual is 0.

    The retired absolute floor accepted any residual up to 1e-9 here. The
    only gate is ``residual <= target * scale``, so scale 0 accepts a
    residual of exactly 0 and rejects a positive residual.
    """
    from snn_opt.solver import KKTCertificate

    A, _, C, d = _random_qp()
    cert = _certificate(A, np.zeros(A.shape[0]), C, d - 1e3, np.zeros(A.shape[0]))
    assert cert.scale == pytest.approx(0.0)
    assert cert.residual == pytest.approx(0.0, abs=0.0)
    assert cert.passed
    positive = KKTCertificate(
        residual=1e-12, stationarity=1e-12, complementarity=0.0,
        scale=0.0, tolerance=1e-9, fit_status="ok")
    assert not positive.passed


def test_nonfinite_scale_fails_closed():
    """A finite residual at a non-finite scale does not pass.

    ``1e-4 * inf`` is inf, so the inequality alone would accept it. Scale 0
    stays on the inequality and still accepts a residual of exactly 0.
    """
    from snn_opt.solver import KKTCertificate

    infinite = KKTCertificate(
        residual=1.0, stationarity=1.0, complementarity=0.0,
        scale=float("inf"), tolerance=float("inf"), fit_status="ok")
    assert not infinite.passed
    zero_at_inf = KKTCertificate(
        residual=0.0, stationarity=0.0, complementarity=0.0,
        scale=float("inf"), tolerance=float("nan"), fit_status="ok",
        kkt_target=0.0)
    assert not zero_at_inf.passed
    for bad in (float("inf"), float("-inf"), float("nan")):
        cert = KKTCertificate(
            residual=0.0, stationarity=0.0, complementarity=0.0,
            scale=bad, tolerance=0.0, fit_status="ok")
        assert not cert.passed
    exact = KKTCertificate(
        residual=0.0, stationarity=0.0, complementarity=0.0,
        scale=0.0, tolerance=0.0, fit_status="ok")
    assert exact.passed


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
    """The decision is residual <= target * scale, including equality.

    ``tolerance`` is not consulted. A stored target of None uses 1e-4.
    """
    from snn_opt.solver import KKTCertificate
    at = KKTCertificate(residual=1.0, stationarity=1.0, complementarity=0.0,
                        scale=1.0, tolerance=1e9, fit_status="ok",
                        kkt_target=1.0)
    above = KKTCertificate(residual=1.0 + 1e-12, stationarity=1.0,
                           complementarity=0.0, scale=1.0, tolerance=1e9,
                           fit_status="ok", kkt_target=1.0)
    assert at.passed
    assert not above.passed
    default_edge = KKTCertificate(
        residual=1e-4, stationarity=1e-4, complementarity=0.0,
        scale=1.0, tolerance=0.0, fit_status="ok")
    assert default_edge.passed


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
    assert cfg.legacy_proj_grad_tol == 1e-5
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
    assert r.kkt_tolerance == pytest.approx(1e-4 * r.kkt_scale)
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


# ---------------------------------------------------------------------------
# Round-2 additions: config round-trips, IVP semantics, sparse certification,
# calibration-sweep lock, absolute-floor crossover, duplicate exactness,
# transient fit failure
# ---------------------------------------------------------------------------

def test_config_replace_and_asdict_round_trips():
    import dataclasses

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning is a failure
        base = ConvergenceConfig()
        r1 = dataclasses.replace(base)
        assert r1.optimality_test == "kkt"
        r2 = dataclasses.replace(base, check_every=25)
        assert r2.optimality_test == "kkt" and r2.check_every == 25
        rebuilt = ConvergenceConfig(**dataclasses.asdict(base))
        assert rebuilt.optimality_test == "kkt"

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        legacy = ConvergenceConfig(proj_grad_tol=1e-5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        legacy2 = dataclasses.replace(legacy)  # must NOT warn or change mode
        assert legacy2.optimality_test == "legacy_projected_gradient"
        assert legacy2.legacy_proj_grad_tol == 1e-5
        none_cfg = dataclasses.replace(ConvergenceConfig(optimality_test="none"))
        assert none_cfg.optimality_test == "none"
        assert none_cfg.kkt_target is None
        custom = dataclasses.replace(ConvergenceConfig(kkt_target=1e-5))
        assert custom.kkt_target == 1e-5
        base_again = dataclasses.replace(ConvergenceConfig())
        assert base_again.kkt_target is None
        assert dataclasses.asdict(base_again)["kkt_target"] is None


def test_explicit_default_selector_with_alias_raises():
    with pytest.raises(ValueError):
        ConvergenceConfig(proj_grad_tol=1e-5, optimality_test="kkt")


def test_post_construction_mutation_keeps_backends_in_sync():
    """Mutating optimality_test after construction must steer BOTH backends
    (dispatch derives from the selector, not from a cached alias)."""
    A, b, C, d = _random_qp()
    results = {}
    for be in ("python",) + (("c",) if HAVE_KERNEL else ()):
        cfg = SolverConfig(backend=be, max_iterations=1000)
        cfg.convergence.optimality_test = "legacy_projected_gradient"
        prob = OptimizationProblem(A=1e10 * A, b=1e10 * b, C=C, d=d - 1e3)
        r = SNNSolver(prob, cfg).solve(np.zeros(A.shape[0]))
        results[be] = r.converged
        # legacy absolute test cannot fire at 1e10 gradient scale
        assert r.converged is False, be
    assert len(set(results.values())) == 1


def test_ivp_stuck_at_boundary_does_not_certify():
    """IVP stall heuristic must not report KKT convergence at a boundary
    point with a feasible descent direction (tangential defect ~ ||g||)."""
    A = np.eye(2)
    b = np.array([-1.0, -1.0])
    C = np.array([[1.0, 0.0]])
    d = np.array([0.0])
    prob = OptimizationProblem(A=A, b=b, C=C, d=d)
    r = SNNSolver(prob, SolverConfig(integration_method="ivp",
                                     t_end=5.0)).solve(np.zeros(2))
    if r.convergence_reason == "stuck_at_boundary":
        assert not r.converged  # certificate fails: descent along x2 remains
    # legacy mode preserves the historical heuristic flag
    r2 = SNNSolver(prob, SolverConfig(
        integration_method="ivp", t_end=5.0,
        convergence=ConvergenceConfig(
            optimality_test="legacy_projected_gradient"))).solve(np.zeros(2))
    if r2.convergence_reason == "stuck_at_boundary":
        assert r2.converged


def test_sparse_certificate_matches_dense():
    import scipy.sparse as sp
    A, b, C, d = _random_qp()
    _, r = _solve(A, b, C, d)
    x = r.final_x
    dense = _certificate(A, b, C, d, x)
    prob = OptimizationProblem(A=A, b=b, C=sp.csr_matrix(C), d=d)
    solver = SNNSolver(prob, SolverConfig())
    sparse_cert = solver._compute_kkt_certificate(x)
    assert sparse_cert.fit_status == "ok"
    # LSMR-based and Lawson-Hanson fits differ at their own optimality
    # tolerance; what matters is that the difference is far below the
    # acceptance threshold so it can never flip the public decision.
    assert abs(sparse_cert.residual - dense.residual) <= 0.01 * dense.tolerance
    assert sparse_cert.passed == dense.passed


def test_sparse_certificate_does_not_densify():
    """A large, very sparse constraint matrix must not be densified: the
    dense facet matrix would need ~4.8 GB; the sparse path needs ~MBs."""
    import scipy.sparse as sp
    n, m = 4000, 150_000
    rng = np.random.default_rng(0)
    A = sp.identity(n, format="csr")
    b = -np.ones(n)
    cols = rng.integers(0, n, size=m)
    C = sp.csr_matrix((np.ones(m), (np.arange(m) % m, cols)), shape=(m, n))
    d = -np.ones(m)
    prob = OptimizationProblem(A=A, b=b, C=C, d=d)
    solver = SNNSolver(prob, SolverConfig())
    cert = solver._compute_kkt_certificate(np.zeros(n))
    assert cert.fit_status == "ok"
    assert np.isfinite(cert.residual)


def test_dense_certificate_size_guard_fails_closed(monkeypatch):
    A, b, C, d = _random_qp()
    prob = OptimizationProblem(A=A, b=b, C=C, d=d)
    solver = SNNSolver(prob, SolverConfig())
    monkeypatch.setattr(SNNSolver, "_DENSE_CERT_MAX_ENTRIES", 10)
    cert = solver._compute_kkt_certificate(np.zeros(A.shape[0]))
    assert cert.fit_status == "too_large"
    assert not cert.passed


def test_calibration_sweep_decisions_locked():
    """Lock the 8-seed sweep decisions the tolerance default was calibrated
    on: seeds 3 and 7 (relative KKT defect above 1e-4) must refuse; the
    others, including boundary seed 4, must certify."""
    backend = "c" if HAVE_KERNEL else "python"
    expected = {0: True, 1: True, 2: True, 3: False,
                4: True, 5: True, 6: True, 7: False}
    for seed, want in expected.items():
        A, b, C, d = _random_qp(seed=100 + seed)
        _, r = _solve(A, b, C, d, backend=backend)
        assert r.converged is want, f"seed {seed}"


def test_absolute_floor_crossover_documented_behavior():
    """Objective scaling leaves the normalized residual and the decision unchanged.

    The retired absolute floor accepted this same point once the gradient
    scale dropped near 1e-30, because 1e-9 dominated 1e-4 * scale. That
    floor is gone, so the Boolean matches the ordinary scales.
    """
    A, b, C, d = _random_qp()
    # 30 iterations: genuinely far from the optimum (the floor is reached
    # around iteration ~150 on this problem).
    _, r = _solve(A, b, C, d, max_iterations=30)
    x = r.final_x
    rels = []
    for s in (1e-20, 1e-6, 1.0, 1e6):
        cert = _certificate(s * A, s * b, C, d, x)
        rels.append(cert.residual / cert.scale)
        assert not cert.passed
    assert max(rels) == pytest.approx(min(rels), rel=1e-9)


def test_exact_duplicate_invariance():
    """Replicating a row k times must leave the residual unchanged (the
    total-complementarity formulation is split-invariant), up to solver
    roundoff."""
    A = np.array([[1.0]])
    b = np.array([-1.0])
    x = np.array([-0.1])
    base = _certificate(A, b, np.array([[1.0]]), np.array([0.0]), x)
    for k in (2, 6, 12):
        dup = _certificate(A, b, np.ones((k, 1)), np.zeros(k), x)
        assert dup.residual == pytest.approx(base.residual, rel=1e-9)


def test_transient_fit_failure_resets_patience(monkeypatch):
    """A fit failure AFTER previous passing checks must reset patience and
    fail closed, not coast through on stale state."""
    import scipy.optimize
    real_nnls = scipy.optimize.nnls
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("forced transient failure")
        return real_nnls(*a, **k)

    monkeypatch.setattr(scipy.optimize, "nnls", flaky)
    A, b, C, d = _random_qp()
    _, r = _solve(A, b, C, d, max_iterations=2000)
    assert not r.converged
    assert r.kkt_fit_status == "fit_failed"


# ---------------------------------------------------------------------------
# Round-3 additions: sparse fit convergence at near-collinear facets,
# pre-allocation guard, IVP master switch, sentinel conflict detection,
# invalid-selector dispatch, non-finite facet data
# ---------------------------------------------------------------------------

def test_near_collinear_sparse_matches_dense_at_tight_tolerance():
    """Exact-KKT point with nearly collinear active normals: the sparse fit
    must genuinely converge (not stop at LSMR's default iteration limit) so
    matrix representation cannot flip the public decision."""
    import scipy.sparse as sp
    theta = 1e-4
    C = np.array([[1.0, 0.0],
                  [np.cos(theta), np.sin(theta)]])
    A = np.eye(2)
    d = np.zeros(2)
    x = np.zeros(2)
    b = -(C[0] + 0.7 * C[1])  # exact cone member: mu = (1, 0.7)
    cfg = dict(convergence=ConvergenceConfig(kkt_abs_tol=0.0,
                                             kkt_rel_tol=1e-6))
    dense = _certificate(A, b, C, d, x, **cfg)
    prob = OptimizationProblem(A=A, b=b, C=sp.csr_matrix(C), d=d)
    solver = SNNSolver(prob, SolverConfig(
        convergence=ConvergenceConfig(kkt_abs_tol=0.0, kkt_rel_tol=1e-6)))
    sparse_cert = solver._compute_kkt_certificate(x)
    assert dense.passed
    assert sparse_cert.passed, (
        f"sparse residual {sparse_cert.residual:.3e} vs tolerance "
        f"{sparse_cert.tolerance:.3e}: inner solver did not converge")
    assert abs(sparse_cert.residual - dense.residual) <= dense.tolerance


def test_box_dominated_family_uses_sparse_blocks():
    """Dense-C-free problem with huge n and bounds: the certificate must not
    allocate dense identity blocks (would be ~2 GB at n=16000 with two
    bounds); it silently switches to sparse facets and certifies."""
    n = 16000
    import scipy.sparse as sp
    A = sp.identity(n, format="csr")
    b = -np.ones(n)
    prob = OptimizationProblem(A=A, b=b, C=np.zeros((0, n)), d=np.zeros(0))
    solver = SNNSolver(prob, SolverConfig(lower_bound=-1.0, upper_bound=1.0))
    cert = solver._compute_kkt_certificate(np.ones(n))  # optimum: upper active
    assert cert.fit_status == "ok"
    assert cert.passed


def test_oversized_dense_rows_fail_closed_without_allocation():
    """When the dense ROW block alone exceeds the guard, the certificate
    fails closed before building facets (no MemoryError propagation)."""
    A, b, C, d = _random_qp()
    prob = OptimizationProblem(A=A, b=b, C=C, d=d)
    solver = SNNSolver(prob, SolverConfig())
    original = SNNSolver._DENSE_CERT_MAX_ENTRIES
    try:
        SNNSolver._DENSE_CERT_MAX_ENTRIES = 10  # every dense family too big
        cert = solver._compute_kkt_certificate(np.zeros(A.shape[0]))
    finally:
        SNNSolver._DENSE_CERT_MAX_ENTRIES = original
    assert cert.fit_status == "too_large"
    assert not cert.passed


def test_ivp_respects_master_early_stopping_switch():
    """A KKT-accepted IVP stall must still report converged=False when
    enable_early_stopping=False (parity with the euler/native paths)."""
    A = np.eye(2)
    b = np.array([-1.0, -1.0])
    C = np.array([[1.0, 0.0]])
    d = np.array([0.0])
    prob = OptimizationProblem(A=A, b=b, C=C, d=d)
    # Lax tolerance so the certificate itself passes; only the switch blocks.
    r = SNNSolver(prob, SolverConfig(
        integration_method="ivp", t_end=5.0,
        convergence=ConvergenceConfig(enable_early_stopping=False,
                                      kkt_abs_tol=2.0,
                                      kkt_rel_tol=0.0))).solve(np.zeros(2))
    assert not r.converged


def test_alias_with_explicit_default_new_settings_raises():
    for kwargs in (dict(proj_grad_tol=1e-5, kkt_abs_tol=1e-9),
                   dict(proj_grad_tol=1e-5, kkt_rel_tol=1e-4),
                   dict(use_projected_gradient=True,
                        kkt_abs_tol=1e-9, kkt_rel_tol=1e-4),
                   dict(proj_grad_tol=1e-5, legacy_proj_grad_tol=2e-6)):
        with pytest.raises(ValueError):
            ConvergenceConfig(**kwargs)


def test_mutated_invalid_selector_raises_at_solve_time():
    A, b, C, d = _random_qp()
    for bogus in ("bogus", None):
        for be in ("python",) + (("c",) if HAVE_KERNEL else ()):
            cfg = SolverConfig(backend=be, max_iterations=300)
            cfg.convergence.optimality_test = bogus
            prob = OptimizationProblem(A=A, b=b, C=C, d=d)
            with pytest.raises(ValueError):
                SNNSolver(prob, cfg).solve(np.zeros(A.shape[0]))


def test_non_finite_constraint_row_fails_closed():
    """A NaN row must not be silently dropped by the degeneracy screen and
    certified around."""
    cert = _certificate(np.array([[1.0]]), np.array([0.0]),
                        np.array([[np.nan]]), np.array([-1.0]),
                        np.array([0.0]))
    assert cert.fit_status == "non_finite"
    assert not cert.passed
    cert2 = _certificate(np.array([[1.0]]), np.array([0.0]),
                         np.array([[0.0]]), np.array([np.nan]),
                         np.array([0.0]))
    assert cert2.fit_status == "non_finite"
    assert not cert2.passed


def test_condition_limit_sparse_matches_dense():
    """theta=1e-8: the facet Gram sits beyond LSMR's default condition limit
    (istop=3). The certificate must either genuinely converge or fail
    closed; it must not accept a condition-limit stop as ok and flip the
    decision vs the identical dense problem."""
    import scipy.sparse as sp
    theta = 1e-8
    C = np.array([[1.0, 0.0],
                  [np.cos(theta), np.sin(theta)]])
    A = np.eye(2)
    d = np.zeros(2)
    x = np.zeros(2)
    b = -(C[0] + 0.7 * C[1])  # exact cone member: mu = (1, 0.7)
    # Contract note: at cond ~ 1e8 the least-squares accuracy floor is
    # ~eps * cond ~ 1e-8, and dense-NNLS accuracy AT that floor varies
    # across scipy versions (observed 2e-16 on one, 7e-9 on another) -- so
    # neither "dense certifies at 1e-10" nor cross-algorithm decision
    # equality below the floor is a guaranteeable property. The defect this
    # test guards against is specific: an UNCONVERGED sparse fit being
    # labeled ok (LSMR condition-limit/iteration-limit stops re-accepted via
    # the lsq_linear shortcut), which reported a mid-range residual (1.5e-9
    # at an exactly-representable optimum) instead of either the achievable
    # floor or a fail-closed status. So: an ok sparse fit at this exact-KKT
    # point must sit at the verified-convergence floor; failing closed is
    # also acceptable. At practical tolerances (>= the 1e-4 default, and
    # anything above the conditioning floor) representations agree -- the
    # theta=1e-4 companion test asserts that stronger form.
    kw = ConvergenceConfig(kkt_abs_tol=0.0, kkt_rel_tol=1e-10)
    prob = OptimizationProblem(A=A, b=b, C=sp.csr_matrix(C), d=d)
    solver = SNNSolver(prob, SolverConfig(convergence=kw))
    sparse_cert = solver._compute_kkt_certificate(x)
    if sparse_cert.fit_status == "ok":
        assert sparse_cert.residual <= 1e-11, (
            f"ok-labeled sparse fit reports residual "
            f"{sparse_cert.residual:.3e} at an exactly-representable KKT "
            f"point: unconverged fit accepted as ok")
    else:
        assert not sparse_cert.passed  # failing closed is acceptable


# ---------------------------------------------------------------------------
# Strict normalized target (kkt_target). None keeps the legacy decision.
# ---------------------------------------------------------------------------

def _interior_pair():
    """2-D QP whose optimum (1, 1) is strictly interior."""
    A = np.eye(2)
    b = np.array([-1.0, -1.0])
    C = np.array([[1.0, 0.0], [0.0, 1.0]])
    d = np.array([-10.0, -10.0])
    return A, b, C, d, np.array([1.0, 1.0])


def test_strict_target_passes_point_legacy_also_passes():
    """A point under both the default 1e-4 target and a 1e-6 target passes."""
    A, b, C, d, x = _interior_pair()
    cfg = ConvergenceConfig(kkt_target=1e-6)
    cert = _certificate(A, b, C, d, x, convergence=cfg)
    assert cert.fit_status == "ok"
    assert cert.kkt_target == 1e-6
    assert cert.scale > 0.0
    assert cert.residual <= cert.tolerance
    assert cert.residual <= cert.kkt_target * cert.scale
    assert cert.passed
    _, result = _solve(A, b, C, d, max_iterations=400, convergence=cfg)
    assert result.converged
    assert result.kkt_target == 1e-6
    assert result.kkt_residual <= result.kkt_tolerance
    assert result.kkt_normalized <= result.kkt_target


def test_strict_target_rejects_point_legacy_tolerance_accepts():
    """Free-coordinate offset lands between the default 1e-4 and a 1e-9 target.

    x = (0, 1e-6) is feasible for x_0 <= 0. The certificate residual is the
    uncovered second coordinate of the gradient, so the normalized defect
    sits between 1e-9 and 1e-4. The default target accepts it; kkt_target=1e-9
    rejects it. ``kkt_tolerance`` is that gate, so the residual is above it.
    The deprecated ``kkt_rel_tol=1e-4`` alias sets the same default target.
    """
    A = np.eye(2)
    b = np.array([-1.0, 0.0])
    C = np.array([[1.0, 0.0]])
    d = np.array([0.0])
    x = np.array([0.0, 1e-6])
    strict = _certificate(
        A, b, C, d, x,
        convergence=ConvergenceConfig(kkt_target=1e-9))
    assert strict.fit_status == "ok"
    assert strict.scale > 0.0
    normalized = strict.residual / strict.scale
    assert 1e-9 < normalized <= 1e-4
    assert strict.tolerance == pytest.approx(strict.kkt_target * strict.scale)
    assert strict.residual > strict.tolerance
    assert strict.residual > strict.kkt_target * strict.scale
    assert not strict.passed
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        legacy = _certificate(
            A, b, C, d, x, convergence=ConvergenceConfig(kkt_rel_tol=1e-4))
    assert legacy.kkt_target == 1e-4
    assert legacy.passed
    assert legacy.residual <= legacy.tolerance


def test_unset_kkt_target_reproduces_legacy_decision():
    """kkt_target=None is the same strict gate as omitting the field.

    Both use effective target 1e-4. ``passed`` follows residual <= tolerance,
    and tolerance is that gate, not the retired absolute-plus-relative
    threshold.
    """
    A, b, C, d, _x = _interior_pair()
    _, default = _solve(A, b, C, d, max_iterations=400)
    _, explicit = _solve(
        A, b, C, d, max_iterations=400,
        convergence=ConvergenceConfig(kkt_target=None))
    assert default.kkt_target is None
    assert explicit.kkt_target is None
    assert default.converged is explicit.converged
    assert default.iterations_used == explicit.iterations_used
    assert default.convergence_reason == explicit.convergence_reason
    np.testing.assert_array_equal(default.final_x, explicit.final_x)
    assert default.kkt_residual == explicit.kkt_residual
    assert default.kkt_tolerance == explicit.kkt_tolerance
    assert default.kkt_normalized == explicit.kkt_normalized
    cert = _certificate(A, b, C, d, default.final_x)
    gate_pass = (cert.fit_status == "ok"
                 and np.isfinite(cert.residual)
                 and bool(cert.residual <= cert.tolerance))
    assert cert.kkt_target is None
    assert cert.tolerance == pytest.approx(1e-4 * cert.scale)
    assert cert.passed is gate_pass


def test_kkt_normalized_equals_residual_over_scale():
    A, b, C, d, _x = _interior_pair()
    _, result = _solve(A, b, C, d, max_iterations=400)
    assert result.kkt_scale > 0.0
    assert result.kkt_fit_status == "ok"
    assert result.kkt_normalized == result.kkt_residual / result.kkt_scale
    assert result.kkt_target is None


def test_scale_zero_accepts_only_exact_zero_residual():
    """Scale 0 accepts a residual of exactly 0 and rejects a positive one.

    The retired strict gate failed closed at scale 0 even for a zero
    residual. The only gate is the inequality: ``0 <= target * 0`` holds,
    and a positive residual does not.
    """
    from snn_opt.solver import KKTCertificate

    A, _b, C, d = _random_qp(n=6, m=4)
    zeros = np.zeros(A.shape[0])
    cert = _certificate(A, zeros, C, d - 1e3, zeros)
    assert cert.kkt_target is None
    assert cert.scale == pytest.approx(0.0)
    assert cert.residual == pytest.approx(0.0, abs=0.0)
    assert cert.passed
    explicit = _certificate(
        A, zeros, C, d - 1e3, zeros,
        convergence=ConvergenceConfig(kkt_target=1e-6))
    assert explicit.kkt_target == 1e-6
    assert explicit.scale == pytest.approx(0.0)
    assert explicit.passed
    positive = KKTCertificate(
        residual=1e-15, stationarity=1e-15, complementarity=0.0, scale=0.0,
        tolerance=0.0, fit_status="ok", kkt_target=1e-6)
    assert not positive.passed
    exact = KKTCertificate(
        residual=0.0, stationarity=0.0, complementarity=0.0, scale=0.0,
        tolerance=1e-9, fit_status="ok", kkt_target=1e-6)
    assert exact.passed


def test_kkt_target_rejects_invalid_values():
    with pytest.raises(ValueError):
        ConvergenceConfig(kkt_target=-1.0)
    with pytest.raises(ValueError):
        ConvergenceConfig(kkt_target=float("nan"))
    with pytest.raises(ValueError):
        ConvergenceConfig(proj_grad_tol=1e-5, kkt_target=1e-6)


@pytestmark_kernel
def test_backend_parity_for_strict_target_fields():
    A, b, C, d, _x = _interior_pair()
    cfg = dict(max_iterations=400,
               convergence=ConvergenceConfig(kkt_target=1e-6))
    _, python = _solve(A, b, C, d, backend="python", **cfg)
    _, compiled = _solve(A, b, C, d, backend="c", **cfg)
    assert python.kkt_target == compiled.kkt_target == 1e-6
    assert python.converged == compiled.converged
    assert python.iterations_used == compiled.iterations_used
    assert python.kkt_normalized == python.kkt_residual / python.kkt_scale
    assert compiled.kkt_normalized == compiled.kkt_residual / compiled.kkt_scale
    np.testing.assert_allclose(python.final_x, compiled.final_x, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        python.kkt_normalized, compiled.kkt_normalized, rtol=1e-9, atol=0.0)
    np.testing.assert_allclose(
        python.kkt_residual, compiled.kkt_residual, rtol=1e-9, atol=0.0)


def test_fix1_strict_gate_accepts_positive_scale_below_legacy_crossover():
    """A zero residual at scale 1e-6 meets target 1e-9.

    The legacy crossover kkt_abs_tol / kkt_rel_tol is 1e-5 and is not part
    of the strict gate. Only a non-positive scale fails closed.
    """
    cert = _certificate(
        np.array([[1e-6]]), np.array([-1e-6]),
        np.zeros((0, 1)), np.zeros(0), np.array([1.0]),
        convergence=ConvergenceConfig(kkt_target=1e-9))
    assert cert.fit_status == "ok"
    assert cert.residual == pytest.approx(0.0, abs=0.0)
    assert cert.scale == pytest.approx(1e-6)
    assert cert.scale > 0.0
    assert cert.passed


def test_fix2_strict_target_requires_kkt_optimality_test():
    """A set target with a non-KKT selector is an error, not a silent no-op."""
    for mode in ("none", "legacy_projected_gradient"):
        with pytest.raises(ValueError, match="kkt_target") as raised:
            ConvergenceConfig(optimality_test=mode, kkt_target=1e-9)
        message = str(raised.value)
        assert "optimality_test" in message
        assert "kkt_target" in message
        assert "KKT" in message


def test_fix3_strict_target_drives_fit_and_dykstra_resolution():
    """Fit budget and Dykstra admission follow effective_kkt_target() * scale."""
    from snn_opt import joint_dykstra_projector, soc_projector

    rng = np.random.default_rng(3)
    for _ in range(3):
        n = 4
        matrix = rng.standard_normal((n, n))
        hessian = matrix.T @ matrix + 0.5 * np.eye(n)
        linear = rng.standard_normal(n) * 3.0
        rows = rng.standard_normal((2, n))
    problem = OptimizationProblem(
        hessian, linear, rows, np.zeros(2),
        nonlinear_candidates=(soc_projector(0, [1, 2, 3]),))
    solver = SNNSolver(problem, SolverConfig(
        convergence=ConvergenceConfig(kkt_target=1e-9)))
    cert = solver._compute_kkt_certificate(np.zeros(n))
    assert cert.fit_status == "ok", cert.fit_status
    assert cert.scale > 0.0
    assert cert.residual <= 1e-9 * cert.scale
    assert cert.passed

    wedge = np.array([[1.0, 0.0],
                      [np.cos(0.7), np.sin(0.7)]])
    dykstra = joint_dykstra_projector(wedge, np.zeros(2), tolerance=1e-8)
    wedge_problem = OptimizationProblem(
        np.eye(2), -np.array([1.0, 0.5]), np.zeros((0, 2)), np.zeros(0),
        nonlinear_candidates=(dykstra,))
    wedge_solver = SNNSolver(wedge_problem, SolverConfig(
        convergence=ConvergenceConfig(kkt_target=1e-9)))
    wedge_cert = wedge_solver._compute_kkt_certificate(np.zeros(2))
    assert wedge_cert.fit_status == "fit_failed"
    assert not wedge_cert.passed


def test_fix4_summary_omits_normalized_line_unless_target_set():
    A, b, C, d, _x = _interior_pair()
    _, plain = _solve(A, b, C, d, max_iterations=1)
    assert plain.kkt_target is None
    assert "KKT normalized" not in plain.summary()
    _, strict = _solve(
        A, b, C, d, max_iterations=1,
        convergence=ConvergenceConfig(kkt_target=1e-4))
    text = strict.summary()
    assert "KKT normalized" in text
    assert "target=1.00e-04" in text


def test_default_config_uses_strict_gate():
    """The default config's reported tolerance is the normalized gate."""
    cfg = ConvergenceConfig()
    assert cfg.kkt_target is None
    assert cfg.effective_kkt_target() == 1e-4
    # A stored None is not a request, so a non-KKT selector is legal.
    ConvergenceConfig(optimality_test="none")
    ConvergenceConfig(optimality_test="none", kkt_target=None)
    A, b, C, d, _x = _interior_pair()
    _, result = _solve(A, b, C, d, max_iterations=400)
    assert result.kkt_target is None
    assert result.kkt_tolerance == pytest.approx(
        cfg.effective_kkt_target() * result.kkt_scale)


def test_small_scale_default_gate_passes_exact_optimum():
    """Hessian 1e-6, linear coefficient -1e-6, optimum x=1, default gate.

    The residual at that point is 0 and the scale is 1e-6, so
    ``0 <= 1e-4 * 1e-6`` passes. The retired threshold was
    ``1e-9 + 1e-4 * scale`` (1.1e-9 here). A residual between the new gate
    (1e-10) and that floor used to pass and now fails.
    """
    from snn_opt.solver import KKTCertificate

    A = np.array([[1e-6]])
    b = np.array([-1e-6])
    cert = _certificate(A, b, np.zeros((0, 1)), np.zeros(0), np.array([1.0]))
    assert cert.fit_status == "ok"
    assert cert.kkt_target is None
    assert cert.residual == pytest.approx(0.0, abs=0.0)
    assert cert.scale == pytest.approx(1e-6)
    assert cert.tolerance == pytest.approx(1e-4 * cert.scale)
    assert cert.passed
    between = KKTCertificate(
        residual=5e-10, stationarity=5e-10, complementarity=0.0,
        scale=1e-6, tolerance=1e-4 * 1e-6, fit_status="ok")
    assert 5e-10 <= 1e-9 + 1e-4 * 1e-6
    assert not between.passed
    _, result = _solve(A, b, np.zeros((0, 1)), np.zeros(0), max_iterations=400)
    assert result.converged
    assert result.kkt_tolerance == pytest.approx(1e-4 * result.kkt_scale)


def test_deprecated_kkt_aliases_warn_and_set_target():
    """kkt_rel_tol sets kkt_target. kkt_abs_tol warns and is ignored."""
    import dataclasses

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rel = ConvergenceConfig(kkt_rel_tol=1e-6)
    assert rel.kkt_target == 1e-6
    assert rel.effective_kkt_target() == 1e-6
    assert any(issubclass(w.category, DeprecationWarning) and "kkt_rel_tol" in str(w.message)
               for w in caught)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        roundtrip = dataclasses.replace(rel)
    assert roundtrip.kkt_target == 1e-6
    assert "kkt_rel_tol" not in dataclasses.asdict(roundtrip)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        absolute = ConvergenceConfig(kkt_abs_tol=1e-9)
    assert absolute.kkt_target is None
    assert absolute.effective_kkt_target() == 1e-4
    assert any(issubclass(w.category, DeprecationWarning) and "absolute floor" in str(w.message)
               for w in caught)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        both = ConvergenceConfig(kkt_abs_tol=1e-3, kkt_rel_tol=2e-5)
    assert both.kkt_target == 2e-5
    categories = [w.category for w in caught]
    assert sum(issubclass(c, DeprecationWarning) for c in categories) >= 2


def test_old_and_new_kkt_styles_raise():
    """An explicit kkt_target together with a deprecated alias raises."""
    with pytest.raises(ValueError):
        ConvergenceConfig(kkt_rel_tol=1e-4, kkt_target=1e-6)
    with pytest.raises(ValueError):
        ConvergenceConfig(kkt_abs_tol=0.0, kkt_target=1e-6)
    with pytest.raises(ValueError):
        ConvergenceConfig(kkt_abs_tol=1e-9, kkt_rel_tol=1e-4, kkt_target=1e-6)


def test_fix5_docs_state_the_measured_step_claim_and_the_kkt_selector():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    changelog = (root / "CHANGELOG.md").read_text()
    api = (root / "docs/api.md").read_text()
    correctness = (root / "docs/correctness.md").read_text()
    assert "The step size is not the bound." not in readme
    assert "0.05" in readme
    assert "absolute-floor crossover" not in api
    assert "absolute-floor crossover" not in correctness
    assert "absolute-floor crossover" not in changelog
    assert "KKT optimality test" in changelog
