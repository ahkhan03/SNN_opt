"""Tests for the verified reference solver ``snn_opt.reference``.

The instances have a KNOWN minimiser. All data are dyadic rationals with few
significant bits, so ``b = -(A x* + C_a^T lam)`` and ``d`` are computed
without rounding and x* is the exact optimum of the floating-point problem
(checked in rational arithmetic by ``_assert_exact_kkt``). Without that, rows
at angles near 1e-9 make the rounded problem's optimum differ from the
constructed x* by more than the certificate, and the test would be measuring
its own construction error.
"""

import pickle
from fractions import Fraction

import numpy as np
import pytest

from benchmarks.qpref import objective as shim_objective
from benchmarks.qpref import solve_exact as shim_solve_exact
from snn_opt import __version__
from snn_opt.reference import (
    ReferenceNotVerified,
    ReferenceResult,
    _certify,
    _stationarity_multipliers,
    objective,
    solve_exact,
    solve_reference,
)


def _spd(n, kappa, rng, *, small_end=False):
    """Dyadic SPD matrix M^T D M with condition number close to `kappa`.

    `small_end` puts the spectrum at [1/kappa, 1] instead of [1, kappa], so
    lambda_min is tiny relative to the forces in b.
    """
    band = np.triu(np.ones((n, n)), 1) - np.triu(np.ones((n, n)), 3)
    M = np.eye(n) + 0.5 * rng.integers(-1, 2, (n, n)) * band
    e = np.round(np.linspace(0.0, np.log2(kappa), n))
    D = 2.0 ** (-e if small_end else e)
    return M.T @ (D[:, None] * M)


def _near_parallel(n, k, log2_angle, rng):
    """k integer rows perturbed by 2**log2_angle: pairwise angles of that order."""
    base = rng.integers(-3, 4, n).astype(float)
    base[0] = 4.0
    return np.array([base + 2.0**log2_angle * i * rng.integers(-2, 3, n) for i in range(k)])


def _instance(A, C_active, lam, C_slack, slack, x_star):
    b = -(A @ x_star + C_active.T @ lam)
    C = np.vstack([C_active, C_slack])
    d = np.r_[-C_active @ x_star, -C_slack @ x_star - slack]
    _assert_exact_kkt(A, b, C, d, x_star, lam)
    return b, C, d


def _assert_exact_kkt(A, b, C, d, x_star, lam_active):
    """x* is the exact optimum of the floating-point data (rational arithmetic)."""
    q = np.vectorize(Fraction, otypes=[object])
    A_, b_, C_, d_, x_ = q(A), q(b), q(C), q(d), q(x_star)
    lam = np.zeros(C.shape[0], dtype=object)
    lam[: len(lam_active)] = q(lam_active)
    lam[len(lam_active):] = Fraction(0)
    assert all(v == 0 for v in A_.dot(x_) + b_ + C_.T.dot(lam))
    s = C_.dot(x_) + d_
    assert all(v == 0 for v in s[: len(lam_active)])
    assert all(v < 0 for v in s[len(lam_active):])
    assert all(v > 0 for v in lam[: len(lam_active)])


def _dyadic(rng, size, scale=8):
    return rng.integers(-scale, scale + 1, size) / 4.0


def _known(n, kappa, k, log2_angle, seed, *, small_end=False):
    rng = np.random.default_rng(seed)
    A = _spd(n, kappa, rng, small_end=small_end)
    Ca = _near_parallel(n, k, log2_angle, rng)
    lam = rng.integers(1, 9, k) / 4.0
    Cs = rng.integers(-3, 4, (5, n)).astype(float)
    x_star = _dyadic(rng, n)
    b, C, d = _instance(A, Ca, lam, Cs, rng.integers(1, 9, 5) / 8.0, x_star)
    return A, b, C, d, x_star


def _check(res, x_star, tol):
    assert res.status == "verified"
    err = float(np.linalg.norm(res.x - x_star))
    assert res.error_bound >= err, (res.error_bound, err)
    assert err <= tol, err
    return err


ANGLES = [-10, -20, -30]  # 2**-10 ~ 1e-3, 2**-20 ~ 1e-6, 2**-30 ~ 1e-9


@pytest.mark.parametrize("kappa", [1e1, 1e3, 1e6])
@pytest.mark.parametrize("k", [2, 6])
@pytest.mark.parametrize("log2_angle", ANGLES)
@pytest.mark.parametrize("seed", range(3))
def test_near_parallel_active_rows(kappa, k, log2_angle, seed):
    A, b, C, d, x_star = _known(8, kappa, k, log2_angle, seed)
    res = solve_reference(A, b, C, d)
    _check(res, x_star, tol=1e-6)
    assert set(range(k)) <= set(res.active_rows.tolist())
    assert np.all(res.multipliers >= 0.0)


@pytest.mark.parametrize("kappa", [1e3, 1e6])
@pytest.mark.parametrize("k", [2, 6])
@pytest.mark.parametrize("log2_angle", ANGLES)
def test_small_lambda_min_is_honest(kappa, k, log2_angle):
    """Spectrum at [1/kappa, 1]: harder (b is large next to A). Whatever the
    route achieves, a verified point is within its bound and an unverified one
    carries no x. The bound is also checked with the sanity ceiling lifted, so
    a loose-but-lying bound cannot hide behind the ceiling."""
    for seed in range(3):
        A, b, C, d, x_star = _known(8, kappa, k, log2_angle, seed, small_end=True)
        for ceiling in (None, np.inf):
            res = solve_reference(A, b, C, d, on_unverified="return", max_error_bound=ceiling)
            if res.status == "verified":
                assert res.error_bound >= np.linalg.norm(res.x - x_star)
            else:
                assert res.status == "unverified" and res.x is None


@pytest.mark.parametrize("inconsistent", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_collision_like_geometry(inconsistent, seed):
    """Six control points against one sphere: nearby normals, nearly equal offsets."""
    rng = np.random.default_rng(seed)
    n = 7
    J = rng.integers(-2, 3, (3, n)).astype(float)
    A = J.T @ J + 2.0**-6 * np.eye(n)  # tracking cost with a small regulariser
    normal = rng.integers(-2, 3, 3).astype(float)
    normal[2] = 2.0
    rows = []
    for _ in range(6):
        Jj = J + 2.0**-14 * rng.integers(-2, 3, (3, n))  # nearby control points
        nj = normal + 2.0**-17 * rng.integers(-2, 3, 3)
        rows.append(-nj @ Jj)  # keep-out row: -n_j^T J_j x + offset <= 0
    rows = np.array(rows)
    x_star = _dyadic(rng, n)
    if inconsistent:
        # Rows 0 and 3 touch; the others sit 2**-27 (about 7e-9) inside.
        tight, loose = [0, 3], [1, 2, 4, 5]
        lam = rng.integers(1, 9, 2) / 4.0
        b, C, d = _instance(A, rows[tight], lam, rows[loose], np.full(4, 2.0**-27), x_star)
    else:
        lam = rng.integers(1, 9, 6) / 4.0
        b, C, d = _instance(A, rows, lam, np.zeros((0, n)), np.zeros(0), x_star)
    res = solve_reference(A, b, C, d)
    _check(res, x_star, tol=1e-6)


def test_bound_never_undercuts_for_arbitrary_points():
    """For ANY x and lam >= 0 the certificate either refuses x as not provably
    feasible or bounds its true error, including points that violate a row by
    roundoff-sized amounts (where a multiplier-based charge would not be safe)."""
    rng = np.random.default_rng(7)
    feasible = refused = 0
    for trial in range(400):
        A, b, C, d, x_star = _known(6, 10 ** rng.uniform(0, 6), 3, int(rng.integers(-30, -3)),
                                    1000 + trial)
        if trial % 2:  # interior-ish points
            x = x_star - 10 ** rng.uniform(-8, 0) * rng.random() * C[:3].sum(axis=0)
            x += 10 ** rng.uniform(-8, -2) * rng.standard_normal(6)
        else:  # points hugging the active rows, on either side, by 1e-16 .. 1e-6
            x = x_star + 10 ** rng.uniform(-16, -6) * rng.standard_normal(6)
        lam = rng.uniform(0, 2, C.shape[0]) * (rng.random(C.shape[0]) < 0.6)
        mu = float(np.linalg.eigvalsh(A)[0])
        bound, _, _, _, _, infeasible = _certify(A, b, C, d, x, mu, [lam],
                                                 np.linalg.norm(C, axis=1))
        if infeasible:
            assert bound == np.inf
            refused += 1
        else:
            assert np.all(C @ x + d <= 0.0)
            assert bound >= np.linalg.norm(x - x_star), trial
            feasible += 1
    assert feasible > 100 and refused > 50


def test_certificate_refuses_slightly_infeasible_point_with_wrong_multipliers():
    """x* = 0 with lam* = [4, 0]; the point violates row 0 by 3.8e-6 and is
    3.9e-3 from x*. Stationarity least squares puts the multiplier on the
    near-copy row 1, where a multiplier-weighted violation charge is ~0."""
    L, delta = 4.0, 2.0**-10
    p = L * delta**2
    A, b = np.eye(2), np.array([L, 0.0])
    C = np.array([[-1.0, 0.0], [-1.0, delta]])
    d = np.zeros(2)
    x = np.array([-p, -p / delta])
    lam = _stationarity_multipliers(A, b, C, x, np.arange(2), 100)
    bound, _, _, _, violation, infeasible = _certify(A, b, C, d, x, 1.0, [lam],
                                                     np.linalg.norm(C, axis=1))
    assert infeasible and bound == np.inf and violation > 0.0


def test_certificate_refuses_equality_repaired_near_parallel_point():
    """kappa 1e6 at the small end, six rows within 2**-30: a float64 equality
    solve on the active rows looks stationary but sits ~1e-6..1e-2 from x*
    and touches a sibling row from outside. It must not be certified."""
    for seed in (4000, 4001, 4002, 4003):
        A, b, C, d, x_star = _known(8, 1e6, 6, -30, seed, small_end=True)
        mu = float(np.linalg.eigvalsh(A)[0])
        for rows in ([0], [0, 1], list(range(6))):
            Ca = C[rows]
            K = np.block([[A, Ca.T], [Ca, np.zeros((len(rows), len(rows)))]])
            x = np.linalg.lstsq(K, np.r_[-b, -d[rows]], rcond=None)[0][:8]
            lam = _stationarity_multipliers(A, b, C, x, np.arange(6), 1000)
            bound, _, _, _, _, infeasible = _certify(A, b, C, d, x, mu, [lam],
                                                     np.linalg.norm(C, axis=1))
            assert infeasible or bound >= np.linalg.norm(x - x_star)


def test_unconstrained_and_inactive():
    rng = np.random.default_rng(3)
    A = _spd(5, 1e2, rng)
    x_star = _dyadic(rng, 5)
    b = -A @ x_star
    res = solve_reference(A, b, np.zeros((0, 5)), np.zeros(0))
    _check(res, x_star, 1e-10)
    assert res.active_rows.size == 0 and res.multipliers.size == 0
    C = np.ones((1, 5))
    d = np.array([-(C @ x_star)[0] - 10.0])  # a loose row changes nothing
    res = solve_reference(A, b, C, d)
    _check(res, x_star, 1e-10)
    assert res.active_rows.size == 0


def test_result_fields_and_serialization():
    A, b, C, d, x_star = _known(4, 10.0, 2, -4, 4)
    res = solve_reference(A, b, C, d)
    assert isinstance(res, ReferenceResult) and res.verified
    assert res.route == "ldp-nnls" and res.version == __version__
    assert res.objective == pytest.approx(objective(A, b, res.x))
    assert res.mu == pytest.approx(np.linalg.eigvalsh(A)[0])
    out = res.as_dict()
    for key in ("x", "objective", "active_rows", "multipliers", "status", "error_bound",
                "mu", "stationarity", "complementarity", "max_violation", "route", "version"):
        assert key in out
    assert isinstance(out["x"], list) and all(type(v) is float for v in out["x"])
    assert all(type(v) is int for v in out["active_rows"])
    assert type(out["error_bound"]) is float


def test_unverified_raises_by_default_and_returns_on_request():
    A, b, C, d, _ = _known(4, 10.0, 2, -4, 5)
    with pytest.raises(ReferenceNotVerified) as info:
        solve_reference(A, b, C, d, max_error_bound=0.0)
    exc = info.value
    assert isinstance(exc, RuntimeError)
    assert exc.status == "unverified" and exc.result.x is None
    assert np.isfinite(exc.error_bound) and exc.error_bound > 0.0
    assert exc.as_dict()["x"] is None
    again = pickle.loads(pickle.dumps(exc))  # survives a process pool
    assert again.status == "unverified" and again.error_bound == exc.error_bound

    res = solve_reference(A, b, C, d, max_error_bound=0.0, on_unverified="return")
    assert res.status == "unverified" and res.x is None and res.objective is None
    assert res.as_dict()["x"] is None


def test_infeasible_set():
    A = np.eye(2)
    b = np.zeros(2)
    C = np.array([[1.0, 0.0], [-1.0, 0.0]])
    d = np.array([1.0, 1.0])  # x0 <= -1 and x0 >= 1
    with pytest.raises(ReferenceNotVerified) as info:
        solve_reference(A, b, C, d)
    assert info.value.status == "infeasible"
    res = solve_reference(A, b, C, d, on_unverified="return")
    assert res.status == "infeasible" and res.x is None
    # A zero row with a positive offset is inconsistent too.
    res = solve_reference(A, b, np.zeros((1, 2)), np.array([1.0]), on_unverified="return")
    assert res.status == "infeasible"


@pytest.mark.parametrize(
    "A",
    [
        np.diag([1.0, -1.0]),  # indefinite
        np.diag([1.0, 0.0]),  # singular
        np.array([[1.0, 0.5], [0.0, 1.0]]),  # not symmetric
        np.array([[1.0, np.nan], [np.nan, 1.0]]),  # non-finite
        np.eye(3),  # wrong shape
    ],
)
def test_malformed_input_raises_value_error(A):
    with pytest.raises(ValueError):
        solve_reference(A, np.zeros(2), np.eye(2), np.zeros(2))


def test_bad_on_unverified_value():
    with pytest.raises(ValueError):
        solve_reference(np.eye(2), np.zeros(2), np.eye(2), np.zeros(2), on_unverified="ignore")


def test_legacy_solve_exact_tuple():
    A, b, C, d, x_star = _known(5, 1e2, 3, -12, 6)
    for fn in (solve_exact, shim_solve_exact):
        x, f, active = fn(A, b, C, d, x_star, max_swaps=5)
        assert isinstance(x, np.ndarray) and isinstance(f, float)
        assert isinstance(active, np.ndarray) and active.dtype.kind == "i"
        assert np.linalg.norm(x - x_star) < 1e-8
        assert f == pytest.approx(shim_objective(A, b, x_star))
    with pytest.raises(RuntimeError):
        solve_exact(np.eye(1), np.zeros(1), np.array([[1.0], [-1.0]]), np.array([1.0, 1.0]))


def test_agrees_with_clarabel():
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("Clarabel not installed")
    rng = np.random.default_rng(8)
    for _ in range(5):
        n, m = 12, 20
        M = rng.standard_normal((n, n))
        A = M.T @ M + 0.1 * np.eye(n)
        b = rng.standard_normal(n) * 3
        C = rng.standard_normal((m, n))
        d = -np.abs(rng.standard_normal(m))
        res = solve_reference(A, b, C, d)
        x = cp.Variable(n)
        cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(A)) + b @ x),
                   [C @ x + d <= 0]).solve(solver="CLARABEL")
        # Two-solver agreement at the interior-point solver's tolerance.
        assert np.linalg.norm(res.x - x.value) < 1e-6


@pytest.mark.parametrize("scale", [1e8, 1e12, 1e-8])
def test_objective_scaling_leaves_the_solution_unchanged(scale):
    """(A, b) -> s (A, b) keeps x*; the route must not drift with the units of
    the objective (MPC weights, SI stiffness). mu and the multipliers scale
    with s, the error bound does not. Rows are well separated here: with
    near-parallel rows, rounding the scaled data alone moves x* by about
    eps / angle, which is the problem's sensitivity, not the route's."""
    for seed in range(3):
        A, b, C, d, _ = _known(8, 1e3, 2, -2, seed)
        base = solve_reference(A, b, C, d)
        res = solve_reference(scale * A, scale * b, C, d)
        assert res.verified
        assert np.linalg.norm(res.x - base.x) <= 1e-11
        assert res.mu == pytest.approx(scale * base.mu, rel=1e-9)
        assert res.error_bound <= 10 * base.error_bound


@pytest.mark.parametrize("scale", [1e-6, 1e6])
def test_variable_scaling_scales_the_bound(scale):
    """x = z / s: (A s^2, b s, C s). The solution and the bound scale by 1/s."""
    A, b, C, d, _ = _known(8, 1e3, 2, -2, 0)
    base = solve_reference(A, b, C, d)
    res = solve_reference(A * scale**2, b * scale, C * scale, d)
    assert res.verified
    assert np.linalg.norm(scale * res.x - base.x) <= 1e-11 * max(1.0, np.linalg.norm(base.x))
    assert scale * res.error_bound == pytest.approx(base.error_bound, rel=0.5)


def test_result_is_not_compared_by_value():
    A, b, C, d, _ = _known(4, 10.0, 2, -4, 4)
    res = solve_reference(A, b, C, d)
    assert res == res and res != solve_reference(A, b, C, d)  # identity, no array __eq__


def test_power_of_two_objective_scaling_is_exact():
    """A power-of-two scale changes no bits after normalisation, even with
    rows 2**-30 apart: the returned point is identical."""
    A, b, C, d, _ = _known(8, 1e6, 6, -30, 1)
    base = solve_reference(A, b, C, d)
    for k in (-40, 40):
        res = solve_reference(2.0**k * A, 2.0**k * b, C, d)
        assert np.array_equal(res.x, base.x) and res.error_bound == base.error_bound
