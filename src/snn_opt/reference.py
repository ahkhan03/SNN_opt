"""Verified reference optimum for strictly convex QPs.

Solves

    minimize    (1/2) x^T A x + b^T x
    subject to  C x + d <= 0

with ``A`` symmetric positive definite, independently of the spiking solver,
and returns the point together with an a-posteriori bound on its distance to
the true minimiser. It is meant as the yardstick that solution errors,
active-set statistics and "verified" periods are measured against, so every
returned point carries its own certificate and a point that cannot be
certified is never returned as an answer.

Route
-----
Cholesky ``A = R^T R`` and ``w = R x + R^{-T} b`` turn the QP into the
least-distance problem ``min ||w|| s.t. G w >= h`` with ``G = -C R^{-1}`` and
``h = d - C A^{-1} b``. That problem is solved by the Lawson-Hanson reduction
to a single non-negative least-squares call (``scipy.optimize.nnls``) on
``E = [G^T; h^T]``, ``f = e_{n+1}``: with ``r = E u - f`` the minimiser is
``w = -r[:n] / r[n]``, and ``r = 0`` means the constraints are inconsistent.
There is no KKT matrix and no active-set guess, so linearly dependent or
nearly parallel active rows (where only the multipliers are non-unique) need
no rank decision. Rows are scaled to unit norm and ``h`` to unit max-norm
before the call; neither changes the solution.

Certificate
-----------
For a point ``x`` with ``C x + d <= 0`` and any multipliers ``lam >= 0``, let
``mu = lambda_min(A)``, ``r = A x + b + C^T lam`` and ``g = -lam^T (C x + d)``
(so ``g >= 0``). With ``f`` the objective and ``L(z) = f(z) + lam^T (C z + d)``,
both ``mu``-strongly convex, and ``delta = ||x - x*||``:

* ``f(x) >= f(x*) + mu/2 delta^2``, because ``x`` is feasible and ``x*`` is
  optimal (``grad f(x*)^T (x - x*) >= 0``);
* ``f(x*) >= L(x*) >= L(x) - ||r|| delta + mu/2 delta^2``, because
  ``lam^T (C x* + d) <= 0`` and ``grad L(x) = r``.

Adding them, with ``L(x) = f(x) - g``, gives ``mu delta^2 - ||r|| delta - g <= 0``,
that is

    ||x - x*|| <= (||r|| + sqrt(||r||^2 + 4 mu g)) / (2 mu),

whatever the active set and whichever ``lam >= 0`` is used. Both steps need
``x`` EXACTLY feasible: with near-parallel rows the optimal multipliers can be
huge, so even a roundoff-sized violation can cost an unbounded amount. The
implementation therefore certifies only a point that is provably feasible in
exact arithmetic, and evaluates every quantity with a rigorous forward-error
bound. Under the standard floating-point model (IEEE double, round to
nearest, no underflow or overflow, ``u = eps / 2``, ``gamma_k = k u / (1 - k u)``)
and assuming LAPACK's symmetric eigensolver has backward error at most
``10 n eps ||A||``:

* ``x`` counts as feasible only if every computed ``s_i = (C x + d)_i`` satisfies
  ``s_i <= -gamma_k (|C_i| |x| + |d_i|)``, so the exact ``s_i`` is ``<= 0``;
* ``||r||`` is replaced by ``||r_hat|| + ||gamma_k (|A||x| + |b| + |C|^T lam)||``,
  an upper bound on the exact residual norm;
* ``g`` is replaced by ``sum_i lam_i (-s_i + gamma_k (|C_i||x| + |d_i|))``,
  an upper bound on the exact gap;
* ``mu`` is replaced by ``lambda_min_computed - 10 n eps lambda_max``,
  a lower bound on the exact ``lambda_min``;

with ``k = 2 (n + 2)`` for the feasibility gate (an ``n``-term dot product
plus one addition needs only ``gamma_{n+1}``) and ``k = 2 (n + m + 2)`` for
``r`` and ``g``, and every derived quantity inflated by a few units of
``gamma_k`` to cover its own rounding. Since the bound is increasing
in ``||r||`` and ``g`` and decreasing in ``mu``, the reported ``error_bound``
bounds the exact distance from the returned (floating-point) ``x`` to the
exact minimiser of the given problem.

Getting a strictly feasible point: the least-distance solution sits on its
active rows to roundoff, on either side. When it is not provably feasible,
the problem is solved again with the offsets tightened, ``d_i + t_i``, where
``t_i`` is a small multiple of the evaluation floor plus the observed
violation, doubling a few times if needed. The tightening only moves where
the candidate point lands; the certificate is always computed against the
ORIGINAL rows, and it absorbs the shift through ``g`` (about
``lam^T t``). A feasible set too thin for any such margin gives
``"unverified"``. Multipliers come from non-negative least squares on the
stationarity condition and from the least-distance solution itself; any
``lam >= 0`` gives a valid bound, so the smallest is reported.

Units: ``(A, b)`` is divided by the power of two nearest ``lambda_max(A)``
before the solve. That is exact, leaves ``x*`` unchanged, and makes the
route independent of the scale of the objective; ``mu``, ``objective``,
``stationarity``, ``complementarity`` and ``multipliers`` are reported in the
units of the given problem. Scaling the variables scales ``x`` and the bound
together.

What the bound can claim: it grows like ``sqrt(g / mu)``, so it cannot
certify below roughly ``sqrt(eps * kappa(A)) * ||x||`` times a modest
constant (of order 1e-7 to 1e-6 on the benchmark problems), even though the
point itself is usually far more accurate than that. Agreement with another
solver below that level is two-solver agreement, not a certified error.
Callers that report a solution error should check that ``error_bound`` sits
well below the smallest error they report. The weakness that remains after
normalisation is dimensionless: a constrained optimum far from the
unconstrained minimiser, measured in the ``A``-norm relative to the row
offsets, with a small ``lambda_min(A)`` (for example spectrum ``[1/kappa, 1]``
at ``kappa >= 1e6`` with order-one multipliers). There the least-distance
point is inaccurate and the call is honestly ``"unverified"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.linalg.lapack import dtrtrs
from scipy.optimize import linprog, nnls

__all__ = [
    "ReferenceNotVerified",
    "ReferenceResult",
    "objective",
    "solve_exact",
    "solve_reference",
]

ROUTE = "ldp-nnls"

_EPS = float(np.finfo(float).eps)
_U = _EPS / 2.0
_ACTIVE_FACTORS = (1e3, 1e6)
_INFEASIBLE_TOL = 1e-12
_SHIFT_ATTEMPTS = 4


def _tri(R: np.ndarray, B: np.ndarray, trans: int = 0) -> np.ndarray:
    """Solve ``R y = B`` (``trans=0``) or ``R^T y = B`` (``trans=1``), R upper.

    LAPACK ``trtrs`` one column at a time: with a matrix right-hand side
    the BLAS may thread even tiny solves, which costs milliseconds per call on
    a loaded machine.
    """
    if B.ndim == 2:
        Y = np.empty_like(B, dtype=float)
        for j in range(B.shape[1]):
            Y[:, j] = _tri(R, B[:, j], trans)
        return Y
    y, info = dtrtrs(R, B, lower=0, trans=trans)
    if info != 0:
        raise np.linalg.LinAlgError(f"triangular solve failed (info={info})")
    return y


def _package_version() -> str:
    from . import __version__

    return __version__


def objective(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    """Evaluate ``(1/2) x^T A x + b^T x``."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    return float(0.5 * x @ A @ x + b @ x)


@dataclass(eq=False)
class ReferenceResult:
    """Outcome of :func:`solve_reference`.

    ``status`` is ``"verified"`` (``x`` is returned with its certificate),
    ``"unverified"`` (the certificate failed; ``x`` is ``None``) or
    ``"infeasible"`` (reported infeasible by an LP, HiGHS at its default
    tolerances; this verdict is not certified, and no point is returned either
    way, so ``x`` is ``None``).

    ``error_bound`` bounds ``||x - x*||`` (Euclidean, in the units of ``x``);
    ``stationarity`` and ``complementarity`` are the upper bounds on ``||r||``
    and ``g`` that entered it, ``mu`` is the computed ``lambda_min(A)`` and
    ``max_violation`` is ``max(0, max_i (C x + d)_i / ||C_i||)``.
    ``multipliers`` has one entry per row of ``C``.
    """

    x: Optional[np.ndarray]
    objective: Optional[float]
    active_rows: Optional[np.ndarray]
    multipliers: Optional[np.ndarray]
    status: str
    error_bound: float
    mu: float
    stationarity: float
    complementarity: float
    max_violation: float
    route: str = ROUTE
    version: str = ""
    reason: str = ""

    @property
    def verified(self) -> bool:
        return self.status == "verified"

    def as_dict(self) -> dict[str, Any]:
        """Plain Python types only (lists, floats, ints, str, None)."""

        def arr(v, kind):
            return None if v is None else [kind(t) for t in np.asarray(v).ravel()]

        return {
            "x": arr(self.x, float),
            "objective": None if self.objective is None else float(self.objective),
            "active_rows": arr(self.active_rows, int),
            "multipliers": arr(self.multipliers, float),
            "status": str(self.status),
            "error_bound": float(self.error_bound),
            "mu": float(self.mu),
            "stationarity": float(self.stationarity),
            "complementarity": float(self.complementarity),
            "max_violation": float(self.max_violation),
            "route": str(self.route),
            "version": str(self.version),
            "reason": str(self.reason),
        }


class ReferenceNotVerified(RuntimeError):  # noqa: N818 (public name)
    """Raised when the reference point cannot be certified.

    ``result`` is the :class:`ReferenceResult` with ``x=None``; its diagnostic
    fields (``status``, ``error_bound``, ``mu``, ``stationarity``,
    ``complementarity``, ``max_violation``, ``route``, ``version``,
    ``reason``) are also readable directly on the exception.
    """

    def __init__(self, result: ReferenceResult):
        self.result = result
        super().__init__(
            f"reference not verified ({result.status}): {result.reason}; "
            f"error_bound={result.error_bound:.3g}, "
            f"max_violation={result.max_violation:.3g}"
        )

    def __getattr__(self, name):
        if name == "result":
            raise AttributeError(name)
        return getattr(self.result, name)

    def __reduce__(self):  # picklable across process pools
        return (type(self), (self.result,))

    def as_dict(self) -> dict[str, Any]:
        return self.result.as_dict()


def _validate(A, b, C, d):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = b.size
    C = np.asarray(C, dtype=float)
    if C.size == 0:
        C = np.zeros((0, n))
    elif C.ndim == 1:
        C = C.reshape(1, -1)
    d = np.asarray(d, dtype=float).reshape(-1)
    if n == 0 or A.shape != (n, n):
        raise ValueError(f"A must be ({n}, {n}) to match b, got {A.shape}")
    if C.ndim != 2 or C.shape != (d.size, n):
        raise ValueError(f"C must be ({d.size}, {n}) to match d and b, got {C.shape}")
    for name, v in (("A", A), ("b", b), ("C", C), ("d", d)):
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{name} contains non-finite entries")
    scale = float(np.max(np.abs(A)))
    if float(np.max(np.abs(A - A.T))) > 1e-10 * scale:
        raise ValueError("A is not symmetric")
    A = 0.5 * (A + A.T)
    eig = np.linalg.eigvalsh(A)
    mu, top = float(eig[0]), float(eig[-1])
    if not (mu > 10.0 * n * _EPS * max(abs(top), 1e-300)):
        raise ValueError(f"A is not positive definite (lambda_min={mu:.3g}, lambda_max={top:.3g})")
    return A, b, C, d, mu, top


def _gamma(k: int) -> float:
    return k * _U / (1.0 - k * _U)


def _eval_floor(C, d, x):
    """Rigorous bound on |fl(C x + d) - (C x + d)|, row by row."""
    k = 2 * (C.shape[1] + 2)
    return _gamma(k) * (np.abs(C) @ np.abs(x) + np.abs(d)) * (1.0 + _gamma(k))


def _bound(r_norm: float, g: float, mu: float) -> float:
    return (r_norm + np.sqrt(r_norm * r_norm + 4.0 * mu * g)) / (2.0 * mu)


def _certify(A, b, C, d, x, mu_cert, lam_candidates, norms):
    """Return (bound, lam, ||r||, g, max_violation, infeasible) for the best candidate.

    ``infeasible`` is True unless ``x`` is provably feasible in exact
    arithmetic; then no bound is formed (``bound = inf``). Otherwise every
    quantity is an upper bound on its exact value (see the module docstring).
    """
    n, m = x.size, C.shape[0]
    s = C @ x + d
    s_err = _eval_floor(C, d, x)
    max_violation = float(max(0.0, (s / norms).max(initial=0.0)))
    if np.any(s > -s_err):
        return float("inf"), None, float("nan"), float("nan"), max_violation, True
    k = 2 * (n + m + 2)
    infl = 1.0 + 4.0 * _gamma(k)
    r_base_err = np.abs(A) @ np.abs(x) + np.abs(b)
    best = None
    for lam in lam_candidates:
        lam = np.maximum(lam, 0.0)
        r = A @ x + b + C.T @ lam
        r_err = _gamma(k) * (r_base_err + np.abs(C).T @ lam)
        r_norm = (float(np.linalg.norm(r)) + float(np.linalg.norm(r_err))) * infl
        g = float(lam @ (s_err - s)) * infl  # every term >= 0 here
        bound = _bound(r_norm, g, mu_cert) * infl
        if best is None or bound < best[0]:
            best = (bound, lam, r_norm, g)
    return (*best, max_violation, False)


def _stationarity_multipliers(A, b, C, x, rows, maxiter):
    lam = np.zeros(C.shape[0])
    if rows.size:
        Cn = C[rows]
        scale = np.linalg.norm(Cn, axis=1)
        sol, _ = nnls((Cn / scale[:, None]).T, -(A @ x + b), maxiter=maxiter)
        lam[rows] = sol / scale
    return lam


def _confirm_infeasible(C, d) -> bool:
    try:
        lp = linprog(np.zeros(C.shape[1]), A_ub=C, b_ub=-d,
                     bounds=[(None, None)] * C.shape[1], method="highs")
    except Exception:
        return False
    return int(lp.status) == 2


def solve_reference(
    A: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    *,
    on_unverified: str = "raise",
    max_error_bound: Optional[float] = None,
    nnls_maxiter: Optional[int] = None,
) -> ReferenceResult:
    """Solve the QP and certify the answer (see the module docstring).

    Parameters
    ----------
    A, b, C, d : array_like
        Problem data for ``min (1/2) x^T A x + b^T x`` s.t. ``C x + d <= 0``.
        ``A`` must be symmetric positive definite. Equality constraints are
        not supported; eliminate them before calling.
    on_unverified : {"raise", "return"}
        ``"raise"`` (default) raises :class:`ReferenceNotVerified` when the
        certificate fails or the constraints are inconsistent. ``"return"``
        returns the result with ``x=None`` and ``status`` ``"unverified"`` or
        ``"infeasible"``, for code that must record the failure and continue.
    max_error_bound : float, optional
        Sanity ceiling on ``error_bound`` for a verified result, in the units
        of ``x``. Defaults to ``1e-3 * max(1, ||x||)``. This is not an
        accuracy target: gate ``error_bound`` against the error scale you
        report.
    nnls_maxiter : int, optional
        Iteration cap for the NNLS calls (default ``50 * (number of rows + n)``).

    Returns
    -------
    ReferenceResult

    Raises
    ------
    ValueError
        Malformed input: mismatched shapes, non-finite data, ``A`` not
        symmetric positive definite.
    ReferenceNotVerified
        With ``on_unverified="raise"``, when no certified point is available.
    """
    if on_unverified not in ("raise", "return"):
        raise ValueError("on_unverified must be 'raise' or 'return'")
    A, b, C, d, mu, top = _validate(A, b, C, d)
    n, m = b.size, d.size
    version = _package_version()
    maxiter = int(nnls_maxiter) if nnls_maxiter is not None else 50 * (m + n + 1)
    # Work on (A, b) / sigma with sigma the power of two nearest lambda_max:
    # the division is exact, x* is unchanged, and the least-distance route
    # stops depending on the units of the objective. Everything below runs on
    # the scaled pair; stationarity, complementarity and multipliers are
    # reported back in the units of the given problem (times sigma, exact).
    sigma = 2.0 ** round(np.log2(top))
    As, bs = A / sigma, b / sigma
    mu_cert = (mu - 10.0 * n * _EPS * top) / sigma

    def fail(status, reason, bound=float("inf"), r_norm=float("nan"),
             g=float("nan"), max_violation=float("nan")):
        res = ReferenceResult(None, None, None, None, status, float(bound), mu,
                              sigma * float(r_norm), sigma * float(g), float(max_violation),
                              ROUTE, version, reason)
        if on_unverified == "raise":
            raise ReferenceNotVerified(res)
        return res

    norms = np.linalg.norm(C, axis=1)
    zero = norms == 0.0
    if np.any(zero & (d > 0.0)):
        return fail("infeasible", "a zero row of C has d > 0")
    keep = np.flatnonzero(~zero)
    Ck, dk, nk = C[keep], d[keep], norms[keep]
    safe_norms = np.where(zero, 1.0, norms)

    try:
        R = np.linalg.cholesky(As).T  # As = R^T R, R upper triangular
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"A is not positive definite (Cholesky failed: {exc})") from exc
    Rtb = _tri(R, bs, 1)  # R^{-T} bs
    if keep.size:
        Cu = Ck / nk[:, None]
        G = -_tri(R, Cu.T, 1).T  # -Cu R^{-1}
        h0 = dk / nk + G @ Rtb  # du - Cu As^{-1} bs

    shift = np.zeros(m)  # row-normalised tightening of the offsets
    lam_ldp = np.zeros(m)
    for attempt in range(_SHIFT_ATTEMPTS):
        if keep.size == 0:
            w = np.zeros(n)
        else:
            h = h0 + shift[keep]
            alpha = float(np.max(np.abs(h)))
            alpha = alpha if alpha > 0.0 else 1.0
            E = np.vstack([G.T, (h / alpha)[None, :]])
            f = np.zeros(n + 1)
            f[n] = 1.0
            try:
                u, _ = nnls(E, f, maxiter=maxiter)
            except Exception as exc:  # iteration cap or LAPACK failure
                return fail("unverified", f"NNLS failed: {type(exc).__name__}: {exc}")
            r = E @ u - f
            r_norm = float(np.linalg.norm(r))
            # At the exact NNLS optimum r[n] = -||r||^2, and r = 0 iff the rows
            # are inconsistent (||f|| = 1 sets the scale). The two tests agree at
            # an exact optimum; the first also catches a residual at roundoff
            # level, where that identity no longer holds. Either is only a
            # suspicion; an LP on the original rows decides.
            if r_norm <= _INFEASIBLE_TOL or abs(r[n]) <= _INFEASIBLE_TOL * r_norm:
                if _confirm_infeasible(C, d):
                    return fail("infeasible", "constraints are inconsistent")
                if attempt or r[n] == 0.0:
                    return fail("unverified", "feasible set too thin for a provably "
                                              "feasible point")
            w = -alpha * r[:n] / r[n]
            lam_ldp = np.zeros(m)
            lam_ldp[keep] = np.maximum(alpha * u / (-r[n]), 0.0) / nk
        x = _tri(R, w - Rtb)
        if not np.all(np.isfinite(x)):
            return fail("unverified", "non-finite point")
        s, s_err = C @ x + d, _eval_floor(C, d, x)
        need = s + s_err  # > 0 where x is not provably feasible
        if not np.any(need[keep] > -s_err[keep]):
            break  # every row clears its floor with room to spare
        # Tighten: a few evaluation floors plus the observed violation.
        shift = np.maximum(2.0 * shift, (4.0 * s_err + 2.0 * np.maximum(need, 0.0)) / safe_norms)
    else:
        if np.any(need > 0.0):
            return fail("unverified", "no provably feasible point after tightening",
                        max_violation=float(max(0.0, (s / safe_norms).max(initial=0.0))))

    # Multiplier candidates: the least-distance multipliers, and NNLS on
    # stationarity over the rows active at x (allowing for the tightening).
    candidates = [lam_ldp]
    if keep.size:
        margin = shift * safe_norms
        for factor in _ACTIVE_FACTORS:
            rows = np.flatnonzero(~zero & (s >= -(factor * s_err + 2.0 * margin)))
            try:
                candidates.append(_stationarity_multipliers(As, bs, C, x, rows, maxiter))
            except Exception:
                pass
    bound, lam, r_norm, g, max_violation, infeasible = _certify(
        As, bs, C, d, x, mu_cert, candidates, safe_norms)

    if infeasible:
        return fail("unverified", "point is not provably feasible",
                    float("inf"), r_norm, g, max_violation)
    ceiling = (float(max_error_bound) if max_error_bound is not None
               else 1e-3 * max(1.0, float(np.linalg.norm(x))))
    if not (np.isfinite(bound) and bound <= ceiling):
        return fail("unverified", f"error bound {bound:.3g} exceeds the ceiling {ceiling:.3g}",
                    bound, r_norm, g, max_violation)

    active = np.flatnonzero(~zero & (s >= -(_ACTIVE_FACTORS[0] * s_err + 2.0 * shift * safe_norms)))
    return ReferenceResult(x, objective(A, b, x), active, sigma * lam, "verified", float(bound),
                           mu, sigma * r_norm, sigma * g, max_violation, ROUTE, version, "")


def solve_exact(
    A: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    x_guess: Optional[np.ndarray] = None,
    *,
    max_swaps: int = 200,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return ``(x_star, f_star, active_rows)``; legacy interface.

    Thin wrapper over :func:`solve_reference`. ``x_guess`` and ``max_swaps``
    belonged to the earlier active-set route and are accepted and ignored.
    Raises :class:`ReferenceNotVerified` (a ``RuntimeError``) when the point
    cannot be certified or the constraints are inconsistent.
    """
    del x_guess, max_swaps
    res = solve_reference(A, b, C, d)
    return res.x, float(res.objective), res.active_rows
