"""Exact reference optimum for the benchmark QPs.

Every convergence figure in this suite reports an error against a *reference*
optimum. Taking that reference from a long run of ``snn_opt`` itself measures
the solver against its own fixed point, which hides any standing offset between
that fixed point and the true minimiser. This module supplies an independent
reference instead.

The problems here are small, so the reference is computed exactly rather than
iteratively: guess an active set, solve the equality-constrained KKT system in
closed form, then repair the guess by dropping rows with negative multipliers
and adding violated rows until both KKT sign conditions hold. On the benchmark
instances this agrees with CVXPY/Clarabel to about 1e-10 and costs under a
millisecond, using nothing beyond NumPy.
"""

from __future__ import annotations

import numpy as np

__all__ = ["solve_exact", "objective"]


def objective(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    """Evaluate ``(1/2) x^T A x + b^T x``."""
    return float(0.5 * x @ A @ x + b @ x)


def solve_exact(
    A: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    x_guess: np.ndarray | None = None,
    *,
    max_swaps: int = 200,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return ``(x_star, f_star, active_rows)`` for the inequality-constrained QP.

    Solves ``min (1/2) x^T A x + b^T x`` subject to ``C x + d <= 0`` by an
    active-set method. ``x_guess`` only seeds the initial active set; the answer
    does not depend on it, so it is safe to seed from an approximate solve.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    C = np.atleast_2d(np.asarray(C, dtype=float))
    d = np.asarray(d, dtype=float)
    n = b.size

    if x_guess is None:
        active = np.zeros(0, dtype=int)
    else:
        active = np.where(C @ np.asarray(x_guess, dtype=float) + d > -1e-7)[0]

    for _ in range(max_swaps):
        if active.size == 0:
            x = np.linalg.solve(A, -b)
            lam = np.zeros(0)
        else:
            Ca = C[active]
            k = active.size
            kkt = np.block([[A, Ca.T], [Ca, np.zeros((k, k))]])
            sol = np.linalg.solve(kkt, np.concatenate([-b, -d[active]]))
            x, lam = sol[:n], sol[n:]

        # Dual feasibility: a negative multiplier means the row should be free.
        if lam.size and lam.min() < -1e-12:
            active = np.delete(active, int(np.argmin(lam)))
            continue

        # Primal feasibility: pull in the worst violated row not already active.
        violation = C @ x + d
        candidates = np.setdiff1d(np.where(violation > 1e-12)[0], active)
        if candidates.size:
            worst = candidates[int(np.argmax(violation[candidates]))]
            active = np.sort(np.append(active, worst))
            continue

        return x, objective(A, b, x), active

    raise RuntimeError("active-set iteration did not settle; problem may be degenerate")
