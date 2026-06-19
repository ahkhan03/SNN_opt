"""Problem transforms for the SNN-QP solver.

A *transform* rewrites the optimization problem into an equivalent one that is
cheaper (or better conditioned) to solve, then maps the solution back to the
original coordinates. Transforms operate on the problem *data* (A, b, C, d), not
on the solve loop, so they are **backend-agnostic** and compose with any backend
(`python`, `c`, `c_serial`, `c_openmp`).

Transforms are an **explicit opt-in** via ``SolverConfig.transform``; the
canonical solver remains the default. This keeps the door open for adding more
transforms (normalization, preconditioning, …) and defers any "which is the
default" decision until experiments single out a winner.

This is the *transform* axis of the solver's design. It is orthogonal to the
*outer-dynamics* axis (integration scheme) and the *inner-projection* axis
(feasibility mechanism).

Currently implemented
---------------------
- :class:`EigenbasisTransform` — rotate a symmetric-PSD Hessian into its
  eigenbasis so the dominant ``O(n^2)`` ``A @ x`` step collapses to an ``O(n)``
  elementwise product. Requires a box-free problem (the rotation does not
  preserve per-coordinate box bounds).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np


@dataclass
class TransformContext:
    """Transformed problem data plus the means to recover the original solution.

    Attributes
    ----------
    A, b, C, d : ndarray
        The transformed problem ``min ½ xᵀAx + bᵀx  s.t.  Cx + d ≤ 0`` in the
        transformed coordinates. Fed to the inner (canonical) solve.
    x0 : ndarray
        The initial iterate mapped into the transformed coordinates.
    a_diag : ndarray or None
        The diagonal of the transformed Hessian when the transform diagonalizes
        ``A`` (lets the solver use its ``O(n)`` diagonal fast path). ``None``
        when the transform leaves ``A`` dense — the solver then uses the dense
        matvec path.
    recover : callable
        Maps a solution in transformed coordinates back to the original ones.
    """

    A: np.ndarray
    b: np.ndarray
    C: np.ndarray
    d: np.ndarray
    x0: np.ndarray
    a_diag: Optional[np.ndarray]
    recover: Callable[[np.ndarray], np.ndarray]


class Transform:
    """Base class for problem transforms.

    Subclasses implement :meth:`forward` (and usually :meth:`check_applicable`).
    """

    name: str = "identity"

    def check_applicable(self, problem, config) -> None:
        """Raise ``ValueError`` if the transform cannot be applied.

        Called before :meth:`forward`. Default: always applicable.
        """

    def forward(self, problem, x0: np.ndarray, config) -> TransformContext:
        """Map ``problem`` + ``x0`` into transformed coordinates."""
        raise NotImplementedError


class EigenbasisTransform(Transform):
    """Diagonalize a constant symmetric-PSD Hessian via its eigenbasis.

    With ``A = V Λ Vᵀ`` (eigendecomposition), solving in the rotated coordinate
    ``ỹ = Vᵀx`` turns the dominant ``O(n^2)`` gradient matvec ``A x`` into the
    ``O(n)`` elementwise step ``Λ ⊙ ỹ``. The constraints rotate to ``Ĉ = C V``
    (one-time ``O(m n^2)`` precompute) with ``d`` unchanged; the constraint Gram
    ``G = ĈĈᵀ = CCᵀ`` and the row norms ``‖ĉ_j‖²`` are **invariant** under the
    orthogonal ``V``, so the projection step is numerically identical. The
    solution maps back as ``x = V ỹ``.

    Restriction: **box constraints are not supported.** Per-coordinate bounds
    ``l ≤ x ≤ u`` are not preserved by the rotation (clipping in ``ỹ`` does not
    correspond to clipping in ``x``), so :meth:`check_applicable` raises when
    ``lower_bound`` / ``upper_bound`` are set.
    """

    name = "eigenbasis"

    def check_applicable(self, problem, config) -> None:
        if config.lower_bound is not None or config.upper_bound is not None:
            raise ValueError(
                "transform='eigenbasis' is incompatible with box constraints "
                "(lower_bound / upper_bound set): the eigenbasis rotation does "
                "not preserve per-coordinate box bounds (box-in-x is not "
                "rotation-invariant). Use transform=None (the default) for "
                "box-constrained problems.")
        from scipy.sparse import issparse
        if issparse(problem.A) or issparse(problem.C):
            raise ValueError(
                "transform='eigenbasis' requires dense A and C "
                "(scipy sparse not supported).")

    def forward(self, problem, x0: np.ndarray, config) -> TransformContext:
        A = np.ascontiguousarray(problem.A, dtype=np.float64)
        # Symmetric eigendecomposition; symmetrize defensively against tiny
        # asymmetry in a user-supplied A. eigh returns ascending real eigenvalues
        # and an orthonormal V.
        w, V = np.linalg.eigh(0.5 * (A + A.T))

        n = problem.n_vars
        m = problem.n_constraints
        b_t = V.T @ np.ascontiguousarray(problem.b, dtype=np.float64)
        if m > 0:
            C_t = np.ascontiguousarray(problem.C, dtype=np.float64).reshape(m, n) @ V
        else:
            C_t = np.zeros((0, n), dtype=np.float64)
        d_t = np.ascontiguousarray(problem.d, dtype=np.float64)
        x0_t = V.T @ np.asarray(x0, dtype=np.float64)

        a_diag = np.ascontiguousarray(w, dtype=np.float64)
        # Dense diagonal A keeps every non-hot-path computation (k0 auto-step,
        # objective/pgnorm in the result builder) correct without special casing;
        # the hot Hessian apply uses a_diag for the O(n) fast path.
        A_t = np.diag(a_diag)

        def recover(x_t: np.ndarray) -> np.ndarray:
            return V @ np.asarray(x_t, dtype=np.float64)

        return TransformContext(A=A_t, b=b_t, C=C_t, d=d_t, x0=x0_t,
                                a_diag=a_diag, recover=recover)


# Registry for string-named transforms (mirrors the backend-string convention).
_TRANSFORMS = {
    "eigenbasis": EigenbasisTransform,
}


def resolve_transform(transform: Union[str, Transform]) -> Transform:
    """Turn a ``SolverConfig.transform`` value into a :class:`Transform`.

    Accepts a registered name (e.g. ``'eigenbasis'``) or a ``Transform``
    instance. Raises ``ValueError`` on an unknown name.
    """
    if isinstance(transform, Transform):
        return transform
    if isinstance(transform, str):
        try:
            return _TRANSFORMS[transform]()
        except KeyError:
            raise ValueError(
                f"unknown transform {transform!r}; expected one of "
                f"{sorted(_TRANSFORMS)} or a Transform instance") from None
    raise TypeError(
        f"transform must be a str or Transform, got {type(transform).__name__}")
