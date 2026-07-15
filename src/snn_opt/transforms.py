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
  elementwise product. Since v0.5.0 box bounds are supported by materializing
  the bound facets as explicit rotated rows (the box is not axis-aligned in the
  eigenbasis, so the implicit O(1)-per-facet representation is surrendered:
  each facet becomes a dense unit-norm row ±V[i,:] and m grows by up to 2n).
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
    consumes_bounds : bool
        True when the transform has folded the config's scalar box bounds into
        explicit rows of ``C`` (e.g. rotated facets under an eigenbasis); the
        inner solve must then run with ``lower_bound = upper_bound = None`` so
        the bounds are not applied a second time in the wrong coordinates.
    """

    A: np.ndarray
    b: np.ndarray
    C: np.ndarray
    d: np.ndarray
    x0: np.ndarray
    a_diag: Optional[np.ndarray]
    recover: Callable[[np.ndarray], np.ndarray]
    consumes_bounds: bool = False


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

    Box bounds (since v0.5.0): the box is not axis-aligned in the eigenbasis
    (clipping in ``ỹ`` does not correspond to clipping in ``x``), so bound
    facets are **materialized as explicit rotated rows**: ``x_i ≤ u`` becomes
    the unit-norm row ``+V[i,:]·ỹ ≤ u`` and ``x_i ≥ l`` becomes ``−V[i,:]·ỹ ≤
    −l`` (rows of an orthogonal ``V`` have unit norm). This surrenders the
    canonical solve's implicit O(1)-per-facet advantage — ``m`` grows by up to
    ``2n`` dense rows — which is the honest cost of combining the transform
    with a box. The inner solve then runs bound-free (``consumes_bounds``).
    """

    name = "eigenbasis"

    def check_applicable(self, problem, config) -> None:
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

        # Materialize box facets as rotated rows (see class docstring). In the
        # original coordinates the facet normals are ±e_i; rotated they are
        # ±(e_i^T V) = ±V[i,:]. Frozen order matches the canonical facet
        # convention: all lower facets 0..n-1, then all upper facets 0..n-1.
        consumes_bounds = False
        lo, hi = config.lower_bound, config.upper_bound
        if lo is not None or hi is not None:
            blocks_C, blocks_d = [C_t], [d_t]
            if lo is not None:
                blocks_C.append(-V)                      # rows -V[i,:]
                blocks_d.append(np.full(n, float(lo)))   # -x_i + l <= 0
            if hi is not None:
                blocks_C.append(V)                       # rows +V[i,:]
                blocks_d.append(np.full(n, -float(hi)))  # x_i - u <= 0
            C_t = np.ascontiguousarray(np.vstack(blocks_C))
            d_t = np.concatenate(blocks_d)
            consumes_bounds = True

        x0_t = V.T @ np.asarray(x0, dtype=np.float64)

        a_diag = np.ascontiguousarray(w, dtype=np.float64)
        # Dense diagonal A keeps every non-hot-path computation (k0 auto-step,
        # objective/pgnorm in the result builder) correct without special casing;
        # the hot Hessian apply uses a_diag for the O(n) fast path.
        A_t = np.diag(a_diag)

        def recover(x_t: np.ndarray) -> np.ndarray:
            return V @ np.asarray(x_t, dtype=np.float64)

        return TransformContext(A=A_t, b=b_t, C=C_t, d=d_t, x0=x0_t,
                                a_diag=a_diag, recover=recover,
                                consumes_bounds=consumes_bounds)


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
