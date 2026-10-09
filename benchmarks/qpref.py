"""Exact reference optimum for the benchmark QPs.

Every convergence figure in this suite reports an error against a *reference*
optimum. Taking that reference from a long run of ``snn_opt`` itself measures
the solver against its own fixed point, which hides any standing offset between
that fixed point and the true minimiser. This module supplies an independent
reference instead.

It is a thin shim over :mod:`snn_opt.reference`, which solves the QP as a
least-distance problem with one non-negative least-squares call and returns a
point only together with a certificate: an a-posteriori bound on
``||x - x*||`` that holds whatever the active set. On the benchmark instances
that bound is of order 1e-6. Agreement with another solver (CVXPY/Clarabel,
say) below the bound is two-solver agreement, not a certified error. A point
that cannot be certified raises ``ReferenceNotVerified`` instead of being
returned.
"""

from __future__ import annotations

from snn_opt.reference import (
    ReferenceNotVerified,
    ReferenceResult,
    objective,
    solve_exact,
    solve_reference,
)

__all__ = [
    "ReferenceNotVerified",
    "ReferenceResult",
    "objective",
    "solve_exact",
    "solve_reference",
]
