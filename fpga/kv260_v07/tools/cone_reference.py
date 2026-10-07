"""Independent references for the v07 friction-cone parity anchors.

``build/run_cone_parity.sh`` compares the fixed-point kernel model against two
references that share no arithmetic with it: a binary64 ``SNNSolver`` run that
records its complete projection-event stream, and the Clarabel conic optimum
via CVXPY.  Clarabel is optional; when it is not installed the reference
reports ``not_available`` and the parity rows keep running.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any, Sequence

import numpy as np

from fpga.kv260_v07.src import kernel_model as km


def _objective(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    xx = np.asarray(x, dtype=float).reshape(-1)
    return float(0.5 * xx @ np.asarray(A, dtype=float) @ xx +
                 np.asarray(b, dtype=float).reshape(-1) @ xx)


def _cone_violation(x: np.ndarray, contacts: int, mu: float) -> float:
    xx = np.asarray(x, dtype=float).reshape(-1)
    violations = []
    for contact in range(int(contacts)):
        t = float(xx[3 * contact])
        z = xx[3 * contact + 1:3 * contact + 3]
        violations.append(max(float(np.linalg.norm(z) - float(mu) * t), 0.0))
    return float(max(violations, default=0.0))


def _step_stream(events: Sequence[tuple[int, str, int]]) -> dict[int, tuple[tuple[str, int], ...]]:
    grouped: dict[int, list[tuple[str, int]]] = {}
    for outer, kind, index in events:
        grouped.setdefault(int(outer), []).append((str(kind), int(index)))
    return {step: tuple(values) for step, values in grouped.items()}


def event_agreement(left: Sequence[tuple[int, str, int]],
                    right: Sequence[tuple[int, str, int]]) -> tuple[float, int | None]:
    """Compare complete outer-step event streams and return first divergence."""
    a = _step_stream(left)
    b = _step_stream(right)
    if not a and not b:
        return 1.0, None
    matches = 0
    total = 0
    first: int | None = None
    for step in sorted(set(a) | set(b)):
        av = a.get(step, ())
        bv = b.get(step, ())
        if first is None and av != bv:
            first = int(step)
        matches += sum(x == y for x, y in zip(av, bv))
        total += max(len(av), len(bv), 1)
    return float(matches / max(total, 1)), first


def double_reference(A: np.ndarray, b: np.ndarray, contacts: int,
                     mu: float, iterations: int, projection_cap: int) -> dict[str, Any]:
    """Run the independent binary64 solver and retain its full event stream."""
    from snn_opt import (
        ConvergenceConfig,
        OptimizationProblem,
        SNNSolver,
        SolverConfig,
        scaled_soc_projector,
    )

    class TapSolver(SNNSolver):
        def __init__(self, *args, **kwargs):
            self.events: list[tuple[int, str, int]] = []
            super().__init__(*args, **kwargs)

        def _observe_nonlinear_event(self, kind, index, outer_iteration,
                                     ordinal, correction_norm):
            self.events.append((int(outer_iteration), str(kind), int(index)))
            return super()._observe_nonlinear_event(
                kind, index, outer_iteration, ordinal, correction_norm)

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = int(b.size)
    candidates = tuple(
        scaled_soc_projector(3 * i, (3 * i + 1, 3 * i + 2), float(mu),
                             name=f"contact-{i}")
        for i in range(int(contacts)))
    problem = OptimizationProblem(A, b, np.zeros((0, n)), np.zeros(0),
                                  nonlinear_candidates=candidates)
    config = SolverConfig(
        k0=km.k0_for(A), max_iterations=int(iterations),
        constraint_tol=float(km.DEFAULT_TOL),
        max_projection_iters=int(projection_cap),
        continue_after_projection_budget=True,
        # The nonlinear recorded path requires the trajectory.  It is also a
        # useful guard that the comparison really ran the complete horizon.
        record_trajectory=True, record_spike_history=False,
        observe_projection_events=True,
        convergence=ConvergenceConfig(enable_early_stopping=False,
                                      feasibility_tol=float(km.DEFAULT_TOL)),
    )
    solver = TapSolver(problem, config)
    result = solver.solve(np.zeros(n, dtype=float))
    final = np.asarray(result.final_x, dtype=float).reshape(-1)
    return {
        "final": final,
        "objective": _objective(A, b, final),
        "events": list(solver.events),
        "iterations": int(result.iterations_used),
        "projection_budget_exhausted": bool(result.projection_budget_exhausted),
    }


def clarabel_reference(A: np.ndarray, b: np.ndarray, contacts: int,
                       mu: float) -> dict[str, Any]:
    """Solve the friction-cone QP with Clarabel, if it is installed."""
    try:
        import clarabel
        import cvxpy as cp
    except Exception as exc:
        return {
            "status": "not_available", "solver": "Clarabel",
            "version": "unavailable", "objective": float("nan"),
            "reason": f"{type(exc).__name__}: {exc}",
        }
    n = int(np.asarray(b).size)
    x = cp.Variable(n)
    constraints = [cp.norm(x[3 * i + 1:3 * i + 3], 2) <= float(mu) * x[3 * i]
                   for i in range(int(contacts))]
    problem = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, A) + b @ x),
                         constraints)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        problem.solve(solver=cp.CLARABEL, verbose=False)
    value = float(problem.value) if x.value is not None else float("nan")
    final = np.asarray(x.value, dtype=float).reshape(-1) if x.value is not None else None
    return {
        "status": str(problem.status), "solver": "Clarabel",
        "version": str(getattr(clarabel, "__version__", "unknown")),
        "objective": value,
        "max_cone_violation": (_cone_violation(final, contacts, mu)
                               if final is not None else float("nan")),
    }
