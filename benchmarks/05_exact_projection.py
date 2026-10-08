"""Exact joint projection removes the step-size floor (prints a table, no figure).

The Figure 1 problem (random 50-D QP, 30 inequalities, 7 active at the
optimum) is solved three ways and every run is scored against the exact
optimum from `qpref.solve_exact`:

  1. the default greedy row sweep, 40k iterations: the objective gap sits on
     the O(k0) floor of the discretised dynamics and does not move;
  2. the same rows passed as ONE `joint_projector(C, d)` candidate, which
     projects exactly onto the polytope with Dykstra's algorithm, default
     certificate tolerance;
  3. the same with a tight certificate (`kkt_rel_tol=1e-9`).

With an exact projection the projected-gradient fixed point is the exact
optimum, so the error keeps falling as the tolerance tightens instead of
stopping at a floor. The price is an inner Dykstra loop per Euler step.
The compiled backend is used when available (it supports one-level Dykstra
over halfspaces); wall times are indicative only.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import qpref  # noqa: E402

from snn_opt import (  # noqa: E402
    ConvergenceConfig,
    OptimizationProblem,
    SNNSolver,
    SolverConfig,
    joint_projector,
)


def figure1_problem():
    spec = importlib.util.spec_from_file_location("fig1", HERE / "01_convergence.py")
    fig1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fig1)
    return fig1.random_qp(50, 30, seed=7)


def backend():
    try:
        from snn_opt import _kernel  # noqa: F401
        return "c"
    except ImportError:
        return "python"


def main() -> int:
    A, b, C, d, x0 = figure1_problem()
    x_star, f_star, active = qpref.solve_exact(A, b, C, d)
    n = A.shape[0]
    be = backend()
    joint = OptimizationProblem(A, b, np.zeros((0, n)), np.zeros(0),
                                nonlinear_candidates=(joint_projector(C, d),))
    runs = [
        ("greedy row sweep", OptimizationProblem(A, b, C, d),
         SolverConfig(max_iterations=40_000, backend=be)),
        ("joint_projector", joint,
         SolverConfig(max_iterations=40_000, backend=be)),
        ("joint_projector, kkt_rel_tol=1e-9", joint,
         SolverConfig(max_iterations=60_000, backend=be,
                      convergence=ConvergenceConfig(kkt_rel_tol=1e-9))),
    ]
    print(f"50-D QP, 30 rows, {len(active)} active at the optimum (backend={be!r})\n")
    print(f"{'run':36s} {'certified':>9s} {'iters':>6s} {'|x - x*|':>9s} {'|f - f*|':>9s} {'time':>7s}")
    for label, problem, cfg in runs:
        t0 = time.perf_counter()
        res = SNNSolver(problem, cfg).solve(x0)
        dt = time.perf_counter() - t0
        print(f"{label:36s} {str(res.converged):>9s} {res.iterations_used:6d} "
              f"{np.linalg.norm(res.final_x - x_star):9.1e} {abs(res.final_objective - f_star):9.1e} {dt:6.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
