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

A second table adds the box ``-0.3 <= x <= 0.3`` (two box facets join the
five active rows) and passes it as one more Dykstra member,
``joint_projector(C, d, members=(box_projector(-0.3, 0.3),))``. The box is a
built-in set, so the whole projection runs natively on the compiled backend.
Accuracy is scored against the reference optimum of the rows-plus-box QP.
Speed is the median of five 200-iteration runs per backend; the Dykstra
cycle counts are identical on every backend, so only the time differs. The
last speed row is a negative control: the same box written as 2n halfspace
rows, which raises the members per Dykstra cycle from m + 1 to m + 2n.
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
    box_projector,
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
    if be != "python":
        print()
        box_tables(A, b, C, d, x0)
    return 0


BOX = 0.3
SPEED_ITERATIONS = 200
SPEED_REPEATS = 5


def box_tables(A, b, C, d, x0) -> None:
    """Rows plus a box: accuracy to the certificate, then speed per backend."""
    n = A.shape[0]
    C_box = np.vstack([C, np.eye(n), -np.eye(n)])
    d_box = np.concatenate([d, -BOX * np.ones(n), -BOX * np.ones(n)])
    x_star, f_star, active = qpref.solve_exact(A, b, C_box, d_box)
    n_box = sum(1 for i in active if i >= C.shape[0])

    def dykstra_problem(candidate):
        return OptimizationProblem(A, b, np.zeros((0, n)), np.zeros(0),
                                   nonlinear_candidates=(candidate,))

    def violation(x):
        return max(0.0, float(np.max(C_box @ x + d_box)))

    rows_box = dykstra_problem(joint_projector(C, d, members=(box_projector(-BOX, BOX),)))
    print(f"same QP plus box [-{BOX}, {BOX}]: {len(active) - n_box} rows + "
          f"{n_box} box facets active at the optimum\n")
    print(f"{'run':36s} {'certified':>9s} {'iters':>6s} {'|x - x*|':>9s} "
          f"{'|f - f*|':>9s} {'max viol':>9s} {'cycles/it':>9s}")
    runs = [
        ("greedy sweep, scalar bounds (c)", OptimizationProblem(A, b, C, d),
         SolverConfig(max_iterations=40_000, backend="c", lower_bound=-BOX, upper_bound=BOX)),
        ("Dykstra rows + box_projector (c)", rows_box,
         SolverConfig(max_iterations=40_000, backend="c")),
    ]
    for label, problem, cfg in runs:
        res = SNNSolver(problem, cfg).solve(x0)
        cycles = np.asarray(res.dykstra_inner_iterations_per_step)
        cyc = f"{cycles.mean():9.1f}" if cycles.size else f"{'-':>9s}"
        print(f"{label:36s} {str(res.converged):>9s} {res.iterations_used:6d} "
              f"{np.linalg.norm(res.final_x - x_star):9.1e} "
              f"{abs(res.final_objective - f_star):9.1e} {violation(res.final_x):9.1e} {cyc}")

    halfspaces = dykstra_problem(joint_projector(C_box, d_box))
    speed = [("Dykstra rows + box_projector", rows_box, backend)
             for backend in ("python", "c_serial", "c_openmp")]
    speed.append(("negative control: box as 2n rows", halfspaces, "c_serial"))
    print(f"\nspeed, {SPEED_ITERATIONS} iterations from x0, median of {SPEED_REPEATS} "
          f"runs (indicative)\n")
    print(f"{'run':36s} {'backend':>8s} {'ms/it':>8s} {'cycles/it':>9s} "
          f"{'events/it':>9s} {'|x - x_py|':>10s}")
    x_python = None
    for label, problem, backend in speed:
        cfg = SolverConfig(max_iterations=SPEED_ITERATIONS, backend=backend,
                           record_trajectory=backend == "python",
                           record_spike_history=False,
                           convergence=ConvergenceConfig(enable_early_stopping=False))
        try:
            solver = SNNSolver(problem, cfg)
        except (RuntimeError, ValueError) as exc:
            print(f"{label:36s} {backend:>8s} unavailable: {exc}")
            continue
        times = []
        for _ in range(SPEED_REPEATS):
            t0 = time.perf_counter()
            res = solver.solve(x0)
            times.append(time.perf_counter() - t0)
        ms = 1e3 * float(np.median(times)) / SPEED_ITERATIONS
        cycles = float(np.mean(res.dykstra_inner_iterations_per_step))
        events = float(np.mean(res.dykstra_inner_projection_events_per_step))
        if x_python is None:
            x_python = res.final_x
        dx = float(np.max(np.abs(res.final_x - x_python)))
        print(f"{label:36s} {backend:>8s} {ms:8.3f} {cycles:9.1f} {events:9.0f} {dx:10.1e}")


if __name__ == "__main__":
    raise SystemExit(main())
