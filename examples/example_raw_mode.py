"""Example: raw mode vs optimized mode.

Contrasts the solver's two projection strategies on the same problem:

* **raw** uses a fixed gradient step `k0` and a fixed projection step `k1`, so a
  violated constraint is walked back to its face over several small corrections.
  This is the behaviour that looks most like integrate-and-fire: overshoot, then
  a short burst of spikes bringing the state back.
* **optimized** (the default) computes `k0` from the Hessian's Lipschitz
  constant and projects with the exact step to the boundary, so one spike per
  violation is normally enough.

The problem is chosen so that a constraint is *active at the optimum*. That
matters. If the unconstrained minimiser is already feasible the solver simply
descends to it, no constraint ever binds, and the comparison degenerates into a
plot of an unconstrained quadratic decaying to zero. An earlier version of this
example had exactly that defect, which is why its convergence panel ran down to
1e-113: it was plotting the objective *value* as it underflowed toward f* = 0,
not a meaningful convergence rate.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# The figure style is shared with the benchmark suite so that every figure in
# the repository and on the website reads as one visual system.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import numpy as np
import matplotlib.pyplot as plt

import figstyle
from snn_opt import ConvergenceConfig, OptimizationProblem, SNNSolver, SolverConfig

_ROOT = Path(__file__).resolve().parent.parent


def exact_optimum(A, b, C, d):
    """Exact QP optimum by enumerating active sets.

    Only sensible because this problem has two rows; the benchmark suite uses
    the iterative active-set solver in ``benchmarks/qpref.py`` instead.
    """
    n, m = len(b), len(d)
    best = None
    for size in range(m + 1):
        for act in itertools.combinations(range(m), size):
            act = list(act)
            if act:
                ca = C[act]
                kkt = np.block([[A, ca.T], [ca, np.zeros((len(act), len(act)))]])
                try:
                    sol = np.linalg.solve(kkt, np.concatenate([-b, -d[act]]))
                except np.linalg.LinAlgError:
                    continue
                x, lam = sol[:n], sol[n:]
                if lam.min() < -1e-9:          # dual feasibility
                    continue
            else:
                x = np.linalg.solve(A, -b)
            if (C @ x + d).max() > 1e-9:       # primal feasibility
                continue
            f = float(0.5 * x @ A @ x + b @ x)
            if best is None or f < best[1]:
                best = (x, f)
    if best is None:
        raise RuntimeError("no feasible active set found")
    return best


def raw_config(k0: float = 0.1, k1: float = 0.05, max_iterations: int = 300) -> SolverConfig:
    """Fixed gradient step, fixed projection step, no early stopping."""
    return SolverConfig(
        k0=k0,
        k1=k1,
        projection_method="fixed",
        integration_method="euler",
        max_iterations=max_iterations,
        convergence=ConvergenceConfig(enable_early_stopping=False),
    )


def optimized_config(max_iterations: int = 300) -> SolverConfig:
    """Auto step size from the Lipschitz constant, exact projection, early stopping."""
    return SolverConfig(
        k0=None,
        projection_method="adaptive",
        integration_method="euler",
        max_iterations=max_iterations,
        k0_scale=0.5,
        convergence=ConvergenceConfig(
            enable_early_stopping=True, check_every=50, min_iterations=100, patience=3
        ),
    )


def main() -> int:
    # minimize (1/2)||x - c||^2 with c outside the feasible wedge, so the
    # optimum lies ON a constraint face and the projection actually does work.
    n = 2
    c = np.array([1.5, 1.0])
    A = np.eye(n)
    b = -c
    C = np.array([[1.0, 2.0], [-1.0, 3.0]])
    d = np.array([-1.0, -1.0])
    problem = OptimizationProblem(A=A, b=b, C=C, d=d)
    x0 = np.array([1.0, 1.0])            # infeasible start: 1 + 2*1 = 3 > 1

    x_star, f_star = exact_optimum(A, b, C, d)

    print("=" * 66)
    print("SNN solver: raw mode vs optimized mode")
    print("=" * 66)
    print("minimize (1/2)||x - c||^2,  c = [1.5, 1.0]")
    print("subject to   x1 + 2 x2 <= 1")
    print("            -x1 + 3 x2 <= 1")
    print(f"start x0 = {x0} (infeasible)")
    print(f"exact optimum x* = [{x_star[0]:.6f}, {x_star[1]:.6f}], f* = {f_star:.9f}")
    print(f"active rows at x*: {np.where(C @ x_star + d > -1e-9)[0].tolist()}\n")

    raw = SNNSolver(problem, raw_config()).solve(x0.copy(), verbose=False)
    opt = SNNSolver(problem, optimized_config()).solve(x0.copy(), verbose=False)

    print(f"{'metric':<28}{'raw':>17}{'optimized':>17}")
    print("-" * 62)
    rows = (
        ("iterations used", raw.iterations_used, opt.iterations_used, "d"),
        ("projections (spikes)", raw.n_projections, opt.n_projections, "d"),
        ("total projection distance", raw.total_projection_distance,
         opt.total_projection_distance, ".6f"),
        ("objective gap vs f*", abs(raw.final_objective - f_star),
         abs(opt.final_objective - f_star), ".3e"),
        ("max row violation", raw.max_distance_rows, opt.max_distance_rows, ".3e"),
    )
    for label, r_val, o_val, fmt in rows:
        print(f"{label:<28}{r_val:>17{fmt}}{o_val:>17{fmt}}")
    print(f"{'converged':<28}{str(raw.converged):>17}{str(opt.converged):>17}\n")

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))

    # Window covers the start point, both trajectories and the optimum, so
    # nothing referenced in the legend can fall outside the axes.
    pts = np.vstack([raw.X, opt.X, x0[None, :], x_star[None, :]])
    pad = 0.2
    xlo, ylo = pts.min(axis=0) - pad
    xhi, yhi = pts.max(axis=0) + pad

    gx_line = np.linspace(xlo, xhi, 300)
    gx, gy = np.meshgrid(gx_line, np.linspace(ylo, yhi, 300))
    feasible = ((gx + 2 * gy - 1 <= 0) & (-gx + 3 * gy - 1 <= 0)).astype(float)

    for ax, res, title in (
        (axes[0], raw, "(a) raw: fixed $k_0$, fixed $k_1$"),
        (axes[1], opt, "(b) optimized: auto $k_0$, exact projection"),
    ):
        ax.contourf(gx, gy, feasible, levels=[0.5, 1.5], colors=[figstyle.GREEN], alpha=0.10)
        ax.plot(gx_line, (1 - gx_line) / 2, "-", color=figstyle.MUTED, linewidth=0.9, zorder=1)
        ax.plot(gx_line, (1 + gx_line) / 3, "--", color=figstyle.MUTED, linewidth=0.9, zorder=1)

        ax.plot(res.X[:, 0], res.X[:, 1], color=figstyle.INK, linewidth=0.9,
                alpha=0.85, zorder=3, label="trajectory")
        idx = res.spike_times.astype(int)
        idx = idx[idx < len(res.X)]
        if idx.size:
            ax.scatter(res.X[idx, 0], res.X[idx, 1], s=28, color=figstyle.VERMILION,
                       zorder=5, label=f"spikes ({res.n_projections})")
        ax.scatter(*x0, s=44, color=figstyle.BLUE, marker="o", zorder=6, label="start")
        ax.scatter(*x_star, s=95, color=figstyle.PURPLE, marker="*", zorder=6,
                   label=r"exact $x^\star$")

        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        figstyle.panel_title(ax, title)
        ax.legend(loc="lower left", fontsize=7.5)

    ax = axes[2]
    for res, color, label in ((raw, figstyle.BLUE, "raw"), (opt, figstyle.VERMILION, "optimized")):
        gap = np.maximum(np.abs(res.objective_values - f_star), 1e-16)
        ax.semilogy(gap, color=color, linewidth=1.3, label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$|f(x_t) - f^\star|$")
    figstyle.panel_title(ax, "(c) objective gap vs exact optimum")
    ax.legend(loc="upper right")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "raw_vs_optimized.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"wrote {out.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
