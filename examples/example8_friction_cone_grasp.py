#!/usr/bin/env python3
"""
Example 8: grasp forces inside friction cones (conic constraints)
==================================================================

Three fingertips hold a ball of radius 1. Each fingertip may push along the
inward surface normal and rub along the surface, but Coulomb friction limits
the tangential part:

    ||f_t,i|| <= mu * f_n,i          (a second-order cone per contact)

The contact forces must balance an external wrench w (gravity plus a small
push and twist), and among all balancing force sets we want the gentlest
grip:

    minimise    1/2 * sum_i ||f_i||^2
    subject to  G f + w = 0          (force and torque balance, 6 equations)
                f_i in friction cone i,  i = 1, 2, 3

This is a second-order-cone program, not a QP, so it uses the opt-in
`snn_opt.nonlinear` API. The equilibrium subspace and the three cones are
wrapped in ONE Dykstra projector, which projects exactly onto their
intersection. With an exact projector, the projected-gradient fixed point is
the exact optimum, so the solver lands within about 4e-13 of a
Newton-polished reference. The Dykstra wrapper also gives the state-unit
certificate, a direct bound on the distance to the optimum.

The second half of the script shows why the Dykstra wrapper matters: handing
the same sets to the solver as separate candidates makes the projection
sweep alternate between them, and the run stops at its iteration cap with a
much larger optimality defect.

Run:  python examples/example8_friction_cone_grasp.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from snn_opt import (  # noqa: E402
    AffineSubspaceProjector,
    OptimizationProblem,
    SNNSolver,
    SolverConfig,
    dykstra_projector,
    scaled_soc_projector,
)

MU = 0.6  # friction coefficient


def build_grasp():
    """Contact points, contact frames, grasp matrix and external wrench."""
    azimuth = np.deg2rad([0.0, 120.0, 240.0])
    elevation = np.deg2rad(-15.0)
    points = np.stack([np.cos(azimuth) * np.cos(elevation),
                       np.sin(azimuth) * np.cos(elevation),
                       np.full(3, np.sin(elevation))], axis=1)
    frames = []
    for p in points:
        n = -p                               # inward normal
        t1 = np.cross([0.0, 0.0, 1.0], n)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        frames.append(np.stack([t1, t2, n], axis=1))   # columns t1, t2, n

    # Decision vector: local forces (f_t1, f_t2, f_n) per contact, stacked.
    G = np.zeros((6, 9))
    for i, (p, R) in enumerate(zip(points, frames)):
        G[:3, 3 * i:3 * i + 3] = R                          # force
        G[3:, 3 * i:3 * i + 3] = np.cross(p, R.T).T         # torque p x (R f)
    w = np.array([0.3, 0.0, -1.0, 0.0, 0.05, 0.0])          # external wrench
    return points, frames, G, w


def friction_cones():
    # ||(f_t1, f_t2)|| <= MU * f_n on coordinates (3i, 3i+1 | 3i+2).
    return [scaled_soc_projector(3 * i + 2, [3 * i, 3 * i + 1], MU, name=f"cone {i + 1}")
            for i in range(3)]


def main():
    points, frames, G, w = build_grasp()
    n = 9
    A, b = np.eye(n), np.zeros(n)
    no_rows = (np.zeros((0, n)), np.zeros(0))

    equilibrium = AffineSubspaceProjector(G, -w, name="equilibrium")   # G f = -w
    grasp_set = dykstra_projector([equilibrium, *friction_cones()], name="grasp set")
    problem = OptimizationProblem(A, b, *no_rows, nonlinear_candidates=(grasp_set,))

    result = SNNSolver(problem, SolverConfig()).solve(np.zeros(n))
    f = result.final_x

    print("=" * 66)
    print("Grasp forces in friction cones (one Dykstra candidate)")
    print("=" * 66)
    print(f"converged          : {result.converged} ({result.iterations_used} iterations)")
    print(f"objective          : {result.final_objective:.12f}")
    print(f"balance residual   : {np.linalg.norm(G @ f + w):.2e}  (||G f + w||)")
    print(f"max cone violation : {result.max_violation_nonlinear:.2e}")
    bound = result.kkt_stationarity_residual + result.kkt_complementarity_residual
    print(f"state certificate  : ||x - x*|| <= {bound:.1e}  (trusts the Dykstra tolerance, 1e-12)")
    print()
    print("contact   f_t1      f_t2      f_n     ||f_t|| / (mu f_n)")
    for i in range(3):
        ft, fn = f[3 * i:3 * i + 2], f[3 * i + 2]
        use = np.linalg.norm(ft) / (MU * fn)
        note = "  <- on the cone: about to slip" if use > 1 - 1e-9 else ""
        print(f"   {i + 1}   {ft[0]:8.4f}  {ft[1]:8.4f}  {fn:8.4f}     {use:6.4f}{note}")

    # Optional independent check against a conic interior-point solver.
    try:
        import warnings

        import cvxpy as cp
        x = cp.Variable(n)
        cons = [G @ x + w == 0] + [cp.norm(x[3 * i:3 * i + 2]) <= MU * x[3 * i + 2] for i in range(3)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x)), cons).solve(
                solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
        # The gap below is the interior-point solver's own accuracy limit;
        # Newton's method on the active-cone KKT system agrees with snn_opt
        # to ~4e-13.
        print(f"\nClarabel (tight tolerances) agrees to {np.linalg.norm(x.value - f):.0e}; "
              f"objectives {result.final_objective:.12f} vs {0.5 * x.value @ x.value:.12f}")
    except Exception:  # cvxpy is an optional dev dependency
        print("\n(install cvxpy to cross-check against Clarabel)")

    # Same sets, handed to the solver separately: the winner-take-all sweep
    # alternates between the equilibrium subspace and the cones instead of
    # projecting onto their intersection.
    separate = OptimizationProblem(A, b, *no_rows,
                                   nonlinear_candidates=(equilibrium, *friction_cones()))
    alt = SNNSolver(separate, SolverConfig(max_iterations=1500)).solve(np.zeros(n))
    print("\nWithout the Dykstra wrapper (separate candidates):")
    print(f"converged          : {alt.converged} ({alt.iterations_used} iterations, "
          f"{alt.convergence_reason.split('(')[0]})")
    print(f"relative KKT defect: {alt.kkt_residual / alt.kkt_scale:.1e}")
    print(f"distance to the Dykstra solution: {np.linalg.norm(alt.final_x - f):.1e}")

    plot(points, frames, f, w)


def plot(points, frames, f, w):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig = plt.figure(figsize=(6.6, 5.4))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    u, v = np.mgrid[0:2 * np.pi:48j, 0:np.pi:24j]
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                    color="#e4e0d6", alpha=0.25, linewidth=0, zorder=0)
    colors = ["#2f6f9f", "#3a9a6a", "#8a5cb8"]
    h, s = 0.75, 1.6           # drawn cone height, force arrow scale
    for i, (p, R, c) in enumerate(zip(points, frames, colors)):
        # The cone of admissible contact forces, drawn on the finger's side:
        # a force is admissible when its arrow, ending at the contact, lies
        # inside this cone.
        th = np.linspace(0, 2 * np.pi, 60)
        rim = np.stack([MU * h * np.cos(th), MU * h * np.sin(th), np.full_like(th, h)], 1)
        rim_w = p - rim @ R.T
        for k in range(0, 60, 5):
            ax.plot(*np.stack([p, rim_w[k]]).T, color=c, alpha=0.3, lw=0.8, zorder=2)
        ax.plot(*rim_w.T, color=c, alpha=0.8, lw=1.2, zorder=2)
        F = R @ f[3 * i:3 * i + 3]
        tail = p - s * F
        ax.quiver(*tail, *(s * F), color=c, lw=2.8, arrow_length_ratio=0.22, zorder=5)
        use = np.linalg.norm(f[3 * i:3 * i + 2]) / (MU * f[3 * i + 2])
        tag = "on the cone (slip limit)" if use > 1 - 1e-9 else f"{100 * use:.0f}% of the friction limit"
        lab = p - 1.75 * h * (R[:, 2])
        ax.text(*lab, f"contact {i + 1}\n{tag}", color=c, fontsize=8, ha="center", zorder=6)
    ax.quiver(0, 0, 0, *(s * w[:3]), color="#d1495b", lw=2.4, arrow_length_ratio=0.15, zorder=5)
    ax.text(*(s * w[:3] * 1.12), "external load", color="#d1495b", fontsize=8, ha="center", zorder=6)
    ax.set_box_aspect((1, 1, 0.9))
    for set_lim in (ax.set_xlim, ax.set_ylim):
        set_lim(-1.7, 1.7)
    ax.set_zlim(-1.9, 1.2)
    ax.set_axis_off()
    ax.view_init(elev=55, azim=-90)
    ax.set_title("Minimum-effort grasp: forces inside their friction cones (mu = 0.6)", fontsize=10)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example8_friction_cone_grasp.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()
