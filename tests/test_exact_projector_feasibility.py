"""An inexact event oracle must not let the convergence gate certify an
infeasible point when the candidate carries an exact projector."""

import numpy as np

from snn_opt import (
    ConvergenceConfig,
    CutterCandidate,
    OptimizationProblem,
    SNNSolver,
    SolverConfig,
    spectral_ball_projector,
)


def _power_cutter(k=3):
    """k warm power steps; the warm vector is carried between calls, so the
    oracle underestimates sigma_1 and its score can read feasible while the
    true spectral norm exceeds 1."""
    exact = spectral_ball_projector(3)
    state = {"warm": np.ones(3) / np.sqrt(3.0)}

    def pair(r):
        A = np.asarray(r, dtype=float).reshape(3, 3)
        v = state["warm"].copy()
        for _ in range(k):
            w = A.T @ (A @ v)
            nw = np.linalg.norm(w)
            if nw == 0.0:
                break
            v = w / nw
        state["warm"] = v.copy()
        Av = A @ v
        s = np.linalg.norm(Av)
        u = Av / s if s > 0.0 else np.array([1.0, 0.0, 0.0])
        return s, u, v

    return CutterCandidate(
        value=lambda r: float(pair(r)[0] - 1.0),
        jacobian=lambda r: np.outer(*pair(r)[1:]).ravel(),
        name="power3", kkt_data=dict(exact.kkt_data))


def test_inexact_oracle_does_not_certify_infeasible_point():
    Hn = np.array([[-0.37134864237835624, -0.26300972053902316, -0.4331579601459586],
                   [0.3656096163776305, 0.19066425210355067, 0.39145252597903324],
                   [-0.35372739043154744, -0.21992253003098108, -0.32947373127127794]])
    eps = 0.01034664183707094
    prob = OptimizationProblem(np.eye(9), -Hn.ravel() / eps, np.zeros((0, 9)),
                               np.zeros(0), nonlinear_candidates=(_power_cutter(),))
    cfg = SolverConfig(k0=0.5, max_iterations=1000, constraint_tol=1e-10,
                       convergence=ConvergenceConfig(enable_early_stopping=True))
    res = SNNSolver(prob, cfg).solve(np.zeros(9))
    s1 = np.linalg.svd(np.asarray(res.X[-1]).reshape(3, 3), compute_uv=False)[0]
    assert s1 - 1.0 > 1e-3          # the inexact oracle stops at an infeasible point
    assert not res.converged
