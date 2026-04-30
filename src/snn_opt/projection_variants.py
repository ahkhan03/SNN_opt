"""
Neuromorphic-pure projection variants for the SNN-QP solver.
============================================================

The canonical ``solve_qp`` in ``snn_opt.solver`` alternates a forward Euler
gradient step with an event-triggered "adaptive projection" inner loop. The
inner loop selects the most-violated constraint, computes the closed-form
exact step ``k1 = g_j / ||c_j||^2`` to its boundary, applies it, and repeats
until all constraints are satisfied. That inner loop is *event-triggered* and
*sequentially Gauss-Seidel*; it is mappable to spike dynamics in principle, but
the inner-loop control flow ("re-check argmax of g, decide whether to fire
again") is more naturally expressed as a digital co-processor than as a pool
of LIF neurons whose dynamics integrate continuously in time.

This module implements two alternative projection strategies whose entire
state evolves under continuous-time ODEs that are integrated by forward Euler.
There is no inner projection loop, no per-iteration argmax, and no sub-iterate
"snap-to-boundary" step: every neuron simply integrates its membrane voltage
forward at every clock tick.

Variant 1 -- Penalty method
---------------------------
Augment the cost with a quadratic penalty on each constraint violation::

    F(x) = 1/2 x^T A x + b^T x + (k_p / 2) sum_i max(0, c_i^T x + d_i)^2

Forward-Euler dynamics::

    dx/dt = -(A x + b) - k_p * C^T * relu(C x + d)
        x_{t+1} = x_t + h * dx/dt

This is *strictly* a single neural population integrating its own gradient.
The only nonlinearity is element-wise ReLU on the constraint residual, which
is the canonical LIF threshold. No matrix inverse, no Cholesky, no division
beyond the constant timestep ``h``.

Trade-off: solutions are biased; the bias scales as ``O(1/k_p)`` (hence we
typically run with ``k_p`` ramping up over time, mimicking a quadratic-penalty
homotopy). Stiffness limits the integration step.

Variant 2 -- Augmented Lagrangian / dual-ascent
-----------------------------------------------
Two coupled neural populations: primal ``x in R^n`` and dual ``lambda in
R_{>=0}^m``. Saddle-point dynamics::

    dx/dt     = -(A x + b) - C^T lambda
    dlambda/dt = relu(C x + d)        # dual ascends on violation, projected to >= 0

Forward-Euler with separate primal step ``h_x`` and dual step ``h_l``. The
dynamics are bilinear (``C^T lambda`` couples primal and dual through the
constraint matrix) but every elementary operation is a vector-matrix product
or element-wise nonlinearity. No matrix inverse anywhere.

This is the "purest neural" form: at convergence, ``lambda`` is the KKT
multiplier vector and the dynamics reproduce the saddle-point of the
Lagrangian, which is exactly what spiking-network literature calls the
"two-population E-I network."

Hardware note
-------------
On FPGA, both variants compile to vectorised MAC + ReLU pipelines, no linear
solver IP needed. On Loihi-2, both variants map to standard LIF compartments;
the augmented-Lagrangian variant requires only excitatory/inhibitory pairs,
no plastic synapses or special blocks.

Public API
----------
- ``solve_qp_penalty(...)``               -- Variant 1: forward-Euler + quadratic penalty
- ``solve_qp_lagrangian(...)``            -- Variant 2: primal-dual Euler / augmented Lagrangian
- ``solve_qp_heun_penalty(...)``          -- Variant 3: Heun (explicit RK2) + quadratic penalty
- ``solve_qp_heavyball_penalty(...)``     -- Variant 4: Polyak heavy-ball momentum + quadratic penalty
- ``solve_qp_nesterov_penalty(...)``      -- Variant 5: Nesterov-accelerated flow + quadratic penalty
- ``solve_qp_expeuler_penalty(...)``      -- Variant 6: exponential Euler on the linear part + penalty

All take ``A, b, C, d, x0`` (matching ``solve_qp``) and return a
``SolverResult`` populated with the same fields. Per-variant ``spike_*``
fields are reinterpreted as the iteration-count of "spike events" (positive
constraint residual triggers a contribution to ``dx``). See each docstring.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

from .solver import OptimizationProblem, SolverResult


# ----------------------------------------------------------------------------
# Variant 1: Quadratic penalty method
# ----------------------------------------------------------------------------

def solve_qp_penalty(
    A: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    x0: np.ndarray,
    *,
    k0: Optional[float] = None,
    k0_scale: float = 0.5,
    k_p: Optional[float] = None,
    k_p_ramp: bool = True,
    k_p_final: Optional[float] = None,
    max_iterations: int = 5000,
    h: Optional[float] = None,
    feasibility_tol: float = 1e-2,
    proj_grad_tol: float = 1e-6,
    enable_early_stopping: bool = True,
    record_full_trajectory: bool = True,
    verbose: bool = False,
) -> SolverResult:
    """
    Variant 1 — quadratic-penalty Euler integration.

    Dynamics
    --------
    dx/dt = -(A x + b) - k_p * C^T relu(Cx + d)

    Forward Euler with step ``h``::

        v        = relu(C x + d)        # constraint residual (positive parts only)
        x_next   = x - h * (A x + b) - h * k_p * C^T v

    All operations are vector adds, element-wise ReLU, and one matrix-vector
    multiply by C and C^T. Mapping to LIF: each entry of ``v`` is a population
    rate-coded threshold neuron; the recurrent term ``-A x - b`` is the
    intrinsic dynamics of the primal LIF pool.

    Parameters
    ----------
    k_p : float
        Penalty strength. Larger -> tighter feasibility, smaller -> larger
        unconstrained-optimum bias but stiffer dynamics.
    k_p_ramp : bool
        If True, geometrically ramp k_p from ``k_p`` (initial) to ``k_p_final``
        (default 100x) over ``max_iterations``. Mimics a homotopy on the penalty
        coefficient.
    k_p_final : float
        Final penalty value when ramping; defaults to 100 * k_p.
    h : float, optional
        Euler step size. If None, set to ``k0_scale / (L + k_p_final * ||C||^2_2)``
        where L is the Lipschitz constant of the QP gradient — this keeps the
        stiff penalty term inside the stability region.
    record_full_trajectory : bool
        If True, store every iterate (matches canonical solve_qp behaviour).

    Returns
    -------
    SolverResult
        ``spike_times``/``spike_deltas`` record iterations where any constraint
        residual was positive (the penalty contribution was nonzero) and the
        contribution it made to ``dx``.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    C = np.asarray(C, dtype=float)
    d = np.asarray(d, dtype=float)
    x0 = np.asarray(x0, dtype=float).copy()

    n = A.shape[0]
    m = C.shape[0] if C.ndim == 2 else 0

    # Lipschitz of unconstrained gradient
    L_A = _spectral_norm_symmetric(A)
    if L_A < 1e-12:
        L_A = 1.0
    if k0 is None:
        k0_grad = k0_scale / L_A  # informational only; used for ref dynamics
    else:
        k0_grad = k0

    L_C = _spectral_norm(C) if m > 0 else 0.0

    # Auto-tune k_p so that the penalty force at a unit violation matches the
    # scale of the unconstrained gradient (||A|| + ||b||). This is the
    # smallest k_p that can hold the iterate inside the feasible region; the
    # ramp boosts it to k_p_fin for tight feasibility.
    grad_scale = max(L_A, np.linalg.norm(b))
    if k_p is None:
        k_p_init = 10.0 * grad_scale / max(L_C ** 2, 1e-12) if L_C > 0 else 1.0
    else:
        k_p_init = float(k_p)
    if k_p_ramp:
        # Ramp by 10^4 over the run -- empirically gives 1e-4 rel_err on MPC
        k_p_fin = float(k_p_final) if k_p_final is not None else 1e4 * k_p_init
    else:
        k_p_fin = k_p_init

    # Penalty term contributes a Lipschitz-like factor of k_p * ||C||_2^2 to
    # the joint dynamics; the safe Euler step is k0_scale / (L_A + k_p_max * ||C||^2_2)
    # Time-varying step: h adapts to *current* k_p, not the asymptotic value.
    # This lets the iterate cover the full unconstrained gradient-flow path
    # while k_p is still small, then decreases as k_p ramps up to maintain
    # stability when the penalty term becomes stiff.
    h_user = h
    if h is None:
        h_init = k0_scale / max(L_A, 1e-12)
    else:
        h_init = h

    if verbose:
        print(f"[penalty] n={n}, m={m}, L_A={L_A:.3e}, ||C||_2={L_C:.3e}")
        print(f"[penalty] k_p init={k_p_init:.2e} -> fin={k_p_fin:.2e}, h_init={h_init:.3e} (adaptive)")

    x = x0.copy()
    trajectory: List[np.ndarray] = [x.copy()]
    obj_history: List[float] = []
    spike_times: List[float] = []
    spike_deltas: List[np.ndarray] = []
    spike_constraints: List[np.ndarray] = []
    spike_violations: List[np.ndarray] = []

    converged = False
    convergence_reason = "max_iterations"
    iterations_used = max_iterations

    # plateau / stationarity tracking
    obj_window: List[float] = []
    window_size = 50
    plateau_tol = 1e-9
    min_iterations = max(200, int(0.05 * max_iterations))
    patience_counter = 0
    patience_required = 3
    check_every = 25

    for k in range(max_iterations):
        if k_p_ramp and max_iterations > 1:
            # alpha^2 schedule: k_p stays near k_p_init for the first ~half
            # of iterations (giving the unconstrained gradient flow time to
            # traverse), then ramps up to k_p_fin to refine feasibility.
            alpha = k / (max_iterations - 1)
            k_p_curr = k_p_init * (k_p_fin / k_p_init) ** (alpha ** 2)
        else:
            k_p_curr = k_p_init

        # adaptive Euler step: stable for current k_p
        if h_user is None:
            h_curr = k0_scale / max(L_A + k_p_curr * (L_C ** 2), 1e-12)
        else:
            h_curr = h_user

        # gradient of unconstrained QP
        grad_f = A @ x + b

        # constraint residual (positive part)
        if m > 0:
            g = C @ x + d
            v = np.maximum(g, 0.0)
            penalty_term = C.T @ v
            penalty_force = k_p_curr * penalty_term
        else:
            v = np.zeros(0)
            penalty_force = np.zeros(n)

        dx_dt = -(grad_f + penalty_force)
        x_new = x + h_curr * dx_dt
        delta = x_new - x

        # spike bookkeeping: every iteration with any positive residual is a "spike"
        if m > 0:
            active = np.where(v > 0)[0]
            if active.size > 0:
                spike_times.append(float(k))
                spike_deltas.append(-h_curr * penalty_force.copy())
                spike_constraints.append(active.copy())
                spike_violations.append(v[active].copy())

        x = x_new
        if record_full_trajectory:
            trajectory.append(x.copy())

        obj_curr = 0.5 * x @ A @ x + b @ x
        obj_history.append(obj_curr)
        obj_window.append(obj_curr)
        if len(obj_window) > window_size:
            obj_window.pop(0)

        # convergence check
        if enable_early_stopping and k >= min_iterations and k % check_every == 0:
            max_viol = float(np.max(np.maximum(C @ x + d, 0.0))) if m > 0 else 0.0
            if max_viol <= feasibility_tol and len(obj_window) == window_size:
                obj_range = max(obj_window) - min(obj_window)
                obj_scale = max(abs(obj_window[-1]), 1e-10)
                if obj_range / obj_scale < plateau_tol:
                    proj_grad = _projected_gradient_norm(x, A, b, C, d)
                    if proj_grad < proj_grad_tol:
                        patience_counter += 1
                        if patience_counter >= patience_required:
                            converged = True
                            convergence_reason = (
                                f"converged(obj_plateau={obj_range/obj_scale:.2e};"
                                f" proj_grad={proj_grad:.2e}; max_viol={max_viol:.2e})"
                            )
                            iterations_used = k + 1
                            if verbose:
                                print(f"[penalty] iter {k}: {convergence_reason}")
                            break
                    else:
                        patience_counter = 0
                else:
                    patience_counter = 0
            else:
                patience_counter = 0

        if verbose and k % 500 == 0:
            max_viol = float(np.max(np.maximum(C @ x + d, 0.0))) if m > 0 else 0.0
            print(f"[penalty] iter {k:>5d}: obj={obj_curr:.4e}, "
                  f"max_viol={max_viol:.2e}, k_p={k_p_curr:.2e}")

    # final projected-gradient norm
    final_proj_grad = _projected_gradient_norm(x, A, b, C, d)

    return _build_result(
        trajectory=trajectory,
        obj_history=obj_history,
        A=A, b=b, C=C, d=d,
        n_projections=len(spike_times),
        converged=converged,
        convergence_reason=convergence_reason,
        iterations_used=iterations_used,
        final_x=x,
        final_proj_grad_norm=final_proj_grad,
        spike_times=spike_times,
        spike_deltas=spike_deltas,
        spike_constraints=spike_constraints,
        spike_violations=spike_violations,
    )


# ----------------------------------------------------------------------------
# Variant 2: Augmented Lagrangian / dual-ascent
# ----------------------------------------------------------------------------

def solve_qp_lagrangian(
    A: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    x0: np.ndarray,
    *,
    k0: Optional[float] = None,
    k0_scale: float = 0.5,
    h_x: Optional[float] = None,
    h_l: Optional[float] = None,
    rho: Optional[float] = None,
    max_iterations: int = 5000,
    feasibility_tol: float = 1e-2,
    proj_grad_tol: float = 1e-6,
    enable_early_stopping: bool = True,
    record_full_trajectory: bool = True,
    verbose: bool = False,
) -> SolverResult:
    """
    Variant 2 — augmented-Lagrangian primal-dual Euler integration.

    Dynamics
    --------
    Augmented Lagrangian:
        L_rho(x, lambda) = f(x) + lambda^T relu(g) + (rho/2) ||relu(g)||^2

    Primal-dual gradient flow::

        dx/dt       = -(A x + b) - C^T (lambda + rho * relu(g))
        dlambda/dt  = relu(C x + d)         # then projected to >= 0

    Forward Euler with primal step ``h_x`` and dual step ``h_l``::

        v            = relu(C x + d)
        x_next       = x  - h_x * [(A x + b) + C^T (lambda + rho * v)]
        lambda_next  = max(0, lambda + h_l * v)

    The augmented (rho > 0) form damps the saddle-point oscillation that
    plagues vanilla primal-dual gradient (rho = 0): rho acts as a primal-
    side stabiliser that pulls the iterate toward feasibility *before*
    lambda has time to adapt, eliminating the multiplier-overshoot mode.

    LIF mapping
    -----------
    Two coupled neural populations: primal x (excitatory) and dual lambda
    (inhibitory, rectified). The augmentation rho * relu(g) is a recurrent
    excitatory-to-inhibitory shortcut at the constraint nodes. All ops are
    matrix-vector products and element-wise ReLU. No closed-form projection.

    Parameters
    ----------
    h_x, h_l : float, optional
        Euler step sizes. Auto-set to ``k0_scale / L`` where L is the relevant
        Lipschitz factor.
    rho : float
        Augmentation strength on the constraint quadratic. rho = 0 recovers
        the vanilla saddle-point flow (oscillation-prone). rho > 0 damps it.

    Returns
    -------
    SolverResult
        ``spike_times`` records iterations where any dual variable's update
        was nonzero (i.e., at least one primal-induced violation triggered an
        ascent on lambda). ``spike_deltas`` records the *primal* displacement
        contributed by ``-h_x * C^T lambda`` at that iteration, so the spike
        norm tracks how much the dual feedback is currently bending the primal
        trajectory.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    C = np.asarray(C, dtype=float)
    d = np.asarray(d, dtype=float)
    x0 = np.asarray(x0, dtype=float).copy()

    n = A.shape[0]
    m = C.shape[0] if C.ndim == 2 else 0

    L_A = _spectral_norm_symmetric(A)
    if L_A < 1e-12:
        L_A = 1.0
    L_C = _spectral_norm(C) if m > 0 else 0.0

    # Auto-tune rho to balance with QP curvature, same logic as the
    # k_p heuristic in the penalty variant
    if rho is None:
        rho = 10.0 * L_A / max(L_C ** 2, 1e-12) if L_C > 0 else 1.0

    # Primal Lipschitz now includes augmentation: L_total = L_A + rho * ||C||^2
    L_total = L_A + rho * (L_C ** 2)
    if h_x is None:
        h_x = k0_scale / max(L_total, 1e-12)
    if h_l is None:
        # Dual step paced with primal: h_l = h_x. (Saddle-point convergence
        # theory says h_l <= 1 / ||C||^2_2 for stability when rho=0; the
        # augmentation adds further damping so we can match h_x.)
        h_l = h_x

    if verbose:
        print(f"[lagrangian] n={n}, m={m}, L_A={L_A:.3e}, ||C||_2={L_C:.3e}")
        print(f"[lagrangian] h_x={h_x:.3e}, h_l={h_l:.3e}, rho={rho:.3e}")

    x = x0.copy()
    lam = np.zeros(m)
    trajectory: List[np.ndarray] = [x.copy()]
    obj_history: List[float] = []
    spike_times: List[float] = []
    spike_deltas: List[np.ndarray] = []
    spike_constraints: List[np.ndarray] = []
    spike_violations: List[np.ndarray] = []

    converged = False
    convergence_reason = "max_iterations"
    iterations_used = max_iterations

    obj_window: List[float] = []
    window_size = 50
    plateau_tol = 1e-9
    min_iterations = max(200, int(0.05 * max_iterations))
    patience_counter = 0
    patience_required = 3
    check_every = 25

    for k in range(max_iterations):
        # Augmented Lagrangian primal-dual flow
        # primal: dx/dt = -(Ax+b) - C^T (lambda + rho * relu(g))
        if m > 0:
            g_at_x = C @ x + d
            v = np.maximum(g_at_x, 0.0)
            dual_force = C.T @ (lam + rho * v)
        else:
            v = np.zeros(0)
            dual_force = np.zeros(n)

        grad_f = A @ x + b
        dx = -(grad_f + dual_force)
        x_new = x + h_x * dx

        # dual: dlambda/dt = relu(g) — evaluated at *current* primal x
        # (the augmented term in the primal already provides damping)
        if m > 0:
            lam_new = np.maximum(lam + h_l * v, 0.0)
            active = np.where(v > 0)[0]
            if active.size > 0:
                primal_displacement_from_dual = -h_x * dual_force.copy()
                spike_times.append(float(k))
                spike_deltas.append(primal_displacement_from_dual)
                spike_constraints.append(active.copy())
                spike_violations.append(v[active].copy())
        else:
            lam_new = lam

        x = x_new
        lam = lam_new

        if record_full_trajectory:
            trajectory.append(x.copy())

        obj_curr = 0.5 * x @ A @ x + b @ x
        obj_history.append(obj_curr)
        obj_window.append(obj_curr)
        if len(obj_window) > window_size:
            obj_window.pop(0)

        if enable_early_stopping and k >= min_iterations and k % check_every == 0:
            max_viol = float(np.max(np.maximum(C @ x + d, 0.0))) if m > 0 else 0.0
            if max_viol <= feasibility_tol and len(obj_window) == window_size:
                obj_range = max(obj_window) - min(obj_window)
                obj_scale = max(abs(obj_window[-1]), 1e-10)
                if obj_range / obj_scale < plateau_tol:
                    proj_grad = _projected_gradient_norm(x, A, b, C, d)
                    if proj_grad < proj_grad_tol:
                        patience_counter += 1
                        if patience_counter >= patience_required:
                            converged = True
                            convergence_reason = (
                                f"converged(obj_plateau={obj_range/obj_scale:.2e};"
                                f" proj_grad={proj_grad:.2e}; max_viol={max_viol:.2e})"
                            )
                            iterations_used = k + 1
                            if verbose:
                                print(f"[lagrangian] iter {k}: {convergence_reason}")
                            break
                    else:
                        patience_counter = 0
                else:
                    patience_counter = 0
            else:
                patience_counter = 0

        if verbose and k % 500 == 0:
            max_viol = float(np.max(np.maximum(C @ x + d, 0.0))) if m > 0 else 0.0
            print(f"[lagrangian] iter {k:>5d}: obj={obj_curr:.4e}, "
                  f"max_viol={max_viol:.2e}, ||lambda||={np.linalg.norm(lam):.2e}")

    final_proj_grad = _projected_gradient_norm(x, A, b, C, d)

    return _build_result(
        trajectory=trajectory,
        obj_history=obj_history,
        A=A, b=b, C=C, d=d,
        n_projections=len(spike_times),
        converged=converged,
        convergence_reason=convergence_reason,
        iterations_used=iterations_used,
        final_x=x,
        final_proj_grad_norm=final_proj_grad,
        spike_times=spike_times,
        spike_deltas=spike_deltas,
        spike_constraints=spike_constraints,
        spike_violations=spike_violations,
    )


# ----------------------------------------------------------------------------
# Variant 3: Heun (explicit RK2) on penalty-augmented gradient flow
# ----------------------------------------------------------------------------

def solve_qp_heun_penalty(
    A: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    x0: np.ndarray,
    *,
    k0_scale: float = 0.5,
    k_p: Optional[float] = None,
    k_p_ramp: bool = True,
    k_p_final: Optional[float] = None,
    max_iterations: int = 5000,
    h: Optional[float] = None,
    feasibility_tol: float = 1e-2,
    proj_grad_tol: float = 1e-6,
    enable_early_stopping: bool = True,
    record_full_trajectory: bool = True,
    verbose: bool = False,
) -> SolverResult:
    """
    Variant 3 — Heun's method (explicit RK2) on the penalty-augmented flow.

    Dynamics (continuous): same as Variant 1.
    Discretisation::

        f(x) = -(A x + b) - k_p * C^T relu(C x + d)
        x_pred  = x_k + h * f(x_k)
        x_{k+1} = x_k + (h/2) * (f(x_k) + f(x_pred))

    Two function evaluations per step. LIF interpretation: the predictor stage
    is a fast feed-forward sweep (one membrane-update tick); the corrector
    stage averages the slope at predicted-position with the slope at current
    position. Maps to a two-phase clock with synapse-resampling at half-tick.

    Local error O(h^3) instead of O(h^2), so we can take ~sqrt larger steps.
    """
    A = np.asarray(A, dtype=float); b = np.asarray(b, dtype=float)
    C = np.asarray(C, dtype=float); d = np.asarray(d, dtype=float)
    x0 = np.asarray(x0, dtype=float).copy()

    n = A.shape[0]
    m = C.shape[0] if C.ndim == 2 else 0

    L_A = _spectral_norm_symmetric(A) or 1.0
    L_C = _spectral_norm(C) if m > 0 else 0.0

    grad_scale = max(L_A, np.linalg.norm(b))
    if k_p is None:
        k_p_init = 10.0 * grad_scale / max(L_C ** 2, 1e-12) if L_C > 0 else 1.0
    else:
        k_p_init = float(k_p)
    k_p_fin = float(k_p_final) if k_p_final is not None else (
        1e4 * k_p_init if k_p_ramp else k_p_init
    )

    h_user = h
    if h is None:
        # Heun is stable up to ~2/L for linear problems; 1.4x larger than Euler
        h_init = 1.4 * k0_scale / max(L_A, 1e-12)
    else:
        h_init = h

    def _f(x_arg, k_p_arg):
        if m > 0:
            v = np.maximum(C @ x_arg + d, 0.0)
            return -(A @ x_arg + b) - k_p_arg * (C.T @ v)
        return -(A @ x_arg + b)

    if verbose:
        print(f"[heun] n={n}, m={m}, h={h:.3e}, k_p {k_p_init:.2e}->{k_p_fin:.2e}")

    x = x0.copy()
    trajectory = [x.copy()]
    obj_history = []
    spike_times: List[float] = []
    spike_deltas: List[np.ndarray] = []
    spike_constraints: List[np.ndarray] = []
    spike_violations: List[np.ndarray] = []

    converged = False
    convergence_reason = "max_iterations"
    iterations_used = max_iterations
    obj_window: List[float] = []
    window_size = 50
    plateau_tol = 1e-9
    min_iterations = max(200, int(0.05 * max_iterations))
    patience_counter = 0
    patience_required = 3
    check_every = 25

    for k in range(max_iterations):
        if k_p_ramp and max_iterations > 1:
            # alpha^2 schedule: k_p stays near k_p_init for the first ~half
            # of iterations (giving the unconstrained gradient flow time to
            # traverse), then ramps up to k_p_fin to refine feasibility.
            alpha = k / (max_iterations - 1)
            k_p_curr = k_p_init * (k_p_fin / k_p_init) ** (alpha ** 2)
        else:
            k_p_curr = k_p_init

        if h_user is None:
            h_curr = 1.4 * k0_scale / max(L_A + k_p_curr * (L_C ** 2), 1e-12)
        else:
            h_curr = h_user

        f1 = _f(x, k_p_curr)
        x_pred = x + h_curr * f1
        f2 = _f(x_pred, k_p_curr)
        x_new = x + (h_curr / 2.0) * (f1 + f2)

        if m > 0:
            v_at_x = np.maximum(C @ x + d, 0.0)
            active = np.where(v_at_x > 0)[0]
            if active.size > 0:
                spike_times.append(float(k))
                spike_deltas.append(x_new - x)
                spike_constraints.append(active.copy())
                spike_violations.append(v_at_x[active].copy())

        x = x_new
        if record_full_trajectory:
            trajectory.append(x.copy())

        obj_curr = 0.5 * x @ A @ x + b @ x
        obj_history.append(obj_curr)
        obj_window.append(obj_curr)
        if len(obj_window) > window_size:
            obj_window.pop(0)

        if enable_early_stopping and k >= min_iterations and k % check_every == 0:
            max_viol = float(np.max(np.maximum(C @ x + d, 0.0))) if m > 0 else 0.0
            if max_viol <= feasibility_tol and len(obj_window) == window_size:
                obj_range = max(obj_window) - min(obj_window)
                obj_scale = max(abs(obj_window[-1]), 1e-10)
                if obj_range / obj_scale < plateau_tol:
                    proj_grad = _projected_gradient_norm(x, A, b, C, d)
                    if proj_grad < proj_grad_tol:
                        patience_counter += 1
                        if patience_counter >= patience_required:
                            converged = True
                            convergence_reason = (
                                f"converged(obj_plateau={obj_range/obj_scale:.2e};"
                                f" proj_grad={proj_grad:.2e}; max_viol={max_viol:.2e})"
                            )
                            iterations_used = k + 1
                            break
                    else:
                        patience_counter = 0
                else:
                    patience_counter = 0
            else:
                patience_counter = 0

    final_proj_grad = _projected_gradient_norm(x, A, b, C, d)
    return _build_result(
        trajectory=trajectory, obj_history=obj_history,
        A=A, b=b, C=C, d=d,
        n_projections=len(spike_times), converged=converged,
        convergence_reason=convergence_reason, iterations_used=iterations_used,
        final_x=x, final_proj_grad_norm=final_proj_grad,
        spike_times=spike_times, spike_deltas=spike_deltas,
        spike_constraints=spike_constraints, spike_violations=spike_violations,
    )


# ----------------------------------------------------------------------------
# Variant 4: Heavy-ball (Polyak) momentum on penalty-augmented flow
# ----------------------------------------------------------------------------

def solve_qp_heavyball_penalty(
    A: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    x0: np.ndarray,
    *,
    k0_scale: float = 0.5,
    k_p: Optional[float] = None,
    k_p_ramp: bool = True,
    k_p_final: Optional[float] = None,
    max_iterations: int = 5000,
    h: Optional[float] = None,
    beta: Optional[float] = None,
    feasibility_tol: float = 1e-2,
    proj_grad_tol: float = 1e-6,
    enable_early_stopping: bool = True,
    record_full_trajectory: bool = True,
    verbose: bool = False,
) -> SolverResult:
    """
    Variant 4 — heavy-ball / Polyak momentum on the penalty-augmented flow.

    Continuous-time second-order ODE::

        m * d2x/dt2 + gamma * dx/dt + nabla F(x) = 0

    Equivalent first-order system on (x, v=dx/dt). Discrete update::

        v_{k+1} = beta * v_k + (1-beta) * (-grad_F(x_k))
                = beta * v_k + (1-beta) * f(x_k)            with f = -grad_F
        x_{k+1} = x_k + h * v_{k+1}

    Two coupled neural populations: primal x and momentum v. Pure LIF
    integrators with leak ``beta`` on v.

    On strongly-convex QPs, heavy-ball achieves linear rate with condition
    number sqrt(kappa) instead of kappa — substantially faster than vanilla
    gradient flow on poorly-conditioned MPC QPs.
    """
    A = np.asarray(A, dtype=float); b = np.asarray(b, dtype=float)
    C = np.asarray(C, dtype=float); d = np.asarray(d, dtype=float)
    x0 = np.asarray(x0, dtype=float).copy()
    n = A.shape[0]
    m = C.shape[0] if C.ndim == 2 else 0

    L_A = _spectral_norm_symmetric(A) or 1.0
    L_C = _spectral_norm(C) if m > 0 else 0.0
    grad_scale = max(L_A, np.linalg.norm(b))
    if k_p is None:
        k_p_init = 10.0 * grad_scale / max(L_C ** 2, 1e-12) if L_C > 0 else 1.0
    else:
        k_p_init = float(k_p)
    k_p_fin = float(k_p_final) if k_p_final is not None else (
        1e4 * k_p_init if k_p_ramp else k_p_init
    )
    h_user = h
    if h is None:
        h_init = k0_scale / max(L_A, 1e-12)
    else:
        h_init = h
    if beta is None:
        # Conservative momentum coefficient; tuned for stability with k_p ramp
        beta = 0.9

    if verbose:
        print(f"[heavyball] n={n}, m={m}, h_init={h_init:.3e} (adaptive), beta={beta}, "
              f"k_p {k_p_init:.2e}->{k_p_fin:.2e}")

    x = x0.copy()
    v = np.zeros(n)
    trajectory = [x.copy()]
    obj_history = []
    spike_times: List[float] = []
    spike_deltas: List[np.ndarray] = []
    spike_constraints: List[np.ndarray] = []
    spike_violations: List[np.ndarray] = []

    converged = False
    convergence_reason = "max_iterations"
    iterations_used = max_iterations
    obj_window: List[float] = []
    window_size = 50
    plateau_tol = 1e-9
    min_iterations = max(200, int(0.05 * max_iterations))
    patience_counter = 0
    patience_required = 3
    check_every = 25

    for k in range(max_iterations):
        if k_p_ramp and max_iterations > 1:
            # alpha^2 schedule: k_p stays near k_p_init for the first ~half
            # of iterations (giving the unconstrained gradient flow time to
            # traverse), then ramps up to k_p_fin to refine feasibility.
            alpha = k / (max_iterations - 1)
            k_p_curr = k_p_init * (k_p_fin / k_p_init) ** (alpha ** 2)
        else:
            k_p_curr = k_p_init

        if h_user is None:
            h_curr = k0_scale / max(L_A + k_p_curr * (L_C ** 2), 1e-12)
        else:
            h_curr = h_user

        if m > 0:
            v_resid = np.maximum(C @ x + d, 0.0)
            f_x = -(A @ x + b) - k_p_curr * (C.T @ v_resid)
            active = np.where(v_resid > 0)[0]
            had_violation = active.size > 0
        else:
            f_x = -(A @ x + b)
            had_violation = False
            v_resid = np.zeros(0)
            active = np.array([], dtype=int)

        v = beta * v + (1.0 - beta) * f_x
        x_new = x + h_curr * v

        if had_violation:
            spike_times.append(float(k))
            spike_deltas.append(x_new - x)
            spike_constraints.append(active.copy())
            spike_violations.append(v_resid[active].copy())

        x = x_new
        if record_full_trajectory:
            trajectory.append(x.copy())

        obj_curr = 0.5 * x @ A @ x + b @ x
        obj_history.append(obj_curr)
        obj_window.append(obj_curr)
        if len(obj_window) > window_size:
            obj_window.pop(0)

        if enable_early_stopping and k >= min_iterations and k % check_every == 0:
            max_viol = float(np.max(np.maximum(C @ x + d, 0.0))) if m > 0 else 0.0
            if max_viol <= feasibility_tol and len(obj_window) == window_size:
                obj_range = max(obj_window) - min(obj_window)
                obj_scale = max(abs(obj_window[-1]), 1e-10)
                if obj_range / obj_scale < plateau_tol:
                    proj_grad = _projected_gradient_norm(x, A, b, C, d)
                    if proj_grad < proj_grad_tol:
                        patience_counter += 1
                        if patience_counter >= patience_required:
                            converged = True
                            convergence_reason = (
                                f"converged(obj_plateau={obj_range/obj_scale:.2e};"
                                f" proj_grad={proj_grad:.2e}; max_viol={max_viol:.2e})"
                            )
                            iterations_used = k + 1
                            break
                    else:
                        patience_counter = 0
                else:
                    patience_counter = 0
            else:
                patience_counter = 0

    final_proj_grad = _projected_gradient_norm(x, A, b, C, d)
    return _build_result(
        trajectory=trajectory, obj_history=obj_history,
        A=A, b=b, C=C, d=d,
        n_projections=len(spike_times), converged=converged,
        convergence_reason=convergence_reason, iterations_used=iterations_used,
        final_x=x, final_proj_grad_norm=final_proj_grad,
        spike_times=spike_times, spike_deltas=spike_deltas,
        spike_constraints=spike_constraints, spike_violations=spike_violations,
    )


# ----------------------------------------------------------------------------
# Variant 5: Nesterov-accelerated flow on penalty-augmented gradient
# ----------------------------------------------------------------------------

def solve_qp_nesterov_penalty(
    A: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    x0: np.ndarray,
    *,
    k0_scale: float = 0.5,
    k_p: Optional[float] = None,
    k_p_ramp: bool = True,
    k_p_final: Optional[float] = None,
    max_iterations: int = 5000,
    h: Optional[float] = None,
    feasibility_tol: float = 1e-2,
    proj_grad_tol: float = 1e-6,
    enable_early_stopping: bool = True,
    record_full_trajectory: bool = True,
    verbose: bool = False,
) -> SolverResult:
    """
    Variant 5 — Nesterov-accelerated flow on the penalty-augmented gradient.

    Continuous-time ODE (Su-Boyd-Candès 2014)::

        d2x/dt2 + (3/t) dx/dt + nabla F(x) = 0

    Discrete Nesterov FISTA-style update::

        y_k     = x_k + ((k-1)/(k+2)) * (x_k - x_{k-1})
        x_{k+1} = y_k + h * f(y_k)         with f = -grad_F

    Same population structure as heavy-ball (primal x and a "lookahead"
    auxiliary y) but with a time-varying coupling rate (k-1)/(k+2). Achieves
    O(1/t^2) error in continuous time vs O(1/t) for vanilla GD.

    Mapping to LIF: y is a synaptic average of recent x's with a time-varying
    coefficient (one extra parameter neuron whose firing rate equals k/(k+2)).
    """
    A = np.asarray(A, dtype=float); b = np.asarray(b, dtype=float)
    C = np.asarray(C, dtype=float); d = np.asarray(d, dtype=float)
    x0 = np.asarray(x0, dtype=float).copy()
    n = A.shape[0]
    m = C.shape[0] if C.ndim == 2 else 0

    L_A = _spectral_norm_symmetric(A) or 1.0
    L_C = _spectral_norm(C) if m > 0 else 0.0
    grad_scale = max(L_A, np.linalg.norm(b))
    if k_p is None:
        k_p_init = 10.0 * grad_scale / max(L_C ** 2, 1e-12) if L_C > 0 else 1.0
    else:
        k_p_init = float(k_p)
    k_p_fin = float(k_p_final) if k_p_final is not None else (
        1e4 * k_p_init if k_p_ramp else k_p_init
    )
    h_user = h
    if h is None:
        # FISTA needs h <= 1/L; auto-step adapts to current k_p (see loop)
        h_init = k0_scale / max(L_A, 1e-12)
    else:
        h_init = h

    if verbose:
        print(f"[nesterov] n={n}, m={m}, h_init={h_init:.3e} (adaptive), "
              f"k_p {k_p_init:.2e}->{k_p_fin:.2e}")

    x_prev = x0.copy()
    x = x0.copy()
    trajectory = [x.copy()]
    obj_history = []
    spike_times: List[float] = []
    spike_deltas: List[np.ndarray] = []
    spike_constraints: List[np.ndarray] = []
    spike_violations: List[np.ndarray] = []

    converged = False
    convergence_reason = "max_iterations"
    iterations_used = max_iterations
    obj_window: List[float] = []
    window_size = 50
    plateau_tol = 1e-9
    min_iterations = max(200, int(0.05 * max_iterations))
    patience_counter = 0
    patience_required = 3
    check_every = 25

    for k in range(max_iterations):
        if k_p_ramp and max_iterations > 1:
            # alpha^2 schedule: k_p stays near k_p_init for the first ~half
            # of iterations (giving the unconstrained gradient flow time to
            # traverse), then ramps up to k_p_fin to refine feasibility.
            alpha = k / (max_iterations - 1)
            k_p_curr = k_p_init * (k_p_fin / k_p_init) ** (alpha ** 2)
        else:
            k_p_curr = k_p_init

        if h_user is None:
            h_curr = k0_scale / max(L_A + k_p_curr * (L_C ** 2), 1e-12)
        else:
            h_curr = h_user

        # Nesterov lookahead
        if k == 0:
            y = x
        else:
            theta = (k - 1.0) / (k + 2.0)
            y = x + theta * (x - x_prev)

        if m > 0:
            v_resid = np.maximum(C @ y + d, 0.0)
            f_y = -(A @ y + b) - k_p_curr * (C.T @ v_resid)
            active = np.where(v_resid > 0)[0]
            had_violation = active.size > 0
        else:
            f_y = -(A @ y + b)
            had_violation = False
            v_resid = np.zeros(0); active = np.array([], dtype=int)

        x_new = y + h_curr * f_y

        if had_violation:
            spike_times.append(float(k))
            spike_deltas.append(x_new - x)
            spike_constraints.append(active.copy())
            spike_violations.append(v_resid[active].copy())

        x_prev = x
        x = x_new
        if record_full_trajectory:
            trajectory.append(x.copy())

        obj_curr = 0.5 * x @ A @ x + b @ x
        obj_history.append(obj_curr)
        obj_window.append(obj_curr)
        if len(obj_window) > window_size:
            obj_window.pop(0)

        if enable_early_stopping and k >= min_iterations and k % check_every == 0:
            max_viol = float(np.max(np.maximum(C @ x + d, 0.0))) if m > 0 else 0.0
            if max_viol <= feasibility_tol and len(obj_window) == window_size:
                obj_range = max(obj_window) - min(obj_window)
                obj_scale = max(abs(obj_window[-1]), 1e-10)
                if obj_range / obj_scale < plateau_tol:
                    proj_grad = _projected_gradient_norm(x, A, b, C, d)
                    if proj_grad < proj_grad_tol:
                        patience_counter += 1
                        if patience_counter >= patience_required:
                            converged = True
                            convergence_reason = (
                                f"converged(obj_plateau={obj_range/obj_scale:.2e};"
                                f" proj_grad={proj_grad:.2e}; max_viol={max_viol:.2e})"
                            )
                            iterations_used = k + 1
                            break
                    else:
                        patience_counter = 0
                else:
                    patience_counter = 0
            else:
                patience_counter = 0

    final_proj_grad = _projected_gradient_norm(x, A, b, C, d)
    return _build_result(
        trajectory=trajectory, obj_history=obj_history,
        A=A, b=b, C=C, d=d,
        n_projections=len(spike_times), converged=converged,
        convergence_reason=convergence_reason, iterations_used=iterations_used,
        final_x=x, final_proj_grad_norm=final_proj_grad,
        spike_times=spike_times, spike_deltas=spike_deltas,
        spike_constraints=spike_constraints, spike_violations=spike_violations,
    )


# ----------------------------------------------------------------------------
# Variant 6: Exponential Euler on linear part + penalty
# ----------------------------------------------------------------------------

def solve_qp_expeuler_penalty(
    A: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    x0: np.ndarray,
    *,
    k0_scale: float = 0.5,
    k_p: Optional[float] = None,
    k_p_ramp: bool = True,
    k_p_final: Optional[float] = None,
    max_iterations: int = 5000,
    h: Optional[float] = None,
    feasibility_tol: float = 1e-2,
    proj_grad_tol: float = 1e-6,
    enable_early_stopping: bool = True,
    record_full_trajectory: bool = True,
    verbose: bool = False,
) -> SolverResult:
    """
    Variant 6 — exponential Euler on the linear part + explicit penalty.

    Background
    ----------
    The unconstrained gradient flow ``dx/dt = -(Ax + b)`` is a linear ODE
    whose exact solution is::

        x(t+h) = exp(-h A) x(t) + (-A^{-1} (I - exp(-h A))) b

    We split the dynamics into linear (Ax + b) plus nonlinear penalty force
    g(x) = -k_p C^T relu(Cx + d) and apply exponential Euler::

        x_{k+1} = E_h x_k + B_h ( -b + g(x_k) )       (ETD1)

    where ``E_h = exp(-h A)`` and ``B_h = A^{-1}(I - E_h)`` are *precomputed
    once*. The per-step computation is two matrix-vector multiplies +
    elementwise ReLU + one matrix-vector multiply for ``C^T v`` — no inverse
    at runtime.

    LIF mapping
    -----------
    On Loihi-2 the leak-and-integrate dynamics already realise ``exp(-h * tau^{-1})``
    decay per neuron analytically; an exponential Euler scheme is therefore
    the *native* per-tick update for spiking hardware that supports leak.
    The matrix exponential precompute is one offline operation analogous to
    setting the leak constant per synapse.

    Caveat: ``A^{-1}`` in the precompute requires invertibility. For QPs where
    A may have zero eigenvalues, we fall back to the series expansion
    ``B_h = h I - h^2 A / 2 + h^3 A^2 / 6 - ...`` truncated at 4 terms.
    """
    from scipy.linalg import expm

    A = np.asarray(A, dtype=float); b = np.asarray(b, dtype=float)
    C = np.asarray(C, dtype=float); d = np.asarray(d, dtype=float)
    x0 = np.asarray(x0, dtype=float).copy()
    n = A.shape[0]
    m = C.shape[0] if C.ndim == 2 else 0

    L_A = _spectral_norm_symmetric(A) or 1.0
    L_C = _spectral_norm(C) if m > 0 else 0.0
    grad_scale = max(L_A, np.linalg.norm(b))

    # Exponential Euler precomputes exp(-h*A) once, so h is fixed for the
    # whole run. Strategy: pick k_p target first (so we have a useful
    # penalty), then back-compute h so that the explicit penalty force
    # stays stable: h * k_p_fin * L_C^2 < 0.5.
    if k_p is None:
        k_p_init = 10.0 * grad_scale / max(L_C ** 2, 1e-12) if L_C > 0 else 1.0
    else:
        k_p_init = float(k_p)
    # Note: expeuler uses a *much* smaller ramp than the time-varying-h
    # variants because it cannot reduce h as k_p_curr grows. We trade
    # asymptotic feasibility tightness for a longer effective gradient-flow
    # horizon (so the iterate can actually reach the optimum).
    k_p_fin = float(k_p_final) if k_p_final is not None else (
        10.0 * k_p_init if k_p_ramp else k_p_init
    )
    if h is None:
        if L_C > 0:
            h_penalty_lim = 0.5 / max(k_p_fin * (L_C ** 2), 1e-12)
        else:
            h_penalty_lim = float("inf")
        h = min(k0_scale / max(L_A, 1e-12), h_penalty_lim)

    # Precompute E_h = exp(-h A) and B_h = A^{-1} (I - E_h)
    E_h = expm(-h * A)
    I = np.eye(n)
    try:
        # If A is not too singular, this path is exact
        B_h = np.linalg.solve(A + 1e-12 * I, I - E_h)
    except np.linalg.LinAlgError:
        # Truncated series: B_h ~ h I - h^2 A / 2 + h^3 A^2 / 6 - h^4 A^3 / 24
        Apow = I.copy()
        B_h = np.zeros_like(A)
        coef = h
        sign = 1.0
        for k_term in range(1, 5):
            B_h = B_h + sign * coef * Apow
            Apow = Apow @ A
            coef = coef * h / (k_term + 1)
            sign = -sign

    if verbose:
        print(f"[expeuler] n={n}, m={m}, h={h:.3e}, "
              f"k_p {k_p_init:.2e}->{k_p_fin:.2e} (E_h, B_h precomputed)")

    x = x0.copy()
    trajectory = [x.copy()]
    obj_history = []
    spike_times: List[float] = []
    spike_deltas: List[np.ndarray] = []
    spike_constraints: List[np.ndarray] = []
    spike_violations: List[np.ndarray] = []

    converged = False
    convergence_reason = "max_iterations"
    iterations_used = max_iterations
    obj_window: List[float] = []
    window_size = 50
    plateau_tol = 1e-9
    min_iterations = max(200, int(0.05 * max_iterations))
    patience_counter = 0
    patience_required = 3
    check_every = 25

    for k in range(max_iterations):
        if k_p_ramp and max_iterations > 1:
            # alpha^2 schedule: k_p stays near k_p_init for the first ~half
            # of iterations (giving the unconstrained gradient flow time to
            # traverse), then ramps up to k_p_fin to refine feasibility.
            alpha = k / (max_iterations - 1)
            k_p_curr = k_p_init * (k_p_fin / k_p_init) ** (alpha ** 2)
        else:
            k_p_curr = k_p_init

        if m > 0:
            v_resid = np.maximum(C @ x + d, 0.0)
            penalty_force = -k_p_curr * (C.T @ v_resid)
            active = np.where(v_resid > 0)[0]
            had_violation = active.size > 0
        else:
            penalty_force = np.zeros(n); v_resid = np.zeros(0)
            active = np.array([], dtype=int); had_violation = False

        forcing = -b + penalty_force
        x_new = E_h @ x + B_h @ forcing

        if had_violation:
            spike_times.append(float(k))
            spike_deltas.append(x_new - x)
            spike_constraints.append(active.copy())
            spike_violations.append(v_resid[active].copy())

        x = x_new
        if record_full_trajectory:
            trajectory.append(x.copy())

        obj_curr = 0.5 * x @ A @ x + b @ x
        obj_history.append(obj_curr)
        obj_window.append(obj_curr)
        if len(obj_window) > window_size:
            obj_window.pop(0)

        if enable_early_stopping and k >= min_iterations and k % check_every == 0:
            max_viol = float(np.max(np.maximum(C @ x + d, 0.0))) if m > 0 else 0.0
            if max_viol <= feasibility_tol and len(obj_window) == window_size:
                obj_range = max(obj_window) - min(obj_window)
                obj_scale = max(abs(obj_window[-1]), 1e-10)
                if obj_range / obj_scale < plateau_tol:
                    proj_grad = _projected_gradient_norm(x, A, b, C, d)
                    if proj_grad < proj_grad_tol:
                        patience_counter += 1
                        if patience_counter >= patience_required:
                            converged = True
                            convergence_reason = (
                                f"converged(obj_plateau={obj_range/obj_scale:.2e};"
                                f" proj_grad={proj_grad:.2e}; max_viol={max_viol:.2e})"
                            )
                            iterations_used = k + 1
                            break
                    else:
                        patience_counter = 0
                else:
                    patience_counter = 0
            else:
                patience_counter = 0

    final_proj_grad = _projected_gradient_norm(x, A, b, C, d)
    return _build_result(
        trajectory=trajectory, obj_history=obj_history,
        A=A, b=b, C=C, d=d,
        n_projections=len(spike_times), converged=converged,
        convergence_reason=convergence_reason, iterations_used=iterations_used,
        final_x=x, final_proj_grad_norm=final_proj_grad,
        spike_times=spike_times, spike_deltas=spike_deltas,
        spike_constraints=spike_constraints, spike_violations=spike_violations,
    )


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------

def _spectral_norm_symmetric(A: np.ndarray) -> float:
    """Largest |eigenvalue| of symmetric A (or 0 if A is empty/zero)."""
    if A.size == 0:
        return 0.0
    if np.allclose(A, 0.0):
        return 0.0
    if np.allclose(A, A.T):
        return float(np.max(np.abs(np.linalg.eigvalsh(A))))
    return float(np.linalg.norm(A, 2))


def _spectral_norm(M: np.ndarray) -> float:
    if M.size == 0:
        return 0.0
    return float(np.linalg.norm(M, 2))


def _projected_gradient_norm(
    x: np.ndarray, A: np.ndarray, b: np.ndarray, C: np.ndarray, d: np.ndarray,
    active_tol: float = 1e-5,
) -> float:
    """Norm of feasible-descent-projected gradient at x."""
    grad = A @ x + b
    proj_grad = grad.copy()
    if C.ndim == 2 and C.shape[0] > 0:
        g = C @ x + d
        c_norms_sq = np.sum(C ** 2, axis=1)
        for j in range(C.shape[0]):
            if g[j] >= -active_tol:
                if c_norms_sq[j] < 1e-12:
                    continue
                cj = C[j]
                comp = (proj_grad @ cj) / c_norms_sq[j]
                if comp < 0:
                    proj_grad = proj_grad - comp * cj
    return float(np.linalg.norm(proj_grad))


def _build_result(
    *,
    trajectory: List[np.ndarray],
    obj_history: List[float],
    A: np.ndarray, b: np.ndarray, C: np.ndarray, d: np.ndarray,
    n_projections: int,
    converged: bool,
    convergence_reason: str,
    iterations_used: int,
    final_x: np.ndarray,
    final_proj_grad_norm: float,
    spike_times: List[float],
    spike_deltas: List[np.ndarray],
    spike_constraints: List[np.ndarray],
    spike_violations: List[np.ndarray],
) -> SolverResult:
    """Pack into a SolverResult identical-shape to the canonical solver."""
    X = np.array(trajectory)
    t = np.arange(len(trajectory), dtype=float)
    n_vars = A.shape[0]

    if X.shape[0] >= 1:
        objective_values = 0.5 * np.einsum("ti,ij,tj->t", X, A, X) + X @ b
    else:
        objective_values = np.array([])

    if C.ndim == 2 and C.shape[0] > 0:
        cv = X @ C.T + d  # (T, m)
        constraint_violations = np.max(np.maximum(cv, 0.0), axis=1)
    else:
        constraint_violations = np.zeros(X.shape[0])

    if spike_times:
        spike_times_arr = np.asarray(spike_times, dtype=float)
    else:
        spike_times_arr = np.array([], dtype=float)

    if spike_deltas:
        spike_deltas_arr = np.vstack(spike_deltas)
        spike_norms = np.linalg.norm(spike_deltas_arr, axis=1)
    else:
        spike_deltas_arr = np.empty((0, n_vars))
        spike_norms = np.empty((0,), dtype=float)

    return SolverResult(
        t=t,
        X=X,
        objective_values=objective_values,
        constraint_violations=constraint_violations,
        n_projections=n_projections,
        converged=converged,
        convergence_reason=convergence_reason,
        iterations_used=iterations_used,
        final_x=final_x,
        final_objective=float(0.5 * final_x @ A @ final_x + b @ final_x),
        final_proj_grad_norm=final_proj_grad_norm,
        spike_times=spike_times_arr,
        spike_deltas=spike_deltas_arr,
        spike_norms=spike_norms,
        spike_constraints=[np.array(idx, dtype=int) for idx in spike_constraints],
        spike_violation_values=[np.array(v, dtype=float) for v in spike_violations],
        total_projection_distance=float(spike_norms.sum()) if spike_norms.size else 0.0,
    )
