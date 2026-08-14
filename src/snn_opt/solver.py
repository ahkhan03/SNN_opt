"""
SNN-Inspired Constrained Optimization Solver

Solves convex optimization problems with linear inequality constraints:
    minimize    (1/2) x^T A x + b^T x
    subject to  C x + d <= 0   (optionally including scalar box bounds l <= x <= u)

The algorithm alternates between gradient descent and discrete boundary projections,
inspired by spiking neural network dynamics.

Since v0.5.0, box bounds are handled INSIDE the projection sweep as implicit
unit-normal constraint facets (one constraint neuron per bound facet), selected
by the same normalized-distance winner-take-all rule as the general rows. The
pre-v0.5 terminal clip (applied after the halfspace sweep, with nothing
re-projecting behind it) was a structural correctness defect: composing the two
mechanisms is not a projection onto the intersection, so on problems where the
box and an interacting row are simultaneously active the solver stalled at an
infeasible point whose objective can undercut the true optimum. Box-only
problems (m == 0) still dispatch to the exact vectorized box projection.
"""

import numpy as np
import scipy.sparse as _sp
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple, List, Callable, Union


def _issparse(x):
    """Check if x is a scipy sparse matrix."""
    return _sp.issparse(x)


# Cap on the constraint count m for precomputing the m x m constraint Gram
# matrix G = C C^T (used by the event-driven adaptive projection). Above this,
# the m x m matrix is judged too costly; the Python path falls back to
# recomputing the residual and backend='c' raises a clear error.
_MAX_GRAM_M = 4096

# Compiled-kernel backends. All three route to the same C++ kernel; they differ
# only in the matvec threading the kernel uses:
#   'c'        -- auto: multicore if the wheel was built with OpenMP, else serial
#   'c_serial' -- force single-threaded (SIMD only)
#   'c_openmp' -- force OpenMP multicore (raises if the wheel lacks OpenMP)
_C_BACKENDS = frozenset({'c', 'c_serial', 'c_openmp'})
_VALID_BACKENDS = frozenset({'python'}) | _C_BACKENDS

# Canonical constant-memory projection-event stream digest. Each committed
# event updates three unsigned 64-bit words in order: outer iteration, ordinal
# within that projection sweep, and canonical candidate ID. Every word uses
#     h <- (h xor (word + 1)) * prime  (mod 2^64).
# The C++ observer uses the same constants and unsigned-wrap semantics.
_EVENT_DIGEST_ALGORITHM = "fnv1a64-word-v2"
_EVENT_DIGEST_OFFSET = 14695981039346656037
_EVENT_DIGEST_PRIME = 1099511628211
_UINT64_MASK = (1 << 64) - 1


@dataclass
class ConvergenceConfig:
    """Configuration for convergence detection.

    Since v0.6.0 the authoritative optimality criterion is a scale-invariant
    KKT certificate (``optimality_test="kkt"``): a nonnegative least-squares
    fit of the gradient onto the cone of all facet normals, augmented with a
    complementarity row, accepted when

        r_kkt <= kkt_abs_tol + kkt_rel_tol * max(||A x||, ||b||, ||N^T mu||).

    Both residual components carry gradient units, so the test is invariant
    under any positive rescaling of the objective (the v0.5 projected-gradient
    test compared an absolute norm against ``1e-6`` and therefore could never
    fire on problems with a large gradient scale, and stalled structurally at
    constrained optima with correlated active normals).

    The cheap criteria (objective plateau, solution stability) and the
    feasibility gate are evaluated first; the KKT fit only runs when they
    already pass, so its cost is confined to near-termination checkpoints.
    """
    # Enable/disable early stopping
    enable_early_stopping: bool = True

    # Authoritative optimality criterion:
    #   "kkt"                       -- scale-invariant KKT certificate (default)
    #   "legacy_projected_gradient" -- pre-v0.6 absolute projected-gradient test
    #   "none"                      -- no optimality test (cheap criteria only)
    optimality_test: str = "kkt"

    # KKT certificate tolerances: accept when
    # r_kkt <= kkt_abs_tol + kkt_rel_tol * scale. The relative term dominates
    # at every practical scale; the absolute floor only matters for problems
    # whose gradient scale is ~0. kkt_rel_tol=1e-4 is calibrated to the O(k0)
    # fixed-point floor of the default dynamics (k0_scale=0.5): it certifies
    # the residual the solver genuinely reaches, roughly 1e-3 relative
    # solution error; it is a KKT-residual tolerance, not an error bound.
    kkt_abs_tol: float = 1e-9
    kkt_rel_tol: float = 1e-4

    # Tolerances (cheap criteria)
    obj_rel_tol: float = 1e-8          # Relative objective change over window
    x_rel_tol: float = 1e-8            # Relative solution change
    feasibility_tol: float = 1e-2      # Max constraint violation for convergence (default: relaxed)

    # Timing control
    check_every: int = 50              # Check convergence every N iterations
    min_iterations: int = 100          # Minimum iterations before checking
    window_size: int = 10              # Window size for objective plateau detection
    patience: int = 3                  # Consecutive converged checks needed

    # Which cheap criteria to use (ALL enabled criteria must be satisfied,
    # together with the selected optimality test)
    use_objective_plateau: bool = True
    use_solution_stable: bool = False   # Disabled by default (can cause false positives)
    require_feasibility: bool = True    # Require feasibility for convergence

    # ------------------------------------------------------------------
    # Deprecated constructor-only aliases (pre-v0.6 projected-gradient
    # criterion). Explicitly supplying either selects the legacy test and
    # warns; supplying them together with a non-default new-style criterion
    # raises. They are resolved in __post_init__ and normalized away.
    # ------------------------------------------------------------------
    use_projected_gradient: Optional[bool] = None
    proj_grad_tol: Optional[float] = None

    def __post_init__(self):
        valid = ("kkt", "legacy_projected_gradient", "none")
        if self.optimality_test not in valid:
            raise ValueError(
                f"optimality_test must be one of {valid}, "
                f"got {self.optimality_test!r}")
        legacy_supplied = (self.use_projected_gradient is not None
                          or self.proj_grad_tol is not None)
        if legacy_supplied:
            new_style_supplied = (self.optimality_test != "kkt"
                                  or self.kkt_abs_tol != 1e-9
                                  or self.kkt_rel_tol != 1e-4)
            if new_style_supplied:
                raise ValueError(
                    "deprecated projected-gradient options "
                    "(use_projected_gradient / proj_grad_tol) cannot be "
                    "combined with explicit optimality_test / kkt_* settings; "
                    "supply exactly one criterion family")
            import warnings
            if self.use_projected_gradient is False:
                self.optimality_test = "none"
                warnings.warn(
                    "use_projected_gradient=False is deprecated; use "
                    "optimality_test='none'", DeprecationWarning, stacklevel=3)
            else:
                self.optimality_test = "legacy_projected_gradient"
                warnings.warn(
                    "use_projected_gradient / proj_grad_tol are deprecated; "
                    "the scale-invariant default is optimality_test='kkt'. "
                    "Selecting optimality_test='legacy_projected_gradient' "
                    "for backward compatibility",
                    DeprecationWarning, stacklevel=3)
        if self.proj_grad_tol is None:
            self.proj_grad_tol = 1e-6
        self.use_projected_gradient = (
            self.optimality_test == "legacy_projected_gradient")


@dataclass
class SolverConfig:
    """Configuration parameters for the SNN solver."""
    k0: float = None  # Gradient descent step size (None = auto-compute from Lipschitz constant)
    t_end: float = 100.0  # Simulation end time (for IVP mode)
    max_step: float = 0.1  # Maximum integration step size (for IVP mode)
    # Geometric (Euclidean-distance) constraint tolerance: a row counts as
    # satisfied when its violation DISTANCE (raw residual / ||c_j||) is <= tol.
    # Box facets have unit normals, so their raw violation is already a distance.
    constraint_tol: float = 1e-6
    # Safety watchdog for the inner projection sweep. None (default) resolves to
    # max(1000, 10 * (m + number of box facets)). The sweep is meant to run to
    # joint tolerance; hitting this cap aborts the solve with
    # convergence_reason='projection_budget_exhausted' (never silently continued
    # from a knowingly infeasible point).
    max_projection_iters: Optional[int] = None
    
    # Integration method: 'ivp' (continuous ODE) or 'euler' (discrete steps)
    integration_method: str = 'euler'  # Default to euler for better stability
    max_iterations: int = 2000  # Maximum iterations (for Euler mode)
    
    # Projection method: 'adaptive' (exact step to boundary) or 'fixed' (fixed k1 step)
    projection_method: str = 'adaptive'  # Default to adaptive (eliminates k1 hyperparameter)
    k1: float = 0.05  # Projection step size (only used when projection_method='fixed')
    
    # Scaling factor for auto-computed k0 (k0 = scale / L where L is Lipschitz constant)
    # Values < 1.0 are more conservative (slower but safer), > 1.0 are aggressive
    k0_scale: float = 0.5  # Default to 0.5 for stability
    
    # Scalar box bounds (for problems like SVM with 0 <= x <= C). Since v0.5.0
    # these are NOT enforced by a terminal clip: each bound facet participates
    # in the unified projection sweep as an implicit unit-normal constraint
    # (facet spike = exact single-coordinate correction). Box-only problems
    # (m == 0) use the exact vectorized box projection instead.
    lower_bound: float = None  # If set, enforce x >= lower_bound
    upper_bound: float = None  # If set, enforce x <= upper_bound
    
    # Convergence detection
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)

    # Instrumentation / performance
    # record_trajectory=True (default) keeps the full iterate trajectory and
    # per-projection spike-event metadata -- needed for figures and diagnostics.
    # record_trajectory=False runs the lean solve path: no trajectory or spike
    # storage, one fused A@x matvec per iteration. Use this for benchmarking.
    record_trajectory: bool = True

    # If False, do not retain per-projection spike_info (delta_x, constraints,
    # violations) across outer iterations. Saves O(max_iterations *
    # max_projection_iters) memory which is the dominant cost for long runs at
    # large projection budgets. Default True preserves backward-compat with
    # figure / illustration scripts that read result.spike_*.
    record_spike_history: bool = True

    # Solve backend (euler + adaptive projection only; implies
    # record_trajectory=False for every compiled variant):
    #   'python'   -- pure-NumPy reference
    #   'c'        -- compiled pybind11 kernel; auto-uses OpenMP multicore matvec
    #                 when the build supports it, else single-threaded
    #   'c_serial' -- compiled kernel, forced single-threaded (SIMD only)
    #   'c_openmp' -- compiled kernel, forced OpenMP multicore (raises if the
    #                 build lacks OpenMP)
    # The three C variants are numerically identical; only the matvec threading
    # differs (the Euler recurrence + greedy projection are inherently serial).
    backend: str = 'python'

    # Problem transform (the "transform axis"): an explicit, backend-agnostic
    # rewrite of the problem that is solved in transformed coordinates and mapped
    # back. None (default) = canonical solve. A name ('eigenbasis') or a
    # snn_opt.transforms.Transform instance opts in. Composes with any backend;
    # implies the lean result (final-state fields only). See snn_opt.transforms.
    transform: Optional[Union[str, "object"]] = None

    # Constant-memory observation of committed adaptive-projection events.
    # Default False preserves the numerical and allocation behavior of v0.5.
    # When enabled, SolverResult reports per-row and per-coordinate counts plus
    # a canonical event-stream digest and its first/last candidate IDs.
    observe_projection_events: bool = False


@dataclass
class OptimizationProblem:
    """
    Defines a constrained quadratic/linear program.
    
    Minimize: (1/2) x^T A x + b^T x
    Subject to: C x + d <= 0
    
    Parameters
    ----------
    A : ndarray, shape (n, n)
        Hessian matrix (for QP) or zeros (for LP)
    b : ndarray, shape (n,)
        Linear cost vector
    C : ndarray, shape (m, n)
        Constraint matrix (m constraints, n variables)
    d : ndarray, shape (m,)
        Constraint offset vector
    """
    A: np.ndarray
    b: np.ndarray
    C: np.ndarray
    d: np.ndarray
    
    def __post_init__(self):
        """Validate problem dimensions."""
        n = self.A.shape[0]
        assert self.A.shape == (n, n), "A must be square"
        assert self.b.shape == (n,), f"b must have shape ({n},)"
        assert self.C.shape[1] == n, f"C must have {n} columns"
        assert self.C.shape[0] == self.d.shape[0], "C rows must match d length"
    
    @property
    def n_vars(self) -> int:
        """Number of optimization variables."""
        return self.A.shape[0]
    
    @property
    def n_constraints(self) -> int:
        """Number of inequality constraints."""
        return self.C.shape[0]
    
    def objective(self, x: np.ndarray) -> float:
        """Evaluate objective function value."""
        return 0.5 * x.T @ self.A @ x + self.b.T @ x
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Evaluate objective gradient."""
        return self.A @ x + self.b
    
    def constraint_values(self, x: np.ndarray) -> np.ndarray:
        """Evaluate constraint function g(x) = Cx + d."""
        return self.C @ x + self.d
    
    def is_feasible(self, x: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if x satisfies all constraints."""
        return np.all(self.constraint_values(x) <= tol)
    
    def max_violation(self, x: np.ndarray) -> float:
        """Return maximum constraint violation (positive means violated, 0 means satisfied)."""
        if self.n_constraints == 0:
            return 0.0
        g = self.constraint_values(x)
        return np.max(np.maximum(g, 0.0))


@dataclass
class KKTCertificate:
    """Scale-invariant KKT residual at a point (see ``optimality_test="kkt"``).

    ``residual = hypot(stationarity, complementarity)`` where both components
    carry gradient units:

        stationarity     ||grad f(x) + N^T mu||_2
        complementarity  |s|^T mu / max(1, ||x||_2)

    with ``mu >= 0`` fitted by one augmented nonnegative least-squares over
    ALL nondegenerate facets (explicit rows and box bounds, unit-normalized),
    no active-set window. ``scale = max(||A x||, ||b||, ||N^T mu||)`` and the
    acceptance threshold is ``tolerance = kkt_abs_tol + kkt_rel_tol * scale``.
    ``fit_status`` is ``"ok"``, ``"non_finite"``, or ``"fit_failed"``; any
    non-``"ok"`` status fails the convergence gate closed.
    """
    residual: float
    stationarity: float
    complementarity: float
    scale: float
    tolerance: float
    fit_status: str

    @property
    def passed(self) -> bool:
        return (self.fit_status == "ok"
                and np.isfinite(self.residual)
                and self.residual <= self.tolerance)


@dataclass
class SolverResult:
    """
    Results from optimization solve.
    
    Attributes
    ----------
    t : ndarray
        Time points / iteration indices
    X : ndarray, shape (len(t), n)
        State trajectory
    objective_values : ndarray
        Objective function values along trajectory
    constraint_violations : ndarray
        Maximum constraint violation at each time point
    n_projections : int
        Total number of projection iterations
    converged : bool
        Whether convergence criteria were satisfied
    convergence_reason : str
        Description of why solver stopped
    iterations_used : int
        Actual number of iterations executed
    final_x : ndarray
        Final solution vector
    final_objective : float
        Final objective value
    final_proj_grad_norm : float
        Projected gradient norm at final solution
    spike_times : ndarray
        Time stamps when projection spikes were applied
    spike_deltas : ndarray
        Projection displacements (len(spike_times) × n)
    spike_norms : ndarray
        L2 norm of each spike displacement
    spike_constraints : list of ndarray
        Indices of constraints that were active for each spike event. Row j is
        index j; box facets use the frozen convention: lower facet of coordinate
        i is m + i, upper facet is m + n + i.
    spike_violation_values : list of ndarray
        Positive constraint residuals at the moment each spike was applied
    total_projection_distance : float
        Sum of norms of all spike displacements (cumulative distance)
    joint_feasible : bool
        Whether the final point satisfies rows AND box to feasibility_tol
        (geometric distances). This is THE feasibility verdict; the legacy
        constraint_violations covers rows only, in raw (un-normalized) units.
    max_violation_rows_raw : float
        Max positive row residual max_j (c_j^T x + d_j)_+ in raw user units.
    max_distance_rows : float
        Max row violation as a Euclidean distance (raw residual / ||c_j||).
    max_violation_box : float
        Max box-bound violation (already a distance; unit normals).
    stationarity_residual : float
        LEGACY diagnostic (pre-v0.6): max of stationarity, complementarity,
        and primal defects on an eps-active set with
        eps = max(10 * constraint_tol, 3 * k0 * ||grad f(x)||). The three
        terms carry different units and the value can depend on constraint
        row order at rank-deficient active sets, so it is NOT the
        convergence criterion; see kkt_residual for the authoritative
        certificate. Retained for one compatibility release.
    optimality_test : str
        Which optimality criterion governed the `converged` flag:
        "kkt" (default), "legacy_projected_gradient", or "none".
    kkt_residual : float
        Scale-invariant KKT certificate at the final point:
        hypot(kkt_stationarity_residual, kkt_complementarity_residual), from
        one augmented NNLS over ALL unit-normalized facets (no active-set
        window). Unique under multiplier non-uniqueness; invariant to
        constraint row order, row duplication, and positive objective
        scaling. NaN when the fit failed (see kkt_fit_status).
    kkt_stationarity_residual : float
        ||grad f(x) + N^T mu||_2 component of the certificate.
    kkt_complementarity_residual : float
        |s|^T mu / max(1, ||x||) component (gradient units).
    kkt_scale : float
        Scale reference max(||A x||, ||b||, ||N^T mu||) used by the
        acceptance threshold.
    kkt_tolerance : float
        Acceptance threshold kkt_abs_tol + kkt_rel_tol * kkt_scale that was
        in force at the final point.
    kkt_fit_status : str
        "ok", "non_finite", or "fit_failed". Anything but "ok" means the
        certificate could not be evaluated and convergence failed closed.
    projection_budget_exhausted : bool
        True when an inner sweep hit the safety cap before reaching joint
        tolerance; the solve is aborted at that iteration with
        convergence_reason='projection_budget_exhausted'.
    explicit_row_event_counts : ndarray or None
        Per-explicit-row committed projection-event counts. None when the
        opt-in observer is disabled.
    implicit_lower_event_counts, implicit_upper_event_counts : ndarray or None
        Per-coordinate implicit-bound event counts. None when disabled.
    explicit_row_events, implicit_lower_events, implicit_upper_events : int or None
        Totals of the corresponding count arrays.
    projection_event_digest : int or None
        Canonical unsigned 64-bit digest of committed candidate IDs, in event
        order, including outer-iteration and within-sweep ordinal tokens. The
        empty observed stream has the fixed offset-basis digest.
    projection_event_digest_algorithm : str or None
        Frozen digest identifier, `fnv1a64-word-v2`, when observation is on.
    observed_total_projection_distance : float or None
        Sum of Euclidean norms of all committed corrections, accumulated by
        the optional observer. Distinct from the legacy
        `total_projection_distance`, which depends on retained spike history
        and is therefore zero on lean solves.
    projection_first_candidate_id, projection_last_candidate_id : int or None
        First and last canonical candidate IDs. Rows use j, implicit lower
        facets m+i, and implicit upper facets m+n+i. None for an empty stream
        or when observation is disabled.
    projection_cap_rechecks : int or None
        Number of inner projection sweeps that consumed the configured cap and
        therefore performed a fresh joint-violation recheck. A recheck can pass
        and allow the outer solve to continue, so this is distinct from the
        terminal `projection_budget_exhausted` flag. None when observation is
        disabled.
    """
    t: np.ndarray
    X: np.ndarray
    objective_values: np.ndarray
    constraint_violations: np.ndarray
    n_projections: int
    converged: bool
    convergence_reason: str
    iterations_used: int
    final_x: np.ndarray
    final_objective: float
    final_proj_grad_norm: float
    spike_times: np.ndarray
    spike_deltas: np.ndarray
    spike_norms: np.ndarray
    spike_constraints: List[np.ndarray]
    spike_violation_values: List[np.ndarray]
    total_projection_distance: float
    joint_feasible: bool = False
    max_violation_rows_raw: float = 0.0
    max_distance_rows: float = 0.0
    max_violation_box: float = 0.0
    stationarity_residual: float = float("nan")
    projection_budget_exhausted: bool = False
    # --- v0.6.0 scale-invariant KKT certificate at the final point ---------
    # Always computed at the final iterate regardless of optimality_test, so
    # every solve reports the same authoritative optimality diagnostic. The
    # convergence FLAG uses it only when optimality_test="kkt".
    optimality_test: str = "kkt"
    kkt_residual: float = float("nan")
    kkt_stationarity_residual: float = float("nan")
    kkt_complementarity_residual: float = float("nan")
    kkt_scale: float = float("nan")
    kkt_tolerance: float = float("nan")
    kkt_fit_status: str = "not_computed"
    explicit_row_event_counts: Optional[np.ndarray] = None
    implicit_lower_event_counts: Optional[np.ndarray] = None
    implicit_upper_event_counts: Optional[np.ndarray] = None
    explicit_row_events: Optional[int] = None
    implicit_lower_events: Optional[int] = None
    implicit_upper_events: Optional[int] = None
    projection_event_digest: Optional[int] = None
    projection_event_digest_algorithm: Optional[str] = None
    observed_total_projection_distance: Optional[float] = None
    projection_first_candidate_id: Optional[int] = None
    projection_last_candidate_id: Optional[int] = None
    projection_cap_rechecks: Optional[int] = None
    
    def summary(self) -> str:
        """Generate summary statistics string."""
        lines = [
            "Solver Result Summary:",
            f"  Converged: {self.converged}",
            f"  Convergence reason: {self.convergence_reason}",
            f"  Iterations used: {self.iterations_used}",
            f"  Final objective: {self.final_objective:.6e}",
            f"  Joint feasible: {self.joint_feasible} "
            f"(rows dist={self.max_distance_rows:.2e}, box={self.max_violation_box:.2e})",
            f"  KKT certificate: residual={self.kkt_residual:.6e} "
            f"(tol={self.kkt_tolerance:.2e}, scale={self.kkt_scale:.2e}, "
            f"status={self.kkt_fit_status})",
            f"  eps-KKT residual (legacy NNLS): {self.stationarity_residual:.6e}",
            f"  Final proj. gradient norm (legacy heuristic): {self.final_proj_grad_norm:.6e}",
            f"  Max row violation (raw): {np.max(self.constraint_violations):.6e}",
            f"  Total projections: {self.n_projections}",
            f"  Total spikes recorded: {len(self.spike_times)}",
        ]
        if self.projection_budget_exhausted:
            lines.append("  WARNING: projection budget exhausted (solve aborted)")

        if len(self.spike_norms) > 0:
            lines.append(f"  Avg spike norm: {self.spike_norms.mean():.6e}")
        lines.append(f"  Total projection distance: {self.total_projection_distance:.6e}")
        lines.append(f"  Final solution: {self.final_x}")
        return "\n".join(lines) + "\n"


class SNNSolver:
    """
    SNN-inspired solver for constrained convex optimization.
    
    Implements gradient descent with discrete boundary projections to solve
    quadratic and linear programs with linear inequality constraints.
    
    Parameters
    ----------
    problem : OptimizationProblem
        The optimization problem to solve
    config : SolverConfig, optional
        Solver configuration parameters
    """
    
    def __init__(self, problem: OptimizationProblem, config: Optional[SolverConfig] = None):
        self.problem = problem
        self.config = config or SolverConfig()

        # Diagonal Hessian fast-path hint. None = dense A (the usual case). Set
        # to the length-n diagonal by the transform path (e.g. eigenbasis) so the
        # O(n^2) A @ x step collapses to an O(n) elementwise product.
        self._a_diag = None

        # Auto-compute k0 from Lipschitz constant if not provided
        if self.config.k0 is None:
            self._k0 = self._compute_adaptive_k0()
        else:
            self._k0 = self.config.k0
        
        # Pre-compute constraint norms squared for efficiency
        if self.problem.n_constraints > 0:
            C = self.problem.C
            if _issparse(C):
                self._c_norms_sq = np.asarray(C.multiply(C).sum(axis=1)).ravel()
            else:
                self._c_norms_sq = np.sum(C ** 2, axis=1)
        else:
            self._c_norms_sq = np.array([])

        # Normalized-distance selection scales: dist_j = (c_j^T x + d_j) / ||c_j||.
        # Zero rows are screened here: with c_j = 0 the row is either redundant
        # (d_j <= tol: never violated, scale 0 makes it inert in selection) or a
        # certificate that the problem is infeasible (d_j > tol: 0 <= -d_j < 0).
        self._c_norms = np.sqrt(self._c_norms_sq)
        with np.errstate(divide="ignore"):
            self._row_scale = np.where(self._c_norms > 1e-12,
                                       1.0 / np.maximum(self._c_norms, 1e-300), 0.0)
        if self.problem.n_constraints > 0:
            degenerate = self._c_norms <= 1e-12
            if np.any(degenerate & (np.asarray(self.problem.d).ravel()
                                    > self.config.constraint_tol)):
                bad = int(np.argmax(degenerate & (np.asarray(self.problem.d).ravel()
                                                  > self.config.constraint_tol)))
                raise ValueError(
                    f"constraint row {bad} has a zero normal and d > 0: the "
                    f"problem is certifiably infeasible (0 <= {-self.problem.d[bad]})")

        # The legacy fixed-step projection predates the unified facet sweep and
        # cannot enforce box bounds correctly; fail fast rather than silently
        # reintroduce the clip-after-project defect.
        if (self.config.observe_projection_events
                and self.config.projection_method != 'adaptive'):
            raise ValueError(
                "observe_projection_events=True supports "
                "projection_method='adaptive' only")
        if (self.config.projection_method == 'fixed'
                and (self.config.lower_bound is not None
                     or self.config.upper_bound is not None)):
            raise ValueError(
                "projection_method='fixed' does not support box bounds since "
                "v0.5.0 (the terminal clip was removed as a correctness defect); "
                "use projection_method='adaptive'")

        # Inner-sweep safety watchdog (see SolverConfig.max_projection_iters).
        n_facets = ((self.problem.n_vars if self.config.lower_bound is not None else 0)
                    + (self.problem.n_vars if self.config.upper_bound is not None else 0))
        if self.config.max_projection_iters is None:
            self._proj_cap = max(1000, 10 * (self.problem.n_constraints + n_facets))
        else:
            self._proj_cap = int(self.config.max_projection_iters)
        self._projection_budget_exhausted = False

        # Pre-compute the constraint Gram matrix G = C C^T (the constraint-
        # coupling / recurrent matrix). Both the compiled C kernel and the
        # Python adaptive projection use it for the event-driven update: a
        # projection spike on constraint j applies the lateral update
        # g <- g - k1 * G[:,j] (O(m)) instead of recomputing C x (O(m*n)).
        # Built for dense C with m <= _MAX_GRAM_M; sparse C or larger m keeps
        # the residual-recompute path (and backend='c' then raises).
        self._c_gram = None
        if (self.problem.n_constraints > 0
                and not _issparse(self.problem.C)
                and self.problem.n_constraints <= _MAX_GRAM_M):
            C = np.asarray(self.problem.C, dtype=float)
            self._c_gram = np.ascontiguousarray(C @ C.T, dtype=np.float64)

        # Storage for trajectory segments
        self._t_segments: List[np.ndarray] = []
        self._x_segments: List[np.ndarray] = []
        self._n_projections = 0
        self._spike_times: List[float] = []
        self._spike_deltas: List[np.ndarray] = []
        self._spike_constraints: List[np.ndarray] = []
        self._spike_violation_values: List[np.ndarray] = []
        self._reset_projection_event_observer()
        
        # Convergence tracking
        self._converged = False
        self._convergence_reason = "max_iterations"
        self._iterations_used = 0

    def _reset_projection_event_observer(self) -> None:
        """Reset the optional constant-memory projection-event observer."""
        if self.config.observe_projection_events:
            self._explicit_row_event_counts = np.zeros(
                self.problem.n_constraints, dtype=np.int64)
            self._implicit_lower_event_counts = np.zeros(
                self.problem.n_vars, dtype=np.int64)
            self._implicit_upper_event_counts = np.zeros(
                self.problem.n_vars, dtype=np.int64)
            self._projection_event_digest = _EVENT_DIGEST_OFFSET
            self._observed_total_projection_distance = 0.0
            self._projection_first_candidate_id = None
            self._projection_last_candidate_id = None
            self._projection_cap_rechecks = 0
        else:
            self._explicit_row_event_counts = None
            self._implicit_lower_event_counts = None
            self._implicit_upper_event_counts = None
            self._projection_event_digest = None
            self._observed_total_projection_distance = None
            self._projection_first_candidate_id = None
            self._projection_last_candidate_id = None
            self._projection_cap_rechecks = None

    def _observe_projection_event(self, kind: str, index: int,
                                  outer_iteration: int, ordinal: int,
                                  correction_norm: float) -> None:
        """Record one already-committed adaptive projection event."""
        if self._explicit_row_event_counts is None:
            return

        m = self.problem.n_constraints
        n = self.problem.n_vars
        if kind == "row":
            self._explicit_row_event_counts[index] += 1
            candidate_id = index
        elif kind == "lo":
            self._implicit_lower_event_counts[index] += 1
            candidate_id = m + index
        elif kind == "hi":
            self._implicit_upper_event_counts[index] += 1
            candidate_id = m + n + index
        else:  # pragma: no cover - internal invariant
            raise AssertionError(f"unknown projection event kind {kind!r}")

        if self._projection_first_candidate_id is None:
            self._projection_first_candidate_id = int(candidate_id)
        self._projection_last_candidate_id = int(candidate_id)
        for word in (outer_iteration, ordinal, candidate_id):
            self._projection_event_digest = (
                (self._projection_event_digest ^ (int(word) + 1))
                * _EVENT_DIGEST_PRIME
            ) & _UINT64_MASK
        self._observed_total_projection_distance += float(correction_norm)

    def _projection_event_result_fields(self) -> dict:
        """Return detached observer fields for SolverResult construction."""
        if self._explicit_row_event_counts is None:
            return {
                "explicit_row_event_counts": None,
                "implicit_lower_event_counts": None,
                "implicit_upper_event_counts": None,
                "explicit_row_events": None,
                "implicit_lower_events": None,
                "implicit_upper_events": None,
                "projection_event_digest": None,
                "projection_event_digest_algorithm": None,
                "observed_total_projection_distance": None,
                "projection_first_candidate_id": None,
                "projection_last_candidate_id": None,
                "projection_cap_rechecks": None,
            }

        row_counts = self._explicit_row_event_counts.copy()
        lower_counts = self._implicit_lower_event_counts.copy()
        upper_counts = self._implicit_upper_event_counts.copy()
        return {
            "explicit_row_event_counts": row_counts,
            "implicit_lower_event_counts": lower_counts,
            "implicit_upper_event_counts": upper_counts,
            "explicit_row_events": int(row_counts.sum(dtype=np.int64)),
            "implicit_lower_events": int(lower_counts.sum(dtype=np.int64)),
            "implicit_upper_events": int(upper_counts.sum(dtype=np.int64)),
            "projection_event_digest": int(self._projection_event_digest),
            "projection_event_digest_algorithm": _EVENT_DIGEST_ALGORITHM,
            "observed_total_projection_distance": float(
                self._observed_total_projection_distance),
            "projection_first_candidate_id": self._projection_first_candidate_id,
            "projection_last_candidate_id": self._projection_last_candidate_id,
            "projection_cap_rechecks": int(self._projection_cap_rechecks),
        }
    
    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """
        Exact projection onto the box (vectorized clip).

        Since v0.5.0 this is ONLY the box-only (m == 0) fast path of the unified
        projection: with no general rows the box projection is separable and
        exact, so a single vectorized clip is the correct projection. For mixed
        problems the bound facets are handled inside the sweep instead --
        clipping AFTER the halfspace sweep was the clip-after-project defect.
        """
        if self.config.lower_bound is not None:
            x = np.maximum(x, self.config.lower_bound)
        if self.config.upper_bound is not None:
            x = np.minimum(x, self.config.upper_bound)
        return x
    
    def _violation_split(self, x: np.ndarray) -> Tuple[float, float, float]:
        """(max raw row violation, max row violation DISTANCE, max box violation).

        Row distances divide the raw residual by ||c_j||; box violations are
        already distances (unit facet normals). All three are 0 when satisfied.
        """
        raw = dist = 0.0
        if self.problem.n_constraints > 0:
            g = np.asarray(self.problem.constraint_values(x)).ravel()
            raw = float(np.max(np.maximum(g, 0.0)))
            dist = float(np.max(np.maximum(g * self._row_scale, 0.0)))
        box = 0.0
        if self.config.lower_bound is not None:
            box = max(box, float(np.max(np.maximum(self.config.lower_bound - x, 0.0))))
        if self.config.upper_bound is not None:
            box = max(box, float(np.max(np.maximum(x - self.config.upper_bound, 0.0))))
        return raw, dist, box

    def _joint_max_violation(self, x: np.ndarray) -> float:
        """Joint geometric infeasibility: max of row distances and box violations.

        This is the feasibility quantity used by the convergence gate and the
        result report. The legacy `OptimizationProblem.max_violation` covers
        rows only, in raw units, and never sees the box.
        """
        _, dist, box = self._violation_split(x)
        return max(dist, box)

    def _stationarity_residual(self, x: np.ndarray) -> float:
        """eps-KKT residual at x (host-side instrumentation).

        Returns max of the three KKT defects, measured on the eps-active set:

            stationarity     min_{mu >= 0} ||grad f(x) + N^T mu||
            complementarity  max_i mu_i * s_i
            primal           max_i (-s_i)_+

        with s_i the normalized slack of row i and

            eps = max(10 * constraint_tol, 3 * k0 * ||grad f(x)||).

        The window MUST scale with k0: between projections the iterate coasts
        O(k0) off each facet, so a fixed window drops truly-active rows and the
        NNLS then reports ~||grad f(x)|| regardless of solution quality (the
        value is then flat in k0 and anti-correlated with the true error). The
        complementarity term is what keeps the widened window honest: without
        it the fit may load a slack row's normal and report ~0 at a point that
        is not optimal. Returns ||grad|| when nothing is eps-active, NaN if the
        fit fails.
        """
        grad = np.asarray(self.problem.gradient(x), dtype=float).ravel()
        n = self.problem.n_vars
        # Detection window tied to the limit-cycle amplitude, not to a constant.
        active_tol = max(self.config.constraint_tol * 10,
                         3.0 * self._k0 * float(np.linalg.norm(grad)))
        normals, slacks = [], []
        if self.problem.n_constraints > 0:
            g = np.asarray(self.problem.constraint_values(x)).ravel()
            dist = g * self._row_scale          # signed; > 0 means violated
            for j in np.nonzero(dist >= -active_tol)[0]:
                if self._c_norms[j] <= 1e-12:
                    continue
                c_j = self.problem.C[j]
                c_j = (np.asarray(c_j.todense()).ravel() if _issparse(c_j)
                       else np.asarray(c_j, dtype=float).ravel())
                normals.append(c_j * self._row_scale[j])
                slacks.append(-dist[j])
        if self.config.lower_bound is not None:
            for i in np.nonzero(self.config.lower_bound - x >= -active_tol)[0]:
                e = np.zeros(n); e[i] = -1.0
                normals.append(e); slacks.append(x[i] - self.config.lower_bound)
        if self.config.upper_bound is not None:
            for i in np.nonzero(x - self.config.upper_bound >= -active_tol)[0]:
                e = np.zeros(n); e[i] = 1.0
                normals.append(e); slacks.append(self.config.upper_bound - x[i])
        if not normals:
            return float(np.linalg.norm(grad))
        try:
            from scipy.optimize import nnls
            N = np.asarray(normals)              # (k, n)
            s = np.asarray(slacks, dtype=float)  # (k,) signed slack, >=0 feasible
            mu, rnorm = nnls(N.T, -grad)         # min ||N^T mu + grad||, mu >= 0
            comp = float(np.max(mu * np.abs(s))) if mu.size else 0.0
            prim = float(max(0.0, np.max(-s)))
            return float(max(rnorm, comp, prim))
        except Exception:
            return float("nan")

    def _kkt_result_fields(self, final_x: np.ndarray) -> dict:
        """KKT-certificate result fields at the final point (all backends)."""
        cert = self._compute_kkt_certificate(np.asarray(final_x, dtype=float))
        return dict(
            kkt_residual=cert.residual,
            kkt_stationarity_residual=cert.stationarity,
            kkt_complementarity_residual=cert.complementarity,
            kkt_scale=cert.scale,
            kkt_tolerance=cert.tolerance,
            kkt_fit_status=cert.fit_status,
        )

    def _certificate_facets(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unified unit-normalized facet family at x: (N, s).

        N stacks every nondegenerate facet normal as a row (explicit rows
        divided by ||c_j||, box facets as +/- e_i); s holds the matching
        signed slacks (>= 0 feasible). Both empty when there are no facets.
        """
        n = self.problem.n_vars
        normals, slacks = [], []
        if self.problem.n_constraints > 0:
            g = np.asarray(self.problem.constraint_values(x), dtype=float).ravel()
            C = self.problem.C
            for j in range(self.problem.n_constraints):
                if self._c_norms[j] <= 1e-12:
                    continue  # degenerate zero row; preflight owns infeasibility
                c_j = C[j]
                c_j = (np.asarray(c_j.todense()).ravel() if _issparse(c_j)
                       else np.asarray(c_j, dtype=float).ravel())
                normals.append(c_j * self._row_scale[j])
                slacks.append(-g[j] * self._row_scale[j])
        if self.config.lower_bound is not None:
            lo = float(self.config.lower_bound)
            for i in range(n):
                e = np.zeros(n); e[i] = -1.0
                normals.append(e); slacks.append(x[i] - lo)
        if self.config.upper_bound is not None:
            up = float(self.config.upper_bound)
            for i in range(n):
                e = np.zeros(n); e[i] = 1.0
                normals.append(e); slacks.append(up - x[i])
        if not normals:
            return np.empty((0, n)), np.empty((0,))
        return np.asarray(normals, dtype=float), np.asarray(slacks, dtype=float)

    def _compute_kkt_certificate(self, x: np.ndarray) -> KKTCertificate:
        """Scale-invariant KKT certificate at x (host-side; all backends).

        One augmented NNLS over ALL nondegenerate facets:

            mu = argmin_{mu >= 0} || [N^T; |s|^T/lx] mu - [-g; 0] ||_2,
            lx = max(1, ||x||_2)

        The appended complementarity row makes loading a slack facet's normal
        expensive inside the fit itself, so no active-set window is needed
        and the residual is independent of the integrator step k0. The
        components are

            stationarity     = ||g + N^T mu||_2      (gradient units)
            complementarity  = |s|^T mu / lx          (gradient units)
            residual         = hypot(stationarity, complementarity)

        residual is unique even when mu is not (the projected augmented
        vector is unique), so constraint row order and duplicated rows cannot
        change the convergence decision. With no facets the residual reduces
        to ||g||. Failures (non-finite data, NNLS failure) return fit_status
        != "ok" and the convergence gate fails closed.
        """
        conv = self.config.convergence
        g = np.asarray(self.problem.gradient(x), dtype=float).ravel()
        b = np.asarray(self.problem.b, dtype=float).ravel()
        Ax = g - b
        scale_gb = max(float(np.linalg.norm(Ax)), float(np.linalg.norm(b)))

        def _cert(res, stat, comp, scale, status):
            tol = conv.kkt_abs_tol + conv.kkt_rel_tol * scale
            return KKTCertificate(residual=res, stationarity=stat,
                                  complementarity=comp, scale=scale,
                                  tolerance=tol, fit_status=status)

        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(g)):
            return _cert(float("nan"), float("nan"), float("nan"),
                         scale_gb, "non_finite")

        N, s = self._certificate_facets(x)
        if N.shape[0] == 0:
            gnorm = float(np.linalg.norm(g))
            return _cert(gnorm, gnorm, 0.0, scale_gb, "ok")
        if not np.all(np.isfinite(s)):
            return _cert(float("nan"), float("nan"), float("nan"),
                         scale_gb, "non_finite")

        lx = max(1.0, float(np.linalg.norm(x)))
        abs_s = np.abs(s)
        M = np.concatenate([N.T, (abs_s / lx)[None, :]], axis=0)  # (n+1, p)
        rhs = np.concatenate([-g, [0.0]])
        try:
            from scipy.optimize import nnls
            mu, _ = nnls(M, rhs)
        except Exception:
            return _cert(float("nan"), float("nan"), float("nan"),
                         scale_gb, "fit_failed")
        if not np.all(np.isfinite(mu)):
            return _cert(float("nan"), float("nan"), float("nan"),
                         scale_gb, "fit_failed")
        stat = float(np.linalg.norm(g + N.T @ mu))
        comp = float(abs_s @ mu / lx)
        resid = float(np.hypot(stat, comp))
        scale = max(scale_gb, float(np.linalg.norm(N.T @ mu)))
        return _cert(resid, stat, comp, scale, "ok")

    def _compute_adaptive_k0(self) -> float:
        """
        Compute adaptive step size based on Lipschitz constant of the gradient.
        
        For QP: f(x) = (1/2) x^T A x + b^T x
        Gradient: ∇f(x) = A x + b
        Lipschitz constant: L = ||A||_2 = λ_max(A) (largest eigenvalue)
        
        Safe step size: k0 = 1/L ensures convergence for convex QP.
        We use k0 = k0_scale / L for additional stability margin.
        """
        A = self.problem.A

        # For zero Hessian (linear program), use default step
        if _issparse(A):
            if A.nnz == 0:
                return 0.01
        elif np.allclose(A, 0):
            return 0.01

        # Compute largest eigenvalue (Lipschitz constant)
        if _issparse(A):
            from scipy.sparse.linalg import eigsh
            try:
                eigenvalues, _ = eigsh(A.astype(float), k=1, which='LM')
                L = np.abs(eigenvalues[0])
            except Exception:
                # Fallback: use Frobenius norm as upper bound
                L = _sp.linalg.norm(A, 'fro')
        elif np.allclose(A, A.T):
            # For symmetric dense matrices, use eigvalsh (faster)
            eigenvalues = np.linalg.eigvalsh(A)
            L = np.max(np.abs(eigenvalues))
        else:
            # For non-symmetric dense, use spectral norm
            L = np.linalg.norm(A, 2)
        
        # Avoid division by zero
        if L < 1e-10:
            return 0.01
        
        # Safe step size with scaling factor
        k0 = self.config.k0_scale / L
        
        return k0
    
    def _compute_projected_gradient_norm(self, x: np.ndarray) -> float:
        """
        Compute norm of gradient projected onto feasible descent directions.
        
        LEGACY heuristic diagnostic (pre-v0.6 convergence criterion; still
        drives optimality_test="legacy_projected_gradient"). It removes each
        near-active facet's gradient component INDEPENDENTLY (no joint fit),
        so at a constrained optimum with correlated active normals the
        cross-terms leave an O(mu * cos-angle) residue: the value is
        structurally nonzero at valid optima unless the active normals are
        mutually orthogonal, and it scales with the objective. Use the
        scale-invariant KKT certificate (kkt_residual) as the optimality
        measure; this quantity is retained for diagnostics and backward
        compatibility only.

        Mathematical basis:
        - Constraint: c_j · x + d_j ≤ 0
        - At boundary: c_j · x + d_j = 0
        - Descent direction: -∇f
        - Feasible if: c_j · (-∇f) ≤ 0, i.e., c_j · ∇f ≥ 0
        - If c_j · ∇f < 0: descent would violate, so project out this component
        """
        grad = self.problem.gradient(x)
        proj_grad = grad.copy()
        
        # Handle linear inequality constraints
        if self.problem.n_constraints > 0:
            g = self.problem.constraint_values(x)
            active_tol = self.config.constraint_tol * 10  # Slightly larger for "near-active"
            
            for j in range(self.problem.n_constraints):
                # Check if constraint is active (at or very near boundary)
                if g[j] >= -active_tol:
                    c_j = self.problem.C[j]
                    # Convert sparse row to dense 1D array
                    if _issparse(c_j):
                        c_j = np.asarray(c_j.todense()).ravel()
                    else:
                        c_j = np.asarray(c_j).ravel()
                    c_norm_sq = self._c_norms_sq[j]

                    if c_norm_sq < 1e-12:
                        continue

                    # Component of gradient in constraint normal direction
                    component = np.dot(grad, c_j) / c_norm_sq

                    # If component < 0, descent (-grad) would push in +c_j direction (violating)
                    # So we need to remove this component from the gradient
                    if component < 0:
                        proj_grad = proj_grad - component * c_j
        
        # Handle box constraints
        if self.config.lower_bound is not None:
            lower_tol = self.config.lower_bound + self.config.constraint_tol * 10
            at_lower = x <= lower_tol
            # At lower bound: can only increase. If grad > 0, descent would decrease → infeasible
            proj_grad[at_lower & (grad > 0)] = 0
        
        if self.config.upper_bound is not None:
            upper_tol = self.config.upper_bound - self.config.constraint_tol * 10
            at_upper = x >= upper_tol
            # At upper bound: can only decrease. If grad < 0, descent would increase → infeasible
            proj_grad[at_upper & (grad < 0)] = 0
        
        return np.linalg.norm(proj_grad)
    
    def _check_convergence(self, iteration: int, x_curr: np.ndarray, 
                           obj_history: List[float], x_history: List[np.ndarray]) -> Tuple[bool, str, bool]:
        """
        Check multiple convergence criteria with safeguards against false detection.
        
        Returns (converged: bool, reason: str, should_check: bool)
        
        The third return value indicates whether this was a check iteration.
        If False, the caller should NOT reset the patience counter.
        """
        conv_cfg = self.config.convergence
        
        # Don't check before minimum iterations
        if iteration < conv_cfg.min_iterations:
            return False, "", False  # Not a check iteration
        
        # Only check at specified intervals
        if iteration % conv_cfg.check_every != 0:
            return False, "", False  # Not a check iteration
        
        # 1. Feasibility gate first (if required) -- JOINT: rows + box. When
        # infeasible, the outcome is "not converged" regardless of the other
        # criteria, so nothing else needs evaluating.
        if conv_cfg.require_feasibility:
            max_viol = self._joint_max_violation(x_curr)
            if max_viol > conv_cfg.feasibility_tol:
                return False, "still_infeasible", True  # Check happened but failed

        # 2. Cheap criteria (objective plateau, solution stability). ALL
        # enabled cheap criteria must pass BEFORE the optimality test runs, so
        # the NNLS certificate is a near-termination cost, not an
        # every-checkpoint cost.
        cheap_ok, reasons_met = self._cheap_criteria_pass(x_curr, obj_history,
                                                          x_history)
        if not cheap_ok:
            return False, "", True

        # 3. Authoritative optimality test.
        opt_ok, opt_reason = self._optimality_criterion_pass(x_curr)
        if not opt_ok:
            return False, "", True
        if opt_reason:
            reasons_met = [opt_reason] + reasons_met

        if not reasons_met:
            # Nothing is enabled at all: never report convergence.
            return False, "", True
        return True, f"converged({'; '.join(reasons_met)})", True

    def _cheap_criteria_pass(self, x_curr: np.ndarray,
                             obj_history: List[float],
                             x_history: List[np.ndarray]
                             ) -> Tuple[bool, List[str]]:
        """Evaluate the enabled cheap criteria. Returns (all_pass, reasons).

        A criterion whose window is not yet filled counts as failed (matching
        the historical behavior of requiring the full window).
        """
        conv_cfg = self.config.convergence
        reasons: List[str] = []

        if conv_cfg.use_objective_plateau:
            if len(obj_history) < conv_cfg.window_size:
                return False, reasons
            window = obj_history[-conv_cfg.window_size:]
            obj_range = max(window) - min(window)
            obj_scale = max(abs(window[-1]), 1e-10)
            obj_rel_change = obj_range / obj_scale
            if obj_rel_change >= conv_cfg.obj_rel_tol:
                return False, reasons
            reasons.append(f"obj_plateau(range={obj_rel_change:.2e})")

        if conv_cfg.use_solution_stable:
            if len(x_history) < conv_cfg.window_size:
                return False, reasons
            recent = x_history[-conv_cfg.window_size:]
            x_norm = max(np.linalg.norm(x_curr), 1e-10)
            max_dist = max(np.linalg.norm(x - x_curr) for x in recent)
            x_rel_change = max_dist / x_norm
            if x_rel_change >= conv_cfg.x_rel_tol:
                return False, reasons
            reasons.append(f"x_stable(range={x_rel_change:.2e})")

        return True, reasons

    def _optimality_criterion_pass(self, x_curr: np.ndarray
                                   ) -> Tuple[bool, Optional[str]]:
        """Evaluate the configured optimality test at x_curr.

        Returns (passed, reason-fragment). "none" always passes with no
        fragment; "kkt" runs the scale-invariant certificate (fails closed on
        fit failure); "legacy_projected_gradient" reproduces the pre-v0.6
        absolute projected-gradient test.
        """
        conv_cfg = self.config.convergence
        mode = conv_cfg.optimality_test
        if mode == "none":
            return True, None
        if mode == "legacy_projected_gradient":
            proj_grad_norm = self._compute_projected_gradient_norm(x_curr)
            if proj_grad_norm < conv_cfg.proj_grad_tol:
                return True, f"proj_grad(norm={proj_grad_norm:.2e})"
            return False, None
        cert = self._compute_kkt_certificate(x_curr)
        self._last_kkt_certificate = cert
        if cert.passed:
            return True, (f"kkt(residual={cert.residual:.2e}"
                          f"<=tol={cert.tolerance:.2e})")
        return False, None
    
    def solve(self, x0: np.ndarray, verbose: bool = False) -> SolverResult:
        """
        Solve the optimization problem starting from x0.
        
        Parameters
        ----------
        x0 : ndarray, shape (n,)
            Initial guess
        verbose : bool, optional
            Print progress information
            
        Returns
        -------
        result : SolverResult
            Optimization results including trajectory and statistics
        """
        x0 = np.asarray(x0, dtype=float).copy()
        assert x0.shape == (self.problem.n_vars,), f"x0 must have shape ({self.problem.n_vars},)"
        
        # Reset trajectory storage
        self._t_segments = []
        self._x_segments = []
        self._n_projections = 0
        self._spike_times = []
        self._spike_deltas = []
        self._spike_constraints = []
        self._spike_violation_values = []
        self._reset_projection_event_observer()
        self._converged = False
        self._last_kkt_certificate = None
        self._chunked_reason = None
        self._convergence_reason = "max_iterations"
        self._iterations_used = 0
        self._projection_budget_exhausted = False

        # Transform axis: an explicit problem transform (e.g. eigenbasis) rewrites
        # the problem, solves the equivalent system, and maps the solution back.
        if self.config.transform is not None:
            return self._solve_with_transform(x0, verbose)

        # Dispatch to appropriate solver
        if self.config.backend not in _VALID_BACKENDS:
            raise ValueError(
                f"unknown backend {self.config.backend!r}; expected one of "
                f"{sorted(_VALID_BACKENDS)}")
        if self.config.backend in _C_BACKENDS:
            return self._solve_euler_c(x0, verbose)
        if self.config.integration_method == 'euler':
            if self.config.record_trajectory:
                return self._solve_euler(x0, verbose)
            return self._solve_euler_lean(x0, verbose)
        return self._solve_ivp(x0, verbose)

    def _solve_with_transform(self, x0: np.ndarray, verbose: bool = False) -> SolverResult:
        """Solve via an explicit problem transform (the transform axis).

        Resolves ``config.transform``, checks applicability (e.g. eigenbasis
        rejects box constraints), rewrites the problem into transformed
        coordinates, solves the equivalent system with the canonical dynamics on
        the chosen backend (using the diagonal Hessian fast path when the
        transform diagonalizes A), then maps the solution back. Always returns
        the lean result (final-state fields); the transform is a performance
        path, so the inner solve runs lean regardless of ``record_trajectory``.
        """
        from .transforms import resolve_transform

        transform = resolve_transform(self.config.transform)
        transform.check_applicable(self.problem, self.config)
        ctx = transform.forward(self.problem, x0, self.config)

        # Inner solve on the transformed problem: canonical dynamics (no nested
        # transform), lean, same backend + convergence config. When the
        # transform folded the box into explicit rows (consumes_bounds), the
        # inner solve must run bound-free -- the scalar bounds are meaningless
        # in the transformed coordinates.
        inner_problem = OptimizationProblem(A=ctx.A, b=ctx.b, C=ctx.C, d=ctx.d)
        inner_kwargs = dict(transform=None, record_trajectory=False)
        if getattr(ctx, "consumes_bounds", False):
            inner_kwargs.update(lower_bound=None, upper_bound=None)
        inner_config = replace(self.config, **inner_kwargs)
        inner = SNNSolver(inner_problem, inner_config)
        inner._a_diag = ctx.a_diag  # enable the O(n) diagonal Hessian fast path
        inner_result = inner.solve(ctx.x0, verbose=verbose)

        # Map the solution back to the original coordinates and report metrics
        # against the ORIGINAL problem.
        final_x = ctx.recover(inner_result.final_x)
        self._converged = inner._converged
        self._convergence_reason = inner._convergence_reason
        self._iterations_used = inner._iterations_used
        self._n_projections = inner._n_projections
        self._projection_budget_exhausted = inner._projection_budget_exhausted
        self._explicit_row_event_counts = inner._explicit_row_event_counts
        self._implicit_lower_event_counts = inner._implicit_lower_event_counts
        self._implicit_upper_event_counts = inner._implicit_upper_event_counts
        self._projection_event_digest = inner._projection_event_digest
        self._observed_total_projection_distance = (
            inner._observed_total_projection_distance)
        self._projection_first_candidate_id = inner._projection_first_candidate_id
        self._projection_last_candidate_id = inner._projection_last_candidate_id
        self._projection_cap_rechecks = inner._projection_cap_rechecks
        return self._build_lean_result(final_x)

    def _solve_euler(self, x0: np.ndarray, verbose: bool = False) -> SolverResult:
        """
        Solve using discrete Euler integration with convergence detection.
        
        More stable for tightly constrained problems like SVM.
        """
        x_current = x0.copy()
        trajectory = [x_current.copy()]
        
        # History for convergence checking
        obj_history: List[float] = []
        x_history: List[np.ndarray] = []
        patience_counter = 0
        
        if verbose:
            print(f"Using k0 = {self._k0:.6e} (auto-computed: {self.config.k0 is None})")
            if self.config.convergence.enable_early_stopping:
                print(f"Early stopping enabled: check every {self.config.convergence.check_every} iters, "
                      f"min {self.config.convergence.min_iterations} iters")
        
        for iteration in range(self.config.max_iterations):
            # Phase 1: Gradient descent step
            gradient = self.problem.gradient(x_current)
            x_current = x_current - self._k0 * gradient
            
            # Phase 2: Project to feasible region
            x_current, n_proj, spike_info = self._project_to_feasible(
                x_current, outer_iteration=iteration)
            self._n_projections += n_proj

            if n_proj > 0 and self.config.record_spike_history:
                for info in spike_info:
                    self._spike_times.append(float(iteration))
                    self._spike_deltas.append(info["delta_x"])
                    self._spike_constraints.append(info["constraints"])
                    self._spike_violation_values.append(info["violations"])

            # (v0.5.0: the former Phase-3 terminal box clip is gone -- bounds
            # are facets inside the Phase-2 sweep; clipping here broke rows.)
            if self._projection_budget_exhausted:
                self._convergence_reason = "projection_budget_exhausted"
                self._iterations_used = iteration + 1
                trajectory.append(x_current.copy())
                break

            trajectory.append(x_current.copy())
            
            # Track history for convergence
            obj_current = self.problem.objective(x_current)
            obj_history.append(obj_current)
            x_history.append(x_current.copy())
            
            # Keep history bounded
            max_history = self.config.convergence.window_size * 2
            if len(obj_history) > max_history:
                obj_history = obj_history[-max_history:]
                x_history = x_history[-max_history:]
            
            # Verbose output
            if verbose and iteration % 100 == 0:
                viol = self.problem.max_violation(x_current)
                proj_grad = self._compute_projected_gradient_norm(x_current)
                print(f"Iter {iteration}: obj={obj_current:.6e}, max_viol={viol:.6e}, "
                      f"proj_grad={proj_grad:.6e}")
            
            # Check convergence (with patience)
            if self.config.convergence.enable_early_stopping:
                converged, reason, was_check = self._check_convergence(iteration, x_current, obj_history, x_history)
                
                if was_check:  # Only update patience on actual check iterations
                    if converged:
                        patience_counter += 1
                        if patience_counter >= self.config.convergence.patience:
                            self._converged = True
                            self._convergence_reason = reason
                            self._iterations_used = iteration + 1
                            if verbose:
                                print(f"Iter {iteration}: Early stop - {reason}")
                            break
                    else:
                        patience_counter = 0  # Reset patience only if check failed
        
        # If we didn't converge early (and didn't abort), record final state
        if (not self._converged
                and self._convergence_reason != "projection_budget_exhausted"):
            self._iterations_used = self.config.max_iterations
            self._convergence_reason = "max_iterations"
        
        # Build result
        X = np.array(trajectory)
        t = np.arange(len(trajectory), dtype=float)
        
        self._t_segments = [t]
        self._x_segments = [X]

        return self._build_result()

    def _solve_euler_lean(self, x0: np.ndarray, verbose: bool = False) -> SolverResult:
        """
        Lean discrete-Euler solve: no trajectory or spike-event storage.

        Numerically tracks :meth:`_solve_euler` but drops every per-iteration
        instrumentation cost: the full iterate trajectory, the per-projection
        spike-event dicts, and the full-trajectory recompute in the result
        builder. The objective needed for the plateau check reuses the
        gradient's ``A @ x`` matvec instead of computing a second one, which
        roughly halves the dense matvec count per iteration.

        This method is the reference implementation that the compiled C/C++
        backend mirrors; keep the two in lockstep.
        """
        conv_cfg = self.config.convergence
        A = self.problem.A
        b = self.problem.b
        track_x_history = conv_cfg.use_solution_stable

        x_current = np.asarray(x0, dtype=float).copy()
        obj_history: List[float] = []
        x_history: List[np.ndarray] = []
        patience_counter = 0

        if verbose:
            print(f"[lean] Using k0 = {self._k0:.6e} (auto-computed: {self.config.k0 is None})")

        # A @ x for the current iterate; recomputed once per iteration and
        # reused for both the gradient step and the plateau-check objective.
        # When a transform has diagonalized A (self._a_diag set), use the O(n)
        # elementwise product instead of the dense O(n^2) matvec.
        a_diag = self._a_diag
        Ax = (a_diag * x_current) if a_diag is not None else A @ x_current

        for iteration in range(self.config.max_iterations):
            # Phase 1: gradient descent step (gradient = A x + b, A x cached)
            gradient = Ax + b
            x_current = x_current - self._k0 * gradient

            # Phase 2: project to feasible region (no spike-info dicts)
            x_current, n_proj, _ = self._project_to_feasible(
                x_current, build_info=False, outer_iteration=iteration)
            self._n_projections += n_proj

            # (v0.5.0: the former Phase-3 terminal box clip is gone -- bounds
            # are facets inside the Phase-2 sweep; clipping here broke rows.)
            if self._projection_budget_exhausted:
                self._convergence_reason = "projection_budget_exhausted"
                self._iterations_used = iteration + 1
                break

            # Objective for the plateau check: reuse the A @ x we need anyway
            # for the next iteration's gradient (O(n^2) once, not twice; O(n)
            # elementwise when the Hessian is diagonal).
            Ax = (a_diag * x_current) if a_diag is not None else A @ x_current
            obj_current = 0.5 * float(x_current @ Ax) + float(b @ x_current)
            obj_history.append(obj_current)
            if track_x_history:
                x_history.append(x_current.copy())

            # Keep history bounded
            max_history = conv_cfg.window_size * 2
            if len(obj_history) > max_history:
                obj_history = obj_history[-max_history:]
                if track_x_history:
                    x_history = x_history[-max_history:]

            if verbose and iteration % 100 == 0:
                viol = self.problem.max_violation(x_current)
                print(f"[lean] Iter {iteration}: obj={obj_current:.6e}, max_viol={viol:.6e}")

            # Check convergence (with patience)
            if conv_cfg.enable_early_stopping:
                converged, reason, was_check = self._check_convergence(
                    iteration, x_current, obj_history, x_history)
                if was_check:
                    if converged:
                        patience_counter += 1
                        if patience_counter >= conv_cfg.patience:
                            self._converged = True
                            self._convergence_reason = reason
                            self._iterations_used = iteration + 1
                            if verbose:
                                print(f"[lean] Iter {iteration}: Early stop - {reason}")
                            break
                    else:
                        patience_counter = 0

        if (not self._converged
                and self._convergence_reason != "projection_budget_exhausted"):
            self._iterations_used = self.config.max_iterations
            self._convergence_reason = "max_iterations"

        return self._build_lean_result(x_current)

    def _build_lean_result(self, final_x: np.ndarray) -> SolverResult:
        """
        Build a minimal :class:`SolverResult` for the lean solve path.

        Only final-state fields are populated; trajectory and spike-event
        arrays are intentionally empty (``record_trajectory=False``). The
        reported ``final_objective`` is computed with the exact objective
        formula, not the matvec-reuse approximation used internally for the
        plateau check.
        """
        final_x = np.asarray(final_x, dtype=float)
        final_objective = self.problem.objective(final_x)
        final_proj_grad_norm = self._compute_projected_gradient_norm(final_x)
        final_violation = self.problem.max_violation(final_x)
        raw_rows, dist_rows, box_viol = self._violation_split(final_x)
        n = self.problem.n_vars

        return SolverResult(
            t=np.array([float(self._iterations_used)]),
            X=final_x.reshape(1, -1),
            objective_values=np.array([final_objective]),
            constraint_violations=np.array([final_violation]),
            n_projections=self._n_projections,
            converged=self._converged,
            convergence_reason=self._convergence_reason,
            iterations_used=self._iterations_used,
            final_x=final_x,
            final_objective=final_objective,
            final_proj_grad_norm=final_proj_grad_norm,
            spike_times=np.array([], dtype=float),
            spike_deltas=np.empty((0, n)),
            spike_norms=np.empty((0,), dtype=float),
            spike_constraints=[],
            spike_violation_values=[],
            total_projection_distance=0.0,
            joint_feasible=(max(dist_rows, box_viol)
                            <= self.config.convergence.feasibility_tol),
            max_violation_rows_raw=raw_rows,
            max_distance_rows=dist_rows,
            max_violation_box=box_viol,
            stationarity_residual=self._stationarity_residual(final_x),
            optimality_test=self.config.convergence.optimality_test,
            **self._kkt_result_fields(final_x),
            projection_budget_exhausted=self._projection_budget_exhausted,
            **self._projection_event_result_fields(),
        )

    def _drive_chunked_kernel(self, kernel_call, x0c: np.ndarray):
        """Drive the compiled kernel in checkpoint-sized chunks (kkt mode).

        The kernel advances the dynamics with early stopping disabled; between
        chunks the host applies EXACTLY the same convergence policy as the
        Python backend (feasibility gate, cheap window criteria via
        :meth:`_cheap_criteria_pass`, KKT certificate via
        :meth:`_optimality_criterion_pass`, and the patience counter), so the
        two backends share one stopping-policy implementation. Chunk
        boundaries land on the iterations the Python loop would check
        (iteration >= min_iterations and iteration % check_every == 0), and
        the kernel exports its objective / iterate window tails so windowed
        criteria see the same history a monolithic run would. Observer
        telemetry is chained across chunks (absolute iteration tokens, seeded
        digest), producing the identical event stream to a monolithic run.

        Returns the same tuple shape as a monolithic ``solve_euler`` call:
        (final_x, iterations_used, n_projections, converged, reason_code).
        """
        conv = self.config.convergence
        max_it = self.config.max_iterations
        ce = max(1, conv.check_every)
        W = max(1, conv.window_size)
        n = self.problem.n_vars

        # First checkpoint: smallest k >= min_iterations with k % ce == 0.
        k1 = ((max(conv.min_iterations, 0) + ce - 1) // ce) * ce
        boundaries = list(range(k1, max_it, ce))

        obj_tail = np.zeros(W, dtype=np.float64)
        obj_tail_len = np.zeros(1, dtype=np.int64)
        tails = {"obj_tail_out": obj_tail, "obj_tail_len_out": obj_tail_len}
        want_x = conv.use_solution_stable
        if want_x:
            x_tail = np.zeros((W, n), dtype=np.float64)
            x_tail_len = np.zeros(1, dtype=np.int64)
            tails.update({"x_tail_out": x_tail, "x_tail_len_out": x_tail_len})

        x_cur = np.asarray(x0c, dtype=np.float64)
        abs_iter = 0
        n_proj_total = 0
        patience_counter = 0
        obj_hist: List[float] = []
        x_hist: List[np.ndarray] = []

        def run(length):
            nonlocal x_cur, abs_iter, n_proj_total, obj_hist, x_hist
            fx, iters, n_proj, _, reason_code = kernel_call(
                x_cur, length, early_stop=False,
                iter_offset=abs_iter, resume=(abs_iter > 0), extra=tails)
            x_cur = np.asarray(fx, dtype=np.float64)
            abs_iter += int(iters)
            n_proj_total += int(n_proj)
            cnt = int(obj_tail_len[0])
            obj_hist = (obj_hist + [float(v) for v in obj_tail[:cnt]])[-W:]
            if want_x:
                xcnt = int(x_tail_len[0])
                x_hist = (x_hist + [x_tail[i].copy()
                                    for i in range(xcnt)])[-W:]
            return int(reason_code)

        for kb in boundaries:
            reason_code = run(kb + 1 - abs_iter)
            if reason_code == 2:  # projection budget exhausted mid-chunk
                return x_cur, abs_iter, n_proj_total, False, 2

            # --- host-side convergence policy (same helpers as backend='python')
            if (conv.require_feasibility
                    and self._joint_max_violation(x_cur) > conv.feasibility_tol):
                patience_counter = 0
                continue
            cheap_ok, reasons = self._cheap_criteria_pass(x_cur, obj_hist,
                                                          x_hist)
            passed = False
            if cheap_ok:
                opt_ok, opt_reason = self._optimality_criterion_pass(x_cur)
                if opt_ok and (reasons or opt_reason):
                    if opt_reason:
                        reasons = [opt_reason] + reasons
                    passed = True
            if passed:
                patience_counter += 1
                if patience_counter >= conv.patience:
                    self._chunked_reason = f"converged({'; '.join(reasons)})"
                    return x_cur, kb + 1, n_proj_total, True, 1
            else:
                patience_counter = 0

        # No stop: run out the remaining iterations (no further checkpoints).
        if abs_iter < max_it:
            reason_code = run(max_it - abs_iter)
            if reason_code == 2:
                return x_cur, abs_iter, n_proj_total, False, 2
        return x_cur, max_it, n_proj_total, False, 0

    def _solve_euler_c(self, x0: np.ndarray, verbose: bool = False) -> SolverResult:
        """
        Solve via the compiled C++ kernel (``snn_opt._kernel``).

        A faithful compiled port of :meth:`_solve_euler_lean`. Supported
        configuration: ``integration_method='euler'``,
        ``projection_method='adaptive'``, dense ``A`` and ``C``. Always lean
        (``record_trajectory`` is ignored) -- only final-state fields of the
        returned :class:`SolverResult` are populated.
        """
        try:
            from . import _kernel
        except ImportError as exc:
            raise ImportError(
                "backend='c' requires the compiled snn_opt._kernel extension. "
                "Build it with `python setup.py build_ext --inplace`."
            ) from exc

        if self.config.projection_method != 'adaptive':
            raise ValueError(
                "backend='c' supports projection_method='adaptive' only "
                f"(got {self.config.projection_method!r})")
        if _issparse(self.problem.A) or _issparse(self.problem.C):
            raise ValueError(
                "backend='c' requires dense A and C (scipy sparse not supported)")

        prob = self.problem
        conv = self.config.convergence
        n, m = prob.n_vars, prob.n_constraints

        A = np.ascontiguousarray(prob.A, dtype=np.float64)
        b = np.ascontiguousarray(prob.b, dtype=np.float64)
        C = np.ascontiguousarray(prob.C, dtype=np.float64).reshape(m, n)
        d = np.ascontiguousarray(prob.d, dtype=np.float64)
        c_norms_sq = np.ascontiguousarray(self._c_norms_sq, dtype=np.float64).reshape(m)
        if self._c_gram is not None:
            c_gram = np.ascontiguousarray(self._c_gram, dtype=np.float64).reshape(m, m)
        elif m == 0:
            c_gram = np.zeros((0, 0), dtype=np.float64)
        else:
            raise ValueError(
                f"backend='c' needs the constraint Gram matrix, but m={m} "
                f"exceeds the precompute cap (_MAX_GRAM_M={_MAX_GRAM_M}); "
                f"use backend='python'")
        x0c = np.ascontiguousarray(x0, dtype=np.float64)

        has_lower = self.config.lower_bound is not None
        has_upper = self.config.upper_bound is not None

        # Resolve the matvec threading from the backend string. Only the matvec
        # is data-parallel (the Euler recurrence + greedy projection are serial
        # -- the Amdahl ceiling), so results are identical across all three; the
        # flag only changes how the matvec rows are distributed.
        has_omp = bool(getattr(_kernel, 'HAS_OPENMP', False))
        backend = self.config.backend
        if backend == 'c_serial':
            parallel = False
        elif backend == 'c_openmp':
            if not has_omp:
                raise ValueError(
                    "backend='c_openmp' requires the compiled kernel to be "
                    "built with OpenMP (-fopenmp), but this build is SIMD-only "
                    "(_kernel.HAS_OPENMP is False). Use backend='c' (auto) or "
                    "'c_serial', or rebuild with an OpenMP-capable compiler.")
            parallel = True
        else:  # 'c' -- auto: multicore when the build supports it, else serial
            parallel = has_omp

        # Diagonal Hessian fast path: when a transform has diagonalized A, hand
        # the kernel the length-n diagonal so its A @ x step is O(n) elementwise.
        use_diag = self._a_diag is not None
        a_diag_c = (np.ascontiguousarray(self._a_diag, dtype=np.float64)
                    if use_diag else np.zeros(0, dtype=np.float64))

        row_scale = np.ascontiguousarray(self._row_scale, dtype=np.float64).reshape(m)

        observer_kwargs = {}
        observer_meta = None
        observer_distance = None
        if self.config.observe_projection_events:
            row_event_counts = np.zeros(m, dtype=np.int64)
            lower_event_counts = np.zeros(n, dtype=np.int64)
            upper_event_counts = np.zeros(n, dtype=np.int64)
            observer_meta = np.zeros(4, dtype=np.uint64)
            observer_distance = np.zeros(1, dtype=np.float64)
            observer_kwargs = {
                "row_event_counts": row_event_counts,
                "lower_event_counts": lower_event_counts,
                "upper_event_counts": upper_event_counts,
                "observer_meta": observer_meta,
                "observer_distance": observer_distance,
            }

        def kernel_call(x_start, iterations, *, early_stop, iter_offset=0,
                        resume=False, extra=None):
            kwargs = dict(observer_kwargs)
            if resume and observer_kwargs:
                kwargs["resume_observer"] = True
            if extra:
                kwargs.update(extra)
            return _kernel.solve_euler(
                A, b, C, d, c_norms_sq, row_scale, c_gram,
                np.ascontiguousarray(x_start, dtype=np.float64),
                self._k0, self.config.constraint_tol,
                iterations, self._proj_cap,
                early_stop, conv.check_every, conv.min_iterations,
                conv.window_size, conv.patience,
                conv.obj_rel_tol, conv.x_rel_tol, conv.proj_grad_tol,
                conv.feasibility_tol,
                conv.use_objective_plateau, conv.use_projected_gradient,
                conv.use_solution_stable, conv.require_feasibility,
                has_lower, float(self.config.lower_bound) if has_lower else 0.0,
                has_upper, float(self.config.upper_bound) if has_upper else 0.0,
                parallel=parallel,
                a_diag=a_diag_c, use_diag=use_diag,
                iter_offset=iter_offset,
                **kwargs,
            )

        # The KKT certificate is host-side policy shared by every backend, so
        # the compiled kernel is driven in fixed chunks ending at each
        # convergence checkpoint; the host evaluates the same feasibility /
        # cheap-criteria / certificate gates the Python backend uses (kernel
        # early stopping stays off). The legacy projected-gradient and "none"
        # criteria still run monolithically with the in-kernel checks, which
        # preserves the pre-v0.6 native behavior exactly.
        chunked = (conv.enable_early_stopping
                   and conv.optimality_test == "kkt"
                   and self.config.max_iterations > 0)
        if not chunked:
            final_x, iters, n_proj, converged, reason_code = kernel_call(
                x0c, self.config.max_iterations,
                early_stop=conv.enable_early_stopping)
        else:
            final_x, iters, n_proj, converged, reason_code = (
                self._drive_chunked_kernel(kernel_call, x0c))

        self._n_projections = int(n_proj)
        if self.config.observe_projection_events:
            self._explicit_row_event_counts = row_event_counts
            self._implicit_lower_event_counts = lower_event_counts
            self._implicit_upper_event_counts = upper_event_counts
            self._projection_event_digest = int(observer_meta[0])
            self._observed_total_projection_distance = float(
                observer_distance[0])
            no_candidate = np.iinfo(np.uint64).max
            self._projection_first_candidate_id = (
                None if observer_meta[1] == no_candidate else int(observer_meta[1]))
            self._projection_last_candidate_id = (
                None if observer_meta[2] == no_candidate else int(observer_meta[2]))
            self._projection_cap_rechecks = int(observer_meta[3])
        self._converged = bool(converged)
        self._iterations_used = int(iters)
        if reason_code == 1:
            # The chunked (host-policy) driver supplies the same reason string
            # the Python backend produces; the monolithic legacy path keeps
            # its historical label.
            self._convergence_reason = (self._chunked_reason
                                        or "converged(c-backend)")
        elif reason_code == 2:
            self._convergence_reason = "projection_budget_exhausted"
            self._projection_budget_exhausted = True
        else:
            self._convergence_reason = "max_iterations"

        if verbose:
            print(f"[c] iterations={iters}, n_projections={n_proj}, "
                  f"converged={bool(converged)}")

        return self._build_lean_result(np.asarray(final_x, dtype=float))

    def _solve_ivp(self, x0: np.ndarray, verbose: bool = False) -> SolverResult:
        """
        Solve using continuous ODE integration (original method).
        
        Uses solve_ivp with event detection for constraint violations.
        """
        # Store initial point
        self._t_segments.append(np.array([0.0]))
        self._x_segments.append(x0.reshape(1, -1))
        
        # Main optimization loop
        t_current = 0.0
        t_previous = -1.0
        x_current = x0
        projection_sweep_index = 0
        
        while t_current < self.config.t_end:
            # Phase 1: Project back into feasible region
            x_current, n_proj, spike_info = self._project_to_feasible(
                x_current, outer_iteration=projection_sweep_index)
            projection_sweep_index += 1
            self._n_projections += n_proj

            if n_proj > 0 and self.config.record_spike_history:
                for info in spike_info:
                    self._spike_times.append(t_current)
                    self._spike_deltas.append(info["delta_x"])
                    self._spike_constraints.append(info["constraints"])
                    self._spike_violation_values.append(info["violations"])
            
            if verbose and n_proj > 0:
                print(f"t={t_current:.3f}: Applied {n_proj} projections")
            
            # Phase 2: Gradient descent until constraint hit
            t_span = (t_current, self.config.t_end)
            result = self._integrate_gradient_descent(t_span, x_current)
            
            # Store trajectory segment, dropping duplicate boundary sample if needed
            t_segment = result.t
            x_segment = result.y.T

            if self._t_segments:
                last_t = self._t_segments[-1][-1]
                if t_segment.size > 0 and np.isclose(t_segment[0], last_t, atol=1e-12, rtol=0.0):
                    t_segment = t_segment[1:]
                    x_segment = x_segment[1:]

            if t_segment.size > 0:
                self._t_segments.append(t_segment)
                self._x_segments.append(x_segment)
            
            # Update current state
            t_previous = t_current
            t_current = result.t[-1]
            x_current = result.y[:, -1]
            
            # Check if time is not advancing (stuck at constraint boundary)
            if abs(t_current - t_previous) < 1e-9:
                if verbose:
                    print(f"t={t_current:.3f}: Terminating - No progress (optimal on boundary)")
                self._converged = True
                self._convergence_reason = "stuck_at_boundary"
                break
            
            # Check if we've reached the end
            if t_current >= self.config.t_end:
                break
            
            if verbose:
                obj = self.problem.objective(x_current)
                viol = self.problem.max_violation(x_current)
                print(f"t={t_current:.3f}: obj={obj:.6e}, max_viol={viol:.6e}")
        
        if not self._converged:
            self._convergence_reason = "t_end_reached"
        
        self._iterations_used = len(self._t_segments)
        
        # Compile results
        return self._build_result()
    
    def _project_to_feasible(self, x: np.ndarray,
                             build_info: bool = True,
                             outer_iteration: int = 0
                             ) -> Tuple[np.ndarray, int, List[dict]]:
        """
        Project x back into feasible region using discrete corrections.

        Supports two projection methods:
        - 'adaptive': Computes exact step to reach constraint boundary (k1 = g_j / ||c_j||²)
        - 'fixed': Uses fixed step size k1 (original method)

        Parameters
        ----------
        x : ndarray
            Point to project.
        build_info : bool, optional
            If True (default) build per-projection spike-event metadata. The
            lean solve path passes False to skip the dict allocations; the
            projected point and iteration count are unaffected.
        outer_iteration : int, optional
            Zero-based outer/sweep index included only in the optional event
            digest. It never enters winner selection or numerical updates.

        Returns
        -------
        x_proj : ndarray
            Projected point
        n_iters : int
            Number of projection iterations performed
        spike_info : list of dict
            Metadata for each projection applied (empty when build_info=False)
        """
        if self.config.projection_method == 'adaptive':
            return self._project_adaptive(
                x, build_info=build_info, outer_iteration=outer_iteration)
        else:
            return self._project_fixed(x, build_info=build_info)

    def _project_adaptive(self, x: np.ndarray,
                          build_info: bool = True,
                          outer_iteration: int = 0
                          ) -> Tuple[np.ndarray, int, List[dict]]:
        """
        Unified adaptive projection: exact steps onto the currently most-violated
        constraint, where "constraints" means the general rows AND the implicit
        box facets in one candidate family.

        Selection is winner-take-all on the NORMALIZED violation distance
        (raw row residual / ||c_j||; box facets have unit normals), so the
        winner choice is invariant to positive row rescaling and
        `constraint_tol` is a geometric distance. For a violated row the exact
        step is k1 = g_j / ||c_j||^2 (one step to the boundary); for a violated
        facet it is the exact single-coordinate correction to the bound (an
        O(1) primal update). The frozen candidate order for ties is: rows in
        input order, then lower facets 0..n-1, then upper facets 0..n-1
        (first maximal index wins) -- identical across all backends.

        The sweep runs until JOINT tolerance (rows + box) or the safety cap
        `self._proj_cap`; cap exhaustion sets
        `self._projection_budget_exhausted` and the caller aborts the solve.

        When the constraint Gram matrix G = C C^T has been precomputed (dense C,
        m <= _MAX_GRAM_M), the row residual g = C x_proj + d is maintained
        incrementally: a row spike on j applies the lateral update
        g <- g - k1 * G[:,j] (O(m)) and a facet spike on coordinate i applies
        g <- g + delta * C[:,i] (O(m)). For sparse C or large m it falls back
        to recomputing the residual each micro-step. Facet residuals are read
        directly off x, so they are always fresh.

        Box-only problems (m == 0) dispatch to the exact vectorized box
        projection (separable, so the clip IS the projection).
        """
        cfg = self.config
        lo, hi = cfg.lower_bound, cfg.upper_bound
        has_box = lo is not None or hi is not None
        m = self.problem.n_constraints
        n = self.problem.n_vars
        x_proj = x.copy()
        spike_info: List[dict] = []

        # Unconstrained case
        if m == 0 and not has_box:
            return x_proj, 0, spike_info

        # Box-only fast path: separable set, vectorized clip is the exact
        # projection (counted as one population event per clipped coordinate).
        if m == 0:
            clipped = self._clip_to_bounds(x_proj.copy())
            changed = np.nonzero(clipped != x_proj)[0]
            if changed.size and build_info:
                deltas = clipped - x_proj
                lo_side = deltas[changed] > 0  # pushed up -> lower facet
                ids = np.where(lo_side, m + changed, m + n + changed)
                spike_info.append({
                    "constraints": ids,
                    "delta_x": deltas,
                    "violations": np.abs(deltas[changed]),
                })
            if changed.size and self._explicit_row_event_counts is not None:
                # The box projection is simultaneous. Canonicalize only the
                # observer stream as lower coordinates first, then upper
                # coordinates, matching the frozen candidate-family order.
                deltas = clipped - x_proj
                lower_changed = changed[deltas[changed] > 0]
                upper_changed = changed[deltas[changed] < 0]
                ordinal = 0
                for i in lower_changed:
                    self._observe_projection_event(
                        "lo", int(i), outer_iteration, ordinal,
                        abs(float(deltas[i])))
                    ordinal += 1
                for i in upper_changed:
                    self._observe_projection_event(
                        "hi", int(i), outer_iteration, ordinal,
                        abs(float(deltas[i])))
                    ordinal += 1
            return clipped, int(changed.size), spike_info

        gram = self._c_gram  # None -> recompute path; else event-driven path
        tol = cfg.constraint_tol
        g = self.problem.constraint_values(x_proj)
        n_iters = 0

        for _ in range(self._proj_cap):
            if gram is None:
                g = self.problem.constraint_values(x_proj)

            # Winner-take-all over normalized distances, frozen candidate order
            # (rows, lower facets, upper facets; first maximal index wins).
            j_row = int(np.argmax(g * self._row_scale)) if m else -1
            best_val = g[j_row] * self._row_scale[j_row] if m else -np.inf
            kind = "row"
            j = j_row
            if lo is not None:
                v_lo = lo - x_proj
                i = int(np.argmax(v_lo))
                if v_lo[i] > best_val:
                    best_val, kind, j = float(v_lo[i]), "lo", i
            if hi is not None:
                v_hi = x_proj - hi
                i = int(np.argmax(v_hi))
                if v_hi[i] > best_val:
                    best_val, kind, j = float(v_hi[i]), "hi", i

            if best_val <= tol:
                break  # jointly satisfied (geometric tolerance)

            if kind == "row":
                c_j = self.problem.C[j]
                if _issparse(c_j):
                    c_j = np.asarray(c_j.todense()).ravel()
                else:
                    c_j = np.asarray(c_j).ravel()
                violation = g[j]  # raw residual; step uses raw / ||c||^2
                k1_adaptive = violation / self._c_norms_sq[j]
                delta_x = -k1_adaptive * c_j
                x_proj = x_proj + delta_x
                if gram is not None:
                    # Spike j propagates a lateral update to coupled rows.
                    g = g - k1_adaptive * gram[j]
                event_distance = abs(k1_adaptive) * self._c_norms[j]
                event_id, event_viol = j, violation
            else:
                # Facet spike: exact single-coordinate correction to the bound.
                delta = best_val if kind == "lo" else -best_val
                if build_info:
                    delta_x = np.zeros(n)
                    delta_x[j] = delta
                x_proj[j] += delta
                if gram is not None:
                    col = self.problem.C[:, j]
                    if _issparse(col):
                        col = np.asarray(col.todense()).ravel()
                    else:
                        col = np.asarray(col).ravel()
                    g = g + delta * col
                event_distance = abs(delta)
                event_id = (m + j) if kind == "lo" else (m + n + j)
                event_viol = best_val
            # Observe only after the selected event's primal and residual
            # updates have completed. This branch never feeds numerical state.
            if self._explicit_row_event_counts is not None:
                self._observe_projection_event(
                    kind, j, outer_iteration, n_iters, event_distance)
            n_iters += 1

            if build_info:
                spike_info.append({
                    "constraints": np.array([event_id]),
                    "delta_x": delta_x,
                    "violations": np.array([event_viol])
                })
        else:
            # Cap hit: re-check joint violation; if still above tolerance the
            # sweep failed and the solve must abort (see the euler loops).
            if self._projection_cap_rechecks is not None:
                self._projection_cap_rechecks += 1
            if self._joint_max_violation(x_proj) > tol:
                self._projection_budget_exhausted = True

        return x_proj, n_iters, spike_info
    
    def _project_fixed(self, x: np.ndarray,
                       build_info: bool = True) -> Tuple[np.ndarray, int, List[dict]]:
        """
        Fixed projection: use constant step size k1 for all constraints.

        Original method that requires tuning k1 hyperparameter.
        """
        x_proj = x.copy()
        n_iters = 0
        spike_info: List[dict] = []

        for _ in range(self._proj_cap):
            g = self.problem.constraint_values(x_proj)
            violations = g > self.config.constraint_tol

            if not np.any(violations):
                break

            # Apply projection: x <- x - k1 * C^T * violations
            direction = self.problem.C.T @ violations.astype(float)
            delta_x = -self.config.k1 * direction
            x_proj = x_proj + delta_x
            n_iters += 1
            if build_info:
                spike_info.append({
                    "constraints": np.where(violations)[0],
                    "delta_x": delta_x,
                    "violations": g[violations].copy()
                })

        return x_proj, n_iters, spike_info
    
    def _integrate_gradient_descent(self, t_span: Tuple[float, float], x0: np.ndarray):
        """
        Integrate gradient descent dynamics until constraint violation.
        
        Uses scipy's solve_ivp with event detection to stop when constraints
        are violated.
        """
        def dynamics(t, x):
            """dx/dt = -k0 * grad(f)"""
            return -self._k0 * self.problem.gradient(x)
        
        def constraint_event(t, x):
            """Event: returns negative in feasible region, zero on boundary, positive when violated."""
            if self.problem.n_constraints == 0:
                return -1.0
            g_max = np.max(self.problem.constraint_values(x))
            return g_max - self.config.constraint_tol
        
        constraint_event.terminal = True
        constraint_event.direction = 1  # Detect only crossings from feasible (negative) to violated (positive)
        
        events = constraint_event if self.problem.n_constraints > 0 else None
        
        result = solve_ivp(
            dynamics,
            t_span,
            x0,
            events=events,
            max_step=self.config.max_step,
            dense_output=False,
            method='RK45'
        )
        
        return result
    
    def _build_result(self) -> SolverResult:
        """Compile trajectory segments into final result."""
        # Concatenate all segments
        t = np.concatenate(self._t_segments)
        X = np.vstack(self._x_segments)
        
        # Compute objective values and constraint violations
        objective_values = np.array([self.problem.objective(x) for x in X])
        constraint_violations = np.array([self.problem.max_violation(x) for x in X])
        
        # Spike metadata
        if self._spike_times:
            spike_times = np.array(self._spike_times, dtype=float)
        else:
            spike_times = np.array([], dtype=float)

        if self._spike_deltas:
            spike_deltas = np.vstack(self._spike_deltas)
            spike_norms = np.linalg.norm(spike_deltas, axis=1)
        else:
            spike_deltas = np.empty((0, self.problem.n_vars))
            spike_norms = np.empty((0,), dtype=float)

        spike_constraints = [np.array(idx, dtype=int) for idx in self._spike_constraints]
        spike_violation_values = [np.array(vals, dtype=float) for vals in self._spike_violation_values]
        total_projection_distance = float(spike_norms.sum()) if spike_norms.size else 0.0

        # Final solution
        final_x = X[-1]
        final_objective = objective_values[-1]
        final_proj_grad_norm = self._compute_projected_gradient_norm(final_x)
        raw_rows, dist_rows, box_viol = self._violation_split(final_x)

        return SolverResult(
            t=t,
            X=X,
            objective_values=objective_values,
            constraint_violations=constraint_violations,
            n_projections=self._n_projections,
            converged=self._converged,
            convergence_reason=self._convergence_reason,
            iterations_used=self._iterations_used,
            final_x=final_x,
            final_objective=final_objective,
            final_proj_grad_norm=final_proj_grad_norm,
            spike_times=spike_times,
            spike_deltas=spike_deltas,
            spike_norms=spike_norms,
            spike_constraints=spike_constraints,
            spike_violation_values=spike_violation_values,
            total_projection_distance=total_projection_distance,
            joint_feasible=(max(dist_rows, box_viol)
                            <= self.config.convergence.feasibility_tol),
            max_violation_rows_raw=raw_rows,
            max_distance_rows=dist_rows,
            max_violation_box=box_viol,
            stationarity_residual=self._stationarity_residual(final_x),
            optimality_test=self.config.convergence.optimality_test,
            **self._kkt_result_fields(final_x),
            projection_budget_exhausted=self._projection_budget_exhausted,
            **self._projection_event_result_fields(),
        )


def solve_qp(A: np.ndarray, b: np.ndarray, C: np.ndarray, d: np.ndarray, 
             x0: np.ndarray, k0: float = None,
             t_end: float = 100.0, max_iterations: int = 2000,
             integration_method: str = 'euler',
             projection_method: str = 'adaptive',
             k0_scale: float = 0.5,
             lower_bound: float = None,
             upper_bound: float = None,
             enable_early_stopping: bool = True,
             record_trajectory: bool = True,
             backend: str = 'python',
             verbose: bool = False) -> SolverResult:
    """
    Convenience function to solve a QP without creating objects explicitly.
    
    Solves: minimize (1/2) x^T A x + b^T x, subject to C x + d <= 0
    
    Parameters
    ----------
    A : ndarray, shape (n, n)
        Hessian matrix
    b : ndarray, shape (n,)
        Linear cost vector
    C : ndarray, shape (m, n)
        Constraint matrix
    d : ndarray, shape (m,)
        Constraint offset
    x0 : ndarray, shape (n,)
        Initial guess
    k0 : float, optional
        Gradient descent step size. If None (default), auto-computed from 
        Lipschitz constant: k0 = k0_scale / λ_max(A)
    t_end : float
        Simulation end time (for IVP mode)
    max_iterations : int
        Maximum iterations (for Euler mode)
    integration_method : str
        'euler' (discrete steps) or 'ivp' (continuous ODE)
    projection_method : str
        'adaptive' (exact step to boundary) or 'fixed' (uses fixed k1)
    k0_scale : float
        Scaling factor for auto-computed k0 (only used when k0=None)
    lower_bound : float, optional
        Lower bound for box constraint clipping (e.g., 0 for SVM)
    upper_bound : float, optional
        Upper bound for box constraint clipping (e.g., C for SVM)
    enable_early_stopping : bool
        Whether to enable convergence-based early stopping
    record_trajectory : bool
        If True (default) record the full iterate trajectory and spike-event
        metadata. If False, run the lean solve path (no trajectory/spike
        storage, one fused matvec per iteration) -- use for benchmarking.
    backend : str
        'python' (reference), or one of the compiled pybind11 kernels (euler +
        adaptive projection only, implies record_trajectory=False): 'c' (auto,
        OpenMP multicore when available), 'c_serial' (single-threaded), or
        'c_openmp' (forced multicore). The C variants are numerically identical.
    verbose : bool
        Print progress

    Returns
    -------
    result : SolverResult
        Optimization results
    """
    problem = OptimizationProblem(A=A, b=b, C=C, d=d)
    conv_config = ConvergenceConfig(enable_early_stopping=enable_early_stopping)
    config = SolverConfig(k0=k0, t_end=t_end,
                          max_iterations=max_iterations,
                          integration_method=integration_method,
                          projection_method=projection_method,
                          k0_scale=k0_scale,
                          lower_bound=lower_bound,
                          upper_bound=upper_bound,
                          record_trajectory=record_trajectory,
                          backend=backend,
                          convergence=conv_config)
    solver = SNNSolver(problem, config)
    return solver.solve(x0, verbose=verbose)
