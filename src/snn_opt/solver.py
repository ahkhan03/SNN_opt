"""
SNN-Inspired Constrained Optimization Solver

Solves convex optimization problems with linear inequality constraints:
    minimize    (1/2) x^T A x + b^T x
    subject to  C x + d <= 0

The algorithm alternates between gradient descent and discrete boundary projections,
inspired by spiking neural network dynamics.
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


@dataclass
class ConvergenceConfig:
    """Configuration for convergence detection."""
    # Enable/disable early stopping
    enable_early_stopping: bool = True
    
    # Tolerances
    obj_rel_tol: float = 1e-8          # Relative objective change over window
    x_rel_tol: float = 1e-8            # Relative solution change
    proj_grad_tol: float = 1e-6        # Projected gradient norm tolerance
    feasibility_tol: float = 1e-2      # Max constraint violation for convergence (default: relaxed)
    
    # Timing control
    check_every: int = 50              # Check convergence every N iterations
    min_iterations: int = 100          # Minimum iterations before checking
    window_size: int = 10              # Window size for objective plateau detection
    patience: int = 3                  # Consecutive converged checks needed
    
    # Which criteria to use (require ALL enabled criteria to be satisfied)
    use_objective_plateau: bool = True
    use_projected_gradient: bool = True
    use_solution_stable: bool = False   # Disabled by default (can cause false positives)
    require_feasibility: bool = True    # Require feasibility for convergence


@dataclass
class SolverConfig:
    """Configuration parameters for the SNN solver."""
    k0: float = None  # Gradient descent step size (None = auto-compute from Lipschitz constant)
    t_end: float = 100.0  # Simulation end time (for IVP mode)
    max_step: float = 0.1  # Maximum integration step size (for IVP mode)
    constraint_tol: float = 1e-6  # Constraint violation tolerance
    max_projection_iters: int = 100  # Maximum projection iterations
    
    # Integration method: 'ivp' (continuous ODE) or 'euler' (discrete steps)
    integration_method: str = 'euler'  # Default to euler for better stability
    max_iterations: int = 2000  # Maximum iterations (for Euler mode)
    
    # Projection method: 'adaptive' (exact step to boundary) or 'fixed' (fixed k1 step)
    projection_method: str = 'adaptive'  # Default to adaptive (eliminates k1 hyperparameter)
    k1: float = 0.05  # Projection step size (only used when projection_method='fixed')
    
    # Scaling factor for auto-computed k0 (k0 = scale / L where L is Lipschitz constant)
    # Values < 1.0 are more conservative (slower but safer), > 1.0 are aggressive
    k0_scale: float = 0.5  # Default to 0.5 for stability
    
    # Box constraint clipping (for problems like SVM with 0 <= x <= C)
    # If provided, variables are clipped to [lower_bound, upper_bound] after each projection
    lower_bound: float = None  # If set, clip x >= lower_bound
    upper_bound: float = None  # If set, clip x <= upper_bound
    
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
        Indices of constraints that were active for each spike event
    spike_violation_values : list of ndarray
        Positive constraint residuals at the moment each spike was applied
    total_projection_distance : float
        Sum of norms of all spike displacements (cumulative distance)
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
    
    def summary(self) -> str:
        """Generate summary statistics string."""
        lines = [
            "Solver Result Summary:",
            f"  Converged: {self.converged}",
            f"  Convergence reason: {self.convergence_reason}",
            f"  Iterations used: {self.iterations_used}",
            f"  Final objective: {self.final_objective:.6e}",
            f"  Final proj. gradient norm: {self.final_proj_grad_norm:.6e}",
            f"  Max constraint violation: {np.max(self.constraint_violations):.6e}",
            f"  Total projections: {self.n_projections}",
            f"  Total spikes recorded: {len(self.spike_times)}",
        ]

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
        
        # Convergence tracking
        self._converged = False
        self._convergence_reason = "max_iterations"
        self._iterations_used = 0
    
    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """
        Clip variables to box constraints if specified.
        
        This is used for problems with simple bounds (like SVM with 0 <= alpha <= C)
        where box constraints are better handled by clipping than by iterative projection.
        """
        if self.config.lower_bound is not None:
            x = np.maximum(x, self.config.lower_bound)
        if self.config.upper_bound is not None:
            x = np.minimum(x, self.config.upper_bound)
        return x
    
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
        
        At the constrained optimum, the projected gradient is zero because:
        - For active constraints, the gradient component that would violate is zeroed
        - What remains is the feasible descent direction, which is zero at optimum
        
        This is the gold standard for constrained optimization convergence.
        
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
        
        reasons_met = []
        
        # 1. Objective plateau over window (not just single step)
        if conv_cfg.use_objective_plateau and len(obj_history) >= conv_cfg.window_size:
            window = obj_history[-conv_cfg.window_size:]
            obj_range = max(window) - min(window)
            obj_scale = max(abs(window[-1]), 1e-10)
            obj_rel_change = obj_range / obj_scale
            
            if obj_rel_change < conv_cfg.obj_rel_tol:
                reasons_met.append(f"obj_plateau(range={obj_rel_change:.2e})")
        
        # 2. Projected gradient norm (most reliable for constrained)
        if conv_cfg.use_projected_gradient:
            proj_grad_norm = self._compute_projected_gradient_norm(x_curr)
            if proj_grad_norm < conv_cfg.proj_grad_tol:
                reasons_met.append(f"proj_grad(norm={proj_grad_norm:.2e})")
        
        # 3. Solution stable over window
        if conv_cfg.use_solution_stable and len(x_history) >= conv_cfg.window_size:
            # Check that all recent solutions are close to current
            recent = x_history[-conv_cfg.window_size:]
            x_norm = max(np.linalg.norm(x_curr), 1e-10)
            max_dist = max(np.linalg.norm(x - x_curr) for x in recent)
            x_rel_change = max_dist / x_norm
            
            if x_rel_change < conv_cfg.x_rel_tol:
                reasons_met.append(f"x_stable(range={x_rel_change:.2e})")
        
        # 4. Check feasibility (if required)
        if conv_cfg.require_feasibility:
            max_viol = self.problem.max_violation(x_curr)
            if max_viol > conv_cfg.feasibility_tol:
                return False, "still_infeasible", True  # Check happened but failed
        
        # Count how many criteria are enabled
        n_enabled = sum([
            conv_cfg.use_objective_plateau,
            conv_cfg.use_projected_gradient,
            conv_cfg.use_solution_stable
        ])
        
        # Require ALL enabled criteria to be met
        if len(reasons_met) >= n_enabled and n_enabled > 0:
            return True, f"converged({'; '.join(reasons_met)})", True
        
        return False, "", True  # This was a check iteration, but criteria not met
    
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
        self._converged = False
        self._convergence_reason = "max_iterations"
        self._iterations_used = 0
        
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
        # transform), lean, same backend + convergence config.
        inner_problem = OptimizationProblem(A=ctx.A, b=ctx.b, C=ctx.C, d=ctx.d)
        inner_config = replace(self.config, transform=None, record_trajectory=False)
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
            x_current, n_proj, spike_info = self._project_to_feasible(x_current)
            self._n_projections += n_proj

            if n_proj > 0 and self.config.record_spike_history:
                for info in spike_info:
                    self._spike_times.append(float(iteration))
                    self._spike_deltas.append(info["delta_x"])
                    self._spike_constraints.append(info["constraints"])
                    self._spike_violation_values.append(info["violations"])
            
            # Phase 3: Clip to box constraints if specified
            x_current = self._clip_to_bounds(x_current)
            
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
        
        # If we didn't converge early, record final state
        if not self._converged:
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
            x_current, n_proj, _ = self._project_to_feasible(x_current, build_info=False)
            self._n_projections += n_proj

            # Phase 3: clip to box constraints if specified
            x_current = self._clip_to_bounds(x_current)

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

        if not self._converged:
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
        )

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

        final_x, iters, n_proj, converged, reason_code = _kernel.solve_euler(
            A, b, C, d, c_norms_sq, c_gram, x0c,
            self._k0, self.config.constraint_tol,
            self.config.max_iterations, self.config.max_projection_iters,
            conv.enable_early_stopping, conv.check_every, conv.min_iterations,
            conv.window_size, conv.patience,
            conv.obj_rel_tol, conv.x_rel_tol, conv.proj_grad_tol,
            conv.feasibility_tol,
            conv.use_objective_plateau, conv.use_projected_gradient,
            conv.use_solution_stable, conv.require_feasibility,
            has_lower, float(self.config.lower_bound) if has_lower else 0.0,
            has_upper, float(self.config.upper_bound) if has_upper else 0.0,
            parallel=parallel,
            a_diag=a_diag_c, use_diag=use_diag,
        )

        self._n_projections = int(n_proj)
        self._converged = bool(converged)
        self._iterations_used = int(iters)
        self._convergence_reason = (
            "converged(c-backend)" if reason_code == 1 else "max_iterations")

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
        
        while t_current < self.config.t_end:
            # Phase 1: Project back into feasible region
            x_current, n_proj, spike_info = self._project_to_feasible(x_current)
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
                             build_info: bool = True) -> Tuple[np.ndarray, int, List[dict]]:
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
            return self._project_adaptive(x, build_info=build_info)
        else:
            return self._project_fixed(x, build_info=build_info)

    def _project_adaptive(self, x: np.ndarray,
                          build_info: bool = True) -> Tuple[np.ndarray, int, List[dict]]:
        """
        Adaptive projection: compute exact step to reach each constraint boundary.

        For violated constraint g_j(x) = c_j^T x + d_j > 0, the exact step is
        k1_j = g_j / ||c_j||^2; this projects exactly onto the boundary in one
        step per constraint, eliminating k1 as a hyperparameter.

        When the constraint Gram matrix G = C C^T has been precomputed (dense C,
        m <= _MAX_GRAM_M), the residual g = C x_proj + d is maintained
        incrementally: each projection event on constraint j applies the lateral
        update g <- g - k1 * G[:,j] (O(m)), the constraint-coupling form of a
        spike-triggered update, instead of recomputing C x_proj (O(m*n)). For
        sparse C or large m it falls back to recomputing the residual.
        """
        x_proj = x.copy()
        n_iters = 0
        spike_info: List[dict] = []

        # Handle unconstrained case
        if self.problem.n_constraints == 0:
            return x_proj, 0, spike_info

        gram = self._c_gram  # None -> recompute path; else event-driven path
        g = self.problem.constraint_values(x_proj)

        for _ in range(self.config.max_projection_iters):
            if gram is None:
                g = self.problem.constraint_values(x_proj)

            # Find most violated constraint
            j = np.argmax(g)
            if g[j] <= self.config.constraint_tol:
                break  # All constraints satisfied

            # Compute exact step to reach boundary: k1 = g_j / ||c_j||^2
            c_j = self.problem.C[j]
            # Convert sparse row to dense 1D array
            if _issparse(c_j):
                c_j = np.asarray(c_j.todense()).ravel()
            else:
                c_j = np.asarray(c_j).ravel()
            violation = g[j]

            # Avoid division by zero for degenerate constraints
            if self._c_norms_sq[j] < 1e-12:
                continue

            k1_adaptive = violation / self._c_norms_sq[j]

            # Project: x <- x - k1 * c_j
            delta_x = -k1_adaptive * c_j
            x_proj = x_proj + delta_x
            if gram is not None:
                # Spike j propagates a lateral update to coupled constraints.
                g = g - k1_adaptive * gram[j]
            n_iters += 1

            if build_info:
                spike_info.append({
                    "constraints": np.array([j]),
                    "delta_x": delta_x,
                    "violations": np.array([violation])
                })

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

        for _ in range(self.config.max_projection_iters):
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
            total_projection_distance=total_projection_distance
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
