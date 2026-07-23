# API Reference

Public symbols re-exported from `snn_opt`. All importable as

```python
from snn_opt import (
    OptimizationProblem,
    SolverConfig,
    ConvergenceConfig,
    SolverResult,
    SNNSolver,
    solve_qp,
)
```

## `solve_qp(A, b, C, d, x0, ...) -> SolverResult`

Convenience function that wraps `OptimizationProblem` + `SolverConfig` +
`SNNSolver.solve` for one-shot QPs.

| Argument | Type | Notes |
|---|---|---|
| `A` | `(n,n) array` | PSD Hessian (use `np.zeros((n,n))` for an LP). |
| `b` | `(n,) array` | Linear cost. |
| `C` | `(m,n) array` | Inequality matrix. |
| `d` | `(m,) array` | Inequality offset; constraints are `Cx + d ≤ 0`. |
| `x0` | `(n,) array` | Initial iterate (may be infeasible). |
| `k0` | `float` or `None` | Gradient step. `None` ⇒ auto from `λ_max(A)`. |
| `t_end` | `float` | Simulation horizon for `'ivp'` mode. |
| `max_iterations` | `int` | Cap for `'euler'` mode. |
| `integration_method` | `'euler'` (default) or `'ivp'` | |
| `projection_method` | `'adaptive'` (default) or `'fixed'` | Adaptive eliminates `k1`. |
| `k0_scale` | `float` | Conservatism factor on auto step. Default `0.5`. |
| `lower_bound`, `upper_bound` | `float` or `None` | Box clipping (e.g. SVM dual). |
| `enable_early_stopping` | `bool` | Convergence-based termination, default on. |
| `record_trajectory` | `bool` | Keep the full iterate trajectory + spike events (default `True`). `False` runs the lean path; the compiled backends imply `False`. |
| `backend` | `str` | `'python'` (default), `'c'` (auto), `'c_serial'`, or `'c_openmp'`. See [`SolverConfig`](#solverconfig). |
| `verbose` | `bool` | Print solver progress. |

Returns: a [`SolverResult`](#solverresult).

## `OptimizationProblem`

Dataclass holding `A, b, C, d`. Methods:

- `objective(x)`: evaluate `½ xᵀAx + bᵀx`
- `gradient(x)`: `Ax + b`
- `constraint_values(x)`: `Cx + d`
- `is_feasible(x)`: boolean
- `max_violation(x)`: scalar

## `SolverConfig`

Solver hyper-parameters with sensible defaults. Most users only ever set
`max_iterations`, `lower_bound`, `upper_bound`, and `convergence`.

| Field | Default | Meaning |
|---|---|---|
| `k0` | `None` | Step size; `None` auto-computes from `λ_max(A)`. |
| `k0_scale` | `0.5` | Multiplier on the auto step (lower = safer). |
| `t_end` | `100.0` | IVP mode horizon. |
| `max_step` | `0.1` | IVP mode max ODE step. |
| `constraint_tol` | `1e-6` | Tolerance for "constraint violated". |
| `max_projection_iters` | `100` | Per-iteration projection cap. |
| `integration_method` | `'euler'` | `'euler'` or `'ivp'`. |
| `max_iterations` | `2000` | Outer-iteration cap (Euler). |
| `projection_method` | `'adaptive'` | `'adaptive'` or `'fixed'`. |
| `k1` | `0.05` | Projection step (only used when `projection_method='fixed'`). |
| `lower_bound`, `upper_bound` | `None` | Box clipping. |
| `record_trajectory` | `True` | Store the full iterate trajectory + per-spike events. `False` runs the lean solve (final state only); the compiled backends always run lean. |
| `backend` | `'python'` | Solve backend. `'python'` is the NumPy reference. The compiled pybind11 kernel (dense + `projection_method='adaptive'` only) comes in three numerically identical variants differing only in matvec threading: `'c'` (auto: OpenMP multicore when the wheel was built with it *and* the problem is large enough to amortize it, else single-thread), `'c_serial'` (forced single-thread), `'c_openmp'` (forced multicore; raises if the build lacks OpenMP). Only the matvec is parallel; the Euler recurrence + greedy projection are serial. Honours `OMP_NUM_THREADS`; `snn_opt._kernel.HAS_OPENMP` / `max_threads()` report the build's capability. |
| `transform` | `None` | Optional problem transform (the *transform axis*). `None` = canonical solve. A name (`'eigenbasis'`) or a `Transform` instance opts in; the problem is solved in transformed coordinates and mapped back. Composes with any backend; implies the lean result. See [Transforms](#transforms). |
| `convergence` | `ConvergenceConfig()` | See below. |

## `ConvergenceConfig`

| Field | Default | Meaning |
|---|---|---|
| `enable_early_stopping` | `True` | Master switch. |
| `obj_rel_tol` | `1e-8` | Relative-objective plateau over `window_size`. |
| `x_rel_tol` | `1e-8` | Relative iterate change. |
| `proj_grad_tol` | `1e-6` | Projected-gradient norm tolerance. |
| `feasibility_tol` | `1e-2` | Maximum violation to count as converged. |
| `check_every` | `50` | Stride between convergence checks. |
| `min_iterations` | `100` | No early-stop before this. |
| `window_size` | `10` | Plateau-detection window. |
| `patience` | `3` | Consecutive passing checks needed. |
| `use_objective_plateau` | `True` | Enable plateau criterion. |
| `use_projected_gradient` | `True` | Enable projected-gradient criterion. |
| `use_solution_stable` | `False` | Off by default, prone to false positives. |
| `require_feasibility` | `True` | Insist on feasibility for "converged". |

## `SNNSolver(problem, config=None)`

The full solver. Use this (rather than `solve_qp`) when you want to amortize
problem construction across many warm-started solves.

- `solver.solve(x0, verbose=False) -> SolverResult`: run the dynamics from
  `x0` and return diagnostics.

## `SolverResult`

Returned by `solve_qp` and `SNNSolver.solve`. Notable fields:

- `final_x`, `final_objective`, `final_proj_grad_norm`: solution and summary.
- `converged`, `convergence_reason`, `iterations_used`: termination info.
- `t`, `X`: full trajectory `(T,)` and `(T, n)`.
- `objective_values`, `constraint_violations`: `(T,)` per iteration.
- `n_projections`: total projection sub-iterations.
- `spike_times`, `spike_deltas`, `spike_norms`, `spike_constraints`,
  `spike_violation_values`: per-spike diagnostics, the raw material for
  the projection-spike raster (see [`02_spike_raster.py`](../benchmarks/02_spike_raster.py)).
- `total_projection_distance`: sum of spike norms.
- `summary()`: human-readable one-line-per-statistic string.

### Correctness fields (v0.5.0)

These four report whether the answer should be trusted, and are the ones to
check on any nontrivial problem. `converged` alone is not sufficient: it says
the network reached a fixed point, not that the fixed point solves your QP.

| Field | Meaning |
|---|---|
| `joint_feasible` | Feasibility of the rows of `C` **and** the bounds together. Before v0.5.0 the convergence gate was rows-only, so a bound violation could not fail it. This is the flag to check. |
| `stationarity_residual` | An NNLS KKT certificate, `min_{mu >= 0} ‖∇f + Σ mu_i a_i‖`, over the active unified normals at the final point. Large values mean the fixed point is far from stationary even if `converged` is `True`. |
| `projection_budget_exhausted` | The inner sweep hit its `max_projection_iters` watchdog. The solve **aborts** with `convergence_reason='projection_budget_exhausted'` rather than reporting success from a knowingly infeasible point. |
| `max_violation_rows_raw`, `max_distance_rows`, `max_violation_box` | The components behind `joint_feasible`: raw row residual, row residual as a Euclidean distance (`residual / ‖c_j‖`), and the worst bound violation. |

Spike IDs in `spike_constraints` cover the implicit bound facets too, in a
frozen order: rows in input order, then lower facets `0..n-1`, then upper facets
`0..n-1`. So lower facet `i` is reported as `m + i` and upper facet `i` as
`m + n + i`.

See the README's [Accuracy and tuning](../README.md#accuracy-and-tuning) section
for what to do when `stationarity_residual` is large.

## Transforms

`snn_opt.transforms` is the **transform axis**: an explicit, backend-agnostic
rewrite of the problem that is solved in transformed coordinates and mapped back.
Transforms operate on the problem data (`A, b, C, d`), not the solve loop, so they
compose with every backend. Opt in via `SolverConfig(transform=...)`; the
canonical solver is the default.

```python
from snn_opt import solve_qp, EigenbasisTransform
solve_qp(A, b, C, d, x0, ...)                                   # canonical
# via SolverConfig:
cfg = SolverConfig(transform='eigenbasis')                     # by name
cfg = SolverConfig(transform=EigenbasisTransform())            # by instance
```

| Symbol | Notes |
|---|---|
| `Transform` | Base class. Subclass and implement `forward(problem, x0, config)` (and usually `check_applicable`). |
| `EigenbasisTransform` (`'eigenbasis'`) | Rotates a symmetric-PSD Hessian into its eigenbasis (`A = VΛVᵀ`), so the dominant `O(n²)` `A @ x` gradient step becomes an `O(n)` elementwise product `Λ ⊙ ỹ`; constraints rotate to `Ĉ = CV` with the Gram/row-norms invariant, so the projection is unchanged. Recovers `x = V ỹ`. Since v0.5.0 **box bounds are accepted**: they are not rotation-invariant, so they are materialized as explicit rotated unit-norm rows (`m` grows by up to `2n`), giving up the implicit `O(1)` facet advantage under a transform. Best on the compiled backends and larger `n`. |

## Versioning

`snn_opt` follows [SemVer](https://semver.org). The public API listed above
is the *commitment surface*: anything else (`snn_opt.solver._private_helper`,
internal config defaults that are not in `ConvergenceConfig` /
`SolverConfig`) may change between minor releases.
