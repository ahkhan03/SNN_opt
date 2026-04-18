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
| `verbose` | `bool` | Print solver progress. |

Returns: a [`SolverResult`](#solverresult).

## `OptimizationProblem`

Dataclass holding `A, b, C, d`. Methods:

- `objective(x)` — evaluate `½ xᵀAx + bᵀx`
- `gradient(x)` — `Ax + b`
- `constraint_values(x)` — `Cx + d`
- `is_feasible(x)` — boolean
- `max_violation(x)` — scalar

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
| `use_solution_stable` | `False` | Off by default — false positives. |
| `require_feasibility` | `True` | Insist on feasibility for "converged". |

## `SNNSolver(problem, config=None)`

The full solver. Use this (rather than `solve_qp`) when you want to amortize
problem construction across many warm-started solves.

- `solver.solve(x0, verbose=False) -> SolverResult` — run the dynamics from
  `x0` and return diagnostics.

## `SolverResult`

Returned by `solve_qp` and `SNNSolver.solve`. Notable fields:

- `final_x`, `final_objective`, `final_proj_grad_norm` — solution + summary.
- `converged`, `convergence_reason`, `iterations_used` — termination info.
- `t`, `X` — full trajectory `(T,)` and `(T, n)`.
- `objective_values`, `constraint_violations` — `(T,)` per iteration.
- `n_projections` — total projection sub-iterations.
- `spike_times`, `spike_deltas`, `spike_norms`, `spike_constraints`,
  `spike_violation_values` — per-spike diagnostics, the raw material for
  the projection-spike raster (see [`figure 02_spike_raster.py`](../benchmarks/02_spike_raster.py)).
- `total_projection_distance` — sum of spike norms.
- `summary()` — human-readable one-line-per-statistic string.

## Versioning

`snn_opt` follows [SemVer](https://semver.org). The public API listed above
is the *commitment surface*: anything else (`snn_opt.solver._private_helper`,
internal config defaults that are not in `ConvergenceConfig` /
`SolverConfig`) may change between minor releases.
