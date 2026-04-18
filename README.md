# SNN-Inspired Optimization Solver

Python implementation of a spiking neural network-inspired solver for constrained convex optimization.

> **For detailed mathematical background and theory**, see [`snn_optimization_framework.md`](snn_optimization_framework.md).

## Overview

This solver tackles optimization problems of the form:

```
minimize    (1/2) x^T A x + b^T x
subject to  C x + d <= 0
```

The algorithm alternates between:
1. **Gradient descent**: Following the negative gradient (Euler or continuous ODE integration)
2. **Boundary projections**: Discrete corrections when constraints are violated (analogous to neural spikes)

### Key Features

- **Auto k0**: Automatically computes gradient step size from Hessian eigenvalue—eliminates `k0` tuning
- **Adaptive projection**: Automatically computes exact step size to reach constraint boundaries—eliminates `k1` as a hyperparameter
- **Box constraint clipping**: Handles simple bounds (0 ≤ x ≤ C) via clipping—more stable and neuromorphic than projection
- **Euler integration**: More stable for tightly constrained problems (e.g., SVM dual)
- **Early stopping**: Multi-criteria convergence detection (objective plateau, projected gradient norm, feasibility)
- **Spike diagnostics**: Records projection timestamps, displacements, active constraints, and magnitudes
- **Warm starting**: Excellent performance for sequences of similar problems (e.g., MPC)

## Installation

### Requirements

- Python 3.8+
- NumPy
- SciPy

Install dependencies:

```bash
pip install numpy scipy
```

Or use the provided requirements file:

```bash
pip install -r requirements.txt
```

## Project Structure

```
SSN_robot/
├── snn_solver.py           # Core solver implementation
├── requirements.txt        # Python dependencies
├── test_installation.py    # Quick verification test
├── README.md              # This file
├── snn_optimization_report.md  # Detailed mathematical background
├── examples/              # Python example scripts
│   ├── example1_simple_2d.py
│   ├── example2_3d_polytope.py
│   ├── example3_linear_program.py
│   ├── example4_warm_start.py
│   ├── example5_infeasible_recovery.py
│   ├── run_all_examples.py
│   └── README.md
└── solver_matlab_code/    # MATLAB reference implementation
    ├── snn_solver.m
    ├── example1.m
    ├── example2.m
    └── main_manip.m
```

## Quick Start

### Basic Usage

```python
import numpy as np
from snn_solver import solve_qp

# Define problem: minimize ||x||^2 subject to x1 + 2*x2 <= 1
A = np.eye(2)
b = np.zeros(2)
C = np.array([[1.0, 2.0]])
d = np.array([-1.0])
x0 = np.array([1.0, 1.0])

# Solve (uses auto k0, Euler integration, adaptive projection by default)
result = solve_qp(A, b, C, d, x0, max_iterations=1000)

print(f"Solution: {result.final_x}")
print(f"Objective: {result.final_objective}")
```

### With Box Constraints (e.g., SVM)

```python
# For SVM-style problems with 0 <= x <= C bounds
result = solve_qp(A, b, C, d, x0,
    lower_bound=0.0,   # x >= 0 (handled by clipping)
    upper_bound=1.0,   # x <= C (handled by clipping)
    max_iterations=2000
)
```

### Advanced Usage with Objects

```python
from snn_solver import OptimizationProblem, SNNSolver, SolverConfig

# Define problem
problem = OptimizationProblem(A=A, b=b, C=C, d=d)

# Configure solver
config = SolverConfig(
    k0=0.05,                    # Gradient descent step size
    max_iterations=1000,        # Max iterations (Euler mode)
    integration_method='euler', # 'euler' (recommended) or 'ivp'
    projection_method='adaptive' # 'adaptive' (no k1 needed) or 'fixed'
)

# Create and run solver
solver = SNNSolver(problem, config)
result = solver.solve(x0, verbose=True)

# Access detailed results
print(result.summary())
print(f"Trajectory length: {len(result.t)}")
print(f"Number of projections: {result.n_projections}")
```

## Examples

Five comprehensive examples are provided in the `examples/` directory:

### Example 1: Simple 2D QP
Basic quadratic program with two linear constraints. Good starting point.

```bash
python examples/example1_simple_2d.py
```

### Example 2: 3D Polytope Constraints
Three-dimensional problem with four constraints defining a polytope. Demonstrates handling multiple active constraints.

```bash
python examples/example2_3d_polytope.py
```

### Example 3: Linear Program
Pure linear program (A=0) with box constraints. Shows LP solving capability and vertex solutions.

```bash
python examples/example3_linear_program.py
```

### Example 4: Warm Starting
Demonstrates receding horizon control scenario where optimization problems are solved sequentially with warm starts. Mimics the manipulator control application.

```bash
python examples/example4_warm_start.py
```

### Example 5: Infeasible Recovery
Shows how the solver handles infeasible starting points and automatically projects them to feasibility.

```bash
python examples/example5_infeasible_recovery.py
```

### Run All Examples

```bash
python examples/run_all_examples.py
```

## API Reference

### `OptimizationProblem`

Encapsulates problem definition.

**Parameters:**
- `A`: Hessian matrix (n × n)
- `b`: Linear cost vector (n,)
- `C`: Constraint matrix (m × n)
- `d`: Constraint offset vector (m,)

**Methods:**
- `objective(x)`: Evaluate objective function
- `gradient(x)`: Evaluate gradient
- `constraint_values(x)`: Evaluate g(x) = Cx + d
- `is_feasible(x)`: Check constraint satisfaction
- `max_violation(x)`: Maximum constraint violation

### `SolverConfig`

Solver configuration parameters.

**Parameters:**
- `k0`: Gradient descent step size. **Set to `None` (default) to auto-compute from Hessian eigenvalue**
- `k0_scale`: Scaling factor for auto-computed k0 (default: 0.5). Final k0 = k0_scale / λ_max(A)
- `t_end`: Simulation end time for IVP mode (default: 100.0)
- `max_step`: Maximum ODE integration step for IVP mode (default: 0.1)
- `constraint_tol`: Constraint violation tolerance (default: 1e-6)
- `max_projection_iters`: Max projection iterations per step (default: 100)
- `integration_method`: `'euler'` (recommended) or `'ivp'` (default: `'euler'`)
- `max_iterations`: Maximum iterations for Euler mode (default: 2000)
- `projection_method`: `'adaptive'` (recommended, no k1 needed) or `'fixed'` (default: `'adaptive'`)
- `k1`: Projection step size, only used when `projection_method='fixed'` (default: 0.05)
- `lower_bound`: Lower bound for box constraint clipping (default: None = no clipping)
- `upper_bound`: Upper bound for box constraint clipping (default: None = no clipping)
- `convergence`: `ConvergenceConfig` object for early stopping (see below)

### `ConvergenceConfig`

Configuration for early stopping and convergence detection.

**Parameters:**
- `enable_early_stopping`: Enable/disable early stopping (default: True)
- `obj_rel_tol`: Relative objective change tolerance over window (default: 1e-8)
- `proj_grad_tol`: Projected gradient norm tolerance (default: 1e-6)
- `feasibility_tol`: Max constraint violation for convergence (default: 1e-2)
- `check_every`: Check convergence every N iterations (default: 50)
- `min_iterations`: Minimum iterations before checking (default: 100)
- `window_size`: Window size for objective plateau detection (default: 10)
- `patience`: Consecutive converged checks needed (default: 3)
- `use_objective_plateau`: Use objective plateau criterion (default: True)
- `use_projected_gradient`: Use projected gradient norm criterion (default: True)

### `SNNSolver`

Main solver class.

**Constructor:**
```python
SNNSolver(problem: OptimizationProblem, config: SolverConfig = None)
```

**Methods:**
- `solve(x0, verbose=False)`: Solve optimization from initial guess x0

### `SolverResult`

Contains optimization results.

**Attributes:**
- `t`: Time points array
- `X`: State trajectory (len(t) × n)
- `objective_values`: Objective along trajectory
- `constraint_violations`: Max violation at each point
- `n_projections`: Total projection events
- `converged`: Boolean convergence flag
- `convergence_reason`: String describing why solver stopped
- `iterations_used`: Actual number of iterations executed
- `final_x`: Final solution vector
- `final_objective`: Final objective value
- `final_proj_grad_norm`: Projected gradient norm at final solution
- `spike_times`: Time stamps at which projection spikes occurred
- `spike_deltas`: Spike displacements (rows align with `spike_times`)
- `spike_norms`: L2 norm of each spike displacement
- `spike_constraints`: List of constraint indices active during each spike
- `spike_violation_values`: Positive residuals for the active constraints at each spike
- `total_projection_distance`: Sum of spike norms, measuring cumulative projection effort

**Methods:**
- `summary()`: Print summary statistics

## Parameter Tuning Guidelines

### Gradient Descent Step Size (`k0`)
- **Auto mode (recommended)**: Set `k0=None` to auto-compute as `k0 = k0_scale / λ_max(A)`
- **Manual mode**: Start with `k0 ≈ 0.01 - 0.1`
- Adjust `k0_scale` (default 0.5) if auto mode is too slow (increase) or unstable (decrease)

### Box Constraints (`lower_bound`, `upper_bound`)
- For SVM: `lower_bound=0, upper_bound=C`
- More stable and efficient than treating bounds as linear inequalities
- Neuromorphic interpretation: neuron firing rate saturation

### Projection Method
- **`'adaptive'` (recommended)**: Computes exact step to reach each constraint boundary. Eliminates `k1` as a hyperparameter. Works well for most problems including tightly constrained ones (SVM, etc.).
- **`'fixed'`**: Uses constant step size `k1`. May require tuning.

### Projection Step Size (`k1`) — only for `projection_method='fixed'`
- Typically set `k1 ≈ k0`
- Larger values → fewer projection iterations needed
- Smaller values → more gentle corrections
- If many projections needed, increase `k1`

### Integration Method
- **`'euler'` (recommended)**: Discrete steps, more stable for tightly constrained problems
- **`'ivp'`**: Continuous ODE integration with event detection, original method

### Iterations / Simulation Time
- For Euler mode: `max_iterations` typically 500-2000
- For IVP mode: `t_end` typically 50-200
- Monitor convergence by checking objective stabilization

### Tolerance (`constraint_tol`)
- Default `1e-6` works for most problems
- Decrease for higher precision (may need more iterations)
- Increase for faster convergence with looser constraints

## Performance Characteristics

### Computational Complexity
- Per iteration: O(n² + mn) where n = variables, m = constraints
- Typical convergence: 10-100 iterations
- No matrix factorizations required
- Well-suited for embedded/real-time applications

### When to Use This Solver
✓ Real-time control applications  
✓ Frequent re-solves with warm starts  
✓ Embedded systems with limited computation  
✓ Problems where approximate solutions suffice  
✓ Receding horizon control  

### When NOT to Use
✗ Need high-accuracy solutions (< 1e-8 error)  
✗ Large-scale problems (n > 1000)  
✗ Ill-conditioned problems  
✗ Problems requiring optimality certificates  

## Design Philosophy

The implementation follows these principles:

1. **Simplicity**: Core algorithm in ~200 lines, no complex data structures
2. **Extensibility**: Class-based design allows easy addition of visualization, monitoring, adaptive parameters
3. **Fast-fail**: Errors propagate immediately (per user rules), no defensive programming
4. **Clean API**: Both simple function interface and detailed object-oriented interface

## Comparison with MATLAB Implementation

| Feature | MATLAB | Python |
|---------|--------|--------|
| Core algorithm | ✓ | ✓ |
| Event detection | `odeset('Events')` | `solve_ivp(events=)` |
| Euler integration | - | ✓ |
| Adaptive projection | - | ✓ |
| Results storage | Cell arrays | Dataclasses |
| Problem definition | Loose parameters | `OptimizationProblem` class |
| Configuration | Function args | `SolverConfig` class |
| Trajectory access | Concatenated arrays | `SolverResult` object |

## Recent Improvements

- **Early stopping**: Multi-criteria convergence detection with projected gradient norm, objective plateau, and feasibility checks
- **Auto k0**: Computes gradient step from Hessian eigenvalue—eliminates `k0` tuning
- **Box constraint clipping**: Handles bounds via clipping—more stable and neuromorphic
- **Adaptive projection**: Eliminates `k1` by computing exact steps to constraint boundaries
- **Euler integration**: More stable for tightly constrained problems
- **Relaxed tolerance**: Default `1e-6` (from `1e-10`) for practical convergence
- **Enhanced diagnostics**: `SolverResult` now includes `convergence_reason`, `iterations_used`, and `final_proj_grad_norm`

## Future Extensions

The class-based design facilitates:

### Performance Enhancements
- **Per-iteration adaptive `k0`**: Barzilai-Borwein step size for faster convergence
- **Sparse matrix support**: Efficient handling of large-scale problems with sparse Hessian/constraints
- **Numba JIT compilation**: Accelerate core projection loop for 10-100x speedup
- **Hardware acceleration**: GPU or neuromorphic chip implementations

### API & Usability
- **Builder pattern**: Fluent API for solver configuration
- **Visualization**: Built-in plotting methods for trajectories and convergence
- **Callbacks**: User-defined functions called during solve for monitoring
- **Parallel solves**: Batch optimization for multiple scenarios

### Advanced Features  
- **Line search**: Optional Armijo backtracking for more robust convergence
- **Direct equality constraints**: Native handling without two-inequality conversion
- **Regularization options**: Tikhonov or proximal regularization for ill-conditioned problems

## Theory and References

For detailed mathematical background, derivations, and theoretical analysis, see:

📖 **[`snn_optimization_framework.md`](snn_optimization_framework.md)** — Complete theoretical framework including:
- Connection to spiking neural network dynamics
- Derivation of adaptive projection formula
- Convergence properties
- Extension to equality constraints
- Application to robotic control
