# SNN Solver Examples

This directory contains example scripts demonstrating various features and use cases of the SNN-inspired optimization solver.

## Running Examples

From the project root directory:

```bash
# Run individual examples
python examples/example1_simple_2d.py
python examples/example1_basic_2d.py
python examples/example1_advanced_2d.py
python examples/example2_3d_polytope.py
python examples/example3_linear_program.py
python examples/example4_warm_start.py
python examples/example5_infeasible_recovery.py
python examples/example6_equality_constraint.py
python examples/example7_svm_dual.py
python examples/example_raw_mode.py

# Run all examples in sequence
python examples/run_all_examples.py
```

Or from within the examples directory:

```bash
cd examples
python example1_simple_2d.py
# ... etc
```

## Example Descriptions

### Example 1: Simple 2D QP (`example1_simple_2d.py`)
**Problem**: Minimize `||x||²` subject to linear constraints
- **Difficulty**: Beginner
- **Features**: Basic QP, constraint satisfaction checking
- **Output**: Solution, objective value, constraint status
- **Use Case**: Understanding the solver basics

### Example 1 (Basic figure): 2D trajectory + spikes (`example1_basic_2d.py`)
**Problem**: The canonical 2-D QP, run in fixed-projection mode for illustration
- **Difficulty**: Beginner
- **Features**: Fixed-step projection (`projection_method='fixed'`), spike-pattern plotting
- **Output**: `example1_basic_2d.png`, the trajectory-with-spikes figure used in the README
- **Use Case**: The reference spike-pattern illustration for figures and slides

### Example 1 (Advanced): Diagnostics in 2D (`example1_advanced_2d.py`)
**Problem**: Same QP as Example 1 with the feasible region shifted to exclude the unconstrained minimizer
- **Difficulty**: Intermediate
- **Features**: Projection behaviour visualization, objective/violation plots, infeasible starting point
- **Output**: Plots of objective/constraint violations and 2D trajectory, solver summary
- **Use Case**: Understanding discrete "spikes" when constraints activate

### Example 2: 3D Polytope Constraints (`example2_3d_polytope.py`)
**Problem**: Minimize `||x||²` in 3D with 4 hyperplane constraints
- **Difficulty**: Intermediate
- **Features**: Multiple active constraints, polytope geometry
- **Output**: Solution analysis, active constraint detection
- **Use Case**: Problems with multiple constraints, vertex solutions

### Example 3: Linear Program (`example3_linear_program.py`)
**Problem**: Minimize `b'x` subject to box constraints
- **Difficulty**: Intermediate
- **Features**: LP solving (A=0), vertex solution verification
- **Output**: Comparison with analytical solution
- **Use Case**: Pure linear objective optimization

### Example 4: Warm Starting (`example4_warm_start.py`)
**Problem**: Sequence of 10 QPs with slowly changing constraints
- **Difficulty**: Advanced
- **Features**: Warm starting, receding horizon control simulation
- **Output**: Solve times, projection counts, convergence statistics
- **Use Case**: Model predictive control, real-time optimization
- **Key Insight**: Warm starting removes the projection work entirely on this toy (1 → 0 events) and cuts solve time roughly 2-3x; the quantitative MPC-style benchmark is [`benchmarks/03_warm_start.py`](../benchmarks/03_warm_start.py) (221 → 101 iterations, 2.19x)

### Example 5: Infeasible Recovery (`example5_infeasible_recovery.py`)
**Problem**: Starting from various infeasible points
- **Difficulty**: Intermediate
- **Features**: Automatic projection to feasibility, robustness testing
- **Output**: Recovery analysis for multiple starting points
- **Use Case**: Handling uncertain or infeasible initializations

### Example 6: Equality Constraint Approximation (`example6_equality_constraint.py`)
**Problem**: Enforce an equality constraint `x1 = a` by sandwiching the variable between two inequalities with slack
- **Difficulty**: Intermediate
- **Features**: Equality-as-inequality band, infeasible start, verbose solver output
- **Output**: Solver summary, feasibility band diagnostics for the final solution
- **Use Case**: Demonstrating how to model equality constraints and tolerances with the SNN solver

### Example 7: SVM Dual Problem (`example7_svm_dual.py`)
**Problem**: Support Vector Machine dual optimization with kernel trick
- **Difficulty**: Advanced
- **Features**: **Auto k0**, **box bounds as implicit projection facets**, equality constraint (y^T alpha = 0), **convergence diagnostics**
- **Output**: Support vector analysis, training accuracy, convergence info
- **Use Case**: Machine learning classification, demonstrating new solver features
- **Key Insight**: Box bounds (0 ≤ alpha ≤ C) as facets of the same projection sweep + auto k0 make SVM solving robust and automatic

### Raw vs. optimized modes (`example_raw_mode.py`)
**Problem**: The same QP solved with hand-set raw parameters vs. the auto-configured defaults
- **Difficulty**: Intermediate
- **Features**: Bypassing auto-configuration, side-by-side convergence comparison
- **Output**: `raw_vs_optimized.png`
- **Use Case**: Understanding what the auto step size and adaptive projection buy you

## Example Output Format

Most examples print:
- Problem description
- Initial conditions
- Solver progress (if verbose)
- Final solution and objective value
- Constraint satisfaction verification
- Performance metrics (projections, time steps, etc.)
- **Convergence diagnostics**: `converged`, `convergence_reason`, `iterations_used`, and since v0.6.0 the KKT certificate fields `kkt_residual`, `kkt_tolerance`, `kkt_scale`, `kkt_fit_status` (`final_proj_grad_norm` is a legacy heuristic)

## Extending Examples

After installing the package (`pip install -e .` from the repo root), authoring
a new example reduces to:

```python
import numpy as np
from snn_opt import OptimizationProblem, SNNSolver, SolverConfig

# Define your problem
A  = ...  # Hessian (n × n, PSD)
b  = ...  # Linear cost (n,)
C  = ...  # Constraint matrix (m × n)
d  = ...  # Constraint offset (m,)
x0 = ...  # Initial guess (n,)

problem = OptimizationProblem(A=A, b=b, C=C, d=d)
config  = SolverConfig(
    k0=None,            # Auto-compute step size from λ_max(A)
    k0_scale=0.5,       # Conservatism factor on the auto step
    max_iterations=2000,
    lower_bound=0.0,    # Optional box bounds (implicit facets)
    upper_bound=1.0,    # Optional box bounds (implicit facets)
    # Early stopping is enabled by default; tweak via config.convergence
)
solver  = SNNSolver(problem, config)
result  = solver.solve(x0, verbose=True)

print(result.summary())
print(f"Converged: {result.converged} ({result.convergence_reason})")
```

If running directly from a clean checkout without `pip install`, use the same
`sys.path` bootstrap the bundled examples use:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
```

## Tips for Running Examples

1. **Start with Example 1**: It's the simplest and helps verify installation
2. **Example 4 is most relevant** for control applications
3. **Example 7 demonstrates** auto k0 and box bounds on a real ML problem
4. **Use `k0=None`** to enable automatic step size computation
5. **Use `lower_bound/upper_bound`** for problems with simple box constraints
6. **Use `verbose=True`** to see solver progress during optimization
7. **Check `result.converged`** and `result.convergence_reason` for diagnostics
8. **Early stopping is enabled by default**: set `config.convergence.enable_early_stopping=False` to disable

