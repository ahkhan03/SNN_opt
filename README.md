# snn_opt

**A spiking neural network solver for constrained convex optimization.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.0-informational.svg)](CHANGELOG.md)
[![Cite](https://img.shields.io/badge/cite-CITATION.cff-orange.svg)](CITATION.cff)
[![Docs](https://img.shields.io/badge/docs-snn.ahkhan.me-success.svg)](https://snn.ahkhan.me)

---

## Abstract

`snn_opt` is a Python implementation of the **spiking neural network (SNN) → convex optimization** equivalence, developed as part of an ongoing research program on neuromorphic computation for classical machine-learning problems. It solves quadratic and linear programs of the form

$$
\min_{x \in \mathbb{R}^n}\ \tfrac{1}{2}\, x^\top A x + b^\top x
\quad\text{subject to}\quad C x + d \le 0,
$$

by alternating gradient descent — playing the role of leaky-integrate membrane drift — with discrete projection events that clamp the trajectory to the constraint boundary, the optimization analogue of an integrate-and-fire **spike**. The construction follows Mancoo, Boerlin and Machens ([NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/64714a86909d401f8feb83e8c2d94b23-Abstract.html)) and is the canonical solver underlying the **SNN-X** publication series (PCA, Ridge, TDSVM, Norm, SVM, CF, KRR, Procrustes — see [Applications](#applications)).

This repository is intended both as a **research artifact** — every published SNN-X paper can be reproduced from the code here — and as a **teaching resource** for students entering the area: it ships with annotated examples, a self-contained mathematical writeup, and a benchmark suite that visualizes convergence and projection dynamics.

## The problem

Given a positive semi-definite Hessian $A \in \mathbb{R}^{n\times n}$, a linear cost $b \in \mathbb{R}^n$, and $m$ linear inequality constraints stacked into $C \in \mathbb{R}^{m \times n}$ and $d \in \mathbb{R}^m$, we seek

$$
x^\star \;=\; \arg\min_{x}\ \tfrac{1}{2}\, x^\top A x + b^\top x \quad\text{s.t.}\quad c_i^\top x + d_i \le 0,\ i = 1,\dots,m.
$$

The class subsumes box-constrained QPs (set $C = [I; -I]$), linear programs ($A = 0$), kernel-ridge subproblems, support-vector machine duals, projected-gradient flows on polytopes, and the bulk of the inner solves that arise in receding-horizon control.

## The spiking idea, in one picture

The continuous-time dynamics

$$
\dot x \;=\; -\nabla f(x) \;-\; C^\top s(t)
$$

models a population of $n$ leaky integrators driven by the gradient $\nabla f(x) = Ax + b$, with a corrective spike train $s(t)$ that fires whenever an inequality $c_i^\top x + d_i$ would otherwise become positive. Each spike applies a *minimal* projection that re-enters the feasible set; spike inter-arrival times encode constraint *traffic*. Discretized with forward Euler and an adaptive step that reaches the boundary exactly, this becomes a fast projected-gradient solver with diagnostics that double as a neural raster plot.

See [`docs/theory.md`](docs/theory.md) for the full derivation, including the eigenvalue-based step-size choice that eliminates `k0` as a hyperparameter and the box-clipping shortcut for problems like SVM.

## Installation

`snn_opt` requires Python 3.9+, NumPy, and SciPy. Install from a checkout:

```bash
git clone https://github.com/ahkhan03/snn_opt.git
cd snn_opt
pip install -e .                   # core
pip install -e ".[examples]"       # also installs matplotlib for examples
pip install -e ".[dev]"            # examples + cvxpy + pytest + ruff
```

The package can also be run **without** installation — every example and test sits next to a small `sys.path` bootstrap that points at `src/`. Smoke test:

```bash
python tests/test_installation.py
```

## Quick start

```python
import numpy as np
from snn_opt import solve_qp

# Minimize ||x||^2 subject to  x_1 + 2 x_2 <= 1  (and that's it).
A  = np.eye(2)
b  = np.zeros(2)
C  = np.array([[1.0, 2.0]])
d  = np.array([-1.0])
x0 = np.array([1.0, 1.0])

result = solve_qp(A, b, C, d, x0, max_iterations=1000)

print(result.summary())             # converged?  iterations?  spikes?
print("x* =", result.final_x)
print("f* =", result.final_objective)
```

For repeated solves (warm-started receding-horizon problems), construct an `SNNSolver` once and call `.solve(x0)` per problem instance — see [`examples/example4_warm_start.py`](examples/example4_warm_start.py).

## Examples

All scripts live under [`examples/`](examples/) and are runnable as plain `python examples/example_name.py`.

| # | Script | Problem | Highlights |
|---|---|---|---|
| 1 | [`example1_simple_2d.py`](examples/example1_simple_2d.py) | 2D quadratic with two linear cuts | Smallest possible runnable demo |
| 1b | [`example1_basic_2d.py`](examples/example1_basic_2d.py) | Same problem, with trajectory plot | See `examples/example1_basic_2d.png` |
| 1c | [`example1_advanced_2d.py`](examples/example1_advanced_2d.py) | Shifted feasible region, infeasible start | Spike raster + violation plot |
| 2 | [`example2_3d_polytope.py`](examples/example2_3d_polytope.py) | 3D QP with 4 hyperplanes | Multiple active constraints, vertex solution |
| 3 | [`example3_linear_program.py`](examples/example3_linear_program.py) | Box-constrained LP ($A=0$) | LP via the same machinery |
| 4 | [`example4_warm_start.py`](examples/example4_warm_start.py) | Sequence of related QPs | Receding-horizon / MPC pattern; spikes drop $30 \to 0$ |
| 5 | [`example5_infeasible_recovery.py`](examples/example5_infeasible_recovery.py) | Infeasible initializations | Automatic projection to feasibility |
| 6 | [`example6_equality_constraint.py`](examples/example6_equality_constraint.py) | Equality via a sandwiched band | $x_1 = a$ as a tight $\pm \varepsilon$ inequality pair |
| 7 | [`example7_svm_dual.py`](examples/example7_svm_dual.py) | SVM dual with kernel | Box clipping + auto step size on a real ML task |
| — | [`example_raw_mode.py`](examples/example_raw_mode.py) | Bypass auto-config | Compares raw vs. optimized solver settings |

Run them all in sequence:

```bash
python examples/run_all_examples.py
```

## Documentation

- [`docs/theory.md`](docs/theory.md) — derivation of the SNN/convex-optimization equivalence, step-size analysis, projection geometry, convergence criteria.
- [`docs/applications.md`](docs/applications.md) — *(coming next)* one-page summary of each SNN-X paper with links to PDFs.
- [`docs/api.md`](docs/api.md) — *(coming next)* hand-curated API reference.
- [https://snn.ahkhan.me](https://snn.ahkhan.me) — companion site, designed for a broader audience (students, curious researchers).

## Applications

`snn_opt` is the solver behind the **SNN-X** family of papers — each one casting a classical ML problem as a constrained QP and showing it can be solved by the same spiking dynamics:

| Paper | Problem class | Status |
|---|---|---|
| **SNN-SVM** | Box-constrained QP (SVM dual) | Published, 99.3% acc matching scikit-learn |
| **SNN-LinReg** | Equality-constrained least squares | Published |
| **SNN-CF** | Collaborative filtering via ALS | Published, +3.9% over baseline ALS |
| **SNN-PCA** | Sphere-constrained quadratic (eigenproblem) | Published; under review at IEEE TNNLS |
| **SNN-Procrustes** | Orthogonal Procrustes | Published |
| **SNN-KRR** | Constrained kernel ridge regression | Published; under review at IEEE TCAS-I |
| **SNN-Ridge** | Ridge regression with constraint variants | Published |
| **SNN-TDSVM** | Two-timescale spiking solver for TDSVM | Under review at *Neural Networks* |
| **SNN-Norm** | Homeostatic normalization for SNNs | Under review |

Other downstream uses of the same solver: model-predictive control of a building cooling loop ([SNN-EnergyPlus](https://github.com/ahkhan03)), neuromorphic image denoising, and SUMO-in-the-loop traffic-signal optimization. See [`docs/applications.md`](docs/applications.md) (work in progress) for a curated list with links.

## Citing this work

If `snn_opt` plays a role in your research or teaching, please cite both the software and the framework paper:

```bibtex
@software{khan2026snnopt,
  author  = {Khan, Ameer Hamza},
  title   = {snn\_opt: A Spiking Neural Network Solver for Constrained Convex Optimization},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/ahkhan03/snn_opt},
  license = {Apache-2.0},
}

@inproceedings{mancoo2020understanding,
  author    = {Mancoo, Allan and Boerlin, Martin and Machens, Christian K.},
  title     = {Understanding Spiking Networks Through Convex Optimization},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2020},
}
```

The full per-paper bibliography of the SNN-X series is maintained at [`docs/applications.md`](docs/applications.md).

## License

Apache-2.0 — see [`LICENSE`](LICENSE). Permissive, with an explicit patent grant; suitable for both academic and commercial reuse.

## Acknowledgments

Developed at the **School of Artificial Intelligence, Taizhou University**. The framework rests on Mancoo, Boerlin, and Machens (NeurIPS 2020), and on the broader projection-neural-network lineage (Hopfield–Tank, Kennedy–Chua, Xia–Wang, Liu–Wang). Pull requests, bug reports, and citations of the SNN-X papers in your own work are all warmly welcomed.
