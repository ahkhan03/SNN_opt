# Changelog

All notable changes to `snn_opt` are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Project converted to a `src/`-layout Python package (`snn_opt`) with a stable
  public API exported from `snn_opt.__init__`.
- `pyproject.toml` for modern, pip-installable distribution.
- Apache-2.0 `LICENSE` and `CITATION.cff` for academic citation.
- `docs/`, `tests/`, `benchmarks/`, `figures/` directories establishing a
  conventional research-software layout.
- Top-level `git` history (initialized 2026-04-18).

### Changed
- `snn_solver.py` moved to `src/snn_opt/solver.py`. All public symbols
  (`OptimizationProblem`, `SolverConfig`, `ConvergenceConfig`, `SolverResult`,
  `SNNSolver`, `solve_qp`) are now imported from `snn_opt`.
- Examples and `tests/test_installation.py` updated to import from the new
  package path.
- `README.md` rewritten with an academic-paper structure (problem statement,
  method, examples, citations).
- `snn_optimization_framework.md` moved to `docs/theory.md`.
- `.gitignore` no longer excludes `figures/`, `*.png`, or `*.pdf` — example and
  benchmark figures are now part of the repository so the rendered README and
  documentation work without a build step.

## [0.1.0] — 2026-04-18

Initial public release. The solver itself is unchanged from the development
version that backs the published portfolio-optimization application
(Khan, Mohammed & Li, *Biomimetics* 2025) and supports an ongoing
research program. This release packages, documents, and licenses that
codebase for public consumption.
