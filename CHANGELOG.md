# Changelog

All notable changes to `snn_opt` are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.3.0] — 2026-06-19

### Removed
- **The six projection-variant solvers** (`solve_qp_penalty`,
  `solve_qp_lagrangian`, `solve_qp_heun_penalty`, `solve_qp_heavyball_penalty`,
  `solve_qp_nesterov_penalty`, `solve_qp_expeuler_penalty`) are no longer part of
  the public API. The package now ships only the canonical solver (Euler +
  adaptive projection) across its four backends (`python`, `c`, `c_serial`,
  `c_openmp`). The variants are being re-scoped for a staged, individually
  documented re-release; until then they remain available in `0.2.0`.

## [0.2.0] — 2026-06-19

### Added
- **Multicore C++ backend variants.** The compiled kernel now exposes three
  numerically identical backends differing only in matvec threading:
  `backend='c'` (auto — OpenMP multicore when the wheel was built with it, else
  single-thread), `'c_serial'` (forced single-thread), and `'c_openmp'` (forced
  multicore; raises if the build lacks OpenMP). Only the inner matrix–vector
  products are parallelized; the Euler recurrence and greedy projection remain
  serial (Amdahl-bound). The threaded path is automatically skipped below a work
  threshold so small/medium problems never pay fork/join overhead, and it honours
  `OMP_NUM_THREADS`.
- `snn_opt._kernel.HAS_OPENMP` and `snn_opt._kernel.max_threads()` report the
  build's OpenMP capability.

### Changed
- `backend='c'` now uses the OpenMP multicore matvec on large problems when the
  wheel was built with OpenMP (previously single-thread); results are unchanged.
- The build probes the compiler for `-fopenmp` and enables it when available,
  falling back to SIMD-only (`-fopenmp-simd`) otherwise — the extension always
  builds; only the multicore capability is conditional.

## [0.1.0] — 2026-05-23

Initial PyPI release. The solver itself is unchanged from the development
version that backs the published portfolio-optimization application
(Khan, Mohammed & Li, *Biomimetics* 2025) and supports an ongoing
research program. This release packages, documents, licenses, and ships
that codebase as a `pip install`-able distribution.

### Added
- **PyPI distribution** as `snn-opt` (import name `snn_opt`); prebuilt wheels
  for manylinux x86_64, linux aarch64, macOS x86_64 + arm64, and Windows
  AMD64 across CPython 3.9–3.13.
- **Compiled C++ backend** (`snn_opt._kernel`, pybind11), enabled via
  `backend='c'` in `SolverConfig` / `solve_qp`. Faithful port of the
  reference Python solver — golden-parity tested in
  `tests/test_c_backend_parity.py`. Roughly an order-of-magnitude faster on
  the inner adaptive-projection loop; the same kernel source is HLS-compatible
  and is the basis for the planned FPGA deployment track.
- **Gram-matrix projection** (event-driven lateral update): when the
  constraint Gram matrix fits in memory, the inner-loop residual is
  updated in O(m) per projection event instead of O(m·n) recomputation.
  Implemented in both backends.
- **Lean solve path** for the C backend: skips trajectory recording for
  cases where only the final solution is needed (the common production
  setting). Gated by a config flag in the Python backend; default in C.
- **`src/`-layout Python package** with a stable public API exported from
  `snn_opt.__init__`: `OptimizationProblem`, `SolverConfig`,
  `ConvergenceConfig`, `SolverResult`, `SNNSolver`, `solve_qp`.
- `pyproject.toml` with build-system metadata, optional extras
  (`[examples]`, `[benchmarks]`, `[dev]`), and PEP 621 project URLs.
- Apache-2.0 `LICENSE` and `CITATION.cff` for academic citation.
- `docs/` (`theory.md`, `applications.md`, `api.md`, `index.md`), `tests/`,
  `benchmarks/`, `examples/`, `figures/` directories establishing a
  conventional research-software layout.
- `cibuildwheel` GitHub Actions workflow that builds the full matrix and
  publishes to PyPI via OIDC trusted publishing on `v*` tag push.
- Top-level `git` history (initialized 2026-04-18).

### Changed
- `snn_solver.py` moved to `src/snn_opt/solver.py`. All public symbols are
  now imported from `snn_opt`.
- Examples and `tests/test_installation.py` updated to import from the new
  package path.
- `README.md` rewritten with an academic-paper structure (problem statement,
  method, examples, citations) and now leads with `pip install snn-opt`.
- `snn_optimization_framework.md` moved to `docs/theory.md`.
- `.gitignore` no longer excludes `figures/`, `*.png`, or `*.pdf` — example
  and benchmark figures are part of the repository so the rendered README
  and documentation work without a build step.
