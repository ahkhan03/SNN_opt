// snn_qp_core.hpp -- pure C++ kernel for the SNN-QP euler/adaptive solve path.
//
// This translation unit is written in a restricted C++ subset so it can be
// reused as the seed for a Vitis HLS kernel (the FPGA deployment track):
//
//   * no STL containers, exceptions, recursion or dynamic allocation inside
//     any loop;
//   * scratch buffers are allocated exactly once at function entry. For a CPU
//     build that allocation uses std::vector; for an HLS build, replace the
//     single `Scratch` block with static arrays sized to compile-time MAX_N /
//     MAX_M and the rest of the file is synthesisable as-is;
//   * all numerics reduce to plain double matvec / axpy / dot / ReLU-threshold.
//
// It is a faithful port of snn_opt.solver.SNNSolver._solve_euler_lean together
// with _project_adaptive (the v0.5 unified rows+facets sweep),
// _check_convergence, _compute_projected_gradient_norm and
// _joint_max_violation. Keep the two in lockstep; the golden-parity harness
// (tests/test_c_backend_parity.py) enforces agreement.

#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

// Minimum matvec work (rows*cols MACs) at or above which the OpenMP-threaded
// path is used; below it the per-call fork/join overhead dominates and the
// serial SIMD loop is faster -- by up to ~170x on tiny matvecs on a 32-thread
// box (measured crossover ~9e4 MACs there; lower on fewer cores). This guard is
// why the multicore `'c'`/`'c_openmp'` path is a no-op on small/medium QPs (all
// current application problems are n<=~150) and only spins up threads on genuinely
// large systems. Tunable at build time with -DSNN_QP_OMP_MIN_WORK=<n>.
#ifndef SNN_QP_OMP_MIN_WORK
#define SNN_QP_OMP_MIN_WORK 65536
#endif

namespace snn_qp {

// Terminal convergence reasons. Mirrors the category of
// SNNSolver._convergence_reason (the Python string also embeds the criterion
// values; only the category is needed for parity).
enum ReasonCode {
    REASON_MAX_ITERATIONS    = 0,
    REASON_CONVERGED         = 1,
    // Inner sweep hit its safety cap before reaching joint tolerance; the
    // solve aborts rather than continuing from a knowingly infeasible point.
    REASON_PROJECTION_BUDGET = 2,
};

struct Result {
    int    iterations_used;
    int    n_projections;
    bool   converged;
    int    reason_code;
};

// ----------------------------------------------------------------------------
// Primitives
// ----------------------------------------------------------------------------

// y = M @ x ; M is (rows x cols), row-major.
//
// The `#pragma omp simd reduction` lets the CPU build (compiled with
// -fopenmp-simd or -fopenmp) vectorise the sum reduction -- a plain -O3 will
// not, because floating-point addition is not associative. The loop body stays
// a clean dot product, so it remains an ideal HLS pipeline target: an HLS build
// ignores the omp pragma and applies its own `#pragma HLS PIPELINE` instead.
//
// When `parallel` is true AND the translation unit was compiled with full
// OpenMP (`-fopenmp`, which defines `_OPENMP`), the row loop is distributed
// across a thread team (`backend='c_openmp'` / the multicore path of the
// `'c'` auto backend). Each call forks/joins a team -- a per-call cost the
// inner-loop SIMD vectorisation amortises only once the matrix is large enough,
// which is why the small-problem default leans on the serial path. Without
// `-fopenmp` the `parallel` flag is inert and the serial SIMD loop runs (so a
// SIMD-only build still satisfies `backend='c_serial'` exactly). The matvec is
// the only data-parallel hot spot; the Euler recurrence and the greedy
// projection are strictly serial (the Amdahl ceiling, ~2x at 4 cores).
inline void matvec(const double* M, const double* x, double* y,
                    int rows, int cols, bool parallel) {
#ifdef _OPENMP
    if (parallel &&
        static_cast<long long>(rows) * cols >= SNN_QP_OMP_MIN_WORK) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; ++i) {
            const double* Mi = M + static_cast<std::size_t>(i) * cols;
            double acc = 0.0;
            #pragma omp simd reduction(+ : acc)
            for (int j = 0; j < cols; ++j) acc += Mi[j] * x[j];
            y[i] = acc;
        }
        return;
    }
#else
    (void)parallel;
#endif
    for (int i = 0; i < rows; ++i) {
        const double* Mi = M + static_cast<std::size_t>(i) * cols;
        double acc = 0.0;
        #pragma omp simd reduction(+ : acc)
        for (int j = 0; j < cols; ++j) acc += Mi[j] * x[j];
        y[i] = acc;
    }
}

// Apply the Hessian: y = A @ x. Dispatches to the diagonal fast path when the
// problem has been reduced to a diagonal A (e.g. an eigenbasis transform has
// rotated A = V Lambda V^T into its eigencoordinates, so the dominant O(n^2)
// A @ x step collapses to the O(n) elementwise product Lambda (.) x). When
// `use_diag` is true, `a_diag` is the length-n diagonal and `A` is ignored;
// otherwise the dense matvec runs. This is a backend-agnostic structure
// exploitation -- any future transform that yields a diagonal A benefits.
inline void apply_hessian(const double* A, const double* a_diag, bool use_diag,
                          const double* x, double* y, int n, bool parallel) {
    if (use_diag) {
        for (int i = 0; i < n; ++i) y[i] = a_diag[i] * x[i];
        return;
    }
    matvec(A, x, y, n, n, parallel);
}

inline double dot(const double* a, const double* b, int n) {
    double acc = 0.0;
    #pragma omp simd reduction(+ : acc)
    for (int i = 0; i < n; ++i) acc += a[i] * b[i];
    return acc;
}

// max(0, max_i (C x + d)_i). Returns 0 when there are no constraints.
inline double max_violation(const double* C, const double* d, const double* x,
                            int n, int m, double* r_scratch, bool parallel) {
    if (m == 0) return 0.0;
    matvec(C, x, r_scratch, m, n, parallel);
    double mv = 0.0;
    for (int i = 0; i < m; ++i) {
        double g = r_scratch[i] + d[i];
        if (g > mv) mv = g;
    }
    return mv;
}

// Joint GEOMETRIC infeasibility: max over row violation DISTANCES (raw residual
// * row_scale, where row_scale[j] = 1/||c_j|| or 0 for a degenerate row) and
// box-bound violations (already distances; unit facet normals). Mirrors
// SNNSolver._joint_max_violation. This is the feasibility-gate quantity.
inline double joint_max_violation(const double* C, const double* d,
                                  const double* row_scale, const double* x,
                                  int n, int m,
                                  bool has_lower, double lower,
                                  bool has_upper, double upper,
                                  double* r_scratch, bool parallel) {
    double mv = 0.0;
    if (m > 0) {
        matvec(C, x, r_scratch, m, n, parallel);
        for (int i = 0; i < m; ++i) {
            const double g = (r_scratch[i] + d[i]) * row_scale[i];
            if (g > mv) mv = g;
        }
    }
    if (has_lower)
        for (int i = 0; i < n; ++i)
            if (lower - x[i] > mv) mv = lower - x[i];
    if (has_upper)
        for (int i = 0; i < n; ++i)
            if (x[i] - upper > mv) mv = x[i] - upper;
    return mv;
}

// ----------------------------------------------------------------------------
// Adaptive projection -- Gram-matrix (event-driven lateral-update) form
// ----------------------------------------------------------------------------
// Snaps the most-violated constraint to its boundary with the closed-form exact
// step k1 = g_j / ||c_j||^2, as SNNSolver._project_adaptive does -- but instead
// of recomputing the residual r = C x + d from scratch every sub-iteration,
// each projection event ("spike") on constraint j updates the residual of every
// coupled constraint by -k1 * G[:,j], where G = C C^T is the constraint-
// coupling (recurrent) matrix. This is event-driven lateral propagation:
// O(m) per spike rather than an O(m*n) residual recompute.
//
// The residual is recomputed exactly once per call (C x + d). Incremental drift
// over the bounded sub-iteration loop stays ~1e-12 (<< constraint_tol) and is
// reset by the fresh recompute on the next outer iteration.
//
// G is m x m, row-major, symmetric (so column j == row j); G[j][j] equals
// c_norms_sq[j]. Modifies x in place. Returns the number of projection events.
// Unified sweep: general rows and implicit box facets in ONE winner-take-all
// candidate family, selected by NORMALIZED violation distance (raw row residual
// * row_scale; facet violations are already distances -- unit normals). Frozen
// candidate order for ties: rows in input order, then lower facets 0..n-1, then
// upper facets 0..n-1; strict '>' comparisons give the first maximal index,
// matching the Python reference exactly.
//
// A row spike applies the exact step k1 = g_j / ||c_j||^2 and propagates the
// lateral update r -= k1 * G[:,j] (O(m)). A facet spike is the exact O(1)
// single-coordinate correction to the bound and propagates r += delta * C[:,i]
// (O(m), strided column access). Facet residuals are read off x, always fresh.
//
// The sweep runs until JOINT tolerance or the safety cap; on cap exhaustion
// with the joint violation still above tolerance, *budget_exhausted is set and
// the caller must abort the solve (v0.5 semantics: never continue the outer
// dynamics from a knowingly infeasible point).
//
// Box-only problems (m == 0) use the exact vectorized box projection (the box
// is separable, so the clip IS the metric projection); events counted per
// clipped coordinate.
//
// This replaces the pre-v0.5 pair (project_adaptive + terminal clip_to_bounds),
// whose composition was NOT a projection onto the intersection (the
// clip-after-project defect).
inline int project_unified(double* x,
                           const double* C, const double* d,
                           const double* c_norms_sq, const double* row_scale,
                           const double* G,
                           int n, int m,
                           double constraint_tol, int proj_cap,
                           bool has_lower, double lower,
                           bool has_upper, double upper,
                           double* r, bool parallel,
                           bool* budget_exhausted) {
    // Box-only fast path: exact separable projection.
    if (m == 0) {
        int n_events = 0;
        if (has_lower)
            for (int i = 0; i < n; ++i)
                if (x[i] < lower) { x[i] = lower; ++n_events; }
        if (has_upper)
            for (int i = 0; i < n; ++i)
                if (x[i] > upper) { x[i] = upper; ++n_events; }
        return n_events;
    }

    matvec(C, x, r, m, n, parallel);               // r = C x      (once)
    for (int i = 0; i < m; ++i) r[i] += d[i];      // r = C x + d

    int n_iters = 0;
    for (int it = 0; it < proj_cap; ++it) {
        // Winner-take-all over normalized distances, frozen candidate order.
        int j = 0;
        int kind = 0;  // 0 = row, 1 = lower facet, 2 = upper facet
        double best = r[0] * row_scale[0];
        for (int i = 1; i < m; ++i) {
            const double s = r[i] * row_scale[i];
            if (s > best) { best = s; j = i; }
        }
        if (has_lower)
            for (int i = 0; i < n; ++i) {
                const double v = lower - x[i];
                if (v > best) { best = v; j = i; kind = 1; }
            }
        if (has_upper)
            for (int i = 0; i < n; ++i) {
                const double v = x[i] - upper;
                if (v > best) { best = v; j = i; kind = 2; }
            }
        if (best <= constraint_tol) return n_iters;  // jointly satisfied

        if (kind == 0) {
            // (row_scale[j] > 0 is guaranteed here: degenerate rows have
            // scale 0, so their distance is 0 <= constraint_tol.)
            const double k1 = r[j] / c_norms_sq[j];
            const double* cj = C + static_cast<std::size_t>(j) * n;
            for (int k = 0; k < n; ++k) x[k] -= k1 * cj[k];  // primal update
            // Lateral update: spike j propagates -k1 * G[:,j] to coupled rows.
            const double* Gj = G + static_cast<std::size_t>(j) * m;
            for (int i = 0; i < m; ++i) r[i] -= k1 * Gj[i];
        } else {
            // Facet spike: exact single-coordinate correction to the bound.
            const double delta = (kind == 1) ? best : -best;  // signed move
            x[j] += delta;
            // Lateral update through column j of C (row-major: stride n).
            for (int i = 0; i < m; ++i)
                r[i] += delta * C[static_cast<std::size_t>(i) * n + j];
        }
        ++n_iters;
    }

    // Cap hit: abort-worthy only if the joint violation is still above tol.
    // Recompute the residual fresh (not the incrementally-drifted r) so the
    // abort decision matches the Python reference bit-for-bit.
    const double mv = joint_max_violation(C, d, row_scale, x, n, m,
                                          has_lower, lower, has_upper, upper,
                                          r, parallel);
    if (mv > constraint_tol) *budget_exhausted = true;
    return n_iters;
}

// ----------------------------------------------------------------------------
// Projected-gradient norm -- mirrors SNNSolver._compute_projected_gradient_norm
// ----------------------------------------------------------------------------
// Note: the constraint-normal component uses the *original* gradient, while the
// running projection accumulates into a separate buffer (pg), matching Python.
inline double projected_gradient_norm(const double* x,
                                      const double* A, const double* a_diag,
                                      bool use_diag, const double* b,
                                      const double* C, const double* d,
                                      const double* c_norms_sq,
                                      int n, int m, double constraint_tol,
                                      bool has_lower, double lower,
                                      bool has_upper, double upper,
                                      double* grad, double* pg, double* r,
                                      bool parallel) {
    apply_hessian(A, a_diag, use_diag, x, grad, n, parallel);  // grad = A x + b
    for (int i = 0; i < n; ++i) grad[i] += b[i];
    for (int i = 0; i < n; ++i) pg[i] = grad[i];

    if (m > 0) {
        matvec(C, x, r, m, n, parallel);            // r = C x + d
        for (int i = 0; i < m; ++i) r[i] += d[i];
        const double active_tol = constraint_tol * 10.0;
        for (int j = 0; j < m; ++j) {
            if (r[j] >= -active_tol) {              // constraint (near-)active
                if (c_norms_sq[j] < 1e-12) continue;
                const double* cj = C + static_cast<std::size_t>(j) * n;
                const double component = dot(grad, cj, n) / c_norms_sq[j];
                if (component < 0.0) {
                    for (int k = 0; k < n; ++k) pg[k] -= component * cj[k];
                }
            }
        }
    }
    if (has_lower) {
        const double lower_tol = lower + constraint_tol * 10.0;
        for (int i = 0; i < n; ++i)
            if (x[i] <= lower_tol && grad[i] > 0.0) pg[i] = 0.0;
    }
    if (has_upper) {
        const double upper_tol = upper - constraint_tol * 10.0;
        for (int i = 0; i < n; ++i)
            if (x[i] >= upper_tol && grad[i] < 0.0) pg[i] = 0.0;
    }
    return std::sqrt(dot(pg, pg, n));
}

// ----------------------------------------------------------------------------
// Main solve -- mirrors SNNSolver._solve_euler_lean
// ----------------------------------------------------------------------------
inline Result solve_euler(const double* A, const double* b,
                          const double* C, const double* d,
                          const double* c_norms_sq, const double* row_scale,
                          const double* G,
                          int n, int m,
                          double k0, double constraint_tol,
                          int max_iterations, int proj_cap,
                          bool enable_early_stopping, int check_every,
                          int min_iterations, int window_size, int patience,
                          double obj_rel_tol, double x_rel_tol,
                          double proj_grad_tol, double feasibility_tol,
                          bool use_obj_plateau, bool use_proj_grad,
                          bool use_sol_stable, bool require_feas,
                          bool has_lower, double lower,
                          bool has_upper, double upper,
                          const double* a_diag, bool use_diag,
                          bool parallel,
                          const double* x0, double* x_out) {
    // --- one-time scratch allocation (HLS build: replace with static arrays) -
    std::vector<double> x(n), Ax(n), grad(n), pg(n);
    std::vector<double> r(m > 0 ? m : 1);
    const int W = window_size > 0 ? window_size : 1;
    std::vector<double> obj_ring(W, 0.0);
    // x-history ring only materialised when the solution-stable criterion is on
    std::vector<double> x_ring(use_sol_stable ? static_cast<std::size_t>(W) * n : 1, 0.0);
    // -----------------------------------------------------------------------

    for (int i = 0; i < n; ++i) x[i] = x0[i];

    int obj_head = 0, obj_count = 0;
    int x_head = 0, x_count = 0;
    int patience_counter = 0;
    int n_projections = 0;
    bool converged = false;
    int reason_code = REASON_MAX_ITERATIONS;
    int iterations_used = max_iterations;

    const int n_enabled = (use_obj_plateau ? 1 : 0)
                        + (use_proj_grad ? 1 : 0)
                        + (use_sol_stable ? 1 : 0);

    // A @ x for the current iterate; reused for both the gradient step and the
    // plateau-check objective (one Hessian apply per iteration, not two).
    apply_hessian(A, a_diag, use_diag, x.data(), Ax.data(), n, parallel);

    for (int it = 0; it < max_iterations; ++it) {
        // Phase 1: gradient descent step  (gradient = A x + b)
        for (int i = 0; i < n; ++i) x[i] -= k0 * (Ax[i] + b[i]);

        // Phase 2: unified projection sweep (rows + implicit box facets).
        // (v0.5.0: the former Phase-3 terminal box clip is gone -- bounds are
        // facets inside the sweep; clipping here broke rows.)
        bool budget_exhausted = false;
        n_projections += project_unified(x.data(), C, d, c_norms_sq, row_scale,
                                         G, n, m, constraint_tol, proj_cap,
                                         has_lower, lower, has_upper, upper,
                                         r.data(), parallel, &budget_exhausted);
        if (budget_exhausted) {
            reason_code = REASON_PROJECTION_BUDGET;
            iterations_used = it + 1;
            break;
        }

        // Objective for the plateau check -- reuse the A @ x needed next iter.
        apply_hessian(A, a_diag, use_diag, x.data(), Ax.data(), n, parallel);
        const double obj_current =
            0.5 * dot(x.data(), Ax.data(), n) + dot(b, x.data(), n);
        obj_ring[obj_head] = obj_current;
        obj_head = (obj_head + 1) % W;
        ++obj_count;
        if (use_sol_stable) {
            double* slot = x_ring.data() + static_cast<std::size_t>(x_head) * n;
            for (int i = 0; i < n; ++i) slot[i] = x[i];
            x_head = (x_head + 1) % W;
            ++x_count;
        }

        // --- convergence check (mirrors _check_convergence + patience) ------
        if (!enable_early_stopping) continue;
        if (it < min_iterations) continue;
        if (it % check_every != 0) continue;

        // Feasibility gate first: if infeasible the outcome is "not converged"
        // regardless of the other criteria, so the early skip is exact.
        // v0.5.0: JOINT geometric violation (rows as distances + box facets).
        if (require_feas &&
            joint_max_violation(C, d, row_scale, x.data(), n, m,
                                has_lower, lower, has_upper, upper,
                                r.data(), parallel) > feasibility_tol) {
            patience_counter = 0;
            continue;
        }

        int n_met = 0;
        if (use_obj_plateau && obj_count >= window_size) {
            double wmax = obj_ring[0], wmin = obj_ring[0];
            for (int i = 1; i < W; ++i) {
                if (obj_ring[i] > wmax) wmax = obj_ring[i];
                if (obj_ring[i] < wmin) wmin = obj_ring[i];
            }
            const double most_recent = obj_ring[(obj_head - 1 + W) % W];
            double scale = std::fabs(most_recent);
            if (scale < 1e-10) scale = 1e-10;
            if ((wmax - wmin) / scale < obj_rel_tol) ++n_met;
        }
        if (use_proj_grad) {
            const double pgn = projected_gradient_norm(
                x.data(), A, a_diag, use_diag, b, C, d, c_norms_sq, n, m,
                constraint_tol, has_lower, lower, has_upper, upper,
                grad.data(), pg.data(), r.data(), parallel);
            if (pgn < proj_grad_tol) ++n_met;
        }
        if (use_sol_stable && x_count >= window_size) {
            double xnorm = std::sqrt(dot(x.data(), x.data(), n));
            if (xnorm < 1e-10) xnorm = 1e-10;
            double max_dist = 0.0;
            for (int s = 0; s < W; ++s) {
                const double* slot = x_ring.data() + static_cast<std::size_t>(s) * n;
                double acc = 0.0;
                for (int i = 0; i < n; ++i) {
                    const double dlt = slot[i] - x[i];
                    acc += dlt * dlt;
                }
                const double dist = std::sqrt(acc);
                if (dist > max_dist) max_dist = dist;
            }
            if (max_dist / xnorm < x_rel_tol) ++n_met;
        }

        const bool check_converged = (n_met >= n_enabled) && (n_enabled > 0);
        if (check_converged) {
            ++patience_counter;
            if (patience_counter >= patience) {
                converged = true;
                reason_code = REASON_CONVERGED;
                iterations_used = it + 1;
                break;
            }
        } else {
            patience_counter = 0;
        }
    }

    for (int i = 0; i < n; ++i) x_out[i] = x[i];
    Result res;
    res.iterations_used = iterations_used;
    res.n_projections   = n_projections;
    res.converged       = converged;
    res.reason_code     = reason_code;
    return res;
}

}  // namespace snn_qp
