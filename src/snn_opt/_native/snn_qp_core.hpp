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
// with _project_adaptive, _check_convergence, _compute_projected_gradient_norm
// and _clip_to_bounds. Keep the two in lockstep; the golden-parity harness
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
    REASON_MAX_ITERATIONS = 0,
    REASON_CONVERGED      = 1,
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
inline int project_adaptive(double* x,
                             const double* C, const double* d,
                             const double* c_norms_sq, const double* G,
                             int n, int m,
                             double constraint_tol, int max_projection_iters,
                             double* r, bool parallel) {
    if (m == 0) return 0;
    matvec(C, x, r, m, n, parallel);               // r = C x      (once)
    for (int i = 0; i < m; ++i) r[i] += d[i];      // r = C x + d

    int n_iters = 0;
    for (int it = 0; it < max_projection_iters; ++it) {
        // most-violated constraint (np.argmax: first maximal index)
        int j = 0;
        double gmax = r[0];
        for (int i = 1; i < m; ++i) {
            if (r[i] > gmax) { gmax = r[i]; j = i; }
        }
        if (gmax <= constraint_tol) break;  // all constraints satisfied

        // Degenerate constraint: re-evaluate (x and r unchanged). The outer
        // loop is bounded by max_projection_iters so this terminates.
        if (c_norms_sq[j] < 1e-12) continue;

        const double k1 = gmax / c_norms_sq[j];
        const double* cj = C + static_cast<std::size_t>(j) * n;
        for (int k = 0; k < n; ++k) x[k] -= k1 * cj[k];      // primal update
        // Lateral update: spike j propagates -k1 * G[:,j] to coupled rows.
        const double* Gj = G + static_cast<std::size_t>(j) * m;
        for (int i = 0; i < m; ++i) r[i] -= k1 * Gj[i];
        ++n_iters;
    }
    return n_iters;
}

// ----------------------------------------------------------------------------
// Box clipping -- mirrors SNNSolver._clip_to_bounds
// ----------------------------------------------------------------------------
inline void clip_to_bounds(double* x, int n,
                           bool has_lower, double lower,
                           bool has_upper, double upper) {
    if (has_lower) for (int i = 0; i < n; ++i) if (x[i] < lower) x[i] = lower;
    if (has_upper) for (int i = 0; i < n; ++i) if (x[i] > upper) x[i] = upper;
}

// ----------------------------------------------------------------------------
// Projected-gradient norm -- mirrors SNNSolver._compute_projected_gradient_norm
// ----------------------------------------------------------------------------
// Note: the constraint-normal component uses the *original* gradient, while the
// running projection accumulates into a separate buffer (pg), matching Python.
inline double projected_gradient_norm(const double* x,
                                      const double* A, const double* b,
                                      const double* C, const double* d,
                                      const double* c_norms_sq,
                                      int n, int m, double constraint_tol,
                                      bool has_lower, double lower,
                                      bool has_upper, double upper,
                                      double* grad, double* pg, double* r,
                                      bool parallel) {
    matvec(A, x, grad, n, n, parallel);             // grad = A x + b
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
                          const double* c_norms_sq, const double* G,
                          int n, int m,
                          double k0, double constraint_tol,
                          int max_iterations, int max_projection_iters,
                          bool enable_early_stopping, int check_every,
                          int min_iterations, int window_size, int patience,
                          double obj_rel_tol, double x_rel_tol,
                          double proj_grad_tol, double feasibility_tol,
                          bool use_obj_plateau, bool use_proj_grad,
                          bool use_sol_stable, bool require_feas,
                          bool has_lower, double lower,
                          bool has_upper, double upper,
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
    // plateau-check objective (one O(n^2) matvec per iteration, not two).
    matvec(A, x.data(), Ax.data(), n, n, parallel);

    for (int it = 0; it < max_iterations; ++it) {
        // Phase 1: gradient descent step  (gradient = A x + b)
        for (int i = 0; i < n; ++i) x[i] -= k0 * (Ax[i] + b[i]);

        // Phase 2: project to feasible region
        n_projections += project_adaptive(x.data(), C, d, c_norms_sq, G, n, m,
                                          constraint_tol, max_projection_iters,
                                          r.data(), parallel);

        // Phase 3: clip to box constraints
        clip_to_bounds(x.data(), n, has_lower, lower, has_upper, upper);

        // Objective for the plateau check -- reuse the A @ x needed next iter.
        matvec(A, x.data(), Ax.data(), n, n, parallel);
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
        if (require_feas &&
            max_violation(C, d, x.data(), n, m, r.data(), parallel) > feasibility_tol) {
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
                x.data(), A, b, C, d, c_norms_sq, n, m, constraint_tol,
                has_lower, lower, has_upper, upper,
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
