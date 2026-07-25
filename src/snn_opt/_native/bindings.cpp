// bindings.cpp -- pybind11 glue exposing the SNN-QP C++ kernel to Python.
//
// This is the only part of the native code that is *not* HLS-portable; it does
// nothing but marshal NumPy arrays and config scalars into snn_qp::solve_euler
// (declared in snn_qp_core.hpp) and pack the result back. The numerical kernel
// itself is in snn_qp_core.hpp and is reused verbatim for the FPGA HLS port.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "snn_qp_core.hpp"

namespace py = pybind11;

// C-contiguous float64 input arrays; forcecast copies/casts mismatched inputs.
using darray = py::array_t<double, py::array::c_style | py::array::forcecast>;
// Observer outputs deliberately do not force-cast: they must remain writable
// views of the caller's exact float64/int64/uint64 buffers.
using f64array = py::array_t<double, py::array::c_style>;
using i64array = py::array_t<std::int64_t, py::array::c_style>;
using u64array = py::array_t<std::uint64_t, py::array::c_style>;

static py::tuple solve_euler_py(
        darray A, darray b, darray C, darray d, darray c_norms_sq,
        darray row_scale, darray c_gram, darray x0,
        double k0, double constraint_tol,
        int max_iterations, int max_projection_iters,
        bool enable_early_stopping, int check_every, int min_iterations,
        int window_size, int patience,
        double obj_rel_tol, double x_rel_tol,
        double proj_grad_tol, double feasibility_tol,
        bool use_obj_plateau, bool use_proj_grad, bool use_sol_stable,
        bool require_feas,
        bool has_lower, double lower, bool has_upper, double upper,
        bool parallel, darray a_diag, bool use_diag,
        i64array row_event_counts, i64array lower_event_counts,
        i64array upper_event_counts, u64array observer_meta,
        f64array observer_distance) {
    if (x0.ndim() != 1)
        throw std::invalid_argument("x0 must be a 1-D array");
    if (d.ndim() != 1)
        throw std::invalid_argument("d must be a 1-D array");

    const int n = static_cast<int>(x0.shape(0));
    const int m = static_cast<int>(d.shape(0));

    if (A.ndim() != 2 || A.shape(0) != n || A.shape(1) != n)
        throw std::invalid_argument("A must have shape (n, n)");
    if (b.ndim() != 1 || b.shape(0) != n)
        throw std::invalid_argument("b must have shape (n,)");
    if (C.ndim() != 2 || C.shape(0) != m || C.shape(1) != n)
        throw std::invalid_argument("C must have shape (m, n)");
    if (c_norms_sq.ndim() != 1 || c_norms_sq.shape(0) != m)
        throw std::invalid_argument("c_norms_sq must have shape (m,)");
    if (row_scale.ndim() != 1 || row_scale.shape(0) != m)
        throw std::invalid_argument("row_scale must have shape (m,)");
    if (c_gram.ndim() != 2 || c_gram.shape(0) != m || c_gram.shape(1) != m)
        throw std::invalid_argument("c_gram must have shape (m, m)");
    if (check_every < 1)
        throw std::invalid_argument("check_every must be >= 1");
    if (window_size < 1)
        throw std::invalid_argument("window_size must be >= 1");
    if (max_iterations < 0 || max_projection_iters < 0)
        throw std::invalid_argument("iteration bounds must be non-negative");
    if (use_diag && (a_diag.ndim() != 1 || a_diag.shape(0) != n))
        throw std::invalid_argument(
            "a_diag must have shape (n,) when use_diag is set");

    const bool observe = (row_event_counts.size() != 0
                       || lower_event_counts.size() != 0
                       || upper_event_counts.size() != 0
                       || observer_meta.size() != 0
                       || observer_distance.size() != 0);
    if (observe) {
        if (row_event_counts.ndim() != 1 || row_event_counts.shape(0) != m)
            throw std::invalid_argument(
                "row_event_counts must have shape (m,) when observation is enabled");
        if (lower_event_counts.ndim() != 1 || lower_event_counts.shape(0) != n)
            throw std::invalid_argument(
                "lower_event_counts must have shape (n,) when observation is enabled");
        if (upper_event_counts.ndim() != 1 || upper_event_counts.shape(0) != n)
            throw std::invalid_argument(
                "upper_event_counts must have shape (n,) when observation is enabled");
        if (observer_meta.ndim() != 1 || observer_meta.shape(0) != 4)
            throw std::invalid_argument(
                "observer_meta must have shape (4,) when observation is enabled");
        if (observer_distance.ndim() != 1 || observer_distance.shape(0) != 1)
            throw std::invalid_argument(
                "observer_distance must have shape (1,) when observation is enabled");
        if (m > 0)
            std::fill(row_event_counts.mutable_data(),
                      row_event_counts.mutable_data() + m, std::int64_t{0});
        if (n > 0) {
            std::fill(lower_event_counts.mutable_data(),
                      lower_event_counts.mutable_data() + n, std::int64_t{0});
            std::fill(upper_event_counts.mutable_data(),
                      upper_event_counts.mutable_data() + n, std::int64_t{0});
        }
    }

    auto x_out = darray(n);
    snn_qp::ProjectionObserver observer(
        observe ? row_event_counts.mutable_data() : nullptr,
        observe ? lower_event_counts.mutable_data() : nullptr,
        observe ? upper_event_counts.mutable_data() : nullptr);
    snn_qp::ProjectionObserver* observer_ptr = observe ? &observer : nullptr;

    snn_qp::Result res;
    {
        // The kernel touches no Python objects -- release the GIL so a
        // benchmark harness can run solves concurrently.
        py::gil_scoped_release release;
        res = snn_qp::solve_euler(
            A.data(), b.data(), C.data(), d.data(), c_norms_sq.data(),
            row_scale.data(), c_gram.data(),
            n, m, k0, constraint_tol, max_iterations, max_projection_iters,
            enable_early_stopping, check_every, min_iterations,
            window_size, patience,
            obj_rel_tol, x_rel_tol, proj_grad_tol, feasibility_tol,
            use_obj_plateau, use_proj_grad, use_sol_stable, require_feas,
            has_lower, lower, has_upper, upper,
            a_diag.data(), use_diag,
            parallel,
            x0.data(), x_out.mutable_data(), observer_ptr);
    }

    if (observe) {
        std::uint64_t* meta = observer_meta.mutable_data();
        const std::uint64_t no_candidate =
            std::numeric_limits<std::uint64_t>::max();
        meta[0] = observer.digest;
        meta[1] = (observer.first_candidate_id < 0)
                ? no_candidate
                : static_cast<std::uint64_t>(observer.first_candidate_id);
        meta[2] = (observer.last_candidate_id < 0)
                ? no_candidate
                : static_cast<std::uint64_t>(observer.last_candidate_id);
        meta[3] = observer.projection_cap_rechecks;
        observer_distance.mutable_data()[0] =
            observer.total_projection_distance;
    }

    return py::make_tuple(x_out, res.iterations_used, res.n_projections,
                          res.converged, res.reason_code);
}

PYBIND11_MODULE(_kernel, m) {
    m.doc() = "Compiled C++ kernel for the SNN-QP euler/adaptive solve path.";
    m.def("solve_euler", &solve_euler_py,
          "Run the lean euler + unified-projection SNN-QP solve (v0.5: box\n"
          "facets inside the sweep, normalized-distance WTA, no terminal clip).\n"
          "Returns (x_final, iterations_used, n_projections, converged, "
          "reason_code); reason_code: 0=max_iterations, 1=converged, "
          "2=projection_budget_exhausted. Optional writable observer buffers "
          "collect committed events and projection-cap rechecks without "
          "changing the return tuple.",
          py::arg("A"), py::arg("b"), py::arg("C"), py::arg("d"),
          py::arg("c_norms_sq"), py::arg("row_scale"), py::arg("c_gram"),
          py::arg("x0"),
          py::arg("k0"), py::arg("constraint_tol"),
          py::arg("max_iterations"), py::arg("max_projection_iters"),
          py::arg("enable_early_stopping"), py::arg("check_every"),
          py::arg("min_iterations"), py::arg("window_size"), py::arg("patience"),
          py::arg("obj_rel_tol"), py::arg("x_rel_tol"),
          py::arg("proj_grad_tol"), py::arg("feasibility_tol"),
          py::arg("use_obj_plateau"), py::arg("use_proj_grad"),
          py::arg("use_sol_stable"), py::arg("require_feas"),
          py::arg("has_lower"), py::arg("lower"),
          py::arg("has_upper"), py::arg("upper"),
          py::arg("parallel") = false,
          py::arg("a_diag") = darray(0), py::arg("use_diag") = false,
          py::arg("row_event_counts") = i64array(0),
          py::arg("lower_event_counts") = i64array(0),
          py::arg("upper_event_counts") = i64array(0),
          py::arg("observer_meta") = u64array(0),
          py::arg("observer_distance") = f64array(0));

    // Build-time OpenMP capability. The `'c'` auto backend reads HAS_OPENMP to
    // decide whether to request the multicore path; `'c_openmp'` raises when it
    // is False. Set from the _OPENMP macro, which the compiler defines only when
    // the extension was built with full `-fopenmp` (not merely -fopenmp-simd).
#ifdef _OPENMP
    m.attr("HAS_OPENMP") = true;
    m.def("max_threads", []() { return omp_get_max_threads(); },
          "Maximum OpenMP threads available to the multicore matvec "
          "(honours OMP_NUM_THREADS).");
#else
    m.attr("HAS_OPENMP") = false;
    m.def("max_threads", []() { return 1; },
          "OpenMP unavailable in this build; always 1.");
#endif
}
