// bindings.cpp -- pybind11 glue exposing the SNN-QP C++ kernel to Python.
//
// This is the only part of the native code that is *not* HLS-portable; it does
// nothing but marshal NumPy arrays and config scalars into snn_qp::solve_euler
// (declared in snn_qp_core.hpp) and pack the result back. The numerical kernel
// itself is in snn_qp_core.hpp and is reused verbatim for the FPGA HLS port.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "snn_qp_core.hpp"
#include "snn_qp_extended.hpp"

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
        f64array observer_distance,
        int iter_offset, bool resume_observer,
        f64array obj_tail_out, i64array obj_tail_len_out,
        f64array x_tail_out, i64array x_tail_len_out) {
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
        // Chunked resume keeps the caller's accumulated counts; a fresh solve
        // zeroes them exactly as before.
        if (!resume_observer) {
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
    }

    // Optional window-tail export buffers for host-side (chunked) policy.
    const int W = window_size;
    const bool want_obj_tail = (obj_tail_out.size() != 0);
    if (want_obj_tail) {
        if (obj_tail_out.ndim() != 1 || obj_tail_out.shape(0) < W)
            throw std::invalid_argument(
                "obj_tail_out must be a 1-D array of at least window_size");
        if (obj_tail_len_out.ndim() != 1 || obj_tail_len_out.shape(0) != 1)
            throw std::invalid_argument(
                "obj_tail_len_out must have shape (1,)");
    }
    const bool want_x_tail = (x_tail_out.size() != 0);
    if (want_x_tail) {
        if (x_tail_out.ndim() != 2 || x_tail_out.shape(0) < W
                || x_tail_out.shape(1) != n)
            throw std::invalid_argument(
                "x_tail_out must have shape (window_size, n)");
        if (x_tail_len_out.ndim() != 1 || x_tail_len_out.shape(0) != 1)
            throw std::invalid_argument(
                "x_tail_len_out must have shape (1,)");
    }

    auto x_out = darray(n);
    snn_qp::ProjectionObserver observer(
        observe ? row_event_counts.mutable_data() : nullptr,
        observe ? lower_event_counts.mutable_data() : nullptr,
        observe ? upper_event_counts.mutable_data() : nullptr);
    if (observe && resume_observer) {
        // Seed the observer from the caller's accumulated meta so a chunked
        // run produces the identical digest / candidate IDs / totals a
        // monolithic run would (iteration tokens are made absolute by
        // iter_offset inside the kernel).
        const std::uint64_t* meta = observer_meta.data();
        const std::uint64_t no_candidate =
            std::numeric_limits<std::uint64_t>::max();
        observer.digest = meta[0];
        observer.first_candidate_id = (meta[1] == no_candidate)
            ? -1 : static_cast<std::int64_t>(meta[1]);
        observer.last_candidate_id = (meta[2] == no_candidate)
            ? -1 : static_cast<std::int64_t>(meta[2]);
        observer.projection_cap_rechecks = meta[3];
        observer.total_projection_distance = observer_distance.data()[0];
    }
    snn_qp::ProjectionObserver* observer_ptr = observe ? &observer : nullptr;

    int obj_tail_len = 0;
    int x_tail_len = 0;
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
            x0.data(), x_out.mutable_data(), observer_ptr,
            iter_offset,
            want_obj_tail ? obj_tail_out.mutable_data() : nullptr,
            want_obj_tail ? &obj_tail_len : nullptr,
            want_x_tail ? x_tail_out.mutable_data() : nullptr,
            want_x_tail ? &x_tail_len : nullptr);
    }
    if (want_obj_tail)
        obj_tail_len_out.mutable_data()[0] = obj_tail_len;
    if (want_x_tail)
        x_tail_len_out.mutable_data()[0] = x_tail_len;

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

static py::array_t<std::int64_t> copy_i64_vector(
        const std::vector<std::int64_t>& values) {
    py::array_t<std::int64_t> out(values.size());
    if (!values.empty())
        std::copy(values.begin(), values.end(), out.mutable_data());
    return out;
}

static py::array_t<double> copy_f64_vector(const std::vector<double>& values) {
    py::array_t<double> out(values.size());
    if (!values.empty())
        std::copy(values.begin(), values.end(), out.mutable_data());
    return out;
}

static py::tuple solve_euler_extended_py(
        darray A, darray b, darray C, darray d, darray c_norms_sq,
        darray row_scale, darray c_gram, darray x0,
        double k0, double constraint_tol,
        int max_iterations, int max_projection_iters,
        bool has_lower, double lower, bool has_upper, double upper,
        i64array candidate_meta, i64array coords, darray candidate_data,
        i64array member_meta,
        bool continue_after_budget, bool parallel,
        i64array row_event_counts, i64array lower_event_counts,
        i64array upper_event_counts, u64array observer_meta,
        f64array observer_distance,
        int iter_offset, bool resume_observer,
        f64array obj_tail_out, i64array obj_tail_len_out,
        f64array x_tail_out, i64array x_tail_len_out,
        int window_size, bool use_solution_stable) {
    if (x0.ndim() != 1 || d.ndim() != 1)
        throw std::invalid_argument("x0 and d must be 1-D arrays");
    const int n = static_cast<int>(x0.shape(0));
    const int m = static_cast<int>(d.shape(0));
    if (A.ndim() != 2 || A.shape(0) != n || A.shape(1) != n)
        throw std::invalid_argument("A must have shape (n, n)");
    if (b.ndim() != 1 || b.shape(0) != n)
        throw std::invalid_argument("b must have shape (n,)");
    if (C.ndim() != 2 || C.shape(0) != m || C.shape(1) != n)
        throw std::invalid_argument("C must have shape (m, n)");
    if (c_norms_sq.ndim() != 1 || c_norms_sq.shape(0) != m
            || row_scale.ndim() != 1 || row_scale.shape(0) != m)
        throw std::invalid_argument("constraint metadata must have shape (m,)");
    if (c_gram.ndim() != 2 || c_gram.shape(0) != m || c_gram.shape(1) != m)
        throw std::invalid_argument("c_gram must have shape (m, m)");
    if (candidate_meta.ndim() != 2 || candidate_meta.shape(1) != 11)
        throw std::invalid_argument("candidate_meta must have shape (q, 11)");
    if (member_meta.ndim() != 2 || member_meta.shape(1) != 11)
        throw std::invalid_argument("member_meta must have shape (r, 11)");
    if (coords.ndim() != 1 || candidate_data.ndim() != 1)
        throw std::invalid_argument("extended descriptor arrays must be 1-D");
    if (max_iterations < 0 || max_projection_iters < 0)
        throw std::invalid_argument("iteration bounds must be non-negative");
    if (window_size < 1)
        throw std::invalid_argument("window_size must be >= 1");

    const int n_candidates = static_cast<int>(candidate_meta.shape(0));
    const int n_members = static_cast<int>(member_meta.shape(0));
    if (n_candidates == 0)
        throw std::invalid_argument("candidate_meta must contain a candidate");
    for (py::ssize_t i = 0; i < coords.size(); ++i) {
        if (coords.data()[i] < 0 || coords.data()[i] >= n)
            throw std::invalid_argument("extended coordinate is outside [0, n)");
    }
    for (py::ssize_t i = 0; i < candidate_data.size(); ++i) {
        if (!std::isfinite(candidate_data.data()[i]))
            throw std::invalid_argument("extended candidate data must be finite");
    }
    auto valid_range = [](std::int64_t offset, std::int64_t count,
                          py::ssize_t size) {
        return offset >= 0 && count >= 0 && offset <= size
            && count <= size - offset;
    };
    auto validate_table = [&](const i64array& table, bool is_member) {
        for (py::ssize_t row = 0; row < table.shape(0); ++row) {
            const std::int64_t* p = table.data() + row * 11;
            const std::string label = (is_member ? "member_meta row "
                                                  : "candidate_meta row ")
                                    + std::to_string(row);
            for (int field = 0; field < 11; ++field) {
                if (p[field] < std::numeric_limits<int>::min()
                        || p[field] > std::numeric_limits<int>::max())
                    throw std::invalid_argument(label + " field exceeds int range");
            }
            const std::int64_t kind = p[0];
            if (kind < 0 || kind > 8 || (is_member && (kind == 5 || kind == 7)))
                throw std::invalid_argument(label + " has unsupported kind");
            if (!valid_range(p[1], p[2], coords.size()))
                throw std::invalid_argument(label + " has invalid coordinate range");
            if (!valid_range(p[9], p[10], candidate_data.size()))
                throw std::invalid_argument(label + " has invalid data range");
            if (!valid_range(p[4], p[5], n_members))
                throw std::invalid_argument(label + " has invalid member range");
            if (kind == 5) {
                if (p[5] == 0 || p[6] <= 0 || p[10] != 1
                        || candidate_data.data()[p[9]] <= 0.0)
                    throw std::invalid_argument(label + " has invalid Dykstra fields");
            } else {
                if (p[5] != 0)
                    throw std::invalid_argument(label + " has unexpected members");
                if (kind == 0 && (p[2] == 0 || p[10] != p[2] + 1
                                  || candidate_data.data()[p[9]] < 0.0))
                    throw std::invalid_argument(label + " has invalid ball fields");
                if (kind == 1 || kind == 2) {
                    if (p[2] < 2 || p[3] < 0 || p[3] >= n
                            || coords.data()[p[1]] != p[3]
                            || p[10] != (kind == 2 ? 1 : 0)
                            || (kind == 2 && candidate_data.data()[p[9]] <= 0.0))
                        throw std::invalid_argument(label + " has invalid SOC fields");
                }
                if (kind == 3) {
                    const std::int64_t q = p[6], p_rows = p[7];
                    const std::int64_t local = p[2] > 0 ? p[2] : n;
                    if (q <= 0 || q > n || p_rows <= 0
                            || p_rows > std::numeric_limits<std::int64_t>::max()
                                            / (3 * q + 1)
                            || p[10] != p_rows * (3 * q + 1)
                            || (local != q && local != q + p_rows))
                        throw std::invalid_argument(label + " has invalid affine fields");
                    for (std::int64_t k = 0; k < p[2]; ++k) {
                        if (coords.data()[p[1] + k] != k)
                            throw std::invalid_argument(label + " affine coordinates are not ambient ordered");
                    }
                }
                if (kind == 4 && (p[6] <= 0 || p[6] > n
                                  || p[10] != p[6] + 1
                                  || (p[2] != 0 && p[2] != p[6])
                                  || (p[2] == 0 && p[6] != n)))
                    throw std::invalid_argument(label + " has invalid halfspace fields");
                if (kind == 6 || kind == 7) {
                    const std::int64_t rows = p[6], cols = p[7];
                    if (rows <= 0 || rows > 8 || cols <= 0 || cols > 8
                            || p[3] != -1 || p[4] != 0 || p[5] != 0 || p[8] != 0
                            || p[10] != 1)
                        throw std::invalid_argument(label + " has invalid spectral fields (native cap is 8x8)");
                    const std::int64_t expected = rows * cols;
                    const bool coords_ok = (p[2] == expected)
                        || (p[2] == 0 && expected == n);
                    if (!coords_ok || candidate_data.data()[p[9]] < 0.0)
                        throw std::invalid_argument(label + " has invalid spectral fields (native cap is 8x8)");
                }
                if (kind == 8) {
                    const std::int64_t dim = p[6];
                    if (dim <= 0 || dim > 8 || p[3] != -1 || p[4] != 0
                            || p[5] != 0 || p[7] != 0 || p[8] != 0 || p[10] != 0)
                        throw std::invalid_argument(label + " has invalid PSD fields (native cap is n=8)");
                    const std::int64_t expected = dim * (dim + 1) / 2;
                    const bool coords_ok = (p[2] == expected)
                        || (p[2] == 0 && expected == n);
                    if (!coords_ok)
                        throw std::invalid_argument(label + " has invalid PSD fields (native cap is n=8)");
                }
            }
        }
    };
    validate_table(candidate_meta, false);
    validate_table(member_meta, true);
    const bool have_counts = (row_event_counts.size() != 0
                           || lower_event_counts.size() != 0
                           || upper_event_counts.size() != 0);
    if (have_counts) {
        if (row_event_counts.ndim() != 1 || row_event_counts.shape(0) != m
                || lower_event_counts.ndim() != 1
                || lower_event_counts.shape(0) != n
                || upper_event_counts.ndim() != 1
                || upper_event_counts.shape(0) != n)
            throw std::invalid_argument("extended observer count arrays have wrong shape");
        if (!resume_observer) {
            std::fill(row_event_counts.mutable_data(),
                      row_event_counts.mutable_data() + m, std::int64_t{0});
            std::fill(lower_event_counts.mutable_data(),
                      lower_event_counts.mutable_data() + n, std::int64_t{0});
            std::fill(upper_event_counts.mutable_data(),
                      upper_event_counts.mutable_data() + n, std::int64_t{0});
        }
    }
    if (observer_meta.size() != 0
            && (observer_meta.ndim() != 1 || observer_meta.shape(0) != 4))
        throw std::invalid_argument("observer_meta must have shape (4,)");
    if (observer_distance.size() != 0
            && (observer_distance.ndim() != 1 || observer_distance.shape(0) != 1))
        throw std::invalid_argument("observer_distance must have shape (1,)");

    const int W = window_size;
    const bool want_obj_tail = obj_tail_out.size() != 0;
    if (want_obj_tail && (obj_tail_out.ndim() != 1
                          || obj_tail_out.shape(0) < W
                          || obj_tail_len_out.ndim() != 1
                          || obj_tail_len_out.shape(0) != 1))
        throw std::invalid_argument("invalid extended objective tail buffers");
    const bool want_x_tail = x_tail_out.size() != 0;
    if (want_x_tail && (x_tail_out.ndim() != 2
                        || x_tail_out.shape(0) < W
                        || x_tail_out.shape(1) != n
                        || x_tail_len_out.ndim() != 1
                        || x_tail_len_out.shape(0) != 1))
        throw std::invalid_argument("invalid extended iterate tail buffers");

    std::vector<std::int64_t> event_ids, event_kinds, event_members;
    std::vector<double> event_distances;
    snn_qp::ExtendedObserver observer(
        have_counts ? row_event_counts.mutable_data() : nullptr,
        have_counts ? lower_event_counts.mutable_data() : nullptr,
        have_counts ? upper_event_counts.mutable_data() : nullptr,
        &event_ids, &event_kinds, &event_members, &event_distances);
    if (resume_observer && observer_meta.size() != 0) {
        const std::uint64_t no_candidate = std::numeric_limits<std::uint64_t>::max();
        const std::uint64_t* meta = observer_meta.data();
        observer.digest = meta[0];
        observer.first_candidate_id = (meta[1] == no_candidate)
            ? -1 : static_cast<std::int64_t>(meta[1]);
        observer.last_candidate_id = (meta[2] == no_candidate)
            ? -1 : static_cast<std::int64_t>(meta[2]);
        observer.projection_cap_rechecks = meta[3];
        if (observer_distance.size() != 0)
            observer.total_projection_distance = observer_distance.data()[0];
    }

    int obj_tail_len = 0;
    int x_tail_len = 0;
    snn_qp::ExtendedTrace trace;
    darray x_out(n);
    snn_qp::Result result;
    {
        py::gil_scoped_release release;
        result = snn_qp::solve_euler_extended(
            A.data(), b.data(), C.data(), d.data(), c_norms_sq.data(),
            row_scale.data(), c_gram.data(), n, m, k0, constraint_tol,
            max_iterations, max_projection_iters, has_lower, lower,
            has_upper, upper, candidate_meta.data(), n_candidates, coords.data(),
            candidate_data.data(), member_meta.data(), n_members,
            continue_after_budget, parallel, x0.data(), x_out.mutable_data(),
            &observer, iter_offset,
            want_obj_tail ? obj_tail_out.mutable_data() : nullptr,
            want_obj_tail ? &obj_tail_len : nullptr,
            want_x_tail ? x_tail_out.mutable_data() : nullptr,
            want_x_tail ? &x_tail_len : nullptr,
            window_size, use_solution_stable, &trace);
    }
    if (want_obj_tail) obj_tail_len_out.mutable_data()[0] = obj_tail_len;
    if (want_x_tail) x_tail_len_out.mutable_data()[0] = x_tail_len;

    if (observer_meta.size() != 0) {
        std::uint64_t* meta = observer_meta.mutable_data();
        const std::uint64_t no_candidate = std::numeric_limits<std::uint64_t>::max();
        meta[0] = observer.digest;
        meta[1] = (observer.first_candidate_id < 0)
                ? no_candidate : static_cast<std::uint64_t>(observer.first_candidate_id);
        meta[2] = (observer.last_candidate_id < 0)
                ? no_candidate : static_cast<std::uint64_t>(observer.last_candidate_id);
        meta[3] = observer.projection_cap_rechecks;
        if (observer_distance.size() != 0)
            observer_distance.mutable_data()[0] = observer.total_projection_distance;
    }

    return py::make_tuple(
        x_out, result.iterations_used, result.n_projections,
        result.converged, result.reason_code,
        copy_i64_vector(event_ids), copy_i64_vector(event_kinds),
        copy_i64_vector(event_members), copy_f64_vector(event_distances),
        copy_i64_vector(trace.dykstra_iterations),
        copy_i64_vector(trace.dykstra_events),
        copy_i64_vector(trace.dykstra_converged),
        copy_i64_vector(trace.dykstra_cap_hits),
        copy_i64_vector(trace.projection_truncated));
}

static py::tuple test_jacobi_svd(darray matrix, int cap, double tol) {
    if (matrix.ndim() != 2)
        throw std::invalid_argument("_test_jacobi_svd matrix must be 2-D");
    const int rows = static_cast<int>(matrix.shape(0));
    const int cols = static_cast<int>(matrix.shape(1));
    snn_qp::sets::SVDResult result;
    snn_qp::sets::jacobi_svd(matrix.data(), rows, cols, cap, tol, result);
    py::array_t<double> sigma(result.k), u({rows, result.k}), vt({result.k, cols});
    for (int i = 0; i < result.k; ++i) sigma.mutable_data()[i] = result.sigma[i];
    auto* up = u.mutable_data();
    auto* vp = vt.mutable_data();
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < result.k; ++c) up[r * result.k + c] = result.u[r][c];
    for (int r = 0; r < result.k; ++r)
        for (int c = 0; c < cols; ++c) vp[r * cols + c] = result.v[c][r];
    return py::make_tuple(sigma, u, vt, result.sweeps);
}

static py::tuple test_jacobi_eigh(darray matrix, int cap, double tol) {
    if (matrix.ndim() != 2 || matrix.shape(0) != matrix.shape(1))
        throw std::invalid_argument("_test_jacobi_eigh matrix must be square");
    const int n = static_cast<int>(matrix.shape(0));
    snn_qp::sets::EighResult result;
    snn_qp::sets::jacobi_eigh(matrix.data(), n, cap, tol, result);
    py::array_t<double> values(n), vectors({n, n});
    for (int i = 0; i < n; ++i) values.mutable_data()[i] = result.values[i];
    auto* vp = vectors.mutable_data();
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c) vp[r * n + c] = result.vectors[r][c];
    return py::make_tuple(values, vectors, result.sweeps);
}

PYBIND11_MODULE(_kernel, m) {
    m.doc() = "Compiled C++ kernel for the SNN-QP euler/adaptive solve path.";
    m.def("_test_jacobi_svd", &test_jacobi_svd,
          "Test-only bounded one-sided Jacobi SVD; returns (sigma, U, Vt, sweeps).",
          py::arg("matrix"), py::arg("cap") = snn_qp::sets::DEFAULT_SWEEP_CAP,
          py::arg("tol") = snn_qp::sets::DEFAULT_TOL);
    m.def("_test_jacobi_eigh", &test_jacobi_eigh,
          "Test-only bounded symmetric Jacobi eigensolver; returns (values, Q, sweeps).",
          py::arg("matrix"), py::arg("cap") = snn_qp::sets::DEFAULT_SWEEP_CAP,
          py::arg("tol") = snn_qp::sets::DEFAULT_TOL);
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
          py::arg("observer_distance") = f64array(0),
          // Chunked-execution support (v0.6.0): host-driven fixed chunks with
          // the stopping policy (KKT certificate) evaluated in Python between
          // chunks. iter_offset makes observer tokens absolute;
          // resume_observer seeds the observer from the caller's accumulated
          // meta instead of resetting; the tail buffers export the last
          // min(window_size, iterations) objective values / iterates for the
          // host's cheap window criteria.
          py::arg("iter_offset") = 0,
          py::arg("resume_observer") = false,
          py::arg("obj_tail_out") = f64array(0),
          py::arg("obj_tail_len_out") = i64array(0),
          py::arg("x_tail_out") = f64array(0),
          py::arg("x_tail_len_out") = i64array(0));

    m.def("solve_euler_extended", &solve_euler_extended_py,
          "Run the descriptor-driven native sweep for built-in conic/projector "
          "candidates.  The return value appends event and Dykstra telemetry "
          "arrays to the released solve tuple.",
          py::arg("A"), py::arg("b"), py::arg("C"), py::arg("d"),
          py::arg("c_norms_sq"), py::arg("row_scale"), py::arg("c_gram"),
          py::arg("x0"), py::arg("k0"), py::arg("constraint_tol"),
          py::arg("max_iterations"), py::arg("max_projection_iters"),
          py::arg("has_lower"), py::arg("lower"),
          py::arg("has_upper"), py::arg("upper"),
          py::arg("candidate_meta"), py::arg("coords"),
          py::arg("candidate_data"), py::arg("member_meta"),
          py::arg("continue_after_budget"), py::arg("parallel"),
          py::arg("row_event_counts") = i64array(0),
          py::arg("lower_event_counts") = i64array(0),
          py::arg("upper_event_counts") = i64array(0),
          py::arg("observer_meta") = u64array(0),
          py::arg("observer_distance") = f64array(0),
          py::arg("iter_offset") = 0,
          py::arg("resume_observer") = false,
          py::arg("obj_tail_out") = f64array(0),
          py::arg("obj_tail_len_out") = i64array(0),
          py::arg("x_tail_out") = f64array(0),
          py::arg("x_tail_len_out") = i64array(0),
          py::arg("window_size") = 1,
          py::arg("use_solution_stable") = false);

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
