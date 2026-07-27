#pragma once

#include "msrp_bundle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace msrp_v05 {

constexpr std::uint64_t DIGEST_OFFSET = UINT64_C(14695981039346656037);
constexpr std::uint64_t DIGEST_PRIME = UINT64_C(1099511628211);

struct Observer {
    std::uint64_t digest = DIGEST_OFFSET;
    std::uint64_t total = 0;
    std::uint64_t rows = 0;
    std::uint64_t lower = 0;
    std::uint64_t upper = 0;
    std::uint64_t cap_rechecks = 0;
    std::uint64_t first = NO_CANDIDATE;
    std::uint64_t last = NO_CANDIDATE;

    inline void digest_word(std::uint64_t word) {
        digest = (digest ^ (word + UINT64_C(1))) * DIGEST_PRIME;
    }

    inline void record(int kind, int index, int m, int n, int outer,
                       int ordinal) {
        std::uint64_t candidate;
        if (kind == 0) {
            ++rows;
            candidate = static_cast<std::uint64_t>(index);
        } else if (kind == 1) {
            ++lower;
            candidate = static_cast<std::uint64_t>(m + index);
        } else {
            ++upper;
            candidate = static_cast<std::uint64_t>(m + n + index);
        }
        if (first == NO_CANDIDATE) first = candidate;
        last = candidate;
        digest_word(static_cast<std::uint64_t>(outer));
        digest_word(static_cast<std::uint64_t>(ordinal));
        digest_word(candidate);
        ++total;
    }
};

struct SolveResult {
    std::vector<double> x;
    int status = 0;
    int iterations_executed = 0;
    int threads = 1;
    Observer observer;
};

inline double fresh_joint_violation(const Problem& q,
                                    const std::vector<double>& x,
                                    std::vector<double>& r) {
    double maximum = 0.0;
    for (int i = 0; i < q.m; ++i) {
        const double* row =
            q.C.data() + static_cast<std::size_t>(i) * q.n;
        double sum = 0.0;
        for (int j = 0; j < q.n; ++j) sum += row[j] * x[j];
        r[i] = sum + q.d[i];
        maximum = std::max(maximum, r[i] * q.row_scale[i]);
    }
    if (q.has_lower) {
        for (int i = 0; i < q.n; ++i)
            maximum = std::max(maximum, q.lower - x[i]);
    }
    if (q.has_upper) {
        for (int i = 0; i < q.n; ++i)
            maximum = std::max(maximum, x[i] - q.upper);
    }
    return maximum;
}

inline bool projection_sweep(const Problem& q, std::vector<double>& x,
                             std::vector<double>& r, int outer,
                             Observer& observer) {
    for (int i = 0; i < q.m; ++i) {
        const double* row =
            q.C.data() + static_cast<std::size_t>(i) * q.n;
        double sum = 0.0;
        for (int j = 0; j < q.n; ++j) sum += row[j] * x[j];
        r[i] = sum + q.d[i];
    }

    for (int ordinal = 0; ordinal < q.projection_cap; ++ordinal) {
        int winner = 0;
        int kind = 0;
        double best = r[0] * q.row_scale[0];
        for (int i = 1; i < q.m; ++i) {
            const double score = r[i] * q.row_scale[i];
            if (score > best) {
                best = score;
                winner = i;
            }
        }
        if (q.has_lower) {
            for (int i = 0; i < q.n; ++i) {
                const double score = q.lower - x[i];
                if (score > best) {
                    best = score;
                    winner = i;
                    kind = 1;
                }
            }
        }
        if (q.has_upper) {
            for (int i = 0; i < q.n; ++i) {
                const double score = x[i] - q.upper;
                if (score > best) {
                    best = score;
                    winner = i;
                    kind = 2;
                }
            }
        }
        if (best <= q.constraint_tol) return false;

        if (kind == 0) {
            const double step = r[winner] / q.c_norms_sq[winner];
            const double* row =
                q.C.data() + static_cast<std::size_t>(winner) * q.n;
            for (int j = 0; j < q.n; ++j) x[j] -= step * row[j];
            const double* gram =
                q.G.data() + static_cast<std::size_t>(winner) * q.m;
            for (int i = 0; i < q.m; ++i) r[i] -= step * gram[i];
        } else {
            const double delta = kind == 1 ? best : -best;
            x[winner] += delta;
            for (int i = 0; i < q.m; ++i) {
                r[i] += delta *
                        q.C[static_cast<std::size_t>(i) * q.n + winner];
            }
        }
        observer.record(kind, winner, q.m, q.n, outer, ordinal);
    }

    ++observer.cap_rechecks;
    return fresh_joint_violation(q, x, r) > q.constraint_tol;
}

inline SolveResult solve_serial(const Problem& q) {
    SolveResult result;
    result.x = q.x0;
    result.threads = 1;
    std::vector<double> Ax(q.n, 0.0);
    std::vector<double> residual(q.m, 0.0);

    for (int outer = 0; outer < q.iterations; ++outer) {
        for (int i = 0; i < q.n; ++i) {
            const double* row =
                q.A.data() + static_cast<std::size_t>(i) * q.n;
            double sum = 0.0;
            for (int j = 0; j < q.n; ++j) sum += row[j] * result.x[j];
            Ax[i] = sum;
        }
        for (int i = 0; i < q.n; ++i) {
            result.x[i] -= q.k0 * (Ax[i] + q.b[i]);
        }
        const bool exhausted =
            projection_sweep(q, result.x, residual, outer, result.observer);
        result.iterations_executed = outer + 1;
        if (exhausted) {
            result.status = 2;
            break;
        }
    }
    return result;
}

// The row matvecs are the only parallel phase. A persistent OpenMP team spans
// the complete fixed horizon so the comparison does not pay fork/join cost per
// outer iteration. The projection recurrence remains strictly serial.
inline SolveResult solve_openmp(const Problem& q, int requested_threads) {
#ifndef _OPENMP
    (void)requested_threads;
    return solve_serial(q);
#else
    if (requested_threads <= 1) return solve_serial(q);

    SolveResult result;
    result.x = q.x0;
    std::vector<double> Ax(q.n, 0.0);
    std::vector<double> residual(q.m, 0.0);
    bool stop = false;
    int threads_observed = 1;

    omp_set_dynamic(0);
    omp_set_num_threads(requested_threads);
#pragma omp parallel shared(stop, threads_observed, result, Ax, residual)
    {
#pragma omp single
        { threads_observed = omp_get_num_threads(); }

        for (int outer = 0; outer < q.iterations; ++outer) {
#pragma omp for schedule(static)
            for (int i = 0; i < q.n; ++i) {
                const double* row =
                    q.A.data() + static_cast<std::size_t>(i) * q.n;
                double sum = 0.0;
                for (int j = 0; j < q.n; ++j)
                    sum += row[j] * result.x[j];
                Ax[i] = sum;
            }
#pragma omp single
            {
                for (int i = 0; i < q.n; ++i)
                    result.x[i] -= q.k0 * (Ax[i] + q.b[i]);
                stop = projection_sweep(q, result.x, residual, outer,
                                        result.observer);
                result.iterations_executed = outer + 1;
                if (stop) result.status = 2;
            }
#pragma omp barrier
            if (stop) break;
        }
    }
    result.threads = threads_observed;
    return result;
#endif
}

inline std::array<std::uint64_t, TELEMETRY_WORDS> telemetry(
    const Problem& q, const SolveResult& result) {
    std::array<std::uint64_t, TELEMETRY_WORDS> out{};
    out[0] = TELEMETRY_MAGIC;
    out[1] = static_cast<std::uint64_t>(result.status);
    out[2] = static_cast<std::uint64_t>(q.iterations);
    out[3] = static_cast<std::uint64_t>(result.iterations_executed);
    out[4] = result.status == 0 &&
                     result.iterations_executed == q.iterations
                 ? UINT64_C(1)
                 : UINT64_C(0);
    out[5] = result.observer.total;
    out[6] = result.observer.rows;
    out[7] = result.observer.lower;
    out[8] = result.observer.upper;
    out[9] = result.observer.cap_rechecks;
    out[10] = result.observer.digest;
    out[11] = result.observer.first;
    out[12] = result.observer.last;
    out[13] = static_cast<std::uint64_t>(result.threads);
    out[14] = UINT64_C(64);
    out[15] = UINT64_C(1);
    return out;
}

}  // namespace msrp_v05
