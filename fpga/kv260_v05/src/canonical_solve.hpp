#pragma once

#include "../../../src/snn_opt/_native/snn_qp_core.hpp"

#include "msrp_bundle.hpp"
#include "v05_double_core.hpp"

#include <cstdint>
#include <vector>

namespace msrp_v05 {

// Thin fixed-horizon adapter around the frozen canonical SNN_opt v0.5 C core.
// The adapter changes no recurrence semantics. It disables the stopping-policy
// checks so G2/G3 and the one-core A53 baseline execute exactly the requested
// horizon, while preserving the canonical unified projection and observer.
inline SolveResult solve_canonical(const Problem& q) {
    SolveResult out;
    out.x.assign(q.n, 0.0);
    out.threads = 1;
    std::vector<std::int64_t> row_counts(q.m, 0);
    std::vector<std::int64_t> lower_counts(q.n, 0);
    std::vector<std::int64_t> upper_counts(q.n, 0);
    snn_qp::ProjectionObserver observer(
        row_counts.data(), lower_counts.data(), upper_counts.data());

    const snn_qp::Result result = snn_qp::solve_euler(
        q.A.data(), q.b.data(), q.C.data(), q.d.data(),
        q.c_norms_sq.data(), q.row_scale.data(), q.G.data(), q.n, q.m, q.k0,
        q.constraint_tol, q.iterations, q.projection_cap,
        false,  // fixed horizon, no early stopping
        1, 0, 1, 1, 0.0, 0.0, 0.0, 0.0,
        false, false, false, false, q.has_lower, q.lower, q.has_upper,
        q.upper, nullptr, false, false, q.x0.data(), out.x.data(), &observer);

    out.status = result.reason_code == snn_qp::REASON_PROJECTION_BUDGET ? 2 : 0;
    out.iterations_executed = result.iterations_used;
    out.observer.digest = observer.digest;
    out.observer.rows = 0;
    out.observer.lower = 0;
    out.observer.upper = 0;
    for (const auto value : row_counts)
        out.observer.rows += static_cast<std::uint64_t>(value);
    for (const auto value : lower_counts)
        out.observer.lower += static_cast<std::uint64_t>(value);
    for (const auto value : upper_counts)
        out.observer.upper += static_cast<std::uint64_t>(value);
    out.observer.total =
        out.observer.rows + out.observer.lower + out.observer.upper;
    out.observer.cap_rechecks = observer.projection_cap_rechecks;
    out.observer.first =
        observer.first_candidate_id < 0
            ? NO_CANDIDATE
            : static_cast<std::uint64_t>(observer.first_candidate_id);
    out.observer.last =
        observer.last_candidate_id < 0
            ? NO_CANDIDATE
            : static_cast<std::uint64_t>(observer.last_candidate_id);
    return out;
}

}  // namespace msrp_v05
