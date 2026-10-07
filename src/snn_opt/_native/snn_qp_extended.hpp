// snn_qp_extended.hpp -- native sweep for the built-in conic candidates.
//
// This is deliberately a sibling of snn_qp_core.hpp.  The released
// solve_euler entry point and its arithmetic remain untouched; this header
// contains only the descriptor-driven extension used when the Python host has
// serialized built-in projectors (ball, SOC, scaled SOC, affine, halfspace,
// and cold Dykstra compositions of those projectors).

#pragma once

#include "snn_qp_core.hpp"
#include "snn_qp_sets.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace snn_qp {

enum ExtendedCandidateKind {
    EXT_BALL       = 0,
    EXT_SOC        = 1,
    EXT_SCALED_SOC = 2,
    EXT_AFFINE     = 3,
    EXT_HALFSPACE  = 4,
    EXT_DYKSTRA    = 5,
    EXT_SPECTRAL_BALL = 6,
    EXT_SPECTRAL_CUTTER = 7,
    EXT_PSD_CONE = 8,
};

// The released core observer is intentionally not extended: FPGA asset tests
// pin that header byte-for-byte.  This sibling observer carries the same
// digest/count contract plus the nonlinear trace buffers for the new entry
// point.
struct ExtendedObserver {
    static constexpr std::uint64_t DIGEST_OFFSET = UINT64_C(14695981039346656037);
    static constexpr std::uint64_t DIGEST_PRIME  = UINT64_C(1099511628211);

    std::int64_t* row_counts;
    std::int64_t* lower_counts;
    std::int64_t* upper_counts;
    std::uint64_t digest;
    double total_projection_distance;
    std::int64_t first_candidate_id;
    std::int64_t last_candidate_id;
    std::uint64_t projection_cap_rechecks;
    std::vector<std::int64_t>* event_ids;
    std::vector<std::int64_t>* event_kinds;
    std::vector<std::int64_t>* event_members;
    std::vector<double>* event_distances;

    ExtendedObserver(std::int64_t* rows, std::int64_t* lower,
                     std::int64_t* upper,
                     std::vector<std::int64_t>* ids,
                     std::vector<std::int64_t>* kinds,
                     std::vector<std::int64_t>* members,
                     std::vector<double>* distances)
        : row_counts(rows), lower_counts(lower), upper_counts(upper),
          digest(DIGEST_OFFSET), total_projection_distance(0.0),
          first_candidate_id(-1), last_candidate_id(-1),
          projection_cap_rechecks(0), event_ids(ids), event_kinds(kinds),
          event_members(members), event_distances(distances) {}

    inline void update_digest_word(std::uint64_t word) {
        digest = (digest ^ (word + 1U)) * DIGEST_PRIME;
    }

    inline void record(int kind, int index, int m, int n,
                       int outer_iteration, int ordinal,
                       double correction_norm) {
        std::int64_t candidate_id;
        if (kind == 0) {
            if (row_counts) ++row_counts[index];
            candidate_id = index;
        } else if (kind == 1) {
            if (lower_counts) ++lower_counts[index];
            candidate_id = static_cast<std::int64_t>(m) + index;
        } else {
            if (upper_counts) ++upper_counts[index];
            candidate_id = static_cast<std::int64_t>(m) + n + index;
        }
        if (first_candidate_id < 0) first_candidate_id = candidate_id;
        last_candidate_id = candidate_id;
        update_digest_word(static_cast<std::uint64_t>(outer_iteration));
        update_digest_word(static_cast<std::uint64_t>(ordinal));
        update_digest_word(static_cast<std::uint64_t>(candidate_id));
        total_projection_distance += correction_norm;
        if (event_ids) event_ids->push_back(candidate_id);
        if (event_kinds) event_kinds->push_back(kind);
        if (event_members) event_members->push_back(-1);
        if (event_distances) event_distances->push_back(correction_norm);
    }

    inline void record_nonlinear(int candidate_id, int outer_iteration,
                                 int ordinal, double correction_norm,
                                 int member_index = -1) {
        if (first_candidate_id < 0) first_candidate_id = candidate_id;
        last_candidate_id = candidate_id;
        update_digest_word(static_cast<std::uint64_t>(outer_iteration));
        update_digest_word(static_cast<std::uint64_t>(ordinal));
        update_digest_word(static_cast<std::uint64_t>(candidate_id));
        total_projection_distance += correction_norm;
        if (event_ids) event_ids->push_back(candidate_id);
        if (event_kinds) event_kinds->push_back(3);
        if (event_members) event_members->push_back(member_index);
        if (event_distances) event_distances->push_back(correction_norm);
    }

    inline void record_projection_cap_recheck() {
        ++projection_cap_rechecks;
    }
};

// The Python binding passes each descriptor as eleven int64 words:
//   kind, coord_offset, coord_count, t_index,
//   member_offset, member_count, aux0, aux1, flags,
//   data_offset, data_count.
// The same descriptor layout is used for top-level candidates and Dykstra
// members.  Offsets index the shared packed coordinate/data arrays.
struct ExtendedDescriptor {
    int kind;
    int coord_offset;
    int coord_count;
    int t_index;
    int member_offset;
    int member_count;
    int aux0;
    int aux1;
    int flags;
    int data_offset;
    int data_count;
};

inline ExtendedDescriptor extended_descriptor(const std::int64_t* meta,
                                               int index) {
    const std::int64_t* p = meta + static_cast<std::size_t>(index) * 11U;
    ExtendedDescriptor d;
    d.kind = static_cast<int>(p[0]);
    d.coord_offset = static_cast<int>(p[1]);
    d.coord_count = static_cast<int>(p[2]);
    d.t_index = static_cast<int>(p[3]);
    d.member_offset = static_cast<int>(p[4]);
    d.member_count = static_cast<int>(p[5]);
    d.aux0 = static_cast<int>(p[6]);
    d.aux1 = static_cast<int>(p[7]);
    d.flags = static_cast<int>(p[8]);
    d.data_offset = static_cast<int>(p[9]);
    d.data_count = static_cast<int>(p[10]);
    return d;
}

struct ExtendedTrace {
    std::vector<std::int64_t> event_ids;
    std::vector<std::int64_t> event_kinds;
    std::vector<std::int64_t> event_members;
    std::vector<double> event_distances;
    std::vector<std::int64_t> dykstra_iterations;
    std::vector<std::int64_t> dykstra_events;
    std::vector<std::int64_t> dykstra_converged;
    std::vector<std::int64_t> dykstra_cap_hits;
    std::vector<std::int64_t> projection_truncated;
};

struct ExtendedDykstraEvent {
    int member_index;
    double correction_norm;
    std::vector<double> delta;
};

struct ExtendedDykstraResult {
    std::vector<double> projected;
    std::vector<ExtendedDykstraEvent> events;
    int iterations = 0;
    bool converged = false;
    bool cap_hit = false;
    double max_member_residual = std::numeric_limits<double>::infinity();
    double settled_residual = std::numeric_limits<double>::infinity();
    double correction_settled_residual = std::numeric_limits<double>::infinity();
};

inline double extended_norm(const std::vector<double>& x) {
    double acc = 0.0;
    #pragma omp simd reduction(+ : acc)
    for (std::size_t i = 0; i < x.size(); ++i) acc += x[i] * x[i];
    return std::sqrt(acc);
}

inline double extended_norm_scoped(const std::vector<double>& x,
                                   const std::int64_t* coords,
                                   int coord_offset, int coord_count) {
    if (coord_count <= 0) return extended_norm(x);
    double acc = 0.0;
    for (int k = 0; k < coord_count; ++k) {
        const int i = static_cast<int>(coords[coord_offset + k]);
        acc += x[static_cast<std::size_t>(i)]
             * x[static_cast<std::size_t>(i)];
    }
    return std::sqrt(acc);
}

inline double extended_norm_delta(const std::vector<double>& a,
                                  const std::vector<double>& b,
                                  const std::int64_t* coords,
                                  int coord_offset, int coord_count) {
    double acc = 0.0;
    if (coord_count > 0) {
        for (int k = 0; k < coord_count; ++k) {
            const int i = static_cast<int>(coords[coord_offset + k]);
            const double d = a[static_cast<std::size_t>(i)]
                           - b[static_cast<std::size_t>(i)];
            acc += d * d;
        }
    } else {
        #pragma omp simd reduction(+ : acc)
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double d = a[i] - b[i];
            acc += d * d;
        }
    }
    return std::sqrt(acc);
}

inline double extended_dot(const double* a, const double* b, int n) {
    double acc = 0.0;
    #pragma omp simd reduction(+ : acc)
    for (int i = 0; i < n; ++i) acc += a[i] * b[i];
    return acc;
}

// Forward declaration because a Dykstra member uses the same primitive as a
// top-level candidate.  Members are validated on the host to be non-Dykstra.
inline void extended_project_one(
        const ExtendedDescriptor& desc,
        const std::int64_t* all_meta, int n_meta,
        const std::int64_t* coords, const double* data,
        const std::int64_t* member_meta, int n_members,
        int n, const std::vector<double>& input,
        std::vector<double>& output,
        ExtendedDykstraResult* dykstra_out,
        bool collect_dykstra_events);

inline void extended_project_simple(
        const ExtendedDescriptor& desc,
        const std::int64_t* all_meta, int n_meta,
        const std::int64_t* coords, const double* data,
        const std::int64_t* member_meta, int n_members,
        int n, const std::vector<double>& input,
        std::vector<double>& output) {
    extended_project_one(desc, all_meta, n_meta, coords, data,
                         member_meta, n_members, n, input, output,
                         nullptr, false);
}

inline void extended_project_dykstra(
        const ExtendedDescriptor& desc,
        const std::int64_t* all_meta, int n_meta,
        const std::int64_t* coords, const double* data,
        const std::int64_t* member_meta, int n_members,
        int n, const std::vector<double>& input,
        ExtendedDykstraResult& result, bool collect_dykstra_events) {
    (void)all_meta;
    (void)n_meta;
    result = ExtendedDykstraResult{};
    result.projected = input;
    const int count = desc.member_count;
    std::vector<std::vector<double>> corrections(
        static_cast<std::size_t>(count), std::vector<double>(n, 0.0));
    const double tolerance = data[desc.data_offset];
    const int max_iterations = desc.aux0;
    for (int cycle = 0; cycle < max_iterations; ++cycle) {
        const std::vector<double> previous = result.projected;
        const std::vector<std::vector<double>> previous_corrections = corrections;
        result.correction_settled_residual = 0.0;
        for (int i = 0; i < count; ++i) {
            const int member_index = desc.member_offset + i;
            const ExtendedDescriptor member =
                extended_descriptor(member_meta, member_index);
            std::vector<double> shifted(n);
            for (int k = 0; k < n; ++k)
                shifted[k] = result.projected[k] + corrections[static_cast<std::size_t>(i)][k];
            std::vector<double> projected;
            extended_project_simple(member, all_meta, n_meta, coords, data,
                                    member_meta, n_members, n, shifted,
                                    projected);
            std::vector<double> correction(n);
            for (int k = 0; k < n; ++k)
                correction[k] = shifted[k] - projected[k];
            result.correction_settled_residual += extended_norm_delta(
                correction, previous_corrections[static_cast<std::size_t>(i)],
                nullptr, 0, 0);
            corrections[static_cast<std::size_t>(i)] = correction;
            result.projected.swap(projected);
            const double correction_norm = extended_norm(correction);
            if (!std::isfinite(correction_norm)) {
                throw std::invalid_argument(
                    "Dykstra member returned a non-finite correction");
            }
            ExtendedDykstraEvent event;
            event.member_index = i;
            event.correction_norm = correction_norm;
            event.delta.resize(n);
            for (int k = 0; k < n; ++k)
                event.delta[k] = -correction[k];
            if (collect_dykstra_events) result.events.push_back(std::move(event));
        }
        result.iterations = cycle + 1;
        result.settled_residual = extended_norm_delta(
            result.projected, previous, nullptr, 0, 0);

        double max_residual = 0.0;
        for (int i = 0; i < count; ++i) {
            const ExtendedDescriptor member =
                extended_descriptor(member_meta, desc.member_offset + i);
            std::vector<double> probe;
            extended_project_simple(member, all_meta, n_meta, coords, data,
                                    member_meta, n_members, n,
                                    result.projected, probe);
            max_residual = std::max(
                max_residual,
                extended_norm_delta(probe, result.projected, nullptr, 0, 0));
        }
        result.max_member_residual = max_residual;
        const double scale = std::max(
            1.0, extended_norm_scoped(result.projected, coords,
                                       desc.coord_offset, desc.coord_count));
        const double threshold = tolerance * scale;
        if (result.settled_residual <= threshold
                && max_residual <= threshold
                && result.correction_settled_residual <= threshold) {
            result.converged = true;
            break;
        }
    }
    result.cap_hit = !result.converged;

    // A Dykstra candidate may scope its update to a coordinate block.  The
    // Python implementation checks this invariant after the inner solve and
    // then embeds the scoped result back into the ambient input.
    if (desc.coord_count > 0) {
        result.projected.resize(n);
        std::vector<double> scoped = input;
        for (int k = 0; k < desc.coord_count; ++k) {
            const int j = static_cast<int>(coords[desc.coord_offset + k]);
            scoped[j] = result.projected[j];
        }
        result.projected.swap(scoped);
    }
}

inline void extended_project_one(
        const ExtendedDescriptor& desc,
        const std::int64_t* all_meta, int n_meta,
        const std::int64_t* coords, const double* data,
        const std::int64_t* member_meta, int n_members,
        int n, const std::vector<double>& input,
        std::vector<double>& output,
        ExtendedDykstraResult* dykstra_out,
        bool collect_dykstra_events) {
    output = input;
    if (desc.kind == EXT_DYKSTRA) {
        ExtendedDykstraResult local;
        extended_project_dykstra(desc, all_meta, n_meta, coords, data,
                                 member_meta, n_members, n, input, local,
                                 collect_dykstra_events);
        output = local.projected;
        if (dykstra_out) *dykstra_out = std::move(local);
        return;
    }

    if (desc.kind == EXT_BALL) {
        const double radius = data[desc.data_offset];
        const double* center = data + desc.data_offset + 1;
        double norm_sq = 0.0;
        for (int k = 0; k < desc.coord_count; ++k) {
            const int j = static_cast<int>(coords[desc.coord_offset + k]);
            const double delta = input[j] - center[k];
            norm_sq += delta * delta;
        }
        const double norm = std::sqrt(norm_sq);
        if (norm > radius && norm > 0.0) {
            for (int k = 0; k < desc.coord_count; ++k) {
                const int j = static_cast<int>(coords[desc.coord_offset + k]);
                const double delta = input[j] - center[k];
                output[j] = center[k] + (radius / norm) * delta;
            }
        } else if (radius == 0.0) {
            for (int k = 0; k < desc.coord_count; ++k)
                output[static_cast<int>(coords[desc.coord_offset + k])] = center[k];
        }
        return;
    }

    if (desc.kind == EXT_SOC || desc.kind == EXT_SCALED_SOC) {
        const double mu = (desc.kind == EXT_SCALED_SOC)
                        ? data[desc.data_offset] : 1.0;
        const int t = desc.t_index;
        double zn_sq = 0.0;
        for (int k = 0; k < desc.coord_count - 1; ++k) {
            const int j = static_cast<int>(coords[desc.coord_offset + 1 + k]);
            zn_sq += input[j] * input[j];
        }
        const double zn = std::sqrt(zn_sq);
        const double t_value = input[t];
        if (desc.kind == EXT_SOC) {
            if (zn <= t_value) return;
            if (zn <= -t_value) {
                output[t] = 0.0;
                for (int k = 0; k < desc.coord_count - 1; ++k)
                    output[static_cast<int>(coords[desc.coord_offset + 1 + k])] = 0.0;
                return;
            }
            const double alpha = 0.5 * (zn + t_value);
            output[t] = alpha;
            const double scale = alpha / zn;
            for (int k = 0; k < desc.coord_count - 1; ++k) {
                const int j = static_cast<int>(coords[desc.coord_offset + 1 + k]);
                output[j] = scale * input[j];
            }
            return;
        }
        if (zn <= mu * t_value) return;
        if (t_value + mu * zn <= 0.0) {
            output[t] = 0.0;
            for (int k = 0; k < desc.coord_count - 1; ++k)
                output[static_cast<int>(coords[desc.coord_offset + 1 + k])] = 0.0;
            return;
        }
        const double t_new = (t_value + mu * zn) / (1.0 + mu * mu);
        output[t] = t_new;
        const double z_new_norm = mu * t_new;
        const double scale = z_new_norm / zn;
        for (int k = 0; k < desc.coord_count - 1; ++k) {
            const int j = static_cast<int>(coords[desc.coord_offset + 1 + k]);
            output[j] = scale * input[j];
        }
        return;
    }

    if (desc.kind == EXT_SPECTRAL_CUTTER) {
        throw std::logic_error("spectral cutter cannot be used as a projector");
    }

    if (desc.kind == EXT_SPECTRAL_BALL || desc.kind == EXT_PSD_CONE) {
        const int rows = desc.aux0;
        const int cols = desc.kind == EXT_PSD_CONE ? desc.aux0 : desc.aux1;
        const int local_dim = (desc.kind == EXT_PSD_CONE)
            ? rows * (rows + 1) / 2 : rows * cols;
        std::vector<double> local(static_cast<std::size_t>(local_dim));
        if (desc.coord_count > 0) {
            for (int k = 0; k < local_dim; ++k)
                local[k] = input[static_cast<int>(coords[desc.coord_offset + k])];
        } else {
            if (local_dim > n)
                throw std::invalid_argument("native nonlinear set block exceeds state dimension");
            for (int k = 0; k < local_dim; ++k) local[k] = input[k];
        }
        std::vector<double> projected(static_cast<std::size_t>(local_dim));
        if (desc.kind == EXT_SPECTRAL_BALL) {
            sets::spectral_project(local.data(), rows, cols,
                                   data[desc.data_offset], projected.data());
        } else {
            sets::psd_project(local.data(), rows, projected.data());
        }
        if (desc.coord_count > 0) {
            for (int k = 0; k < local_dim; ++k)
                output[static_cast<int>(coords[desc.coord_offset + k])] = projected[k];
        } else {
            for (int k = 0; k < local_dim; ++k) output[k] = projected[k];
        }
        return;
    }

    if (desc.kind == EXT_HALFSPACE) {
        const double* normal = data + desc.data_offset;
        const int normal_count = desc.aux0;
        const double offset = data[desc.data_offset + normal_count];
        double residual = offset;
        if (desc.coord_count > 0) {
            for (int k = 0; k < normal_count; ++k) {
                const int j = static_cast<int>(coords[desc.coord_offset + k]);
                residual += normal[k] * input[j];
            }
        } else {
            for (int k = 0; k < normal_count; ++k)
                residual += normal[k] * input[k];
        }
        double norm_sq = 0.0;
        for (int k = 0; k < normal_count; ++k) norm_sq += normal[k] * normal[k];
        if (residual > 0.0 && norm_sq > 1e-24) {
            const double scale = -residual / norm_sq;
            if (desc.coord_count > 0) {
                for (int k = 0; k < normal_count; ++k) {
                    const int j = static_cast<int>(coords[desc.coord_offset + k]);
                    output[j] = input[j] + scale * normal[k];
                }
            } else {
                for (int k = 0; k < normal_count; ++k)
                    output[k] = input[k] + scale * normal[k];
            }
        }
        return;
    }

    if (desc.kind == EXT_AFFINE) {
        const int q = desc.aux0;
        const int p = desc.aux1;
        const double* ptr = data + desc.data_offset;
        const double* B = ptr;
        const double* h = B + static_cast<std::size_t>(p) * q;
        const double* correction = h + p;
        const double* graph = correction + static_cast<std::size_t>(q) * p;
        const int local_size = desc.coord_count > 0 ? desc.coord_count : n;
        std::vector<double> local(static_cast<std::size_t>(local_size));
        if (desc.coord_count > 0) {
            for (int k = 0; k < local_size; ++k)
                local[k] = input[static_cast<int>(coords[desc.coord_offset + k])];
        } else {
            local = input;
        }
        if (local_size == q + p) {
            std::vector<double> residual(p, 0.0), updated(q, 0.0);
            for (int i = 0; i < p; ++i) {
                double value = local[q + i] - h[i];
                for (int j = 0; j < q; ++j) value -= B[i * q + j] * local[j];
                residual[i] = value;
            }
            for (int i = 0; i < q; ++i) {
                double value = local[i];
                for (int j = 0; j < p; ++j) value += graph[i * p + j] * residual[j];
                updated[i] = value;
            }
            for (int i = 0; i < q; ++i) {
                local[i] = updated[i];
            }
            for (int i = 0; i < p; ++i) {
                double value = h[i];
                for (int j = 0; j < q; ++j) value += B[i * q + j] * updated[j];
                local[q + i] = value;
            }
        } else if (local_size == q) {
            std::vector<double> residual(p, 0.0), updated(q, 0.0);
            for (int i = 0; i < p; ++i) {
                double value = -h[i];
                for (int j = 0; j < q; ++j) value += B[i * q + j] * local[j];
                residual[i] = value;
            }
            for (int i = 0; i < q; ++i) {
                double value = local[i];
                for (int j = 0; j < p; ++j) value -= correction[i * p + j] * residual[j];
                updated[i] = value;
            }
            local.swap(updated);
        }
        if (desc.coord_count > 0) {
            for (int k = 0; k < local_size; ++k)
                output[static_cast<int>(coords[desc.coord_offset + k])] = local[k];
        } else {
            output.swap(local);
        }
        return;
    }
}

struct ExtendedCandidateEval {
    std::vector<double> projected;
    double score = 0.0;
    bool is_dykstra = false;
    bool is_cutter = false;
    ExtendedDykstraResult dykstra;
};

inline double extended_candidate_score(
        const ExtendedDescriptor& desc,
        const std::int64_t* all_meta, int n_meta,
        const std::int64_t* coords, const double* data,
        const std::int64_t* member_meta, int n_members,
        int n, const std::vector<double>& x,
        ExtendedCandidateEval& eval) {
    eval.is_dykstra = (desc.kind == EXT_DYKSTRA);
    eval.is_cutter = (desc.kind == EXT_SPECTRAL_CUTTER);
    if (eval.is_cutter) {
        const int rows = desc.aux0, cols = desc.aux1;
        const int local_dim = rows * cols;
        std::vector<double> local(static_cast<std::size_t>(local_dim));
        if (desc.coord_count > 0) {
            for (int k = 0; k < local_dim; ++k)
                local[k] = x[static_cast<int>(coords[desc.coord_offset + k])];
        } else {
            if (local_dim > n)
                throw std::invalid_argument("spectral cutter block exceeds state dimension");
            for (int k = 0; k < local_dim; ++k) local[k] = x[k];
        }
        double sigma = 0.0;
        double u[sets::MAX_DIM]{}, v[sets::MAX_DIM]{};
        sets::spectral_top_pair(local.data(), rows, cols, sigma, u, v);
        const double value = sigma - data[desc.data_offset];
        eval.projected = x;
        if (value <= 0.0) { eval.score = 0.0; return 0.0; }
        std::vector<double> gradient(static_cast<std::size_t>(n), 0.0);
        double norm_sq = 0.0;
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                const double g = u[r] * v[c];
                const int k = r * cols + c;
                const int j = desc.coord_count > 0
                    ? static_cast<int>(coords[desc.coord_offset + k]) : k;
                gradient[static_cast<std::size_t>(j)] = g;
                norm_sq += g * g;
            }
        if (!std::isfinite(norm_sq) || norm_sq <= 1e-24)
            throw std::invalid_argument("spectral cutter has a near-zero gradient");
        const double scale = value / norm_sq;
        for (int j = 0; j < n; ++j) eval.projected[j] -= scale * gradient[j];
        eval.score = value / std::sqrt(norm_sq);
        if (!std::isfinite(eval.score))
            throw std::invalid_argument("spectral cutter produced a non-finite correction");
        return eval.score;
    }
    extended_project_one(desc, all_meta, n_meta, coords, data,
                         member_meta, n_members, n, x, eval.projected,
                         eval.is_dykstra ? &eval.dykstra : nullptr,
                         eval.is_dykstra);
    eval.score = extended_norm_delta(eval.projected, x, coords,
                                     desc.coord_offset, desc.coord_count);
    return eval.score;
}

inline double extended_max_violation(
        const double* C, const double* d, const double* row_scale,
        const double* x, int n, int m,
        bool has_lower, double lower, bool has_upper, double upper,
        const std::int64_t* candidate_meta, int n_candidates,
        const std::int64_t* coords, const double* candidate_data,
        const std::int64_t* member_meta, int n_members,
        double* residual, bool parallel) {
    double best = 0.0;
    if (m > 0) {
        matvec(C, x, residual, m, n, parallel);
        for (int i = 0; i < m; ++i)
            best = std::max(best, (residual[i] + d[i]) * row_scale[i]);
    }
    if (has_lower)
        for (int i = 0; i < n; ++i) best = std::max(best, lower - x[i]);
    if (has_upper)
        for (int i = 0; i < n; ++i) best = std::max(best, x[i] - upper);
    std::vector<double> state(x, x + n);
    for (int q = 0; q < n_candidates; ++q) {
        ExtendedCandidateEval eval;
        const ExtendedDescriptor desc = extended_descriptor(candidate_meta, q);
        extended_candidate_score(desc, candidate_meta, n_candidates, coords,
                                 candidate_data, member_meta, n_members, n,
                                 state, eval);
        best = std::max(best, eval.score);
    }
    return best;
}

inline int project_extended(
        double* x,
        const double* C, const double* d,
        const double* c_norms_sq, const double* row_scale,
        const double* G, int n, int m,
        double constraint_tol, int proj_cap,
        bool has_lower, double lower, bool has_upper, double upper,
        const std::int64_t* candidate_meta, int n_candidates,
        const std::int64_t* coords, const double* candidate_data,
        const std::int64_t* member_meta, int n_members,
        bool continue_after_budget, double* residual, bool parallel,
        bool* budget_exhausted, int outer_iteration,
        ExtendedObserver* observer, ExtendedTrace* trace) {
    std::vector<double> state(x, x + n);
    std::vector<double> g(m > 0 ? static_cast<std::size_t>(m) : 1U, 0.0);
    if (m > 0) {
        matvec(C, x, g.data(), m, n, parallel);
        for (int i = 0; i < m; ++i) g[i] += d[i];
    }
    int n_iters = 0;
    int winner_rounds = 0;
    int dykstra_iterations = 0;
    int dykstra_events = 0;
    bool dykstra_converged = true;
    int dykstra_cap_hits = 0;

    for (int it = 0; it < proj_cap; ++it) {
        int kind = 0;  // row, lower, upper, nonlinear
        int index = (m > 0) ? 0 : -1;
        double best = (m > 0) ? g[0] * row_scale[0] : -std::numeric_limits<double>::infinity();
        for (int i = 1; i < m; ++i) {
            const double value = g[i] * row_scale[i];
            if (value > best) { best = value; index = i; }
        }
        if (has_lower) {
            for (int i = 0; i < n; ++i) {
                const double value = lower - state[i];
                if (value > best) { best = value; kind = 1; index = i; }
            }
        }
        if (has_upper) {
            for (int i = 0; i < n; ++i) {
                const double value = state[i] - upper;
                if (value > best) { best = value; kind = 2; index = i; }
            }
        }

        std::vector<ExtendedCandidateEval> cache(
            static_cast<std::size_t>(n_candidates));
        for (int q = 0; q < n_candidates; ++q) {
            const ExtendedDescriptor desc = extended_descriptor(candidate_meta, q);
            const double value = extended_candidate_score(
                desc, candidate_meta, n_candidates, coords, candidate_data,
                member_meta, n_members, n, state, cache[static_cast<std::size_t>(q)]);
            if (value > best) { best = value; kind = 3; index = q; }
        }
        if (best <= constraint_tol) break;
        ++winner_rounds;

        if (kind == 0) {
            const double step = g[index] / c_norms_sq[index];
            const double* row = C + static_cast<std::size_t>(index) * n;
            for (int k = 0; k < n; ++k) state[k] -= step * row[k];
            if (m > 0 && G) {
                const double* column = G + static_cast<std::size_t>(index) * m;
                for (int i = 0; i < m; ++i) g[i] -= step * column[i];
            } else if (m > 0) {
                matvec(C, state.data(), g.data(), m, n, parallel);
                for (int i = 0; i < m; ++i) g[i] += d[i];
            }
            const double distance = std::fabs(step) * std::sqrt(c_norms_sq[index]);
            if (observer) observer->record(0, index, m, n, outer_iteration,
                                           n_iters, distance);
            ++n_iters;
        } else if (kind == 1 || kind == 2) {
            const double delta = (kind == 1) ? best : -best;
            state[index] += delta;
            if (m > 0) {
                for (int i = 0; i < m; ++i)
                    g[i] += delta * C[static_cast<std::size_t>(i) * n + index];
            }
            const double distance = std::fabs(delta);
            if (observer) observer->record(kind, index, m, n, outer_iteration,
                                           n_iters, distance);
            ++n_iters;
        } else {
            ExtendedCandidateEval& eval = cache[static_cast<std::size_t>(index)];
            const std::vector<double> old = state;
            state = eval.projected;
            if (m > 0) {
                for (int i = 0; i < m; ++i) {
                    double delta_g = 0.0;
                    for (int k = 0; k < n; ++k)
                        delta_g += C[static_cast<std::size_t>(i) * n + k]
                                 * (state[k] - old[k]);
                    g[i] += delta_g;
                }
            }
            const int candidate_id = m + 2 * n + index;
            if (eval.is_dykstra) {
                const int local_events = static_cast<int>(eval.dykstra.events.size());
                dykstra_iterations += eval.dykstra.iterations;
                dykstra_events += local_events;
                dykstra_converged = dykstra_converged && eval.dykstra.converged;
                dykstra_cap_hits += eval.dykstra.cap_hit ? 1 : 0;
                if (eval.dykstra.cap_hit && !continue_after_budget)
                    *budget_exhausted = true;
                if (local_events > 0) {
                    for (int local = 0; local < local_events; ++local) {
                        const ExtendedDykstraEvent& event =
                            eval.dykstra.events[static_cast<std::size_t>(local)];
                        if (observer)
                            observer->record_nonlinear(candidate_id,
                                outer_iteration, n_iters + local,
                                event.correction_norm, event.member_index);
                    }
                    n_iters += local_events;
                } else {
                    if (observer)
                        observer->record_nonlinear(candidate_id,
                            outer_iteration, n_iters, eval.score, -1);
                    ++n_iters;
                }
            } else {
                if (observer)
                    observer->record_nonlinear(candidate_id,
                        outer_iteration, n_iters, eval.score, -1);
                ++n_iters;
            }
        }
    }

    if (winner_rounds >= proj_cap) {
        if (observer) observer->record_projection_cap_recheck();
        const double maximum = extended_max_violation(
            C, d, row_scale, state.data(), n, m, has_lower, lower,
            has_upper, upper, candidate_meta, n_candidates, coords,
            candidate_data, member_meta, n_members, residual, parallel);
        if (maximum > constraint_tol && !continue_after_budget)
            *budget_exhausted = true;
    }

    if (trace) {
        trace->dykstra_iterations.push_back(dykstra_iterations);
        trace->dykstra_events.push_back(dykstra_events);
        trace->dykstra_converged.push_back(dykstra_events == 0 || dykstra_converged);
        trace->dykstra_cap_hits.push_back(dykstra_cap_hits);
        trace->projection_truncated.push_back(winner_rounds >= proj_cap ? 1 : 0);
    }
    for (int k = 0; k < n; ++k) x[k] = state[k];
    return n_iters;
}

// Native outer Euler loop.  Convergence is intentionally host-side: the
// Python solver owns the nonlinear KKT certificate and drives this function in
// checkpoint-sized chunks using the same tail protocol as the released kernel.
inline Result solve_euler_extended(
        const double* A, const double* b,
        const double* C, const double* d,
        const double* c_norms_sq, const double* row_scale,
        const double* G, int n, int m,
        double k0, double constraint_tol,
        int max_iterations, int proj_cap,
        bool has_lower, double lower, bool has_upper, double upper,
        const std::int64_t* candidate_meta, int n_candidates,
        const std::int64_t* coords, const double* candidate_data,
        const std::int64_t* member_meta, int n_members,
        bool continue_after_budget,
        bool parallel, const double* x0, double* x_out,
        ExtendedObserver* observer, int iter_offset,
        double* obj_tail_out, int* obj_tail_len_out,
        double* x_tail_out, int* x_tail_len_out,
        int window_size, bool use_solution_stable,
        ExtendedTrace* trace) {
    std::vector<double> x(x0, x0 + n), Ax(n, 0.0);
    std::vector<double> residual(m > 0 ? static_cast<std::size_t>(m) : 1U, 0.0);
    const int W = std::max(1, window_size);
    std::vector<double> objective_ring(static_cast<std::size_t>(W), 0.0);
    std::vector<double> x_ring(use_solution_stable
        ? static_cast<std::size_t>(W) * n : 1U, 0.0);
    int obj_head = 0, obj_count = 0, x_head = 0, x_count = 0;
    int n_projections = 0;
    bool budget_exhausted = false;
    int iterations_used = max_iterations;

    apply_hessian(A, nullptr, false, x.data(), Ax.data(), n, parallel);
    for (int it = 0; it < max_iterations; ++it) {
        for (int i = 0; i < n; ++i) x[i] -= k0 * (Ax[i] + b[i]);
        const int outer_events = project_extended(
            x.data(), C, d, c_norms_sq, row_scale, G, n, m,
            constraint_tol, proj_cap, has_lower, lower, has_upper, upper,
            candidate_meta, n_candidates, coords, candidate_data,
            member_meta, n_members, continue_after_budget, residual.data(),
            parallel, &budget_exhausted, iter_offset + it, observer, trace);
        n_projections += outer_events;
        if (budget_exhausted) {
            iterations_used = it + 1;
            break;
        }
        apply_hessian(A, nullptr, false, x.data(), Ax.data(), n, parallel);
        const double objective = 0.5 * dot(x.data(), Ax.data(), n)
                               + dot(b, x.data(), n);
        objective_ring[static_cast<std::size_t>(obj_head)] = objective;
        obj_head = (obj_head + 1) % W;
        ++obj_count;
        if (use_solution_stable) {
            double* slot = x_ring.data() + static_cast<std::size_t>(x_head) * n;
            for (int i = 0; i < n; ++i) slot[i] = x[i];
            x_head = (x_head + 1) % W;
            ++x_count;
        }
    }

    for (int i = 0; i < n; ++i) x_out[i] = x[i];
    if (obj_tail_out) {
        const int count = std::min(obj_count, W);
        for (int i = 0; i < count; ++i) {
            const int j = ((obj_head - count + i) % W + W) % W;
            obj_tail_out[i] = objective_ring[static_cast<std::size_t>(j)];
        }
        if (obj_tail_len_out) *obj_tail_len_out = count;
    }
    if (x_tail_out && use_solution_stable) {
        const int count = std::min(x_count, W);
        for (int s = 0; s < count; ++s) {
            const int j = ((x_head - count + s) % W + W) % W;
            const double* src = x_ring.data() + static_cast<std::size_t>(j) * n;
            double* dst = x_tail_out + static_cast<std::size_t>(s) * n;
            for (int i = 0; i < n; ++i) dst[i] = src[i];
        }
        if (x_tail_len_out) *x_tail_len_out = count;
    } else if (x_tail_len_out) {
        *x_tail_len_out = 0;
    }

    Result result;
    result.iterations_used = iterations_used;
    result.n_projections = n_projections;
    result.converged = false;
    result.reason_code = budget_exhausted ? REASON_PROJECTION_BUDGET
                                          : REASON_MAX_ITERATIONS;
    return result;
}

}  // namespace snn_qp
