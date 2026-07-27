// v0.5-faithful fixed-horizon SNN-QP kernel for the frozen MSRP workload.
//
// The top function accepts the exact binary64 interface bundle used by both
// same-board CPU baselines. It casts once into the fixed datapath, then
// performs gradient-before-projection, one
// normalized-distance winner scan over explicit rows and implicit box facets,
// strict first-maximal tie breaking, exact row/facet residual propagation, and
// a fresh joint-feasibility recheck on projection-cap exhaustion. There is no
// terminal clip.

#include "dt.h"

#include <ap_int.h>
#include <cstddef>

#ifndef MAXN
#define MAXN 64
#endif

#ifndef MAXM
#define MAXM 64
#endif

#ifndef UF
#define UF 4
#endif

#ifndef RG
#define RG 2
#endif

#if UF != 4 || RG != 2
#error "The HLS array-partition factors must track UF=4 and RG=2"
#endif

namespace {

constexpr unsigned long long TELEMETRY_MAGIC =
    0x4d53525056303531ULL;
constexpr unsigned long long DIGEST_OFFSET =
    14695981039346656037ULL;
constexpr unsigned long long NO_CANDIDATE = ~0ULL;

inline ap_uint<64> digest_word(ap_uint<64> hash, ap_uint<64> word) {
#pragma HLS INLINE
    // 1099511628211 = 2^40 + 2^8 + 2^7 + 2^5 + 2^4 + 2^1 + 1.
    // Expressing the constant product as shifts and adds avoids a general
    // 64-bit multiplier while preserving modulo-2^64 FNV semantics exactly.
    const ap_uint<64> value = hash ^ (word + 1);
    return value + (value << 1) + (value << 4) + (value << 5) +
           (value << 7) + (value << 8) + (value << 40);
}

inline dt cast_dt(acc_t value, ap_uint<1>& range_violation) {
#pragma HLS INLINE
    const acc_t magnitude_limit =
        static_cast<acc_t>(1ULL << (DATA_I - 1));
    if (value < -magnitude_limit || value >= magnitude_limit)
        range_violation = 1;
    return static_cast<dt>(value);
}

}  // namespace

extern "C" void snn_qp_v05(
    const double* A_in, const double* b_in, const double* C_in,
    const double* d_in, const double* cns_in, const double* row_scale_in,
    const double* G_in, const double* x0_in, long long* x_raw_out,
    unsigned long long* telemetry_out, int n, int m, double k0_f,
    double ctol_f, int n_iters, int projmax, int has_lower, double lower_f,
    int has_upper, double upper_f) {
#pragma HLS INTERFACE m_axi port = A_in bundle = g0 offset = slave depth = 4096
#pragma HLS INTERFACE m_axi port = C_in bundle = g1 offset = slave depth = 4096
#pragma HLS INTERFACE m_axi port = G_in bundle = g2 offset = slave depth = 4096
#pragma HLS INTERFACE m_axi port = b_in bundle = g3 offset = slave depth = 64
#pragma HLS INTERFACE m_axi port = d_in bundle = g3 offset = slave depth = 64
#pragma HLS INTERFACE m_axi port = cns_in bundle = g3 offset = slave depth = 64
#pragma HLS INTERFACE m_axi port = row_scale_in bundle = g3 offset = slave depth = 64
#pragma HLS INTERFACE m_axi port = x0_in bundle = g3 offset = slave depth = 64
#pragma HLS INTERFACE m_axi port = x_raw_out bundle = g4 offset = slave depth = 64
#pragma HLS INTERFACE m_axi port = telemetry_out bundle = g4 offset = slave depth = 16
#pragma HLS INTERFACE s_axilite port = A_in bundle = c
#pragma HLS INTERFACE s_axilite port = b_in bundle = c
#pragma HLS INTERFACE s_axilite port = C_in bundle = c
#pragma HLS INTERFACE s_axilite port = d_in bundle = c
#pragma HLS INTERFACE s_axilite port = cns_in bundle = c
#pragma HLS INTERFACE s_axilite port = row_scale_in bundle = c
#pragma HLS INTERFACE s_axilite port = G_in bundle = c
#pragma HLS INTERFACE s_axilite port = x0_in bundle = c
#pragma HLS INTERFACE s_axilite port = x_raw_out bundle = c
#pragma HLS INTERFACE s_axilite port = telemetry_out bundle = c
#pragma HLS INTERFACE s_axilite port = n bundle = c
#pragma HLS INTERFACE s_axilite port = m bundle = c
#pragma HLS INTERFACE s_axilite port = k0_f bundle = c
#pragma HLS INTERFACE s_axilite port = ctol_f bundle = c
#pragma HLS INTERFACE s_axilite port = n_iters bundle = c
#pragma HLS INTERFACE s_axilite port = projmax bundle = c
#pragma HLS INTERFACE s_axilite port = has_lower bundle = c
#pragma HLS INTERFACE s_axilite port = lower_f bundle = c
#pragma HLS INTERFACE s_axilite port = has_upper bundle = c
#pragma HLS INTERFACE s_axilite port = upper_f bundle = c
#pragma HLS INTERFACE s_axilite port = return bundle = c

    static dt A[MAXN][MAXN];
    static dt C[MAXM][MAXN];
    static dt G[MAXM][MAXM];
#pragma HLS ARRAY_PARTITION variable = A cyclic factor = 2 dim = 1
#pragma HLS ARRAY_PARTITION variable = A cyclic factor = 4 dim = 2
#pragma HLS ARRAY_PARTITION variable = C cyclic factor = 2 dim = 1
#pragma HLS ARRAY_PARTITION variable = C cyclic factor = 4 dim = 2
#pragma HLS ARRAY_PARTITION variable = G cyclic factor = 4 dim = 2

    dt b[MAXN];
    dt d[MAXM];
    dt cns[MAXM];
    dt row_scale[MAXM];
    dt x[MAXN];
    dt Ax[MAXN];
    dt residual[MAXM];
    using gradient_sum_t = decltype(acc_t() + acc_t());
    using gradient_product_t = decltype(acc_t() * gradient_sum_t());
    gradient_product_t gradient_product[MAXN];
    acc_t gradient_next[MAXN];
#pragma HLS ARRAY_PARTITION variable = x cyclic factor = 4 dim = 1
#pragma HLS ARRAY_PARTITION variable = residual cyclic factor = 4 dim = 1

    const dt k0 = static_cast<dt>(k0_f);
    const dt ctol = static_cast<dt>(ctol_f);
    const dt lower = static_cast<dt>(lower_f);
    const dt upper = static_cast<dt>(upper_f);
    ap_uint<1> input_range_violation = 0;
    ap_uint<1> scalar_range_violation = 0;
    ap_uint<1> gradient_range_violation = 0;
    ap_uint<1> lane_range_violation[UF];
#pragma HLS ARRAY_PARTITION variable = lane_range_violation complete
init_range_flags:
    for (int lane = 0; lane < UF; ++lane)
        lane_range_violation[lane] = 0;
    const double input_limit =
        static_cast<double>(1ULL << (DATA_I - 1));

load_A:
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            {
                const double value =
                    A_in[static_cast<std::size_t>(i) * n + j];
                if (value < -input_limit || value >= input_limit)
                    input_range_violation = 1;
                A[i][j] = static_cast<dt>(value);
            }
load_C:
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            {
                const double value =
                    C_in[static_cast<std::size_t>(i) * n + j];
                if (value < -input_limit || value >= input_limit)
                    input_range_violation = 1;
                C[i][j] = static_cast<dt>(value);
            }
load_G:
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            {
                const double value =
                    G_in[static_cast<std::size_t>(i) * m + j];
                if (value < -input_limit || value >= input_limit)
                    input_range_violation = 1;
                G[i][j] = static_cast<dt>(value);
            }
load_vectors_n:
    for (int i = 0; i < n; ++i) {
        if (b_in[i] < -input_limit || b_in[i] >= input_limit)
            input_range_violation = 1;
        if (x0_in[i] < -input_limit || x0_in[i] >= input_limit)
            input_range_violation = 1;
        b[i] = static_cast<dt>(b_in[i]);
        x[i] = static_cast<dt>(x0_in[i]);
    }
load_vectors_m:
    for (int i = 0; i < m; ++i) {
        if (d_in[i] < -input_limit || d_in[i] >= input_limit)
            input_range_violation = 1;
        if (cns_in[i] < -input_limit || cns_in[i] >= input_limit)
            input_range_violation = 1;
        if (row_scale_in[i] < -input_limit ||
            row_scale_in[i] >= input_limit)
            input_range_violation = 1;
        d[i] = static_cast<dt>(d_in[i]);
        cns[i] = static_cast<dt>(cns_in[i]);
        row_scale[i] = static_cast<dt>(row_scale_in[i]);
    }

    ap_uint<64> event_digest = DIGEST_OFFSET;
    ap_uint<64> total_events = 0;
    ap_uint<64> row_events = 0;
    ap_uint<64> lower_events = 0;
    ap_uint<64> upper_events = 0;
    ap_uint<64> cap_rechecks = 0;
    ap_uint<64> first_candidate = NO_CANDIDATE;
    ap_uint<64> last_candidate = NO_CANDIDATE;
    int status = 0;
    int executed = 0;

outer_loop:
    for (int outer = 0; outer < n_iters; ++outer) {
#pragma HLS LOOP_TRIPCOUNT min = 120000 max = 120000
    hessian_rows:
        for (int i0 = 0; i0 < n; i0 += RG) {
            acc_t acc[RG];
#pragma HLS ARRAY_PARTITION variable = acc complete
            for (int group = 0; group < RG; ++group) acc[group] = 0;
        hessian_columns:
            for (int j0 = 0; j0 < n; j0 += UF) {
#pragma HLS PIPELINE II = 1
                for (int group = 0; group < RG; ++group) {
                    acc_t partial = 0;
                    for (int lane = 0; lane < UF; ++lane) {
                        const int row = i0 + group;
                        const int column = j0 + lane;
                        if (row < n && column < n)
                            partial += A[row][column] * x[column];
                    }
                    acc[group] += partial;
                }
            }
            for (int group = 0; group < RG; ++group) {
                if (i0 + group < n)
                    Ax[i0 + group] =
                        cast_dt(acc[group],
                                lane_range_violation[group]);
            }
        }

    gradient_product_step:
        for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II = 1
            gradient_product[i] =
                static_cast<acc_t>(k0) *
                (static_cast<acc_t>(Ax[i]) + static_cast<acc_t>(b[i]));
        }
    gradient_apply_step:
        for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II = 1
            gradient_next[i] = static_cast<acc_t>(
                static_cast<acc_t>(x[i]) - gradient_product[i]);
        }
    gradient_commit_step:
        for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II = 1
            x[i] =
                cast_dt(gradient_next[i], gradient_range_violation);
        }

    residual_rows:
        for (int i0 = 0; i0 < m; i0 += RG) {
            acc_t acc[RG];
#pragma HLS ARRAY_PARTITION variable = acc complete
            for (int group = 0; group < RG; ++group) acc[group] = 0;
        residual_columns:
            for (int j0 = 0; j0 < n; j0 += UF) {
#pragma HLS PIPELINE II = 1
                for (int group = 0; group < RG; ++group) {
                    acc_t partial = 0;
                    for (int lane = 0; lane < UF; ++lane) {
                        const int row = i0 + group;
                        const int column = j0 + lane;
                        if (row < m && column < n)
                            partial += C[row][column] * x[column];
                    }
                    acc[group] += partial;
                }
            }
            for (int group = 0; group < RG; ++group) {
                if (i0 + group < m)
                    residual[i0 + group] =
                        cast_dt(
                            acc[group] +
                                static_cast<acc_t>(d[i0 + group]),
                            lane_range_violation[group]);
            }
        }

        bool left_sweep = false;
    projection_loop:
        for (int ordinal = 0; ordinal < projmax; ++ordinal) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 10000
            int winner = 0;
            int kind = 0;
            acc_t best = static_cast<acc_t>(residual[0]) *
                         static_cast<acc_t>(row_scale[0]);
        scan_rows:
            for (int i = 1; i < m; ++i) {
#pragma HLS PIPELINE II = 1
                const acc_t score = static_cast<acc_t>(residual[i]) *
                                    static_cast<acc_t>(row_scale[i]);
                if (score > best) {
                    best = score;
                    winner = i;
                }
            }
        scan_lower:
            for (int i = 0; i < MAXN; ++i) {
#pragma HLS PIPELINE II = 1
                if (has_lower && i < n) {
                    const acc_t score = static_cast<acc_t>(lower) -
                                        static_cast<acc_t>(x[i]);
                    if (score > best) {
                        best = score;
                        winner = i;
                        kind = 1;
                    }
                }
            }
        scan_upper:
            for (int i = 0; i < MAXN; ++i) {
#pragma HLS PIPELINE II = 1
                if (has_upper && i < n) {
                    const acc_t score = static_cast<acc_t>(x[i]) -
                                        static_cast<acc_t>(upper);
                    if (score > best) {
                        best = score;
                        winner = i;
                        kind = 2;
                    }
                }
            }
            if (best <= static_cast<acc_t>(ctol)) {
                left_sweep = true;
                break;
            }

            ap_uint<64> candidate = 0;
            if (kind == 0) {
                const dt step = residual[winner] / cns[winner];
            update_x_row:
                for (int j0 = 0; j0 < n; j0 += UF) {
#pragma HLS PIPELINE II = 1
                    for (int lane = 0; lane < UF; ++lane) {
                        const int j = j0 + lane;
                        if (j < n) {
                            const acc_t next =
                                static_cast<acc_t>(x[j]) -
                                static_cast<acc_t>(step) *
                                    static_cast<acc_t>(C[winner][j]);
                            x[j] =
                                cast_dt(next, lane_range_violation[lane]);
                        }
                    }
                }
            update_residual_row:
                for (int i0 = 0; i0 < m; i0 += UF) {
#pragma HLS PIPELINE II = 1
                    for (int lane = 0; lane < UF; ++lane) {
                        const int i = i0 + lane;
                        if (i < m) {
                            const acc_t next =
                                static_cast<acc_t>(residual[i]) -
                                static_cast<acc_t>(step) *
                                    static_cast<acc_t>(G[winner][i]);
                            residual[i] =
                                cast_dt(next,
                                        lane_range_violation[lane]);
                        }
                    }
                }
                ++row_events;
                candidate = static_cast<ap_uint<64>>(winner);
            } else {
                const dt delta =
                    kind == 1
                        ? cast_dt(best, scalar_range_violation)
                        : cast_dt(-best, scalar_range_violation);
                x[winner] =
                    cast_dt(static_cast<acc_t>(x[winner]) +
                                static_cast<acc_t>(delta),
                            scalar_range_violation);
            update_residual_facet:
                for (int i0 = 0; i0 < m; i0 += UF) {
#pragma HLS PIPELINE II = 1
                    for (int lane = 0; lane < UF; ++lane) {
                        const int i = i0 + lane;
                        if (i < m) {
                            const acc_t next =
                                static_cast<acc_t>(residual[i]) +
                                static_cast<acc_t>(delta) *
                                    static_cast<acc_t>(C[i][winner]);
                            residual[i] =
                                cast_dt(next,
                                        lane_range_violation[lane]);
                        }
                    }
                }
                if (kind == 1) {
                    ++lower_events;
                    candidate =
                        static_cast<ap_uint<64>>(m + winner);
                } else {
                    ++upper_events;
                    candidate =
                        static_cast<ap_uint<64>>(m + n + winner);
                }
            }

            if (first_candidate == static_cast<ap_uint<64>>(NO_CANDIDATE))
                first_candidate = candidate;
            last_candidate = candidate;
            event_digest =
                digest_word(event_digest, static_cast<ap_uint<64>>(outer));
            event_digest =
                digest_word(event_digest, static_cast<ap_uint<64>>(ordinal));
            event_digest = digest_word(event_digest, candidate);
            ++total_events;
        }

        if (!left_sweep) {
            ++cap_rechecks;
        cap_residual_rows:
            for (int i0 = 0; i0 < m; i0 += RG) {
                acc_t acc[RG];
#pragma HLS ARRAY_PARTITION variable = acc complete
                for (int group = 0; group < RG; ++group) acc[group] = 0;
            cap_residual_columns:
                for (int j0 = 0; j0 < n; j0 += UF) {
#pragma HLS PIPELINE II = 1
                    for (int group = 0; group < RG; ++group) {
                        acc_t partial = 0;
                        for (int lane = 0; lane < UF; ++lane) {
                            const int row = i0 + group;
                            const int column = j0 + lane;
                            if (row < m && column < n)
                                partial += C[row][column] * x[column];
                        }
                        acc[group] += partial;
                    }
                }
                for (int group = 0; group < RG; ++group) {
                    if (i0 + group < m)
                        residual[i0 + group] =
                            cast_dt(acc[group] +
                                        static_cast<acc_t>(d[i0 + group]),
                                    lane_range_violation[group]);
                }
            }

            acc_t maximum = 0;
        cap_scan_rows:
            for (int i = 0; i < m; ++i) {
#pragma HLS PIPELINE II = 1
                const acc_t score = static_cast<acc_t>(residual[i]) *
                                    static_cast<acc_t>(row_scale[i]);
                if (score > maximum) maximum = score;
            }
        cap_scan_lower:
            for (int i = 0; i < MAXN; ++i) {
#pragma HLS PIPELINE II = 1
                if (has_lower && i < n) {
                    const acc_t score = static_cast<acc_t>(lower) -
                                        static_cast<acc_t>(x[i]);
                    if (score > maximum) maximum = score;
                }
            }
        cap_scan_upper:
            for (int i = 0; i < MAXN; ++i) {
#pragma HLS PIPELINE II = 1
                if (has_upper && i < n) {
                    const acc_t score = static_cast<acc_t>(x[i]) -
                                        static_cast<acc_t>(upper);
                    if (score > maximum) maximum = score;
                }
            }
            if (maximum > static_cast<acc_t>(ctol)) {
                status = 2;
                executed = outer + 1;
                break;
            }
        }
        executed = outer + 1;
    }

write_state:
    for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II = 1
        const ap_int<DATA_W> raw = x[i].range(DATA_W - 1, 0);
        const ap_int<64> extended = raw;
        x_raw_out[i] = static_cast<long long>(extended);
    }

    telemetry_out[0] = TELEMETRY_MAGIC;
    telemetry_out[1] = static_cast<unsigned long long>(status);
    telemetry_out[2] = static_cast<unsigned long long>(n_iters);
    telemetry_out[3] = static_cast<unsigned long long>(executed);
    telemetry_out[4] =
        status == 0 && executed == n_iters ? 1ULL : 0ULL;
    telemetry_out[5] = static_cast<unsigned long long>(total_events);
    telemetry_out[6] = static_cast<unsigned long long>(row_events);
    telemetry_out[7] = static_cast<unsigned long long>(lower_events);
    telemetry_out[8] = static_cast<unsigned long long>(upper_events);
    telemetry_out[9] = static_cast<unsigned long long>(cap_rechecks);
    telemetry_out[10] = static_cast<unsigned long long>(event_digest);
    telemetry_out[11] = static_cast<unsigned long long>(first_candidate);
    telemetry_out[12] = static_cast<unsigned long long>(last_candidate);
    telemetry_out[13] = static_cast<unsigned long long>(DATA_W);
    telemetry_out[14] = static_cast<unsigned long long>(DATA_I);
    ap_uint<1> any_range_violation =
        input_range_violation | scalar_range_violation |
        gradient_range_violation;
reduce_range_flags:
    for (int lane = 0; lane < UF; ++lane)
        any_range_violation |= lane_range_violation[lane];
    telemetry_out[15] =
        static_cast<unsigned long long>(any_range_violation);
}
