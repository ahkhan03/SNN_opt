// SNN-QP v09b resident command-service kernel.
//
// Semantics, command ABI, mailbox, telemetry words, fixed-point formats, and
// the gradient / scan / event / digest operation order are the v0.8 kernel.
// The resident store is the WINNER row-interleave BRAM image: static 2D
// arrays, RG=4 by UF=8, no URAM and no packed 256-bit words. The cap is
// n,m <= 128 and the only route is FULL. CG and STREAM stay in the ABI and
// are rejected.
//
// Matvec issue width stays RG=4 by UF=8, so one column step still holds 32
// products. The accumulation contract is v0.8 mac_four, four products wide.
// A product of two dt values has 48 fraction bits. part += product rounds
// that product into acc_t (32 fraction bits, AP_RND_CONV) against the bits
// already in part. An eight-long chain rounds columns 4..7 against a partial
// that already holds columns 0..3, which is not the v0.6 / v0.8 result.
// mac_four starts each partial at zero and folds four products in ascending
// column order. The boundaries are absolute columns 0, 4, 8, and so on.
// Each eight-column step is two of those partials, part_lo over j+0..j+3 and
// part_hi over j+4..j+7, then acc += part_lo and acc += part_hi. Two acc_t
// adds match mac_four when a partial saturates. One add of (part_lo+part_hi)
// matches only while that sum does not saturate, so this kernel uses the two
// adds. A tree over the four products would not match either. Event bodies
// are single products and keep the v0.8 casts. The coefficient read is the
// 2D array, not a packed word.

#include "dt.h"
#include "v09_abi.hpp"

#include <ap_int.h>
#include <ap_utils.h>
#include <cstddef>
#include <cstdint>

#ifndef MAXN
#define MAXN 128
#endif

#ifndef MAXM
#define MAXM 128
#endif

#ifndef UF
#define UF 8
#endif

#ifndef RG
#define RG 4
#endif

#ifndef POLL_CYCLES
#define POLL_CYCLES 256
#endif

// The fused scan width. The factor below is the literal 8: this HLS version
// does not expand a macro inside an ARRAY_PARTITION factor.
#ifndef SCAN_LANES
#define SCAN_LANES 8
#endif

#if MAXN != 128 || MAXM != 128
#error "v09b is capped at n,m <= 128"
#endif

#if DATA_W != 32 || DATA_I != 8
#error "v09b ABI requires the v05 ap_fixed<32,8> state"
#endif

#if UF != 8 || RG != 4
#error "The v09b arithmetic shape is fixed at UF=8 and RG=4"
#endif

#if SCAN_LANES != 8
#error "The fused scan is built for SCAN_LANES=8"
#endif

#if UF != SCAN_LANES
#error "Event updates step by UF and the scan steps by SCAN_LANES"
#endif

#if (MAXN % SCAN_LANES) != 0 || (MAXM % SCAN_LANES) != 0
#error "Scan groups must stay inside x and residual"
#endif

#if (MAXN % RG) != 0 || (MAXM % RG) != 0
#error "Row groups must stay inside the resident matrices"
#endif

namespace {

constexpr unsigned long long DIGEST_OFFSET = 14695981039346656037ULL;
constexpr unsigned long long NO_CANDIDATE = ~0ULL;

// WINNER partitioning, literals not macros. A and C are read RG rows by UF
// columns per cycle. G and Ct are read one row by UF columns (event update).
// Ct is row-major by facet: Ct[j][i] = C[i][j].
static dt resident_A[MAXN][MAXN];

static dt resident_C[MAXM][MAXN];

static dt resident_Ct[MAXN][MAXM];

static dt resident_G[MAXM][MAXM];

static dt resident_cns[MAXM];
static dt resident_scale[MAXM];
static dt resident_state[MAXN];

static int configured_n = 0;
static int configured_m = 0;
// AUTO until a configure succeeds, exactly as v0.6/v0.8: the route word of
// a pre-configure or rejected-command acknowledgement reports AUTO. An
// in-cap configure selects FULL (the only route this kernel implements).
static int configured_route = snn_v08::AUTO;
static int configured_iters = 0;
static int configured_projmax = 0;
static int configured_has_lower = 0;
static int configured_has_upper = 0;
static dt configured_k0 = 0;
static dt configured_ctol = 0;
static dt configured_lower = 0;
static dt configured_upper = 0;
static ap_uint<1> is_configured = 0;
static ap_uint<1> is_stopped = 0;
static ap_uint<1> geometry_range_violation = 0;
static ap_uint<1> initial_range_violation = 0;
static ap_uint<1> state_has_committed = 0;
static ap_uint<1> a_range_violation = 0;
static ap_uint<1> c_range_violation = 0;
static ap_uint<1> g_range_violation = 0;
static ap_uint<1> cns_range_violation = 0;
static ap_uint<1> scale_range_violation = 0;

inline ap_uint<64> digest_word(ap_uint<64> hash, ap_uint<64> word) {
#pragma HLS INLINE
    const ap_uint<64> value = hash ^ (word + 1);
    // 1099511628211 = 2^40 + 2^8 + 2^7 + 2^5 + 2^4 + 2^1 + 1.
    return value + (value << 1) + (value << 4) + (value << 5) +
           (value << 7) + (value << 8) + (value << 40);
}

inline dt cast_dt(acc_t value, ap_uint<1>& range_violation) {
#pragma HLS INLINE
    const acc_t magnitude_limit = static_cast<acc_t>(1ULL << (DATA_I - 1));
    if (value < -magnitude_limit || value >= magnitude_limit)
        range_violation = 1;
    return static_cast<dt>(value);
}

inline std::uint32_t dt_to_raw(dt value) {
#pragma HLS INLINE
    const ap_uint<DATA_W> bits = value.range(DATA_W - 1, 0);
    return static_cast<std::uint32_t>(bits);
}

inline void write_image(std::uint32_t* image, std::uint32_t index, dt value) {
#pragma HLS INLINE
    if (image != nullptr) image[index] = dt_to_raw(value);
}

inline bool dimensions_valid(int n, int m) {
#pragma HLS INLINE
    return n >= 1 && m >= 1 && n <= MAXN && m <= MAXM;
}

// AUTO and FORCE_FULL select FULL inside the cap. FORCE_CG and FORCE_STREAM
// are removed routes: the caller maps -1 to ERR_CAPACITY, which stays
// distinct from ERR_BAD_DIMENSIONS for an over-cap geometry.
inline int choose_route(int n, int m, int requested) {
#pragma HLS INLINE
    if (requested == snn_v08::FORCE_CG ||
        requested == snn_v08::FORCE_STREAM)
        return -1;
    if (requested != snn_v08::AUTO && requested != snn_v08::FORCE_FULL)
        return -1;
    if (!dimensions_valid(n, m)) return -1;
    return snn_v08::FORCE_FULL;
}

inline void clear_outputs(long long* x_raw_out,
                          unsigned long long* telemetry_out, int n) {
#pragma HLS INLINE
    if (x_raw_out != nullptr) {
        const int limit = n < MAXN ? n : MAXN;
        for (int i = 0; i < limit; ++i) x_raw_out[i] = 0;
    }
    if (telemetry_out != nullptr) {
        for (int i = 0; i < static_cast<int>(snn_v08::TELEMETRY_WORDS); ++i)
            telemetry_out[i] = 0;
    }
}

inline void write_mailbox(volatile std::uint32_t* mb_out,
                          std::uint32_t sequence, std::uint32_t error,
                          std::uint32_t status, std::uint32_t route,
                          std::uint32_t iterations) {
#pragma HLS INLINE
    if (mb_out == nullptr) return;
    // Commit words first and the sequence word last.  The host uses sequence
    // as the acquire point for the preceding result fields.
    mb_out[snn_v08::OUT_ERROR_CODE] = error;
    mb_out[snn_v08::OUT_STATUS] = status;
    mb_out[snn_v08::OUT_SELECTED_ROUTE] = route;
    mb_out[snn_v08::OUT_ITERATIONS] = iterations;
    mb_out[snn_v08::OUT_DONE_SEQUENCE] = sequence;
}

inline void write_litmus_pattern(
    long long* x_raw_out, unsigned long long* telemetry_out,
    volatile const std::uint32_t* mb_in, int n) {
#pragma HLS INLINE off
    if (x_raw_out != nullptr) {
        for (int i = 0; i < n && i < MAXN; ++i)
            x_raw_out[i] = static_cast<long long>(
                UINT64_C(0x13579bdf00000000) |
                static_cast<unsigned long long>(i));
    }
    if (telemetry_out != nullptr) {
        telemetry_out[0] = mb_in == nullptr
                               ? UINT64_C(0xa5a5000000000000)
                               : static_cast<unsigned long long>(
                                     mb_in[snn_v08::MAILBOX_LITMUS_INPUT]);
        for (std::size_t i = 1; i < snn_v08::TELEMETRY_WORDS; ++i)
            telemetry_out[i] = UINT64_C(0xa5a5000000000000) | i;
    }
}

inline void write_geometry_range_flag() {
#pragma HLS INLINE
    geometry_range_violation = a_range_violation | c_range_violation |
                               g_range_violation | cns_range_violation |
                               scale_range_violation;
}

constexpr int GEOMETRY_A = 0;
constexpr int GEOMETRY_C = 1;
constexpr int GEOMETRY_G = 2;

inline dt binary64_to_dt(double value, double input_limit,
                         ap_uint<1>& range_violation) {
#pragma HLS INLINE off
    if (value < -input_limit || value >= input_limit)
        range_violation = 1;
    return static_cast<dt>(value);
}

// Load A, C, or G from the binary64 configure image into the resident BRAM
// and the fixed-point DDR image. C also fills Ct[j][i] = C[i][j] in the
// same pass. One shared body, matching the v0.8 allocation of the converter.
inline void load_geometry_matrix(
    const double* source, std::uint32_t* image,
    std::uint32_t* transpose_image, int matrix_selector, int transpose,
    int rows, int columns, ap_uint<1>& range_violation) {
#pragma HLS INLINE off
    const double input_limit = static_cast<double>(1ULL << (DATA_I - 1));
load_geometry_rows:
    for (int i = 0; i < rows; ++i) {
        const std::uint32_t row_base =
            static_cast<std::uint32_t>(i) * static_cast<std::uint32_t>(columns);
    load_geometry_columns:
        for (int j = 0; j < columns; ++j) {
            const double value =
                source == nullptr ? 0.0 : source[row_base + static_cast<std::uint32_t>(j)];
            const dt fixed = binary64_to_dt(value, input_limit, range_violation);
            write_image(image, row_base + static_cast<std::uint32_t>(j), fixed);
            if (matrix_selector == GEOMETRY_A) {
                resident_A[i][j] = fixed;
            } else if (matrix_selector == GEOMETRY_C) {
                resident_C[i][j] = fixed;
                if (transpose) {
                    resident_Ct[j][i] = fixed;
                    const std::uint32_t transposed_index =
                        static_cast<std::uint32_t>(j) *
                            static_cast<std::uint32_t>(rows) +
                        static_cast<std::uint32_t>(i);
                    write_image(transpose_image, transposed_index, fixed);
                }
            } else {
                resident_G[i][j] = fixed;
            }
        }
    }
}

inline void load_cns_scale(const double* cns_source,
                           const double* scale_source, int rows,
                           ap_uint<1>& cns_range,
                           ap_uint<1>& scale_range) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION function instances = binary64_to_dt limit = 1
    const double input_limit = static_cast<double>(1ULL << (DATA_I - 1));
load_cns_scale_shared:
    for (int i = 0; i < rows; ++i) {
        const double cns_value = cns_source == nullptr ? 0.0 : cns_source[i];
        const double scale_value =
            scale_source == nullptr ? 0.0 : scale_source[i];
        resident_cns[i] = binary64_to_dt(cns_value, input_limit, cns_range);
        resident_scale[i] =
            binary64_to_dt(scale_value, input_limit, scale_range);
    }
}

inline int configure_geometry(
    const double* cns_cfg, const double* row_scale_cfg,
    const double* x0_in, int n, int m, double k0_f, double ctol_f,
    int n_iters, int projmax, int has_lower, double lower_f, int has_upper,
    double upper_f, int requested_route) {
#pragma HLS INLINE off
    if (!dimensions_valid(n, m) || n_iters <= 0 || projmax <= 0 ||
        (has_lower != 0 && has_lower != 1) ||
        (has_upper != 0 && has_upper != 1))
        return snn_v08::ERR_BAD_DIMENSIONS;

    const int route = choose_route(n, m, requested_route);
    if (route < 0) return snn_v08::ERR_CAPACITY;

    // Validation is complete before the current configuration changes. A
    // rejected FORCE_CG or FORCE_STREAM leaves the previous one intact.
    configured_n = n;
    configured_m = m;
    configured_route = route;
    configured_iters = n_iters;
    configured_projmax = projmax;
    configured_has_lower = has_lower;
    configured_has_upper = has_upper;
    configured_k0 = static_cast<dt>(k0_f);
    configured_ctol = static_cast<dt>(ctol_f);
    configured_lower = static_cast<dt>(lower_f);
    configured_upper = static_cast<dt>(upper_f);

    a_range_violation = 0;
    c_range_violation = 0;
    g_range_violation = 0;
    cns_range_violation = 0;
    scale_range_violation = 0;
    initial_range_violation = 0;
    load_cns_scale(cns_cfg, row_scale_cfg, m, cns_range_violation,
                   scale_range_violation);

    const double input_limit = static_cast<double>(1ULL << (DATA_I - 1));
initialize_state:
    for (int i = 0; i < n; ++i) {
        const double value = x0_in == nullptr ? 0.0 : x0_in[i];
        if (x0_in != nullptr &&
            (value < -input_limit || value >= input_limit))
            initial_range_violation = 1;
        resident_state[i] = static_cast<dt>(value);
    }
    for (int i = n; i < MAXN; ++i) resident_state[i] = static_cast<dt>(0);

    is_configured = 1;
    is_stopped = 0;
    state_has_committed = 0;
    return snn_v08::ERR_OK;
}

inline int refresh_a(int n, int m) {
#pragma HLS INLINE off
    if (!is_configured) return snn_v08::ERR_NOT_CONFIGURED;
    if (n != configured_n || m != configured_m || !dimensions_valid(n, m))
        return snn_v08::ERR_BAD_DIMENSIONS;
    a_range_violation = 0;
    return snn_v08::ERR_OK;
}

inline void apply_shift(dt* x, dt* shifted, int n, int stride, int tail) {
#pragma HLS INLINE off
    if (stride == 0) {
        for (int i = 0; i < n; ++i) shifted[i] = x[i];
    } else {
    shift_body:
        for (int i = 0; i < n; ++i) {
            if (i < n - stride) {
                shifted[i] = x[i + stride];
            } else if (tail == snn_v08::REPEAT_LAST) {
                shifted[i] = x[n - 1];
            } else {
                shifted[i] = x[i];
            }
        }
    }
    for (int i = 0; i < n; ++i) x[i] = shifted[i];
}

inline void write_telemetry(long long* x_raw_out,
                            unsigned long long* telemetry_out, dt* x,
                            int n, int status, int executed,
                            int n_iters, ap_uint<64> total_events,
                            ap_uint<64> row_events,
                            ap_uint<64> lower_events,
                            ap_uint<64> upper_events,
                            ap_uint<64> cap_rechecks,
                            ap_uint<64> event_digest,
                            ap_uint<64> first_candidate,
                            ap_uint<64> last_candidate,
                            ap_uint<1> any_range_violation) {
#pragma HLS INLINE off
    if (x_raw_out != nullptr) {
    write_state:
        for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II = 1
            const ap_int<DATA_W> raw = x[i].range(DATA_W - 1, 0);
            const ap_int<64> extended = raw;
            x_raw_out[i] = static_cast<long long>(extended);
        }
    }
    if (telemetry_out == nullptr) return;
    telemetry_out[0] = snn_v08::TELEMETRY_MAGIC;
    telemetry_out[1] = static_cast<unsigned long long>(status);
    telemetry_out[2] = static_cast<unsigned long long>(n_iters);
    telemetry_out[3] = static_cast<unsigned long long>(executed);
    telemetry_out[4] = status == 0 && executed == n_iters ? 1ULL : 0ULL;
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
    telemetry_out[15] = static_cast<unsigned long long>(any_range_violation);
}

struct CandidateWinner {
    acc_t score;
    int index;
    int kind;
    ap_uint<1> valid;
};

inline CandidateWinner prefer_earlier(CandidateWinner earlier,
                                      CandidateWinner later) {
#pragma HLS INLINE
    if (!later.valid) return earlier;
    if (!earlier.valid) return later;
    if (later.score > earlier.score) return later;
    return earlier;
}

inline CandidateWinner reduce_lanes(const CandidateWinner lane[SCAN_LANES]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = lane complete
    const CandidateWinner w01 = prefer_earlier(lane[0], lane[1]);
    const CandidateWinner w23 = prefer_earlier(lane[2], lane[3]);
    const CandidateWinner w45 = prefer_earlier(lane[4], lane[5]);
    const CandidateWinner w67 = prefer_earlier(lane[6], lane[7]);
    const CandidateWinner w03 = prefer_earlier(w01, w23);
    const CandidateWinner w47 = prefer_earlier(w45, w67);
    return prefer_earlier(w03, w47);
}

inline void consider_group(CandidateWinner& best, CandidateWinner group,
                           int kind) {
#pragma HLS INLINE
    if (!group.valid) return;
    if (!best.valid || group.score > best.score) {
        group.kind = kind;
        best = group;
    }
}

// First maximum over rows (residual * scale), then lower facets
// (lower - x), then upper facets (x - upper). One pipelined pass steps
// SCAN_LANES coordinates. Families stay separate during the pass and combine
// at the end, so a later lower candidate still beats an earlier upper
// candidate on a tie. Within a family the tree plus a strict-greater update
// is the scalar first-maximum rule.
inline CandidateWinner scan_candidates(const dt x[MAXN],
                                      const dt residual[MAXM], dt lower_bound,
                                      dt upper_bound, int n, int m,
                                      int has_lower, int has_upper) {
#pragma HLS INLINE
    CandidateWinner best_row;
    best_row.score = 0;
    best_row.index = 0;
    best_row.kind = 0;
    best_row.valid = 0;
    CandidateWinner best_lower = best_row;
    CandidateWinner best_upper = best_row;

    const int bound_span = (has_lower || has_upper) ? n : 0;
    const int span = bound_span > m ? bound_span : m;

scan_fused:
    for (int base = 0; base < span; base += SCAN_LANES) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 16
        CandidateWinner row_lane[SCAN_LANES];
        CandidateWinner lower_lane[SCAN_LANES];
        CandidateWinner upper_lane[SCAN_LANES];
#pragma HLS ARRAY_PARTITION variable = row_lane complete
#pragma HLS ARRAY_PARTITION variable = lower_lane complete
#pragma HLS ARRAY_PARTITION variable = upper_lane complete
        for (int k = 0; k < SCAN_LANES; ++k) {
#pragma HLS UNROLL
            const int local = base + k;
            const ap_uint<1> row_ok = local < m;
            const ap_uint<1> col_ok = local < n;
            const acc_t row_score = static_cast<acc_t>(residual[local]) *
                                    static_cast<acc_t>(resident_scale[local]);
            const acc_t lower_score = static_cast<acc_t>(lower_bound) -
                                      static_cast<acc_t>(x[local]);
            const acc_t upper_score = static_cast<acc_t>(x[local]) -
                                      static_cast<acc_t>(upper_bound);
            row_lane[k].score = row_ok ? row_score : static_cast<acc_t>(0);
            row_lane[k].index = local;
            row_lane[k].kind = 0;
            row_lane[k].valid = row_ok;
            lower_lane[k].score = col_ok ? lower_score : static_cast<acc_t>(0);
            lower_lane[k].index = local;
            lower_lane[k].kind = 1;
            lower_lane[k].valid = col_ok;
            upper_lane[k].score = col_ok ? upper_score : static_cast<acc_t>(0);
            upper_lane[k].index = local;
            upper_lane[k].kind = 2;
            upper_lane[k].valid = col_ok;
        }
        if (base < m) consider_group(best_row, reduce_lanes(row_lane), 0);
        if (has_lower && base < n)
            consider_group(best_lower, reduce_lanes(lower_lane), 1);
        if (has_upper && base < n)
            consider_group(best_upper, reduce_lanes(upper_lane), 2);
    }
    return prefer_earlier(best_row, prefer_earlier(best_lower, best_upper));
}

// Hessian y = A x. One pipeline step still covers UF=8 columns and RG=4
// rows (32 products). Each row folds those columns as two mac_four partials
// from zero, low columns first, then adds each partial into acc. Edge lanes
// are masked the way the WINNER amac loop masks them. The store is cast_dt,
// the v0.8 saturating store, not a bare dt cast of a partial sum.
inline void hessian_matvec(dt Ax[MAXN], const dt x[MAXN], int n,
                           ap_uint<1> lane_range_violation[SCAN_LANES]) {
#pragma HLS INLINE
hessian_rows:
    for (int i0 = 0; i0 < n; i0 += RG) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
        acc_t acc[RG];
#pragma HLS ARRAY_PARTITION variable = acc complete
        for (int g = 0; g < RG; ++g) acc[g] = 0;
    hessian_columns:
        for (int j = 0; j < n; j += UF) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 16
            for (int g = 0; g < RG; ++g) {
#pragma HLS UNROLL
                const int ii = i0 + g;
                // Low half: absolute columns j+0..j+3. Fresh partial.
                acc_t part_lo = 0;
                for (int u = 0; u < 4; ++u) {
#pragma HLS UNROLL
                    const int jj = j + u;
                    if (jj < n && ii < n)
                        part_lo += resident_A[ii][jj] * x[jj];
                }
                // High half: absolute columns j+4..j+7. Fresh partial, so
                // these four products do not round against part_lo.
                acc_t part_hi = 0;
                for (int u = 0; u < 4; ++u) {
#pragma HLS UNROLL
                    const int jj = j + 4 + u;
                    if (jj < n && ii < n)
                        part_hi += resident_A[ii][jj] * x[jj];
                }
                // Separate adds. acc += (part_lo + part_hi) clips when the
                // sum of the two partials saturates acc_t and mac_four does
                // not, or the other way around.
                acc[g] += part_lo;
                acc[g] += part_hi;
            }
        }
#pragma HLS UNROLL
        for (int g = 0; g < RG; ++g) {
            if (i0 + g < n)
                Ax[i0 + g] = cast_dt(acc[g], lane_range_violation[g]);
        }
    }
}

// Residual r = C x + d. Same two mac_four partials as the Hessian. d is
// added in acc_t before the saturating cast, which is the v0.8 order. A cast
// of the accumulator to dt followed by a dt add of d would not match.
inline void residual_matvec(dt residual[MAXM], const dt x[MAXN], const dt d[MAXM],
                            int m, int n,
                            ap_uint<1> lane_range_violation[SCAN_LANES]) {
#pragma HLS INLINE off
residual_rows:
    for (int i0 = 0; i0 < m; i0 += RG) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
        acc_t acc[RG];
#pragma HLS ARRAY_PARTITION variable = acc complete
        for (int g = 0; g < RG; ++g) acc[g] = 0;
    residual_columns:
        for (int j = 0; j < n; j += UF) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 16
            for (int g = 0; g < RG; ++g) {
#pragma HLS UNROLL
                const int ii = i0 + g;
                acc_t part_lo = 0;
                for (int u = 0; u < 4; ++u) {
#pragma HLS UNROLL
                    const int jj = j + u;
                    if (jj < n && ii < m)
                        part_lo += resident_C[ii][jj] * x[jj];
                }
                acc_t part_hi = 0;
                for (int u = 0; u < 4; ++u) {
#pragma HLS UNROLL
                    const int jj = j + 4 + u;
                    if (jj < n && ii < m)
                        part_hi += resident_C[ii][jj] * x[jj];
                }
                acc[g] += part_lo;
                acc[g] += part_hi;
            }
        }
#pragma HLS UNROLL
        for (int g = 0; g < RG; ++g) {
            if (i0 + g < m)
                residual[i0 + g] = cast_dt(
                    acc[g] + static_cast<acc_t>(d[i0 + g]),
                    lane_range_violation[g]);
        }
    }
}

inline void or_lane_flags(ap_uint<1> lane_range_violation[SCAN_LANES],
                          const ap_uint<1> lane_flag[SCAN_LANES]) {
#pragma HLS INLINE
    for (int lane = 0; lane < SCAN_LANES; ++lane) {
#pragma HLS UNROLL
        lane_range_violation[lane] = lane_range_violation[lane] | lane_flag[lane];
    }
}

inline int solve_current(const double* b_in, const double* d_in,
                         const double* x0_in, long long* x_raw_out,
                         unsigned long long* telemetry_out, int start_mode,
                         int shift_stride, int tail_policy) {
#pragma HLS INLINE off
    if (!is_configured) return snn_v08::ERR_NOT_CONFIGURED;
    if (start_mode < snn_v08::RESIDENT_WARM ||
        start_mode > snn_v08::COLD_ZERO || shift_stride < 0 ||
        shift_stride > configured_n ||
        (tail_policy != snn_v08::HOLD_TAIL &&
         tail_policy != snn_v08::REPEAT_LAST))
        return snn_v08::ERR_BAD_DIMENSIONS;

    dt b[MAXN];
    dt d[MAXM];
    dt x[MAXN];
    dt shifted[MAXN];
    dt Ax[MAXN];
    dt residual[MAXM];
    using gradient_sum_t = decltype(acc_t() + acc_t());
    using gradient_product_t = decltype(acc_t() * gradient_sum_t());
#pragma HLS ARRAY_PARTITION variable = b cyclic factor = 4 dim = 1
#pragma HLS ARRAY_PARTITION variable = d cyclic factor = 4 dim = 1
#pragma HLS ARRAY_PARTITION variable = x cyclic factor = 8 dim = 1
#pragma HLS ARRAY_PARTITION variable = residual cyclic factor = 8 dim = 1
#pragma HLS ARRAY_PARTITION variable = Ax cyclic factor = 4 dim = 1
#ifndef __SYNTHESIS__
    // Padding lanes of a scan group read past n or m. The winner mask
    // discards those lanes. Zeroing keeps the native read defined.
    for (int i = 0; i < MAXN; ++i) x[i] = static_cast<dt>(0);
    for (int i = 0; i < MAXM; ++i) residual[i] = static_cast<dt>(0);
#endif

    ap_uint<1> input_range_violation = geometry_range_violation;
    if (start_mode == snn_v08::RESIDENT_WARM && !state_has_committed)
        input_range_violation |= initial_range_violation;
    ap_uint<1> scalar_range_violation = 0;
    ap_uint<1> gradient_range_violation = 0;
    ap_uint<1> lane_range_violation[SCAN_LANES];
#pragma HLS ARRAY_PARTITION variable = lane_range_violation complete
init_range_flags:
    for (int lane = 0; lane < SCAN_LANES; ++lane) lane_range_violation[lane] = 0;

    const double input_limit = static_cast<double>(1ULL << (DATA_I - 1));
load_b:
    for (int i = 0; i < configured_n; ++i) {
        const double value = b_in == nullptr ? 0.0 : b_in[i];
        if (b_in != nullptr &&
            (value < -input_limit || value >= input_limit))
            input_range_violation = 1;
        b[i] = static_cast<dt>(value);
    }
load_d:
    for (int i = 0; i < configured_m; ++i) {
        const double value = d_in == nullptr ? 0.0 : d_in[i];
        if (d_in != nullptr &&
            (value < -input_limit || value >= input_limit))
            input_range_violation = 1;
        d[i] = static_cast<dt>(value);
    }

select_start:
    for (int i = 0; i < configured_n; ++i) {
        if (start_mode == snn_v08::RESIDENT_WARM) {
            x[i] = resident_state[i];
        } else if (start_mode == snn_v08::HOST_X0) {
            const double value = x0_in == nullptr ? 0.0 : x0_in[i];
            if (x0_in != nullptr &&
                (value < -input_limit || value >= input_limit))
                input_range_violation = 1;
            x[i] = static_cast<dt>(value);
        } else {
            x[i] = static_cast<dt>(0);
        }
    }
    apply_shift(x, shifted, configured_n, shift_stride, tail_policy);

    const dt k0 = configured_k0;
    const dt ctol = configured_ctol;
    const dt lower = configured_lower;
    const dt upper = configured_upper;
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
    for (int outer = 0; outer < configured_iters; ++outer) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 30000
        hessian_matvec(Ax, x, configured_n, lane_range_violation);

    gradient_fused_step:
        for (int i = 0; i < configured_n; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = x type = inter dependent = false
            const gradient_product_t product =
                static_cast<acc_t>(k0) *
                (static_cast<acc_t>(Ax[i]) + static_cast<acc_t>(b[i]));
            const acc_t next = static_cast<acc_t>(
                static_cast<acc_t>(x[i]) - product);
            x[i] = cast_dt(next, gradient_range_violation);
        }

        residual_matvec(residual, x, d, configured_m, configured_n,
                        lane_range_violation);

        bool left_sweep = false;
    projection_loop:
        for (int ordinal = 0; ordinal < configured_projmax; ++ordinal) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 10000
            const CandidateWinner selected = scan_candidates(
                x, residual, lower, upper, configured_n, configured_m,
                configured_has_lower, configured_has_upper);
            const int winner = selected.index;
            const int kind = selected.kind;
            const acc_t best = selected.score;
            if (best <= static_cast<acc_t>(ctol)) {
                left_sweep = true;
                break;
            }

            ap_uint<64> candidate = 0;
            if (kind == 0) {
                // True division. A reciprocal multiply is not the v0.8 step.
                const dt step = residual[winner] / resident_cns[winner];
            update_x_row:
                for (int j0 = 0; j0 < configured_n; j0 += UF) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = x type = inter dependent = false
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 16
                    ap_uint<1> lane_flag[UF];
#pragma HLS ARRAY_PARTITION variable = lane_flag complete
                    for (int lane = 0; lane < UF; ++lane) {
#pragma HLS UNROLL
                        lane_flag[lane] = 0;
                        const int j = j0 + lane;
                        if (j < configured_n) {
                            const acc_t next =
                                static_cast<acc_t>(x[j]) -
                                static_cast<acc_t>(step) *
                                    static_cast<acc_t>(resident_C[winner][j]);
                            x[j] = cast_dt(next, lane_flag[lane]);
                        }
                    }
                    or_lane_flags(lane_range_violation, lane_flag);
                }
                // m <= 8 is one cyclic group. Unrolling it avoids a one-trip
                // pipeline call. Wider rows keep the pipelined loop. Lane k
                // is bank k in both. The arithmetic is the v0.8 residual
                // update either way.
                if (configured_m <= SCAN_LANES) {
                    ap_uint<1> lane_flag[SCAN_LANES];
#pragma HLS ARRAY_PARTITION variable = lane_flag complete
                    for (int lane = 0; lane < SCAN_LANES; ++lane) {
#pragma HLS UNROLL
                        lane_flag[lane] = 0;
                        if (lane < configured_m) {
                            const acc_t next =
                                static_cast<acc_t>(residual[lane]) -
                                static_cast<acc_t>(step) *
                                    static_cast<acc_t>(resident_G[winner][lane]);
                            residual[lane] = cast_dt(next, lane_flag[lane]);
                        }
                    }
                    or_lane_flags(lane_range_violation, lane_flag);
                } else {
                update_residual_row:
                    for (int i0 = 0; i0 < configured_m; i0 += UF) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 16
                        ap_uint<1> lane_flag[UF];
#pragma HLS ARRAY_PARTITION variable = lane_flag complete
                        for (int lane = 0; lane < UF; ++lane) {
#pragma HLS UNROLL
                            lane_flag[lane] = 0;
                            const int i = i0 + lane;
                            if (i < configured_m) {
                                const acc_t next =
                                    static_cast<acc_t>(residual[i]) -
                                    static_cast<acc_t>(step) *
                                        static_cast<acc_t>(resident_G[winner][i]);
                                residual[i] = cast_dt(next, lane_flag[lane]);
                            }
                        }
                        or_lane_flags(lane_range_violation, lane_flag);
                    }
                }
                ++row_events;
                candidate = static_cast<ap_uint<64>>(winner);
            } else {
                const dt delta =
                    kind == 1 ? cast_dt(best, scalar_range_violation)
                              : cast_dt(-best, scalar_range_violation);
                x[winner] = cast_dt(static_cast<acc_t>(x[winner]) +
                                        static_cast<acc_t>(delta),
                                    scalar_range_violation);
                if (configured_m <= SCAN_LANES) {
                    ap_uint<1> lane_flag[SCAN_LANES];
#pragma HLS ARRAY_PARTITION variable = lane_flag complete
                    for (int lane = 0; lane < SCAN_LANES; ++lane) {
#pragma HLS UNROLL
                        lane_flag[lane] = 0;
                        if (lane < configured_m) {
                            const acc_t next =
                                static_cast<acc_t>(residual[lane]) +
                                static_cast<acc_t>(delta) *
                                    static_cast<acc_t>(resident_Ct[winner][lane]);
                            residual[lane] = cast_dt(next, lane_flag[lane]);
                        }
                    }
                    or_lane_flags(lane_range_violation, lane_flag);
                } else {
                update_residual_facet:
                    for (int i0 = 0; i0 < configured_m; i0 += UF) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 16
                        ap_uint<1> lane_flag[UF];
#pragma HLS ARRAY_PARTITION variable = lane_flag complete
                        for (int lane = 0; lane < UF; ++lane) {
#pragma HLS UNROLL
                            lane_flag[lane] = 0;
                            const int i = i0 + lane;
                            if (i < configured_m) {
                                const acc_t next =
                                    static_cast<acc_t>(residual[i]) +
                                    static_cast<acc_t>(delta) *
                                        static_cast<acc_t>(resident_Ct[winner][i]);
                                residual[i] = cast_dt(next, lane_flag[lane]);
                            }
                        }
                        or_lane_flags(lane_range_violation, lane_flag);
                    }
                }
                if (kind == 1) {
                    ++lower_events;
                    candidate = static_cast<ap_uint<64>>(configured_m + winner);
                } else {
                    ++upper_events;
                    candidate = static_cast<ap_uint<64>>(
                        configured_m + configured_n + winner);
                }
            }

            if (first_candidate == static_cast<ap_uint<64>>(NO_CANDIDATE))
                first_candidate = candidate;
            last_candidate = candidate;
            event_digest = digest_word(
                event_digest, static_cast<ap_uint<64>>(outer));
            event_digest = digest_word(
                event_digest, static_cast<ap_uint<64>>(ordinal));
            event_digest = digest_word(event_digest, candidate);
            ++total_events;
        }

        if (!left_sweep) {
            ++cap_rechecks;
            residual_matvec(residual, x, d, configured_m, configured_n,
                            lane_range_violation);
            const CandidateWinner selected = scan_candidates(
                x, residual, lower, upper, configured_n, configured_m,
                configured_has_lower, configured_has_upper);
            acc_t maximum = 0;
            if (selected.valid && selected.score > maximum)
                maximum = selected.score;
            if (maximum > static_cast<acc_t>(ctol)) {
                status = 2;
                executed = outer + 1;
                break;
            }
        }
        executed = outer + 1;
    }

    for (int i = 0; i < configured_n; ++i) resident_state[i] = x[i];
    state_has_committed = 1;
    initial_range_violation = 0;
    // Lane flags are not part of word 15. v0.6 and v0.8 both omit them.
    const ap_uint<1> any_range_violation =
        input_range_violation | scalar_range_violation |
        gradient_range_violation;
    write_telemetry(x_raw_out, telemetry_out, x, configured_n, status,
                    executed, configured_iters, total_events, row_events,
                    lower_events, upper_events, cap_rechecks, event_digest,
                    first_candidate, last_candidate, any_range_violation);
    return snn_v08::ERR_OK;
}

struct DispatchResult {
    int error = snn_v08::ERR_OK;
    int status = 0;
    int iterations = 0;
    int route = snn_v08::AUTO;
};

inline DispatchResult dispatch_command(
    int command, int requested_route, int start_mode, int shift_stride,
    int tail_policy, const double* A_cfg, const double* C_cfg,
    const double* G_cfg, const double* cns_cfg, const double* row_scale_cfg,
    const double* b_in, const double* d_in, const double* x0_in,
    std::uint32_t* A_ddr, std::uint32_t* C_ddr, std::uint32_t* Ct_ddr,
    std::uint32_t* G_ddr, long long* x_raw_out,
    unsigned long long* telemetry_out, int n, int m, double k0_f,
    double ctol_f, int n_iters, int projmax, int has_lower, double lower_f,
    int has_upper, double upper_f) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION function instances = load_geometry_matrix limit = 1
#pragma HLS ALLOCATION function instances = binary64_to_dt limit = 1
    DispatchResult result;
    result.route = configured_route;
    if (command == snn_v08::CONFIGURE) {
        result.error = configure_geometry(
            cns_cfg, row_scale_cfg, x0_in, n, m, k0_f, ctol_f, n_iters,
            projmax, has_lower, lower_f, has_upper, upper_f,
            requested_route);
        if (result.error == snn_v08::ERR_OK) {
            load_geometry_matrix(A_cfg, A_ddr, nullptr, GEOMETRY_A, 0, n, n,
                                 a_range_violation);
            load_geometry_matrix(C_cfg, C_ddr, Ct_ddr, GEOMETRY_C, 1, m, n,
                                 c_range_violation);
            load_geometry_matrix(G_cfg, G_ddr, nullptr, GEOMETRY_G, 0, m, m,
                                 g_range_violation);
            write_geometry_range_flag();
        }
        result.route = configured_route;
        return result;
    }
    if (command == snn_v08::REFRESH_A) {
        result.error = refresh_a(n, m);
        if (result.error == snn_v08::ERR_OK) {
            load_geometry_matrix(A_cfg, A_ddr, nullptr, GEOMETRY_A, 0, n, n,
                                 a_range_violation);
            write_geometry_range_flag();
        }
        result.route = configured_route;
        return result;
    }
    if (command == snn_v08::SOLVE) {
        if (is_stopped) {
            result.error = snn_v08::ERR_STOPPED;
            result.route = configured_route;
            return result;
        }
        result.error = solve_current(
            b_in, d_in, x0_in, x_raw_out, telemetry_out, start_mode,
            shift_stride, tail_policy);
        result.route = configured_route;
        if (telemetry_out != nullptr) {
            result.status = static_cast<int>(telemetry_out[1]);
            result.iterations = static_cast<int>(telemetry_out[3]);
        }
        return result;
    }
    if (command == snn_v08::STOP) {
        is_stopped = 1;
        result.error = snn_v08::ERR_STOPPED;
        result.route = configured_route;
        return result;
    }
    result.error = snn_v08::ERR_BAD_COMMAND;
    return result;
}

inline void acknowledge_dispatch(volatile std::uint32_t* mb_out,
                                 std::uint32_t sequence,
                                 const DispatchResult& result) {
#pragma HLS INLINE
    write_mailbox(
        mb_out, sequence, static_cast<std::uint32_t>(result.error),
        static_cast<std::uint32_t>(result.status),
        static_cast<std::uint32_t>(result.route),
        static_cast<std::uint32_t>(result.iterations));
}

}  // namespace

extern "C" void snn_qp_v09b(
    const double* A_cfg, const double* C_cfg, const double* G_cfg,
    const double* cns_cfg, const double* row_scale_cfg, const double* b_in,
    const double* d_in, const double* x0_in, std::uint32_t* A_ddr,
    std::uint32_t* C_ddr, std::uint32_t* Ct_ddr, std::uint32_t* G_ddr,
    long long* x_raw_out, unsigned long long* telemetry_out,
    volatile const std::uint32_t* mb_in, volatile std::uint32_t* mb_out,
    int command, int launch_mode, int route_mode, int start_mode,
    int shift_stride, int tail_policy, int n, int m, double k0_f,
    double ctol_f, int n_iters, int projmax, int has_lower, double lower_f,
    int has_upper, double upper_f) {
#pragma HLS INTERFACE m_axi port = A_cfg bundle = g0 offset = slave depth = 16384
#pragma HLS INTERFACE m_axi port = C_cfg bundle = g0 offset = slave depth = 16384
#pragma HLS INTERFACE m_axi port = G_cfg bundle = g0 offset = slave depth = 16384
#pragma HLS INTERFACE m_axi port = cns_cfg bundle = g0 offset = slave depth = 128
#pragma HLS INTERFACE m_axi port = row_scale_cfg bundle = g0 offset = slave depth = 128
#pragma HLS INTERFACE m_axi port = A_ddr bundle = g1 offset = slave depth = 16384
#pragma HLS INTERFACE m_axi port = C_ddr bundle = g1 offset = slave depth = 16384
#pragma HLS INTERFACE m_axi port = Ct_ddr bundle = g1 offset = slave depth = 16384
#pragma HLS INTERFACE m_axi port = G_ddr bundle = g1 offset = slave depth = 16384
#pragma HLS INTERFACE m_axi port = b_in bundle = g2 offset = slave depth = 128
#pragma HLS INTERFACE m_axi port = d_in bundle = g2 offset = slave depth = 128
#pragma HLS INTERFACE m_axi port = x0_in bundle = g2 offset = slave depth = 128
#pragma HLS INTERFACE m_axi port = mb_in bundle = g2 offset = slave depth = 64
#pragma HLS INTERFACE m_axi port = mb_out bundle = g2 offset = slave depth = 64
#pragma HLS INTERFACE m_axi port = x_raw_out bundle = g2 offset = slave depth = 128
#pragma HLS INTERFACE m_axi port = telemetry_out bundle = g2 offset = slave depth = 16
#pragma HLS INTERFACE s_axilite port = A_cfg bundle = c
#pragma HLS INTERFACE s_axilite port = C_cfg bundle = c
#pragma HLS INTERFACE s_axilite port = G_cfg bundle = c
#pragma HLS INTERFACE s_axilite port = cns_cfg bundle = c
#pragma HLS INTERFACE s_axilite port = row_scale_cfg bundle = c
#pragma HLS INTERFACE s_axilite port = b_in bundle = c
#pragma HLS INTERFACE s_axilite port = d_in bundle = c
#pragma HLS INTERFACE s_axilite port = x0_in bundle = c
#pragma HLS INTERFACE s_axilite port = A_ddr bundle = c
#pragma HLS INTERFACE s_axilite port = C_ddr bundle = c
#pragma HLS INTERFACE s_axilite port = Ct_ddr bundle = c
#pragma HLS INTERFACE s_axilite port = G_ddr bundle = c
#pragma HLS INTERFACE s_axilite port = x_raw_out bundle = c
#pragma HLS INTERFACE s_axilite port = telemetry_out bundle = c
#pragma HLS INTERFACE s_axilite port = mb_in bundle = c
#pragma HLS INTERFACE s_axilite port = mb_out bundle = c
#pragma HLS INTERFACE s_axilite port = command bundle = c
#pragma HLS INTERFACE s_axilite port = launch_mode bundle = c
#pragma HLS INTERFACE s_axilite port = route_mode bundle = c
#pragma HLS INTERFACE s_axilite port = start_mode bundle = c
#pragma HLS INTERFACE s_axilite port = shift_stride bundle = c
#pragma HLS INTERFACE s_axilite port = tail_policy bundle = c
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

    // Storage directives for the file-scope resident arrays. The HLS
    // frontend rejects '#pragma HLS' at namespace scope (parse error
    // 'expected unqualified-id'; g++ accepts it, so x86 parity cannot
    // catch this). Inside the top function they bind the globals by
    // name: the v08 kernel places its bind_storage block here too.
#pragma HLS ARRAY_PARTITION variable = resident_A cyclic factor = 4 dim = 1
#pragma HLS ARRAY_PARTITION variable = resident_A cyclic factor = 8 dim = 2
#pragma HLS bind_storage variable = resident_A type = ram_t2p impl = bram
#pragma HLS ARRAY_PARTITION variable = resident_C cyclic factor = 4 dim = 1
#pragma HLS ARRAY_PARTITION variable = resident_C cyclic factor = 8 dim = 2
#pragma HLS bind_storage variable = resident_C type = ram_t2p impl = bram
#pragma HLS ARRAY_PARTITION variable = resident_Ct cyclic factor = 8 dim = 2
#pragma HLS bind_storage variable = resident_Ct type = ram_t2p impl = bram
#pragma HLS ARRAY_PARTITION variable = resident_G cyclic factor = 8 dim = 2
#pragma HLS bind_storage variable = resident_G type = ram_t2p impl = bram
#pragma HLS ARRAY_PARTITION variable = resident_scale cyclic factor = 8 dim = 1
#pragma HLS bind_storage variable = resident_cns type = ram_1p impl = bram
#pragma HLS bind_storage variable = resident_scale type = ram_1p impl = bram
#pragma HLS bind_storage variable = resident_state type = ram_1p impl = bram


    clear_outputs(x_raw_out, telemetry_out, n);

    if (launch_mode != snn_v08::PERSISTENT) {
        int effective_command = command;
        int effective_start = start_mode;
        int effective_shift = shift_stride;
        int effective_tail = tail_policy;
        int effective_route = route_mode;
        std::uint32_t sequence = 0;
        if (command == snn_v08::SERVE) {
            if (mb_in != nullptr) {
                sequence = mb_in[snn_v08::MAILBOX_SEQUENCE];
                effective_command =
                    static_cast<int>(mb_in[snn_v08::MAILBOX_COMMAND]);
                effective_start =
                    static_cast<int>(mb_in[snn_v08::MAILBOX_START_MODE]);
                effective_shift =
                    static_cast<int>(mb_in[snn_v08::MAILBOX_SHIFT_STRIDE]);
                effective_tail =
                    static_cast<int>(mb_in[snn_v08::MAILBOX_TAIL_POLICY]);
                const std::uint32_t flags = mb_in[snn_v08::MAILBOX_FLAGS];
                if ((flags & snn_v08::FLAG_ROUTE_MASK) != 0)
                    effective_route =
                        static_cast<int>(flags & snn_v08::FLAG_ROUTE_MASK);
            }
        } else if (mb_in != nullptr) {
            sequence = mb_in[snn_v08::MAILBOX_SEQUENCE];
        }
        DispatchResult result;
        if (effective_command == snn_v08::LITMUS) {
            write_litmus_pattern(x_raw_out, telemetry_out, mb_in, n);
            result.error = snn_v08::ERR_OK;
            result.route = configured_route;
        } else {
            result = dispatch_command(
                effective_command, effective_route, effective_start,
                effective_shift, effective_tail, A_cfg, C_cfg, G_cfg,
                cns_cfg, row_scale_cfg, b_in, d_in, x0_in, A_ddr, C_ddr,
                Ct_ddr, G_ddr, x_raw_out, telemetry_out, n, m, k0_f, ctol_f,
                n_iters, projmax, has_lower, lower_f, has_upper, upper_f);
        }
        acknowledge_dispatch(mb_out, sequence, result);
        return;
    }

    std::uint32_t seen_sequence = mb_in == nullptr
                                       ? 0
                                       : mb_in[snn_v08::MAILBOX_SEQUENCE];
    if (command != snn_v08::SERVE) {
        DispatchResult result;
        if (command == snn_v08::LITMUS) {
            write_litmus_pattern(x_raw_out, telemetry_out, mb_in, n);
            result.error = snn_v08::ERR_OK;
            result.route = configured_route;
        } else {
            result = dispatch_command(
                command, route_mode, start_mode, shift_stride, tail_policy,
                A_cfg, C_cfg, G_cfg, cns_cfg, row_scale_cfg, b_in, d_in,
                x0_in, A_ddr, C_ddr, Ct_ddr, G_ddr, x_raw_out,
                telemetry_out, n, m, k0_f, ctol_f, n_iters, projmax,
                has_lower, lower_f, has_upper, upper_f);
        }
        acknowledge_dispatch(mb_out, seen_sequence, result);
        if (command == snn_v08::STOP) return;
    }

persistent_wait:
    volatile unsigned int poll_counter = 0;
    while (!is_stopped) {
        std::uint32_t sequence =
            mb_in == nullptr ? seen_sequence
                             : mb_in[snn_v08::MAILBOX_SEQUENCE];
        if (sequence != seen_sequence) {
#ifndef __SYNTHESIS__
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
#endif
            int mailbox_command = snn_v08::SERVE;
            int mailbox_start = start_mode;
            int mailbox_shift = shift_stride;
            int mailbox_tail = tail_policy;
            int mailbox_route = route_mode;
            if (mb_in != nullptr) {
                mailbox_command =
                    static_cast<int>(mb_in[snn_v08::MAILBOX_COMMAND]);
                mailbox_start =
                    static_cast<int>(mb_in[snn_v08::MAILBOX_START_MODE]);
                mailbox_shift =
                    static_cast<int>(mb_in[snn_v08::MAILBOX_SHIFT_STRIDE]);
                mailbox_tail =
                    static_cast<int>(mb_in[snn_v08::MAILBOX_TAIL_POLICY]);
                const std::uint32_t flags = mb_in[snn_v08::MAILBOX_FLAGS];
                if ((flags & snn_v08::FLAG_ROUTE_MASK) != 0)
                    mailbox_route =
                        static_cast<int>(flags & snn_v08::FLAG_ROUTE_MASK);
            }
            DispatchResult result;
            if (mailbox_command == snn_v08::LITMUS) {
                write_litmus_pattern(x_raw_out, telemetry_out, mb_in, n);
                result.error = snn_v08::ERR_OK;
                result.route = configured_route;
            } else {
                result = dispatch_command(
                    mailbox_command, mailbox_route, mailbox_start,
                    mailbox_shift, mailbox_tail, A_cfg, C_cfg, G_cfg,
                    cns_cfg, row_scale_cfg, b_in, d_in, x0_in, A_ddr, C_ddr,
                    Ct_ddr, G_ddr, x_raw_out, telemetry_out, n, m, k0_f,
                    ctol_f, n_iters, projmax, has_lower, lower_f, has_upper,
                    upper_f);
            }
            acknowledge_dispatch(mb_out, sequence, result);
            seen_sequence = sequence;
            if (mailbox_command == snn_v08::STOP) break;
        }
    poll_wait:
        for (int poll = 0; poll < POLL_CYCLES; ++poll) {
#pragma HLS LOOP_TRIPCOUNT min = 256 max = 256
            poll_counter = static_cast<unsigned int>(poll);
            ap_wait_n(1);
        }
    }
}
