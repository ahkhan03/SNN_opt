// SNN-QP v0.7 resident command-service kernel.
//
// The recurrence in solve_current is the v0.5 fixed-point recurrence.  v0.6
// adds only storage accessors, a configure/refresh dispatcher, and the
// optional persistent mailbox loop.  Geometry images are written once as
// ap_fixed bit patterns, so streamed and resident reads see the same words.

#include "dt.h"
#include "v07_abi.hpp"
#include "v07_cone_table.hpp"

#include <ap_int.h>
#include <ap_utils.h>
#include <cstddef>
#include <cstdint>

#ifndef MAXN
#define MAXN 1024
#endif

#ifndef MAXM
#define MAXM 1024
#endif

// Logical 32-bit word budgets from the resident plan.  They are deliberately
// guarded so an implementation run can lower them without editing the ABI.
#ifndef MATRIX_WORD_CAP
#define MATRIX_WORD_CAP 440000
#endif

#ifndef BRAM_WORD_CAP
#define BRAM_WORD_CAP 129024
#endif

#ifndef FULL_SQUARE_CAP
#define FULL_SQUARE_CAP 320
#endif

#ifndef CG_SQUARE_CAP
#define CG_SQUARE_CAP 372
#endif

#ifndef STREAM_DIM_CAP
#define STREAM_DIM_CAP 1024
#endif

#ifndef UF
#define UF 4
#endif

#ifndef RG
#define RG 2
#endif

#ifndef POLL_CYCLES
#define POLL_CYCLES 256
#endif

// Eight 32-bit fixed-point values form one 256-bit resident word.  The
// physical row stride is rounded up to this factor, so a two-row/four-lane
// arithmetic group consumes exactly two URAM reads.  Keep this literal: the
// Vitis 2022.1 parser does not expand a macro in an ARRAY_PARTITION factor.
#ifndef MATRIX_PACK_FACTOR
#define MATRIX_PACK_FACTOR 8
#endif

#if MATRIX_PACK_FACTOR != 8
#error "The v07 resident image is defined as eight 32-bit lanes per word"
#endif

#if MAXN < 1 || MAXM < 1 || MATRIX_WORD_CAP < 1
#error "v07 capacities must be positive"
#endif

#if DATA_W != 32 || DATA_I != 8
#error "v07 ABI requires the v05 ap_fixed<32,8> state"
#endif

#if UF != 4 || RG != 2
#error "The v07 arithmetic shape is fixed at UF=4 and RG=2"
#endif

// HLS does not expand a macro in a raw pragma token.  Keep the factors
// explicit through _Pragma, as in the v05 build, so a controller can inspect
// and change the shape from the compile command.
#define V07_STRINGIFY_(x) #x
#define V07_STRINGIFY(x) V07_STRINGIFY_(x)
#define V07_DO_PRAGMA(x) _Pragma(V07_STRINGIFY(x))
#define V07_UNROLL_UF V07_DO_PRAGMA(HLS UNROLL factor = UF)
#define V07_UNROLL_RG V07_DO_PRAGMA(HLS UNROLL factor = RG)

namespace {

constexpr unsigned int MATRIX_LANES = 8;
// All configured matrix dimensions currently share the same 1024 cap, but
// keep the line-buffer extent expressed as the larger of the two ABI caps so
// a controller can lower MAXN/MAXM independently without changing accessors.
constexpr int MAX_ROW_DIM = MAXN > MAXM ? MAXN : MAXM;
constexpr std::size_t MATRIX_PACKED_CAP =
    (static_cast<std::size_t>(MATRIX_WORD_CAP) + MATRIX_LANES - 1) /
    MATRIX_LANES;
using matrix_word_t = ap_uint<MATRIX_LANES * DATA_W>;

constexpr unsigned long long DIGEST_OFFSET =
    14695981039346656037ULL;
constexpr unsigned long long NO_CANDIDATE = ~0ULL;

// These globals are the persistent device state.  A CONFIGURE command is the
// explicit reset point, which also makes one-shot launches deterministic.
// Logical MATRIX_WORD_CAP fixed words are stored as packed 256-bit words.
// Keeping the logical cap in the route predicates preserves the public
// capacity language while reducing the URAM depth by 8x.
static matrix_word_t resident_matrix[MATRIX_PACKED_CAP];
static dt resident_cns[MAXM];
static dt resident_scale[MAXM];
static dt resident_state[MAXN];

static int configured_n = 0;
static int configured_m = 0;
static int configured_route = snn_v07::AUTO;
static int configured_iters = 0;
static int configured_projmax = 0;
static int configured_has_lower = 0;
static int configured_has_upper = 0;
static dt configured_k0 = 0;
// Native cone mode keeps the v13 F32 k0 register.  The legacy resident path
// retains its dt copy for cones-off parity.
static acc_t configured_native_k0 = 0;
static dt configured_ctol = 0;
static dt configured_lower = 0;
static dt configured_upper = 0;
// All image capacities and row offsets are below 2^32.  Keeping these
// indices explicitly 32-bit avoids carrying 64-bit adders/multipliers into
// every line-buffer and geometry address calculation.
static std::uint32_t resident_words = 0;
static std::uint32_t offset_a = 0;
static std::uint32_t offset_c = 0;
static std::uint32_t offset_ct = 0;
static std::uint32_t offset_g = 0;
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

// v07 native reset image.  Metadata is copied at CONFIGURE and remains in
// the resident kernel, so SOLVE/REFRESH mailbox traffic keeps the v06 layout.
static int configured_cone_count = 0;
static std::uint32_t cone_offset[snn_v07::MAX_CONES];
static std::uint32_t cone_length[snn_v07::MAX_CONES];
static snn_v07::ConeKind cone_kind[snn_v07::MAX_CONES];
static dt cone_radius[snn_v07::MAX_CONES];
static dt cone_mu[snn_v07::MAX_CONES];
static dt cone_center[snn_v07::MAX_CONES];

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

// Keep the wide fixed-point round/saturate path behind an explicit register.
// The cone gradient loop calls this after its multiply/add stages, so HLS
// cannot place conversion logic on the same cycle as the arithmetic or x RAM
// write.  Native C simulation remains the cast_dt operation byte-for-byte.
inline dt cone_gradient_commit_stage(
    acc_t value, ap_uint<1>& range_violation) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
#pragma HLS LATENCY min = 1
    return cast_dt(value, range_violation);
}

// The native cone schedule shares the v13 product register contract.  Keep
// the round point explicit here instead of relying on the expression width of
// ap_fixed operator overloads, which is intentionally wider in the resident
// row path for v06 compatibility.
inline acc_t cone_product_mul(acc_t left, acc_t right) {
#pragma HLS INLINE
    acc_t out = left * right;
    return out;
}

inline acc_t cone_product_add(acc_t left, acc_t right) {
#pragma HLS INLINE
    acc_t out = left + right;
    return out;
}

// Timing-closure stages for the cone-only v13 arithmetic.  The argument and
// result formats are unchanged from cone_product_mul/add; only the HLS
// schedule gains registered boundaries.  Native C simulation therefore sees
// exactly the same fixed-point operations and rounding.
inline acc_t cone_product_mul_stage(acc_t left, acc_t right) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
    acc_t out;
#pragma HLS BIND_OP variable=out op=mul impl=dsp latency=2
    out = static_cast<acc_t>(left * right);
    return out;
}

inline acc_t cone_product_add_stage(acc_t left, acc_t right) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
    acc_t out = static_cast<acc_t>(left + right);
    return out;
}

inline dt raw_to_dt(std::uint32_t word) {
#pragma HLS INLINE
    // Assigning the bit range preserves the fixed-point binary point.  A
    // numeric cast through ap_int would interpret the word as an integer and
    // lose the DATA_I fractional bits.
    dt value = 0;
    value.range(DATA_W - 1, 0) = word;
    return value;
}

inline std::uint32_t dt_to_raw(dt value) {
#pragma HLS INLINE
    const ap_uint<DATA_W> bits = value.range(DATA_W - 1, 0);
    return static_cast<std::uint32_t>(bits);
}

inline std::uint32_t padded_row_words(int columns) {
#pragma HLS INLINE
    const std::uint32_t width =
        static_cast<std::uint32_t>(columns < 0 ? 0 : columns);
    return ((width + MATRIX_LANES - 1U) / MATRIX_LANES) * MATRIX_LANES;
}

inline std::uint32_t packed_index(std::uint32_t logical_index) {
#pragma HLS INLINE
    return logical_index / MATRIX_LANES;
}

inline dt unpack_matrix_lane(const matrix_word_t& word, int lane) {
#pragma HLS INLINE
    ap_uint<DATA_W> bits =
        word.range((lane + 1) * DATA_W - 1, lane * DATA_W);
    return raw_to_dt(static_cast<std::uint32_t>(bits));
}

inline void pack_matrix_lane(matrix_word_t& word, int lane, dt value) {
#pragma HLS INLINE
    word.range((lane + 1) * DATA_W - 1, lane * DATA_W) = dt_to_raw(value);
}

inline void write_resident_word(std::uint32_t logical_index, dt value) {
#pragma HLS INLINE
    const std::uint32_t word_index = packed_index(logical_index);
    const int lane = static_cast<int>(logical_index % MATRIX_LANES);
    matrix_word_t word = resident_matrix[word_index];
    pack_matrix_lane(word, lane, value);
    resident_matrix[word_index] = word;
}

inline dt read_image(const std::uint32_t* image, std::uint32_t index) {
#pragma HLS INLINE
    if (image == nullptr) return static_cast<dt>(0);
    return raw_to_dt(image[index]);
}

inline void write_image(std::uint32_t* image, std::uint32_t index, dt value) {
#pragma HLS INLINE
    if (image != nullptr) image[index] = dt_to_raw(value);
}

inline bool dimensions_valid(int n, int m) {
#pragma HLS INLINE
    return n >= 1 && m >= 1 && n <= MAXN && m <= MAXM &&
           n <= STREAM_DIM_CAP && m <= STREAM_DIM_CAP;
}

inline std::uint32_t full_words(int n, int m) {
#pragma HLS INLINE
    return static_cast<std::uint32_t>(n) * padded_row_words(n) +
           static_cast<std::uint32_t>(m) * padded_row_words(n) +
           static_cast<std::uint32_t>(n) * padded_row_words(m) +
           static_cast<std::uint32_t>(m) * padded_row_words(m);
}

inline std::uint32_t cg_words(int n, int m) {
#pragma HLS INLINE
    return static_cast<std::uint32_t>(m) * padded_row_words(n) +
           static_cast<std::uint32_t>(n) * padded_row_words(m) +
           static_cast<std::uint32_t>(m) * padded_row_words(m);
}

inline bool bram_shape_fits(int n, int m) {
#pragma HLS INLINE
    return 16ULL * static_cast<unsigned long long>(n + m) + 4096ULL <=
           static_cast<unsigned long long>(BRAM_WORD_CAP);
}

inline bool route_fits(int n, int m, int route) {
#pragma HLS INLINE
    if (!dimensions_valid(n, m) || !bram_shape_fits(n, m)) return false;
    if (route == snn_v07::FORCE_FULL) {
        if (full_words(n, m) > static_cast<unsigned long long>(MATRIX_WORD_CAP))
            return false;
        if (full_words(n, m) > 440000ULL) return false;
        if (n == m && n > FULL_SQUARE_CAP) return false;
        return true;
    }
    if (route == snn_v07::FORCE_CG) {
        if (cg_words(n, m) > static_cast<unsigned long long>(MATRIX_WORD_CAP))
            return false;
        if (cg_words(n, m) > 440000ULL) return false;
        if (n == m && n > CG_SQUARE_CAP) return false;
        return true;
    }
    if (route == snn_v07::FORCE_STREAM) return true;
    return false;
}

inline int choose_route(int n, int m, int requested) {
#pragma HLS INLINE
    if (requested == snn_v07::AUTO) {
        if (route_fits(n, m, snn_v07::FORCE_FULL))
            return snn_v07::FORCE_FULL;
        if (route_fits(n, m, snn_v07::FORCE_CG))
            return snn_v07::FORCE_CG;
        if (route_fits(n, m, snn_v07::FORCE_STREAM))
            return snn_v07::FORCE_STREAM;
        return -1;
    }
    if (requested < snn_v07::FORCE_FULL ||
        requested > snn_v07::FORCE_STREAM)
        return -1;
    return route_fits(n, m, requested) ? requested : -1;
}

inline void set_offsets(int n, int m, int route) {
#pragma HLS INLINE
    offset_a = 0;
    offset_c = 0;
    offset_ct = 0;
    offset_g = 0;
    if (route == snn_v07::FORCE_FULL) {
        offset_a = 0;
        offset_c = static_cast<std::uint32_t>(n) * padded_row_words(n);
        offset_ct = offset_c +
                    static_cast<std::uint32_t>(m) * padded_row_words(n);
        offset_g = offset_ct +
                   static_cast<std::uint32_t>(n) * padded_row_words(m);
        resident_words = full_words(n, m);
    } else if (route == snn_v07::FORCE_CG) {
        offset_c = 0;
        offset_ct = static_cast<std::uint32_t>(m) * padded_row_words(n);
        offset_g = offset_ct +
                   static_cast<std::uint32_t>(n) * padded_row_words(m);
        resident_words = cg_words(n, m);
    } else {
        resident_words = 0;
    }
}

// Load a pair of matrix rows before entering an arithmetic matvec loop.  A
// resident row is read as 256-bit words (eight fixed values); a streamed row
// is read from its fixed-point DDR image in the same order.  The caller's
// cyclic-four column partition then presents four independent values per
// cycle to the v05 arithmetic body.
// One line-buffer loader is deliberately kept out of its callers.  The same
// module is used for row-group loads and all single-row event loads, so HLS
// can schedule the six sequential call sites through one DDR/URAM reader.
inline void load_line_buffer(
    dt row[MAX_ROW_DIM], int row_index, int row_count, int columns,
    int use_resident, std::uint32_t resident_base,
    const std::uint32_t* image) {
#pragma HLS INLINE off
    const std::uint32_t resident_row_base =
        resident_base + static_cast<std::uint32_t>(row_index) *
                            padded_row_words(columns);
    const std::uint32_t image_row_base =
        static_cast<std::uint32_t>(row_index) *
        static_cast<std::uint32_t>(columns);
load_line_words:
    for (int j0 = 0; j0 < columns; j0 += MATRIX_LANES) {
#pragma HLS PIPELINE II = 1
        matrix_word_t resident_word = 0;
        if (use_resident && row_index < row_count) {
            const std::uint32_t logical =
                resident_row_base + static_cast<std::uint32_t>(j0);
            resident_word = resident_matrix[packed_index(logical)];
        }
        for (int lane = 0; lane < MATRIX_LANES; ++lane) {
            const int column = j0 + lane;
            if (column >= MAX_ROW_DIM) {
                continue;
            } else if (column >= columns || row_index >= row_count) {
                row[column] = static_cast<dt>(0);
            } else if (use_resident) {
                row[column] = unpack_matrix_lane(resident_word, lane);
            } else {
                row[column] = read_image(image, image_row_base + column);
            }
        }
    }
}

inline void load_row_group(dt rows[RG][MAX_ROW_DIM], int row0, int row_count,
                           int columns, int use_resident,
                           std::uint32_t resident_base,
                           const std::uint32_t* image) {
#pragma HLS INLINE
load_group_rows:
    for (int group = 0; group < RG; ++group) {
        load_line_buffer(rows[group], row0 + group, row_count, columns,
                         use_resident, resident_base, image);
    }
}

// C*x+d is needed once for the normal projection sweep and again for the cap
// recheck.  Keep one out-of-line matvec body so the two call sites share the
// 48-DSP/8,185-LUT datapath reported for residual_columns instead of building
// a second cap_residual_columns copy.
inline void residual_matvec(
    dt rows[RG][MAX_ROW_DIM], dt* residual, const dt* x, const dt* d,
    int row_count, int columns, int use_resident,
    std::uint32_t resident_base, const std::uint32_t* image,
    ap_uint<1> lane_range_violation[UF]) {
#pragma HLS INLINE off
residual_rows_shared:
    for (int i0 = 0; i0 < row_count; i0 += RG) {
        acc_t acc[RG];
#pragma HLS ARRAY_PARTITION variable = acc complete
        for (int group = 0; group < RG; ++group) acc[group] = 0;
        load_row_group(rows, i0, row_count, columns, use_resident,
                       resident_base, image);
    residual_columns_shared:
        for (int j0 = 0; j0 < columns; j0 += UF) {
#pragma HLS PIPELINE II = 1
            dt x_chunk[UF];
#pragma HLS ARRAY_PARTITION variable = x_chunk complete
            for (int lane = 0; lane < UF; ++lane) {
                const int column = j0 + lane;
                x_chunk[lane] = column < columns ? x[column]
                                                  : static_cast<dt>(0);
            }
            for (int group = 0; group < RG; ++group) {
                const int row = i0 + group;
                acc_t partial = 0;
                for (int lane = 0; lane < UF; ++lane) {
                    const int column = j0 + lane;
                    if (row < row_count && column < columns)
                        partial += rows[group][column] * x_chunk[lane];
                }
                acc[group] += partial;
            }
        }
        for (int group = 0; group < RG; ++group) {
            if (i0 + group < row_count)
                residual[i0 + group] = cast_dt(
                    acc[group] + static_cast<acc_t>(d[i0 + group]),
                    lane_range_violation[group]);
        }
    }
}

inline void clear_outputs(long long* x_raw_out,
                          unsigned long long* telemetry_out, int n) {
#pragma HLS INLINE
    if (x_raw_out != nullptr) {
        const int limit = n < MAXN ? n : MAXN;
        for (int i = 0; i < limit; ++i) x_raw_out[i] = 0;
    }
    if (telemetry_out != nullptr) {
        for (int i = 0; i < static_cast<int>(snn_v07::TELEMETRY_WORDS); ++i)
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
    mb_out[snn_v07::OUT_ERROR_CODE] = error;
    mb_out[snn_v07::OUT_STATUS] = status;
    mb_out[snn_v07::OUT_SELECTED_ROUTE] = route;
    mb_out[snn_v07::OUT_ITERATIONS] = iterations;
    mb_out[snn_v07::OUT_DONE_SEQUENCE] = sequence;
}

inline void write_litmus_pattern(
    long long* x_raw_out, unsigned long long* telemetry_out,
    volatile const std::uint32_t* mb_in, int n) {
#pragma HLS INLINE off
    // A deliberately nonzero pattern catches stale cache lines and endian
    // mistakes.  Word zero in telemetry echoes the input marker, which lets
    // the host test the reverse (host-to-device) direction in the same run.
    if (x_raw_out != nullptr) {
        for (int i = 0; i < n; ++i)
            x_raw_out[i] = static_cast<long long>(
                UINT64_C(0x13579bdf00000000) |
                static_cast<unsigned long long>(i));
    }
    if (telemetry_out != nullptr) {
        telemetry_out[0] = mb_in == nullptr
                               ? UINT64_C(0xa5a5000000000000)
                               : static_cast<unsigned long long>(
                                     mb_in[snn_v07::MAILBOX_LITMUS_INPUT]);
        for (std::size_t i = 1; i < snn_v07::TELEMETRY_WORDS; ++i)
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

// Keep the binary64-to-fixed operation in one non-inlined function.  The
// configure and refresh paths call this converter through the shared geometry
// loader below, so the expensive floating-point conversion is instantiated
// once instead of once per source loop.
inline dt binary64_to_dt(double value, double input_limit,
                         ap_uint<1>& range_violation) {
#pragma HLS INLINE off
    if (value < -input_limit || value >= input_limit)
        range_violation = 1;
    return static_cast<dt>(value);
}

// Load A, C, or G from the binary64 configure image, write its fixed-point
// backing image, and optionally populate the resident image.  C uses the
// transpose flag to emit Ct in the same pass.  The matrix selector keeps the
// route-specific residency decision inside one shared body, including the
// refresh-A call site.
inline void load_geometry_matrix(
    const double* source, std::uint32_t* image,
    std::uint32_t* transpose_image, int matrix_selector, int transpose,
    int rows, int columns, int route, std::uint32_t resident_base,
    std::uint32_t transpose_base, ap_uint<1>& range_violation) {
#pragma HLS INLINE off
    const double input_limit = static_cast<double>(1ULL << (DATA_I - 1));
    const bool resident_enabled =
        route == snn_v07::FORCE_FULL ||
        (matrix_selector != GEOMETRY_A && route == snn_v07::FORCE_CG);
load_geometry_rows:
    for (int i = 0; i < rows; ++i) {
        const std::uint32_t row_base =
            static_cast<std::uint32_t>(i) * static_cast<std::uint32_t>(columns);
        const std::uint32_t resident_row_base =
            resident_base + static_cast<std::uint32_t>(i) *
                                padded_row_words(columns);
    load_geometry_columns:
        for (int j = 0; j < columns; ++j) {
            const double value =
                source == nullptr ? 0.0 : source[row_base + j];
            const dt fixed = binary64_to_dt(value, input_limit,
                                             range_violation);
            const std::uint32_t index =
                row_base + static_cast<std::uint32_t>(j);
            write_image(image, index, fixed);
            if (transpose) {
                const std::uint32_t transposed_index =
                    static_cast<std::uint32_t>(j) *
                        static_cast<std::uint32_t>(rows) +
                    static_cast<std::uint32_t>(i);
                write_image(transpose_image, transposed_index, fixed);
            }
            if (resident_enabled) {
                write_resident_word(resident_row_base + j, fixed);
                if (transpose)
                    write_resident_word(
                        transpose_base +
                            static_cast<std::uint32_t>(j) *
                                padded_row_words(rows) +
                            static_cast<std::uint32_t>(i),
                        fixed);
            }
        }
    }
}

inline bool cone_table_valid_hls(const std::uint32_t* kinds,
                                  const std::uint32_t* offsets,
                                  const std::uint32_t* lengths,
                                  const double* radii, const double* mus,
                                  const double* centers, int count, int n) {
#pragma HLS INLINE
    if (count < 0 || count > snn_v07::MAX_CONES) return false;
    if (count > 0 && (kinds == nullptr || offsets == nullptr || lengths == nullptr ||
                      radii == nullptr || mus == nullptr || centers == nullptr))
        return false;
    for (int i=0; i<count; ++i) {
        const std::uint32_t kind = kinds == nullptr ? 99U : kinds[i];
        const std::uint32_t off = offsets == nullptr ? 0U : offsets[i];
        const std::uint32_t len = lengths == nullptr ? 0U : lengths[i];
        const double radius = radii == nullptr ? 0.0 : radii[i];
        const double mu = mus == nullptr ? 1.0 : mus[i];
        const double center = centers == nullptr ? 0.0 : centers[i];
        const std::uint32_t minimum = kind == static_cast<std::uint32_t>(snn_v07::SCALED_SOC) ? 3U : 1U;
        if ((kind != static_cast<std::uint32_t>(snn_v07::BALL) &&
             kind != static_cast<std::uint32_t>(snn_v07::SCALED_SOC)) || len < minimum ||
            off > static_cast<std::uint32_t>(n) ||
            len > static_cast<std::uint32_t>(n) - off)
            return false;
        if (!(radius >= 0.0 && radius < 128.0 && center >= -128.0 && center < 128.0))
            return false;
        if (kind == static_cast<std::uint32_t>(snn_v07::SCALED_SOC) &&
            (!(mu >= snn_v07::MIN_CONE_MU && mu < 128.0))) return false;
        if (!snn_v07::cone_norm_fits(kind, len)) return false;
        for (int j=0; j<i; ++j) {
            const std::uint32_t old_off = offsets[j];
            const std::uint32_t old_end = old_off + lengths[j];
            if (off < old_end && old_off < off + len) return false;
        }
    }
    return true;
}

// Fixed reciprocal-root stages ported from the measured v13 implementation.
using norm_sq_t = ap_ufixed<68,20,AP_RND_CONV,AP_SAT>;
using norm_product_t = ap_fixed<64,16,AP_RND_CONV,AP_SAT>;
using rsqrt_t = ap_ufixed<49,25,AP_RND_CONV,AP_SAT>;
using rsqrt_work_t = ap_ufixed<32,2,AP_RND_CONV,AP_SAT>;
using rsqrt_product_t = ap_ufixed<48,4,AP_RND_CONV,AP_SAT>;
using rsqrt_signed_t = ap_fixed<48,4,AP_RND_CONV,AP_SAT>;
using rsqrt_scale_t = ap_ufixed<64,32,AP_RND_CONV,AP_SAT>;
using cone_product_t = ap_fixed<48,16,AP_RND_CONV,AP_SAT>;
constexpr int NORM_WIDTH=68, NORM_FRACTIONAL_BITS=48;
constexpr int RSQRT_WORK_FRACTIONAL_BITS=30, RSQRT_LUT_BITS=5;

inline cone_product_t cone_mul_stage(cone_product_t left,
                                     cone_product_t right) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
    cone_product_t out;
#pragma HLS BIND_OP variable=out op=mul impl=dsp latency=2
    out = static_cast<cone_product_t>(left * right);
    return out;
}

inline cone_product_t cone_add_stage(cone_product_t left,
                                     cone_product_t right) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
    cone_product_t out = static_cast<cone_product_t>(left + right);
    return out;
}

inline norm_product_t norm_square_mul_stage(dt left, dt right) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
    norm_product_t out;
#pragma HLS BIND_OP variable=out op=mul impl=dsp latency=2
    out = static_cast<norm_product_t>(left * right);
    return out;
}

inline norm_product_t norm_cone_mul_stage(cone_product_t left,
                                          cone_product_t right) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
    norm_product_t out;
#pragma HLS BIND_OP variable=out op=mul impl=dsp latency=2
    out = static_cast<norm_product_t>(left * right);
    return out;
}

inline norm_sq_t norm_square_add_stage(norm_sq_t left,
                                       norm_product_t right) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
    norm_sq_t out = static_cast<norm_sq_t>(left + right);
    return out;
}

inline rsqrt_work_t reciprocal_root_seed(ap_uint<RSQRT_LUT_BITS> index) {
#pragma HLS INLINE
    // Midpoint samples of 1/sqrt(m) for 32 equal bins on m in [1,2).
    // A switch keeps the seed visibly constant and synthesizes as a small ROM.
    switch (index) {
        case 0: return static_cast<rsqrt_work_t>(0.99227787671366774);
        case 1: return static_cast<rsqrt_work_t>(0.97735555485044179);
        case 2: return static_cast<rsqrt_work_t>(0.96308682468615359);
        case 3: return static_cast<rsqrt_work_t>(0.94942532655508260);
        case 4: return static_cast<rsqrt_work_t>(0.93632917756904455);
        case 5: return static_cast<rsqrt_work_t>(0.92376043070340108);
        case 6: return static_cast<rsqrt_work_t>(0.91168461167710357);
        case 7: return static_cast<rsqrt_work_t>(0.90007032074081916);
        case 8: return static_cast<rsqrt_work_t>(0.88888888888888884);
        case 9: return static_cast<rsqrt_work_t>(0.87811407991752277);
        case 10: return static_cast<rsqrt_work_t>(0.86772183127462466);
        case 11: return static_cast<rsqrt_work_t>(0.85769002787023585);
        case 12: return static_cast<rsqrt_work_t>(0.84799830400508802);
        case 13: return static_cast<rsqrt_work_t>(0.83862786937753464);
        case 14: return static_cast<rsqrt_work_t>(0.82956135578434020);
        case 15: return static_cast<rsqrt_work_t>(0.82078268166812329);
        case 16: return static_cast<rsqrt_work_t>(0.81227693210689533);
        case 17: return static_cast<rsqrt_work_t>(0.80403025220736968);
        case 18: return static_cast<rsqrt_work_t>(0.79602975216799132);
        case 19: return static_cast<rsqrt_work_t>(0.78826342253143455);
        case 20: return static_cast<rsqrt_work_t>(0.78072005835882663);
        case 21: return static_cast<rsqrt_work_t>(0.77338919123653083);
        case 22: return static_cast<rsqrt_work_t>(0.76626102817692110);
        case 23: return static_cast<rsqrt_work_t>(0.75932639660199919);
        case 24: return static_cast<rsqrt_work_t>(0.75257669470687782);
        case 25: return static_cast<rsqrt_work_t>(0.74600384659225105);
        case 26: return static_cast<rsqrt_work_t>(0.73960026163363879);
        case 27: return static_cast<rsqrt_work_t>(0.73335879762256906);
        case 28: return static_cast<rsqrt_work_t>(0.72727272727272729);
        case 29: return static_cast<rsqrt_work_t>(0.72133570773394584);
        case 30: return static_cast<rsqrt_work_t>(0.71554175279993271);
        default: return static_cast<rsqrt_work_t>(0.70988520753289108);
    }
}

// Keep each Newton multiply in its own registered, reusable unit.  The
// explicit latency binding prevents the scheduler from folding adjacent
// dependent multiplies into one combinational stage at the 200 MHz target.
rsqrt_product_t rsqrt_square_stage(rsqrt_work_t value) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
#pragma HLS LATENCY min = 1
    rsqrt_product_t out;
#pragma HLS BIND_OP variable=out op=mul impl=dsp latency=1
    out = static_cast<rsqrt_product_t>(value * value);
    return out;
}

rsqrt_product_t rsqrt_mantissa_product_stage(
    rsqrt_work_t mantissa, rsqrt_product_t y_sq) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
#pragma HLS LATENCY min = 1
    rsqrt_product_t out;
#pragma HLS BIND_OP variable=out op=mul impl=dsp latency=1
    out = static_cast<rsqrt_product_t>(mantissa * y_sq);
    return out;
}

rsqrt_signed_t rsqrt_half_product_stage(
    rsqrt_product_t xy_sq, rsqrt_signed_t one_half) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
#pragma HLS LATENCY min = 1
    rsqrt_signed_t out;
#pragma HLS BIND_OP variable=out op=mul impl=dsp latency=1
    out = static_cast<rsqrt_signed_t>(
        static_cast<rsqrt_signed_t>(xy_sq) * one_half);
    return out;
}

rsqrt_signed_t rsqrt_correction_stage(
    rsqrt_signed_t one_point_five, rsqrt_signed_t half_xy_sq) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
#pragma HLS LATENCY min = 1
    return static_cast<rsqrt_signed_t>(one_point_five - half_xy_sq);
}

rsqrt_signed_t rsqrt_update_stage(
    rsqrt_work_t y, rsqrt_signed_t correction) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
#pragma HLS LATENCY min = 1
    rsqrt_signed_t out;
#pragma HLS BIND_OP variable=out op=mul impl=dsp latency=1
    out = static_cast<rsqrt_signed_t>(
        static_cast<rsqrt_signed_t>(y) * correction);
    return out;
}

// The Newton update is rounded/saturated back to the work format before the
// next iteration. Keep that conversion in its own registered unit so it does
// not share a clock with the update multiply or the recurrence storage.
rsqrt_work_t rsqrt_update_commit_stage(rsqrt_signed_t value) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
#pragma HLS LATENCY min = 1
    return static_cast<rsqrt_work_t>(value);
}

rsqrt_t reciprocal_root(norm_sq_t norm_sq) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION instances = mul limit = 4 operation
    // The caller performs the <= state-LSB-squared guard.  Normalize the raw
    // unsigned value as norm_sq = mantissa * 2^exponent with mantissa in
    // [1,2); no floating-point conversion exists on this path.
    const ap_uint<NORM_WIDTH> bits = norm_sq.range(NORM_WIDTH - 1, 0);
    int msb = -1;
find_norm_msb:
    for (int bit = NORM_WIDTH - 1; bit >= 0; --bit) {
#pragma HLS LOOP_TRIPCOUNT min = 68 max = 68
        if (msb < 0 && bits[bit]) msb = bit;
    }
    if (msb < 0) return static_cast<rsqrt_t>(0);

    int exponent = msb - NORM_FRACTIONAL_BITS;
    norm_sq_t normalized = norm_sq;
    if (exponent >= 0)
        normalized = static_cast<norm_sq_t>(norm_sq >> exponent);
    else
        normalized = static_cast<norm_sq_t>(norm_sq << (-exponent));
    rsqrt_work_t mantissa = static_cast<rsqrt_work_t>(normalized);
    // Convergent rounding can carry a near-two mantissa to the endpoint.
    if (mantissa >= static_cast<rsqrt_work_t>(2)) {
        mantissa = static_cast<rsqrt_work_t>(mantissa >> 1);
        ++exponent;
    }
    const ap_uint<RSQRT_LUT_BITS> lut_index = mantissa.range(
        RSQRT_WORK_FRACTIONAL_BITS - 1,
        RSQRT_WORK_FRACTIONAL_BITS - RSQRT_LUT_BITS);
    const rsqrt_work_t y0 = reciprocal_root_seed(lut_index);

    const rsqrt_signed_t one_half = static_cast<rsqrt_signed_t>(0.5);
    const rsqrt_signed_t one_point_five = static_cast<rsqrt_signed_t>(1.5);
    // This recurrence is fixed at two Newton steps. Spell both steps out so
    // the registered conversion between them is visible and no loop-carried
    // y store can be chained with the final update and saturation network.
    const rsqrt_product_t y0_sq = rsqrt_square_stage(y0);
    const rsqrt_product_t xy0_sq = rsqrt_mantissa_product_stage(
        mantissa, y0_sq);
    const rsqrt_signed_t half_xy0_sq = rsqrt_half_product_stage(
        xy0_sq, one_half);
    const rsqrt_signed_t correction0 = rsqrt_correction_stage(
        one_point_five, half_xy0_sq);
    const rsqrt_signed_t next0 = rsqrt_update_stage(y0, correction0);
    const rsqrt_work_t y1 = rsqrt_update_commit_stage(next0);

    const rsqrt_product_t y1_sq = rsqrt_square_stage(y1);
    const rsqrt_product_t xy1_sq = rsqrt_mantissa_product_stage(
        mantissa, y1_sq);
    const rsqrt_signed_t half_xy1_sq = rsqrt_half_product_stage(
        xy1_sq, one_half);
    const rsqrt_signed_t correction1 = rsqrt_correction_stage(
        one_point_five, half_xy1_sq);
    const rsqrt_signed_t next1 = rsqrt_update_stage(y1, correction1);
    const rsqrt_work_t y = rsqrt_update_commit_stage(next1);

    // exponent = 2*q + r, r in {0,1}.  Floor division is spelled explicitly
    // because C++ integer division truncates negative values toward zero.
    const int half_exponent = exponent >= 0
        ? exponent / 2 : -((-exponent + 1) / 2);
    const bool odd_exponent = exponent - 2 * half_exponent != 0;
    rsqrt_scale_t scaled = static_cast<rsqrt_scale_t>(y);
    if (odd_exponent) {
        const rsqrt_scale_t inverse_sqrt_two =
            static_cast<rsqrt_scale_t>(0.70710678118654752);
        scaled = static_cast<rsqrt_scale_t>(scaled * inverse_sqrt_two);
    }
    if (half_exponent >= 0)
        scaled = static_cast<rsqrt_scale_t>(scaled >> half_exponent);
    else
        scaled = static_cast<rsqrt_scale_t>(scaled << (-half_exponent));
    return static_cast<rsqrt_t>(scaled);
}

// Evaluate one block into scratch. Return its Euclidean reset distance
// squared, before the axial state cast, as in v13. Strict comparison in the
// sweep retains first-maximal row/lower/upper/cone ordering.
inline norm_sq_t cone_project(const dt* x, dt* projected, int q,
                              ap_uint<1>& range_violation) {
#pragma HLS INLINE off
    const int off = cone_offset[q], len = cone_length[q];
    const bool ball = cone_kind[q] == snn_v07::BALL;
    norm_sq_t n2 = 0;
cone_norm_square:
    for (int j = ball ? 0 : 1; j < len; ++j) {
#pragma HLS PIPELINE II = 1
        const dt z = cast_dt(static_cast<acc_t>(x[off+j]) -
                       (ball ? static_cast<acc_t>(cone_center[q]) : static_cast<acc_t>(0)), range_violation);
        const norm_product_t product = norm_square_mul_stage(z, z);
        n2 = norm_square_add_stage(n2, product);
    }
    const bool guarded = n2 <= static_cast<norm_sq_t>(1.0/281474976710656.0);
    const rsqrt_t inverse = guarded ? static_cast<rsqrt_t>(0) : reciprocal_root(n2);
    const cone_product_t zn = guarded ? static_cast<cone_product_t>(0) :
        cone_mul_stage(static_cast<cone_product_t>(n2),
                       static_cast<cone_product_t>(inverse));
    const cone_product_t norm = zn;
    cone_product_t ratio = static_cast<cone_product_t>(1);
    cone_product_t axial = static_cast<cone_product_t>(x[off]);
    bool changed = false;
    if (ball) {
        if (norm > static_cast<cone_product_t>(cone_radius[q])) {
            ratio = cone_mul_stage(static_cast<cone_product_t>(cone_radius[q]),
                                   static_cast<cone_product_t>(inverse));
            changed = true;
        }
    } else {
        const cone_product_t mu = static_cast<cone_product_t>(cone_mu[q]);
        const cone_product_t rhs = cone_mul_stage(
            mu, static_cast<cone_product_t>(x[off]));
        if (norm > rhs) {
            const cone_product_t numerator = cone_add_stage(
                static_cast<cone_product_t>(x[off]), cone_mul_stage(mu, norm));
            if (numerator <= 0) { axial=0; ratio=0; }
            else {
                const cone_product_t denom = cone_add_stage(
                    static_cast<cone_product_t>(1), cone_mul_stage(mu, mu));
                axial = static_cast<cone_product_t>(numerator/denom);
                const cone_product_t zn_new = cone_mul_stage(mu, axial);
                ratio = cone_mul_stage(zn_new,
                                       static_cast<cone_product_t>(inverse));
            }
            changed = true;
        }
    }
    norm_sq_t distance = 0;
cone_distance_square:
    for (int j=0; j<len; ++j) {
#pragma HLS PIPELINE II = 1
        cone_product_t raw = static_cast<cone_product_t>(x[off+j]);
        if (changed) {
            if (!ball && j==0) raw=axial;
            else {
                const cone_product_t center = ball ? static_cast<cone_product_t>(cone_center[q]) : static_cast<cone_product_t>(0);
                const cone_product_t z=static_cast<cone_product_t>(cast_dt(
                    static_cast<acc_t>(x[off+j])-static_cast<acc_t>(center),range_violation));
                raw=cone_add_stage(center, cone_mul_stage(ratio, z));
                // The v13 lateral coordinate enters F24 before scoring.
                raw=static_cast<acc_t>(cast_dt(raw,range_violation));
            }
        }
        projected[j]=cast_dt(raw,range_violation);
        const cone_product_t delta=static_cast<cone_product_t>(raw-static_cast<cone_product_t>(x[off+j]));
        const norm_product_t term=norm_cone_mul_stage(delta, delta);
        distance=norm_square_add_stage(distance, term);
    }
    return distance;
}

inline norm_sq_t positive_score_square(acc_t value) {
#pragma HLS INLINE
    if (value <= 0) return static_cast<norm_sq_t>(0);
    return static_cast<norm_sq_t>(value * value);
}

inline void load_cns_scale(const double* cns_source,
                           const double* scale_source, int rows,
                           ap_uint<1>& cns_range,
                           ap_uint<1>& scale_range) {
#pragma HLS INLINE off
// Both vectors use the same binary64-to-fixed pipeline.  Keep the two
// sequential call sites on one converter instance.
#pragma HLS ALLOCATION function instances=binary64_to_dt limit=1
    const double input_limit = static_cast<double>(1ULL << (DATA_I - 1));
load_cns_scale_shared:
    for (int i = 0; i < rows; ++i) {
        const double cns_value =
            cns_source == nullptr ? 0.0 : cns_source[i];
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
    double upper_f, int requested_route, const std::uint32_t* cone_kinds,
    const std::uint32_t* cone_offsets, const std::uint32_t* cone_lengths,
    const double* cone_radii, const double* cone_mus,
    const double* cone_centers, int cone_count) {
#pragma HLS INLINE off
    if (!dimensions_valid(n, m) || n_iters <= 0 || projmax <= 0 ||
        (has_lower != 0 && has_lower != 1) ||
        (has_upper != 0 && has_upper != 1))
        return snn_v07::ERR_BAD_DIMENSIONS;
    if (!cone_table_valid_hls(cone_kinds, cone_offsets, cone_lengths,
                              cone_radii, cone_mus, cone_centers,
                              cone_count, n))
        return snn_v07::ERR_BAD_CONES;

    const int route = choose_route(n, m, requested_route);
    if (route < 0) return snn_v07::ERR_CAPACITY;

    // All validation above is complete before changing the current
    // configuration.  A failed FORCE request therefore leaves it intact.
    configured_n = n;
    configured_m = m;
    configured_route = route;
    configured_iters = n_iters;
    configured_projmax = projmax;
    configured_has_lower = has_lower;
    configured_has_upper = has_upper;
    configured_k0 = static_cast<dt>(k0_f);
    configured_native_k0 = static_cast<acc_t>(k0_f);
    configured_ctol = static_cast<dt>(ctol_f);
    configured_lower = static_cast<dt>(lower_f);
    configured_upper = static_cast<dt>(upper_f);
    set_offsets(n, m, route);

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

    configured_cone_count = cone_count;
    for (int q = 0; q < snn_v07::MAX_CONES; ++q) {
        cone_offset[q] = q < cone_count ? cone_offsets[q] : 0;
        cone_length[q] = q < cone_count ? cone_lengths[q] : 0;
        cone_kind[q] = q < cone_count ? static_cast<snn_v07::ConeKind>(cone_kinds[q]) : snn_v07::BALL;
        cone_radius[q] = q < cone_count ? static_cast<dt>(cone_radii[q]) : static_cast<dt>(0);
        cone_mu[q] = q < cone_count ? static_cast<dt>(cone_mus[q]) : static_cast<dt>(1);
        cone_center[q] = q < cone_count ? static_cast<dt>(cone_centers[q]) : static_cast<dt>(0);
    }

    is_configured = 1;
    is_stopped = 0;
    state_has_committed = 0;
    return snn_v07::ERR_OK;
}

inline int refresh_a(int n, int m) {
#pragma HLS INLINE off
    if (!is_configured) return snn_v07::ERR_NOT_CONFIGURED;
    if (n != configured_n || m != configured_m || !dimensions_valid(n, m))
        return snn_v07::ERR_BAD_DIMENSIONS;

    a_range_violation = 0;
    return snn_v07::ERR_OK;
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
            } else if (tail == snn_v07::REPEAT_LAST) {
                shifted[i] = x[n - 1];
            } else {
                // HOLD_TAIL keeps each original tail entry in place.
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
    telemetry_out[0] = snn_v07::TELEMETRY_MAGIC;
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

inline int solve_current(const double* b_in, const double* d_in,
                         const double* x0_in,
                         const std::uint32_t* A_ddr,
                         const std::uint32_t* C_ddr,
                         const std::uint32_t* Ct_ddr,
                         const std::uint32_t* G_ddr,
                         long long* x_raw_out,
                         unsigned long long* telemetry_out, int start_mode,
                         int shift_stride, int tail_policy) {
#pragma HLS INLINE off
// residual_matvec, hessian_rows, and the event updates all call the same
// sequential line-buffer reader.  One instance is sufficient and avoids a
// second 2.6k-LUT reader without changing any II=1 arithmetic loop.
#pragma HLS ALLOCATION function instances=load_line_buffer limit=1
    if (!is_configured) return snn_v07::ERR_NOT_CONFIGURED;
    if (start_mode < snn_v07::RESIDENT_WARM ||
        start_mode > snn_v07::COLD_ZERO || shift_stride < 0 ||
        shift_stride > configured_n ||
        (tail_policy != snn_v07::HOLD_TAIL &&
         tail_policy != snn_v07::REPEAT_LAST))
        return snn_v07::ERR_BAD_DIMENSIONS;

    dt b[MAXN];
    dt d[MAXM];
    dt x[MAXN];
    dt shifted[MAXN];
    dt Ax[MAXN];
    dt residual[MAXM];
    dt cone_scratch[MAXN];
    dt cone_winner[MAXN];
    using gradient_sum_t = decltype(acc_t() + acc_t());
    using gradient_product_t = decltype(acc_t() * gradient_sum_t());
    gradient_product_t gradient_product[MAXN];
    acc_t gradient_next[MAXN];
    dt cone_gradient_committed[MAXN];
    // Geometry is fetched a row group at a time.  These buffers are the
    // BRAM-side banked view used by all six II=1 arithmetic loops; no AXI or
    // URAM access appears in those loops themselves.
    // One partitioned BRAM line-buffer view is reused for A, C, G, and Ct.
    // The loader is out-of-line, so these sequential uses do not replicate
    // the six DDR reader datapaths present in the round-B report.
    dt matrix_rows[RG][MAX_ROW_DIM];
#pragma HLS ARRAY_PARTITION variable = b cyclic factor = 4 dim = 1
#pragma HLS ARRAY_PARTITION variable = d cyclic factor = 4 dim = 1
#pragma HLS ARRAY_PARTITION variable = x cyclic factor = 4 dim = 1
#pragma HLS ARRAY_PARTITION variable = residual cyclic factor = 4 dim = 1
#pragma HLS ARRAY_PARTITION variable = matrix_rows complete dim = 1
#pragma HLS ARRAY_PARTITION variable = matrix_rows cyclic factor = 4 dim = 2
#pragma HLS bind_storage variable = gradient_product type = ram_1p impl = bram
#pragma HLS bind_storage variable = gradient_next type = ram_1p impl = bram

    ap_uint<1> input_range_violation = geometry_range_violation;
    if (start_mode == snn_v07::RESIDENT_WARM && !state_has_committed)
        input_range_violation |= initial_range_violation;
    ap_uint<1> scalar_range_violation = 0;
    ap_uint<1> gradient_range_violation = 0;
    ap_uint<1> lane_range_violation[UF];
#pragma HLS ARRAY_PARTITION variable = lane_range_violation complete
init_range_flags:
    for (int lane = 0; lane < UF; ++lane) lane_range_violation[lane] = 0;

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
        if (start_mode == snn_v07::RESIDENT_WARM) {
            x[i] = resident_state[i];
        } else if (start_mode == snn_v07::HOST_X0) {
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
    const norm_sq_t ctol_sq = positive_score_square(static_cast<acc_t>(ctol));
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
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 120000
        if (configured_cone_count > 0) {
            // Native cone mode follows the measured v13 recurrence.  The
            // resident row path below deliberately keeps its v06 accumulator
            // shape, so this branch makes the product/add round points
            // explicit without changing cones-off parity.
        cone_hessian_rows:
            for (int i0 = 0; i0 < configured_n; i0 += RG) {
                acc_t acc[RG];
#pragma HLS ARRAY_PARTITION variable = acc complete
                acc_t cone_hessian_terms[RG * UF];
#pragma HLS ARRAY_PARTITION variable = cone_hessian_terms complete
                for (int group = 0; group < RG; ++group) acc[group] = 0;
                load_row_group(matrix_rows, i0, configured_n, configured_n,
                               configured_route == snn_v07::FORCE_FULL, offset_a,
                               A_ddr);
            cone_hessian_columns:
                for (int j0 = 0; j0 < configured_n; j0 += UF) {
                cone_hessian_products:
                    for (int flat = 0; flat < RG * UF; ++flat) {
#pragma HLS PIPELINE II = 1
                        const int group = flat / UF;
                        const int lane = flat % UF;
                        const int row = i0 + group;
                        const int column = j0 + lane;
                        if (row < configured_n && column < configured_n) {
                            cone_hessian_terms[flat] = cone_product_mul_stage(
                                static_cast<acc_t>(matrix_rows[group][column]),
                                static_cast<acc_t>(x[column]));
                        }
                    }
                cone_hessian_accumulate:
                    for (int flat = 0; flat < RG * UF; ++flat) {
#pragma HLS PIPELINE II = 1
                        const int group = flat / UF;
                        const int lane = flat % UF;
                        const int row = i0 + group;
                        const int column = j0 + lane;
                        if (row < configured_n && column < configured_n) {
                            acc[group] = cone_product_add_stage(
                                acc[group], cone_hessian_terms[flat]);
                        }
                    }
                }
                for (int group = 0; group < RG; ++group) {
                    const int row = i0 + group;
                    if (row < configured_n)
                        gradient_next[row] = cone_product_add_stage(
                            acc[group], static_cast<acc_t>(b[row]));
                }
            }
        cone_gradient_step:
            for (int i = 0; i < configured_n; ++i) {
#pragma HLS PIPELINE II = 1
                const acc_t scaled = cone_product_mul_stage(
                    configured_native_k0, gradient_next[i]);
                const acc_t updated = cone_product_add_stage(
                    static_cast<acc_t>(x[i]), -scaled);
                cone_gradient_committed[i] = cone_gradient_commit_stage(
                    updated, gradient_range_violation);
            }
        cone_gradient_store:
            for (int i = 0; i < configured_n; ++i) {
#pragma HLS PIPELINE II = 1
                x[i] = cone_gradient_committed[i];
            }
        } else {
    hessian_rows:
        for (int i0 = 0; i0 < configured_n; i0 += RG) {
            acc_t acc[RG];
#pragma HLS ARRAY_PARTITION variable = acc complete
            for (int group = 0; group < RG; ++group) acc[group] = 0;
            load_row_group(matrix_rows, i0, configured_n, configured_n,
                           configured_route == snn_v07::FORCE_FULL, offset_a,
                           A_ddr);
        hessian_columns:
            for (int j0 = 0; j0 < configured_n; j0 += UF) {
#pragma HLS PIPELINE II = 1
                dt x_chunk[UF];
#pragma HLS ARRAY_PARTITION variable = x_chunk complete
                for (int lane = 0; lane < UF; ++lane) {
                    const int column = j0 + lane;
                    x_chunk[lane] = column < configured_n ? x[column]
                                                          : static_cast<dt>(0);
                }
                V07_UNROLL_RG
                for (int group = 0; group < RG; ++group) {
                    const int row = i0 + group;
                    acc_t partial = 0;
                    V07_UNROLL_UF
                    for (int lane = 0; lane < UF; ++lane) {
                        const int column = j0 + lane;
                        if (row < configured_n && column < configured_n)
                            partial += matrix_rows[group][column] * x_chunk[lane];
                    }
                    acc[group] += partial;
                }
            }
            for (int group = 0; group < RG; ++group) {
                if (i0 + group < configured_n)
                    Ax[i0 + group] =
                        cast_dt(acc[group], lane_range_violation[group]);
            }
        }

    gradient_product_step:
        for (int i = 0; i < configured_n; ++i) {
#pragma HLS PIPELINE II = 1
            gradient_product[i] =
                static_cast<acc_t>(k0) *
                (static_cast<acc_t>(Ax[i]) + static_cast<acc_t>(b[i]));
        }
    gradient_apply_step:
        for (int i = 0; i < configured_n; ++i) {
#pragma HLS PIPELINE II = 1
            gradient_next[i] = static_cast<acc_t>(
                static_cast<acc_t>(x[i]) - gradient_product[i]);
        }
    gradient_commit_step:
        for (int i = 0; i < configured_n; ++i) {
#pragma HLS PIPELINE II = 1
            x[i] = cast_dt(gradient_next[i], gradient_range_violation);
        }
        }

        residual_matvec(matrix_rows, residual, x, d, configured_m,
                        configured_n,
                        configured_route != snn_v07::FORCE_STREAM, offset_c,
                        C_ddr, lane_range_violation);

        bool left_sweep = false;
    projection_loop:
        for (int ordinal = 0; ordinal < configured_projmax; ++ordinal) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 10000
            int winner = 0;
            int kind = 0;
            acc_t best = static_cast<acc_t>(residual[0]) *
                         static_cast<acc_t>(resident_scale[0]);
        scan_rows:
            for (int i = 1; i < configured_m; ++i) {
#pragma HLS PIPELINE II = 1
                const acc_t score = static_cast<acc_t>(residual[i]) *
                                    static_cast<acc_t>(resident_scale[i]);
                if (score > best) {
                    best = score;
                    winner = i;
                }
            }
        scan_lower:
            for (int i = 0; i < configured_n; ++i) {
#pragma HLS PIPELINE II = 1
                if (configured_has_lower) {
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
            for (int i = 0; i < configured_n; ++i) {
#pragma HLS PIPELINE II = 1
                if (configured_has_upper) {
                    const acc_t score = static_cast<acc_t>(x[i]) -
                                        static_cast<acc_t>(upper);
                    if (score > best) {
                        best = score;
                        winner = i;
                        kind = 2;
                    }
                }
            }
            norm_sq_t best_sq=positive_score_square(best);
        scan_cones:
            for (int q=0; q<configured_cone_count; ++q) {
                const norm_sq_t score=cone_project(x,cone_scratch,q,scalar_range_violation);
                if (score>best_sq && score>ctol_sq) {
                    best_sq=score; winner=q; kind=3;
                }
            }
            if (kind != 3 && best <= static_cast<acc_t>(ctol)) {
                left_sweep = true;
                break;
            }

            ap_uint<64> candidate = 0;
            if (kind == 0) {
                const dt step = residual[winner] / resident_cns[winner];
                load_line_buffer(matrix_rows[0], winner, configured_m,
                                 configured_n,
                                 configured_route != snn_v07::FORCE_STREAM,
                                 offset_c, C_ddr);
            update_x_row:
                for (int j0 = 0; j0 < configured_n; j0 += UF) {
#pragma HLS PIPELINE II = 1
                    for (int lane = 0; lane < UF; ++lane) {
                        const int j = j0 + lane;
                        if (j < configured_n) {
                            const acc_t next =
                                static_cast<acc_t>(x[j]) -
                                static_cast<acc_t>(step) *
                                    static_cast<acc_t>(matrix_rows[0][j]);
                            x[j] = cast_dt(next, lane_range_violation[lane]);
                        }
                    }
                }
                load_line_buffer(matrix_rows[0], winner, configured_m,
                                 configured_m,
                                 configured_route != snn_v07::FORCE_STREAM,
                                 offset_g, G_ddr);
            update_residual_row:
                for (int i0 = 0; i0 < configured_m; i0 += UF) {
#pragma HLS PIPELINE II = 1
                    for (int lane = 0; lane < UF; ++lane) {
                        const int i = i0 + lane;
                        if (i < configured_m) {
                            const acc_t next =
                                static_cast<acc_t>(residual[i]) -
                                static_cast<acc_t>(step) *
                                    static_cast<acc_t>(matrix_rows[0][i]);
                            residual[i] =
                                cast_dt(next, lane_range_violation[lane]);
                        }
                    }
                }
                ++row_events;
                candidate = static_cast<ap_uint<64>>(winner);
            } else if (kind != 3) {
                const dt delta =
                    kind == 1 ? cast_dt(best, scalar_range_violation)
                              : cast_dt(-best, scalar_range_violation);
                x[winner] = cast_dt(static_cast<acc_t>(x[winner]) +
                                        static_cast<acc_t>(delta),
                                    scalar_range_violation);
                // Ct is stored row-major by facet, so one row load makes the
                // following residual update a pure BRAM calculation.
                load_line_buffer(matrix_rows[0], winner, configured_n,
                                 configured_m,
                                 configured_route != snn_v07::FORCE_STREAM,
                                 offset_ct, Ct_ddr);
            update_residual_facet:
                for (int i0 = 0; i0 < configured_m; i0 += UF) {
#pragma HLS PIPELINE II = 1
                    for (int lane = 0; lane < UF; ++lane) {
                        const int i = i0 + lane;
                        if (i < configured_m) {
                            const acc_t next =
                                static_cast<acc_t>(residual[i]) +
                                static_cast<acc_t>(delta) *
                                    static_cast<acc_t>(matrix_rows[0][i]);
                            residual[i] =
                                cast_dt(next, lane_range_violation[lane]);
                        }
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

            if (kind == 3) {
                cone_project(x,cone_scratch,winner,scalar_range_violation);
                for (int j=0; j<static_cast<int>(cone_length[winner]); ++j)
                    cone_winner[j]=cone_scratch[j];
                for (int j=0; j<static_cast<int>(cone_length[winner]); ++j)
                    x[cone_offset[winner]+j]=cone_winner[j];
                // A nonlinear reset changes a whole block; refresh all row
                // residuals through the same fixed resident matvec before WTA.
                residual_matvec(matrix_rows,residual,x,d,configured_m,configured_n,
                    configured_route != snn_v07::FORCE_STREAM,offset_c,C_ddr,lane_range_violation);
                candidate=static_cast<ap_uint<64>>(configured_m+2*configured_n+winner);
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
            residual_matvec(matrix_rows, residual, x, d, configured_m,
                            configured_n,
                            configured_route != snn_v07::FORCE_STREAM,
                            offset_c, C_ddr, lane_range_violation);

            acc_t maximum = 0;
        cap_scan_rows:
            for (int i = 0; i < configured_m; ++i) {
#pragma HLS PIPELINE II = 1
                const acc_t score = static_cast<acc_t>(residual[i]) *
                                    static_cast<acc_t>(resident_scale[i]);
                if (score > maximum) maximum = score;
            }
        cap_scan_lower:
            for (int i = 0; i < configured_n; ++i) {
#pragma HLS PIPELINE II = 1
                if (configured_has_lower) {
                    const acc_t score = static_cast<acc_t>(lower) -
                                        static_cast<acc_t>(x[i]);
                    if (score > maximum) maximum = score;
                }
            }
        cap_scan_upper:
            for (int i = 0; i < configured_n; ++i) {
#pragma HLS PIPELINE II = 1
                if (configured_has_upper) {
                    const acc_t score = static_cast<acc_t>(x[i]) -
                                        static_cast<acc_t>(upper);
                    if (score > maximum) maximum = score;
                }
            }
            norm_sq_t maximum_sq=positive_score_square(maximum);
            for (int q=0; q<configured_cone_count; ++q) {
                const norm_sq_t score=cone_project(x,cone_scratch,q,scalar_range_violation);
                if (score>maximum_sq) maximum_sq=score;
            }
            const bool unsettled = configured_cone_count > 0
                ? maximum_sq > ctol_sq : maximum > static_cast<acc_t>(ctol);
            if (unsettled) {
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
    const ap_uint<1> any_range_violation =
        input_range_violation | scalar_range_violation |
        gradient_range_violation;
    write_telemetry(x_raw_out, telemetry_out, x, configured_n, status,
                    executed, configured_iters, total_events, row_events,
                    lower_events, upper_events, cap_rechecks, event_digest,
                    first_candidate, last_candidate, any_range_violation);
    return snn_v07::ERR_OK;
}

struct DispatchResult {
    int error = snn_v07::ERR_OK;
    int status = 0;
    int iterations = 0;
    int route = snn_v07::AUTO;
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
    int has_upper, double upper_f, const std::uint32_t* cone_kinds,
    const std::uint32_t* cone_offsets, const std::uint32_t* cone_lengths,
    const double* cone_radii, const double* cone_mus, const double* cone_centers,
    int cone_count) {
#pragma HLS INLINE off
// Keep all configuration and refresh geometry calls at this hierarchy level.
// The allocation limit then covers A, C, Ct, G, and refresh-A uniformly.
#pragma HLS ALLOCATION function instances=load_geometry_matrix limit=1
// The geometry loader and CNS/scale loader use the same conversion pipeline.
#pragma HLS ALLOCATION function instances=binary64_to_dt limit=1
    DispatchResult result;
    result.route = configured_route;
    if (command == snn_v07::CONFIGURE) {
        result.error = configure_geometry(
            cns_cfg, row_scale_cfg, x0_in, n, m, k0_f, ctol_f, n_iters,
            projmax, has_lower, lower_f, has_upper, upper_f,
            requested_route, cone_kinds, cone_offsets, cone_lengths,
            cone_radii, cone_mus, cone_centers, cone_count);
        if (result.error == snn_v07::ERR_OK) {
            load_geometry_matrix(A_cfg, A_ddr, nullptr, GEOMETRY_A, 0, n, n,
                                 configured_route, offset_a, 0,
                                 a_range_violation);
            load_geometry_matrix(C_cfg, C_ddr, Ct_ddr, GEOMETRY_C, 1, m, n,
                                 configured_route, offset_c, offset_ct,
                                 c_range_violation);
            load_geometry_matrix(G_cfg, G_ddr, nullptr, GEOMETRY_G, 0, m, m,
                                 configured_route, offset_g, 0,
                                 g_range_violation);
            write_geometry_range_flag();
        }
        result.route = configured_route;
        return result;
    }
    if (command == snn_v07::REFRESH_A) {
        result.error = refresh_a(n, m);
        if (result.error == snn_v07::ERR_OK) {
            load_geometry_matrix(A_cfg, A_ddr, nullptr, GEOMETRY_A, 0, n, n,
                                 configured_route, offset_a, 0,
                                 a_range_violation);
            write_geometry_range_flag();
        }
        result.route = configured_route;
        return result;
    }
    if (command == snn_v07::SOLVE) {
        if (is_stopped) {
            result.error = snn_v07::ERR_STOPPED;
            result.route = configured_route;
            return result;
        }
        result.error = solve_current(b_in, d_in, x0_in, A_ddr, C_ddr, Ct_ddr, G_ddr,
                x_raw_out, telemetry_out, start_mode, shift_stride, tail_policy);
        result.route = configured_route;
        if (telemetry_out != nullptr) {
            result.status = static_cast<int>(telemetry_out[1]);
            result.iterations = static_cast<int>(telemetry_out[3]);
        }
        return result;
    }
    if (command == snn_v07::STOP) {
        is_stopped = 1;
        result.error = snn_v07::ERR_STOPPED;
        result.route = configured_route;
        return result;
    }
    result.error = snn_v07::ERR_BAD_COMMAND;
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

extern "C" void snn_qp_v07(
    const double* A_cfg, const double* C_cfg, const double* G_cfg,
    const double* cns_cfg, const double* row_scale_cfg, const double* b_in,
    const double* d_in, const double* x0_in, std::uint32_t* A_ddr,
    std::uint32_t* C_ddr, std::uint32_t* Ct_ddr, std::uint32_t* G_ddr,
    long long* x_raw_out, unsigned long long* telemetry_out,
    volatile const std::uint32_t* mb_in, volatile std::uint32_t* mb_out,
    int command, int launch_mode, int route_mode, int start_mode,
    int shift_stride, int tail_policy, int n, int m, double k0_f,
    double ctol_f, int n_iters, int projmax, int has_lower, double lower_f,
    int has_upper, double upper_f, const std::uint32_t* cone_kinds,
    const std::uint32_t* cone_offsets, const std::uint32_t* cone_lengths,
    const double* cone_radii, const double* cone_mus, const double* cone_centers,
    int cone_count) {
#pragma HLS INTERFACE m_axi port = A_cfg bundle = g0 offset = slave depth = 1048576
#pragma HLS INTERFACE m_axi port = C_cfg bundle = g0 offset = slave depth = 1048576
#pragma HLS INTERFACE m_axi port = G_cfg bundle = g0 offset = slave depth = 1048576
#pragma HLS INTERFACE m_axi port = cns_cfg bundle = g0 offset = slave depth = 1024
#pragma HLS INTERFACE m_axi port = row_scale_cfg bundle = g0 offset = slave depth = 1024
#pragma HLS INTERFACE m_axi port = A_ddr bundle = g1 offset = slave depth = 1048576
#pragma HLS INTERFACE m_axi port = C_ddr bundle = g1 offset = slave depth = 1048576
#pragma HLS INTERFACE m_axi port = Ct_ddr bundle = g1 offset = slave depth = 1048576
#pragma HLS INTERFACE m_axi port = G_ddr bundle = g1 offset = slave depth = 1048576
#pragma HLS INTERFACE m_axi port = b_in bundle = g2 offset = slave depth = 1024
#pragma HLS INTERFACE m_axi port = d_in bundle = g2 offset = slave depth = 1024
#pragma HLS INTERFACE m_axi port = x0_in bundle = g2 offset = slave depth = 1024
#pragma HLS INTERFACE m_axi port = mb_in bundle = g2 offset = slave depth = 64
#pragma HLS INTERFACE m_axi port = mb_out bundle = g2 offset = slave depth = 64
#pragma HLS INTERFACE m_axi port = x_raw_out bundle = g2 offset = slave depth = 1024
#pragma HLS INTERFACE m_axi port = telemetry_out bundle = g2 offset = slave depth = 16
#pragma HLS INTERFACE m_axi port = cone_kinds bundle = g0 offset = slave depth = 64
#pragma HLS INTERFACE m_axi port = cone_offsets bundle = g0 offset = slave depth = 64
#pragma HLS INTERFACE m_axi port = cone_lengths bundle = g0 offset = slave depth = 64
#pragma HLS INTERFACE m_axi port = cone_radii bundle = g0 offset = slave depth = 64
#pragma HLS INTERFACE m_axi port = cone_mus bundle = g0 offset = slave depth = 64
#pragma HLS INTERFACE m_axi port = cone_centers bundle = g0 offset = slave depth = 64
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
#pragma HLS INTERFACE s_axilite port = cone_kinds bundle = c
#pragma HLS INTERFACE s_axilite port = cone_offsets bundle = c
#pragma HLS INTERFACE s_axilite port = cone_lengths bundle = c
#pragma HLS INTERFACE s_axilite port = cone_radii bundle = c
#pragma HLS INTERFACE s_axilite port = cone_mus bundle = c
#pragma HLS INTERFACE s_axilite port = cone_centers bundle = c
#pragma HLS INTERFACE s_axilite port = cone_count bundle = c
#pragma HLS INTERFACE s_axilite port = return bundle = c

#pragma HLS bind_storage variable = resident_matrix type = ram_2p impl = uram latency = 2
#pragma HLS bind_storage variable = resident_cns type = ram_1p impl = bram
#pragma HLS bind_storage variable = resident_scale type = ram_1p impl = bram
#pragma HLS bind_storage variable = resident_state type = ram_1p impl = bram

    clear_outputs(x_raw_out, telemetry_out, n);

    // ONESHOT executes exactly one scalar command.  PERSISTENT executes the
    // launch command once, then services mailbox sequence changes until STOP.
    if (launch_mode != snn_v07::PERSISTENT) {
        int effective_command = command;
        int effective_start = start_mode;
        int effective_shift = shift_stride;
        int effective_tail = tail_policy;
        int effective_route = route_mode;
        std::uint32_t sequence = 0;
        if (command == snn_v07::SERVE) {
            if (mb_in != nullptr) {
                sequence = mb_in[snn_v07::MAILBOX_SEQUENCE];
                effective_command =
                    static_cast<int>(mb_in[snn_v07::MAILBOX_COMMAND]);
                effective_start =
                    static_cast<int>(mb_in[snn_v07::MAILBOX_START_MODE]);
                effective_shift =
                    static_cast<int>(mb_in[snn_v07::MAILBOX_SHIFT_STRIDE]);
                effective_tail =
                    static_cast<int>(mb_in[snn_v07::MAILBOX_TAIL_POLICY]);
                const std::uint32_t flags = mb_in[snn_v07::MAILBOX_FLAGS];
                if ((flags & snn_v07::FLAG_ROUTE_MASK) != 0)
                    effective_route =
                        static_cast<int>(flags & snn_v07::FLAG_ROUTE_MASK);
            }
        } else if (mb_in != nullptr) {
            sequence = mb_in[snn_v07::MAILBOX_SEQUENCE];
        }
        DispatchResult result;
        if (effective_command == snn_v07::LITMUS) {
            write_litmus_pattern(x_raw_out, telemetry_out, mb_in, n);
            result.error = snn_v07::ERR_OK;
            result.route = configured_route;
        } else {
            result = dispatch_command(
                effective_command, effective_route, effective_start,
                effective_shift, effective_tail, A_cfg, C_cfg, G_cfg,
                cns_cfg, row_scale_cfg, b_in, d_in, x0_in, A_ddr, C_ddr,
                Ct_ddr, G_ddr, x_raw_out, telemetry_out, n, m, k0_f, ctol_f,
                n_iters, projmax, has_lower, lower_f, has_upper, upper_f,
                cone_kinds, cone_offsets, cone_lengths, cone_radii, cone_mus,
                cone_centers, cone_count);
        }
        acknowledge_dispatch(mb_out, sequence, result);
        return;
    }

    std::uint32_t seen_sequence = mb_in == nullptr
                                       ? 0
                                       : mb_in[snn_v07::MAILBOX_SEQUENCE];
    if (command != snn_v07::SERVE) {
        DispatchResult result;
        if (command == snn_v07::LITMUS) {
            write_litmus_pattern(x_raw_out, telemetry_out, mb_in, n);
            result.error = snn_v07::ERR_OK;
            result.route = configured_route;
        } else {
            result = dispatch_command(
                command, route_mode, start_mode, shift_stride, tail_policy,
                A_cfg, C_cfg, G_cfg, cns_cfg, row_scale_cfg, b_in, d_in,
                x0_in, A_ddr, C_ddr, Ct_ddr, G_ddr, x_raw_out,
                telemetry_out, n, m, k0_f, ctol_f, n_iters, projmax,
                has_lower, lower_f, has_upper, upper_f, cone_kinds,
                cone_offsets, cone_lengths, cone_radii, cone_mus, cone_centers,
                cone_count);
        }
        acknowledge_dispatch(mb_out, seen_sequence, result);
        if (command == snn_v07::STOP) return;
    }

persistent_wait:
    volatile unsigned int poll_counter = 0;
    while (!is_stopped) {
        // There is one sequence read per polling interval, not one read per
        // cycle.  ap_wait_n keeps the trip count visible to HLS; the volatile
        // counter also prevents native/synthesis dead-code elimination.
        std::uint32_t sequence =
            mb_in == nullptr ? seen_sequence
                             : mb_in[snn_v07::MAILBOX_SEQUENCE];
        if (sequence != seen_sequence) {
#ifndef __SYNTHESIS__
            // Native emulation uses ordinary host arrays for the payload.  A
            // compiler acquire fence models the device-side visibility point
            // that the volatile sequence load provides on the AXI path.
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
#endif
            int mailbox_command = snn_v07::SERVE;
            int mailbox_start = start_mode;
            int mailbox_shift = shift_stride;
            int mailbox_tail = tail_policy;
            int mailbox_route = route_mode;
            if (mb_in != nullptr) {
                mailbox_command =
                    static_cast<int>(mb_in[snn_v07::MAILBOX_COMMAND]);
                mailbox_start =
                    static_cast<int>(mb_in[snn_v07::MAILBOX_START_MODE]);
                mailbox_shift =
                    static_cast<int>(mb_in[snn_v07::MAILBOX_SHIFT_STRIDE]);
                mailbox_tail =
                    static_cast<int>(mb_in[snn_v07::MAILBOX_TAIL_POLICY]);
                const std::uint32_t flags = mb_in[snn_v07::MAILBOX_FLAGS];
                if ((flags & snn_v07::FLAG_ROUTE_MASK) != 0)
                    mailbox_route =
                        static_cast<int>(flags & snn_v07::FLAG_ROUTE_MASK);
            }
            DispatchResult result;
            if (mailbox_command == snn_v07::LITMUS) {
                write_litmus_pattern(x_raw_out, telemetry_out, mb_in, n);
                result.error = snn_v07::ERR_OK;
                result.route = configured_route;
            } else {
                result = dispatch_command(
                    mailbox_command, mailbox_route, mailbox_start,
                    mailbox_shift, mailbox_tail, A_cfg, C_cfg, G_cfg,
                    cns_cfg, row_scale_cfg, b_in, d_in, x0_in, A_ddr, C_ddr,
                    Ct_ddr, G_ddr, x_raw_out, telemetry_out, n, m, k0_f,
                    ctol_f, n_iters, projmax, has_lower, lower_f, has_upper,
                    upper_f, cone_kinds, cone_offsets, cone_lengths,
                    cone_radii, cone_mus, cone_centers, cone_count);
            }
            acknowledge_dispatch(mb_out, sequence, result);
            seen_sequence = sequence;
            if (mailbox_command == snn_v07::STOP) break;
        }
    poll_wait:
        for (int poll = 0; poll < POLL_CYCLES; ++poll) {
#pragma HLS LOOP_TRIPCOUNT min = 256 max = 256
            poll_counter = static_cast<unsigned int>(poll);
            ap_wait_n(1);
        }
    }
}
