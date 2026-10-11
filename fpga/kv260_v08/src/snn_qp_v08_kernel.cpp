// SNN-QP v0.8 resident command-service kernel.
//
// The command ABI, mailbox, telemetry words, routes, fixed-point formats,
// and recurrence operation order are the v0.6 kernel.  v0.8 evaluates every
// candidate family in one eight-lane scan, reads resident matrix words
// directly in the FULL Hessian, the resident residual matvec, and the
// resident event updates, and keeps the line-buffer path for streamed rows.
// The gradient is one pass.  Geometry images are written once as ap_fixed
// bit patterns, so streamed and resident reads see the same words.

#include "dt.h"
#include "v08_abi.hpp"

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
#error "The v08 resident image is defined as eight 32-bit lanes per word"
#endif

#if MAXN < 1 || MAXM < 1 || MATRIX_WORD_CAP < 1
#error "v08 capacities must be positive"
#endif

#if DATA_W != 32 || DATA_I != 8
#error "v08 ABI requires the v05 ap_fixed<32,8> state"
#endif

#if UF != 4 || RG != 2
#error "The v08 arithmetic shape is fixed at UF=4 and RG=2"
#endif

// Width of the fused candidate scan.  The reduction tree and the cyclic
// partition factors below are written as the literal 8: this HLS version
// does not expand a macro inside an ARRAY_PARTITION factor.
#ifndef SCAN_LANES
#define SCAN_LANES 8
#endif

#if SCAN_LANES != 8
#error "The fused scan is built for SCAN_LANES=8"
#endif

#if (MAXN % SCAN_LANES) != 0 || (MAXM % SCAN_LANES) != 0
#error "Scan groups must stay inside x and residual"
#endif

// HLS does not expand a macro in a raw pragma token.  Keep the factors
// explicit through _Pragma, as in the v05 build, so a controller can inspect
// and change the shape from the compile command.
#define V08_STRINGIFY_(x) #x
#define V08_STRINGIFY(x) V08_STRINGIFY_(x)
#define V08_DO_PRAGMA(x) _Pragma(V08_STRINGIFY(x))
#define V08_UNROLL_UF V08_DO_PRAGMA(HLS UNROLL factor = UF)
#define V08_UNROLL_RG V08_DO_PRAGMA(HLS UNROLL factor = RG)

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
static int configured_route = snn_v08::AUTO;
static int configured_iters = 0;
static int configured_projmax = 0;
static int configured_has_lower = 0;
static int configured_has_upper = 0;
static dt configured_k0 = 0;
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
    if (route == snn_v08::FORCE_FULL) {
        if (full_words(n, m) > static_cast<unsigned long long>(MATRIX_WORD_CAP))
            return false;
        if (full_words(n, m) > 440000ULL) return false;
        if (n == m && n > FULL_SQUARE_CAP) return false;
        return true;
    }
    if (route == snn_v08::FORCE_CG) {
        if (cg_words(n, m) > static_cast<unsigned long long>(MATRIX_WORD_CAP))
            return false;
        if (cg_words(n, m) > 440000ULL) return false;
        if (n == m && n > CG_SQUARE_CAP) return false;
        return true;
    }
    if (route == snn_v08::FORCE_STREAM) return true;
    return false;
}

inline int choose_route(int n, int m, int requested) {
#pragma HLS INLINE
    if (requested == snn_v08::AUTO) {
        if (route_fits(n, m, snn_v08::FORCE_FULL))
            return snn_v08::FORCE_FULL;
        if (route_fits(n, m, snn_v08::FORCE_CG))
            return snn_v08::FORCE_CG;
        if (route_fits(n, m, snn_v08::FORCE_STREAM))
            return snn_v08::FORCE_STREAM;
        return -1;
    }
    if (requested < snn_v08::FORCE_FULL ||
        requested > snn_v08::FORCE_STREAM)
        return -1;
    return route_fits(n, m, requested) ? requested : -1;
}

inline void set_offsets(int n, int m, int route) {
#pragma HLS INLINE
    offset_a = 0;
    offset_c = 0;
    offset_ct = 0;
    offset_g = 0;
    if (route == snn_v08::FORCE_FULL) {
        offset_a = 0;
        offset_c = static_cast<std::uint32_t>(n) * padded_row_words(n);
        offset_ct = offset_c +
                    static_cast<std::uint32_t>(m) * padded_row_words(n);
        offset_g = offset_ct +
                   static_cast<std::uint32_t>(n) * padded_row_words(m);
        resident_words = full_words(n, m);
    } else if (route == snn_v08::FORCE_CG) {
        offset_c = 0;
        offset_ct = static_cast<std::uint32_t>(m) * padded_row_words(n);
        offset_g = offset_ct +
                   static_cast<std::uint32_t>(n) * padded_row_words(m);
        resident_words = cg_words(n, m);
    } else {
        resident_words = 0;
    }
}

// Load one matrix row before an arithmetic loop on the streamed path.  The
// resident URAM read and the DDR image read are separate loops so a resident
// line fill is not scheduled at the image path's memory latency.  Resident
// FULL Hessian rows, resident C/G/Ct consumers, and resident event updates
// read packed words at the use site instead of calling this loader.  A
// resident row is one 256-bit word (eight fixed values) per cycle.
inline void load_line_buffer(
    dt row[MAX_ROW_DIM], int row_index, int row_count, int columns,
    int use_resident, std::uint32_t resident_base,
    const std::uint32_t* image) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = row cyclic factor = 8 dim = 1
    const std::uint32_t resident_row_base =
        resident_base + static_cast<std::uint32_t>(row_index) *
                            padded_row_words(columns);
    const std::uint32_t image_row_base =
        static_cast<std::uint32_t>(row_index) *
        static_cast<std::uint32_t>(columns);
    if (use_resident) {
    load_line_resident:
        for (int j0 = 0; j0 < columns; j0 += MATRIX_LANES) {
#pragma HLS PIPELINE II = 1
            matrix_word_t resident_word = 0;
            if (row_index < row_count) {
                const std::uint32_t logical =
                    resident_row_base + static_cast<std::uint32_t>(j0);
                resident_word = resident_matrix[packed_index(logical)];
            }
            for (int lane = 0; lane < MATRIX_LANES; ++lane) {
#pragma HLS UNROLL
                const int column = j0 + lane;
                if (column >= MAX_ROW_DIM) {
                    continue;
                } else if (column >= columns || row_index >= row_count) {
                    row[column] = static_cast<dt>(0);
                } else {
                    row[column] = unpack_matrix_lane(resident_word, lane);
                }
            }
        }
    } else {
    load_line_image:
        for (int j0 = 0; j0 < columns; j0 += MATRIX_LANES) {
#pragma HLS PIPELINE II = 1
            for (int lane = 0; lane < MATRIX_LANES; ++lane) {
#pragma HLS UNROLL
                const int column = j0 + lane;
                if (column >= MAX_ROW_DIM) {
                    continue;
                } else if (column >= columns || row_index >= row_count) {
                    row[column] = static_cast<dt>(0);
                } else {
                    row[column] =
                        read_image(image, image_row_base +
                                              static_cast<std::uint32_t>(column));
                }
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

// Four lanes of one resident row.  j0 is a multiple of UF, so the group is
// either the low or the high half of one packed word.  Padding lanes past
// the live row length are not consumed: the MAC masks on the column index,
// because a packed word is only partially written when the width is not a
// multiple of eight.
inline void resident_half(dt out[UF], std::uint32_t row_base, int j0,
                          int row_ok) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = out complete
    dt lane[MATRIX_LANES];
#pragma HLS ARRAY_PARTITION variable = lane complete
    matrix_word_t word = 0;
    if (row_ok) {
        const unsigned aligned = static_cast<unsigned>(j0) & ~7U;
        const std::uint32_t logical =
            row_base + static_cast<std::uint32_t>(aligned);
        word = resident_matrix[packed_index(logical)];
    }
    for (int k = 0; k < MATRIX_LANES; ++k) {
#pragma HLS UNROLL
        lane[k] = unpack_matrix_lane(word, k);
    }
    const int half = static_cast<int>((static_cast<unsigned>(j0) >> 2) & 1U);
    for (int k = 0; k < UF; ++k) {
#pragma HLS UNROLL
        out[k] = lane[half * UF + k];
    }
}

// Eight lanes of one resident row.  i0 is a multiple of SCAN_LANES and every
// stored row is padded to that multiple, so i0 addresses a whole word.
inline void resident_oct(dt out[SCAN_LANES], std::uint32_t row_base, int i0) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = out complete
    const std::uint32_t logical =
        row_base + static_cast<std::uint32_t>(i0);
    const matrix_word_t word = resident_matrix[packed_index(logical)];
    for (int lane = 0; lane < SCAN_LANES; ++lane) {
#pragma HLS UNROLL
        out[lane] = unpack_matrix_lane(word, lane);
    }
}

// One UF-wide partial per output, then one add into the running accumulator.
// That is the v0.6 column-group order: a tree over the four products would
// not match saturation at the accumulator boundary.
inline void mac_four(acc_t acc[RG], int row0, int j0, int row_count,
                     int columns, const dt coeff[RG][UF],
                     const dt x_chunk[UF]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = acc complete
#pragma HLS ARRAY_PARTITION variable = coeff complete dim = 0
#pragma HLS ARRAY_PARTITION variable = x_chunk complete
    V08_UNROLL_RG
    for (int group = 0; group < RG; ++group) {
        const int row = row0 + group;
        acc_t partial = 0;
        V08_UNROLL_UF
        for (int lane = 0; lane < UF; ++lane) {
            const int column = j0 + lane;
            if (row < row_count && column < columns)
                partial += coeff[group][lane] * x_chunk[lane];
        }
        acc[group] += partial;
    }
}

inline void load_x_chunk(dt x_chunk[UF], const dt* x, int j0, int columns) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = x_chunk complete
    for (int lane = 0; lane < UF; ++lane) {
#pragma HLS UNROLL
        const int column = j0 + lane;
        x_chunk[lane] =
            column < columns ? x[column] : static_cast<dt>(0);
    }
}

// C*x+d is needed once for the normal projection sweep and again for the cap
// recheck.  Keep one out-of-line matvec body so the two call sites share the
// column datapath instead of building a second cap copy.  The resident route
// reads each packed C word inside the column loop.  The streamed route still
// fills the line buffer first.  Each output is still a chain of UF-wide
// partials, then a saturating add of d.
inline void residual_matvec(
    dt rows[RG][MAX_ROW_DIM], dt* residual, const dt* x, const dt* d,
    int row_count, int columns, int use_resident,
    std::uint32_t resident_base, const std::uint32_t* image,
    ap_uint<1> lane_range_violation[UF]) {
#pragma HLS INLINE off
    if (use_resident) {
    residual_rows_resident:
        for (int i0 = 0; i0 < row_count; i0 += RG) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
            acc_t acc[RG];
#pragma HLS ARRAY_PARTITION variable = acc complete
            for (int group = 0; group < RG; ++group) acc[group] = 0;
            const std::uint32_t pad = padded_row_words(columns);
            const std::uint32_t base0 =
                resident_base + static_cast<std::uint32_t>(i0) * pad;
            const int row1 = i0 + 1;
            const std::uint32_t base1 =
                resident_base + static_cast<std::uint32_t>(row1) * pad;
        residual_columns_resident:
            for (int j0 = 0; j0 < columns; j0 += UF) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256
                dt x_chunk[UF];
#pragma HLS ARRAY_PARTITION variable = x_chunk complete
                load_x_chunk(x_chunk, x, j0, columns);
                dt coeff[RG][UF];
#pragma HLS ARRAY_PARTITION variable = coeff complete dim = 0
                resident_half(coeff[0], base0, j0, 1);
                resident_half(coeff[1], base1, j0, row1 < row_count);
                mac_four(acc, i0, j0, row_count, columns, coeff, x_chunk);
            }
#pragma HLS UNROLL
            for (int group = 0; group < RG; ++group) {
                if (i0 + group < row_count)
                    residual[i0 + group] = cast_dt(
                        acc[group] + static_cast<acc_t>(d[i0 + group]),
                        lane_range_violation[group]);
            }
        }
    } else {
    residual_rows_shared:
        for (int i0 = 0; i0 < row_count; i0 += RG) {
            acc_t acc[RG];
#pragma HLS ARRAY_PARTITION variable = acc complete
            for (int group = 0; group < RG; ++group) acc[group] = 0;
            load_row_group(rows, i0, row_count, columns, 0, resident_base,
                           image);
        residual_columns_shared:
            for (int j0 = 0; j0 < columns; j0 += UF) {
#pragma HLS PIPELINE II = 1
                dt x_chunk[UF];
#pragma HLS ARRAY_PARTITION variable = x_chunk complete
                load_x_chunk(x_chunk, x, j0, columns);
                dt coeff[RG][UF];
#pragma HLS ARRAY_PARTITION variable = coeff complete dim = 0
                for (int group = 0; group < RG; ++group) {
#pragma HLS UNROLL
                    for (int lane = 0; lane < UF; ++lane) {
#pragma HLS UNROLL
                        const int column = j0 + lane;
                        coeff[group][lane] =
                            column < MAX_ROW_DIM ? rows[group][column]
                                                 : static_cast<dt>(0);
                    }
                }
                mac_four(acc, i0, j0, row_count, columns, coeff, x_chunk);
            }
#pragma HLS UNROLL
            for (int group = 0; group < RG; ++group) {
                if (i0 + group < row_count)
                    residual[i0 + group] = cast_dt(
                        acc[group] + static_cast<acc_t>(d[i0 + group]),
                        lane_range_violation[group]);
            }
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
        route == snn_v08::FORCE_FULL ||
        (matrix_selector != GEOMETRY_A && route == snn_v08::FORCE_CG);
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
    double upper_f, int requested_route) {
#pragma HLS INLINE off
    if (!dimensions_valid(n, m) || n_iters <= 0 || projmax <= 0 ||
        (has_lower != 0 && has_lower != 1) ||
        (has_upper != 0 && has_upper != 1))
        return snn_v08::ERR_BAD_DIMENSIONS;

    const int route = choose_route(n, m, requested_route);
    if (route < 0) return snn_v08::ERR_CAPACITY;

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


// One candidate in the v0.6 order: explicit rows, then lower facets, then
// upper facets.  valid is false for lanes past the live count of a group.
struct CandidateWinner {
    acc_t score;
    int index;
    int kind;
    ap_uint<1> valid;
};

// Earlier wins when both scores are valid and the later score is not
// strictly greater.  That is the sequential test `if (score > best)`.
inline CandidateWinner prefer_earlier(CandidateWinner earlier,
                                      CandidateWinner later) {
#pragma HLS INLINE
    if (!later.valid) return earlier;
    if (!earlier.valid) return later;
    if (later.score > earlier.score) return later;
    return earlier;
}

// Balanced tree over one group of SCAN_LANES.  Each left child is earlier
// in v0.6 order than its right child, so the root is the first maximum.
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
// (lower - x), then upper facets (x - upper).  One pipelined pass steps
// SCAN_LANES coordinates.  Each iteration reads x once and scores the lower
// and upper facets of that group, and scores the row group when the
// coordinate is still inside m.  The three families keep separate winners
// during the pass and are combined only at the end, so a later lower
// candidate still beats an earlier upper candidate when the scores tie.
// Within a family the reduction tree plus a strict-greater update is the
// scalar first-maximum rule.  Disabled box sides never update their winner.
// The same function serves the projection sweep and the cap recheck.
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
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 128
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
            // base is a multiple of SCAN_LANES, so lane k is bank k.  The
            // index stays affine even past the live count; the valid bit
            // drops those lanes.  m and n are at most the array extents, and
            // both extents are multiples of SCAN_LANES, so local is in range.
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
            lower_lane[k].score =
                col_ok ? lower_score : static_cast<acc_t>(0);
            lower_lane[k].index = local;
            lower_lane[k].kind = 1;
            lower_lane[k].valid = col_ok;
            upper_lane[k].score =
                col_ok ? upper_score : static_cast<acc_t>(0);
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
    // Geometry is fetched a row group at a time.  These buffers are the
    // BRAM-side banked view used by the II=1 arithmetic loops; no AXI or
    // URAM access appears in those loops themselves.
    // One partitioned BRAM line-buffer view is reused for A, C, G, and Ct.
    // Column factor 8 matches the resident word and the fused scan so each
    // of those reads or writes hits a distinct bank.
    dt matrix_rows[RG][MAX_ROW_DIM];
#pragma HLS ARRAY_PARTITION variable = b cyclic factor = 4 dim = 1
#pragma HLS ARRAY_PARTITION variable = d cyclic factor = 4 dim = 1
#pragma HLS ARRAY_PARTITION variable = x cyclic factor = 8 dim = 1
#pragma HLS ARRAY_PARTITION variable = resident_scale cyclic factor = 8 dim = 1
#pragma HLS ARRAY_PARTITION variable = residual cyclic factor = 8 dim = 1
#pragma HLS ARRAY_PARTITION variable = matrix_rows complete dim = 1
#pragma HLS ARRAY_PARTITION variable = matrix_rows cyclic factor = 8 dim = 2
#ifndef __SYNTHESIS__
    // Padding lanes of a scan group read past n or m.  The winner mask
    // discards those lanes.  Zeroing keeps the native read defined; the
    // values are not part of the committed state.
    for (int i = 0; i < MAXN; ++i) x[i] = static_cast<dt>(0);
    for (int i = 0; i < MAXM; ++i) residual[i] = static_cast<dt>(0);
#endif

    ap_uint<1> input_range_violation = geometry_range_violation;
    if (start_mode == snn_v08::RESIDENT_WARM && !state_has_committed)
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
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 120000
        if (configured_route == snn_v08::FORCE_FULL) {
        // A is resident only on FULL.  Each column group reads the two row
        // words that hold its four lanes and accumulates in the same order
        // as the line-buffer column loop.  The two-lane store is unrolled so
        // it is not a separate pipelined call.
        hessian_rows_resident:
            for (int i0 = 0; i0 < configured_n; i0 += RG) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
                acc_t acc[RG];
#pragma HLS ARRAY_PARTITION variable = acc complete
                for (int group = 0; group < RG; ++group) acc[group] = 0;
                const std::uint32_t pad = padded_row_words(configured_n);
                const std::uint32_t base0 =
                    offset_a + static_cast<std::uint32_t>(i0) * pad;
                const int row1 = i0 + 1;
                const std::uint32_t base1 =
                    offset_a + static_cast<std::uint32_t>(row1) * pad;
            hessian_columns_resident:
                for (int j0 = 0; j0 < configured_n; j0 += UF) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256
                    dt x_chunk[UF];
#pragma HLS ARRAY_PARTITION variable = x_chunk complete
                    load_x_chunk(x_chunk, x, j0, configured_n);
                    dt coeff[RG][UF];
#pragma HLS ARRAY_PARTITION variable = coeff complete dim = 0
                    resident_half(coeff[0], base0, j0, 1);
                    resident_half(coeff[1], base1, j0, row1 < configured_n);
                    mac_four(acc, i0, j0, configured_n, configured_n, coeff,
                             x_chunk);
                }
#pragma HLS UNROLL
                for (int group = 0; group < RG; ++group) {
                    if (i0 + group < configured_n)
                        Ax[i0 + group] =
                            cast_dt(acc[group], lane_range_violation[group]);
                }
            }
        } else {
        hessian_rows:
            for (int i0 = 0; i0 < configured_n; i0 += RG) {
                acc_t acc[RG];
#pragma HLS ARRAY_PARTITION variable = acc complete
                for (int group = 0; group < RG; ++group) acc[group] = 0;
                load_row_group(matrix_rows, i0, configured_n, configured_n, 0,
                               offset_a, A_ddr);
            hessian_columns:
                for (int j0 = 0; j0 < configured_n; j0 += UF) {
#pragma HLS PIPELINE II = 1
                    dt x_chunk[UF];
#pragma HLS ARRAY_PARTITION variable = x_chunk complete
                    load_x_chunk(x_chunk, x, j0, configured_n);
                    dt coeff[RG][UF];
#pragma HLS ARRAY_PARTITION variable = coeff complete dim = 0
                    for (int group = 0; group < RG; ++group) {
#pragma HLS UNROLL
                        for (int lane = 0; lane < UF; ++lane) {
#pragma HLS UNROLL
                            const int column = j0 + lane;
                            coeff[group][lane] =
                                column < MAX_ROW_DIM ? matrix_rows[group][column]
                                                     : static_cast<dt>(0);
                        }
                    }
                    mac_four(acc, i0, j0, configured_n, configured_n, coeff,
                             x_chunk);
                }
#pragma HLS UNROLL
                for (int group = 0; group < RG; ++group) {
                    if (i0 + group < configured_n)
                        Ax[i0 + group] =
                            cast_dt(acc[group], lane_range_violation[group]);
                }
            }
        }

    gradient_fused_step:
        for (int i = 0; i < configured_n; ++i) {
#pragma HLS PIPELINE II = 1
            // Each index is read and written once.  The next iteration
            // touches a different cyclic bank, so the store does not feed
            // the next load.
#pragma HLS DEPENDENCE variable = x type = inter dependent = false
            const gradient_product_t product =
                static_cast<acc_t>(k0) *
                (static_cast<acc_t>(Ax[i]) + static_cast<acc_t>(b[i]));
            const acc_t next = static_cast<acc_t>(
                static_cast<acc_t>(x[i]) - product);
            x[i] = cast_dt(next, gradient_range_violation);
        }

        residual_matvec(matrix_rows, residual, x, d, configured_m,
                        configured_n,
                        configured_route != snn_v08::FORCE_STREAM, offset_c,
                        C_ddr, lane_range_violation);

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
            const int resident_rows =
                configured_route != snn_v08::FORCE_STREAM;
            if (kind == 0) {
                const dt step = residual[winner] / resident_cns[winner];
                if (resident_rows) {
                    const std::uint32_t c_base =
                        offset_c + static_cast<std::uint32_t>(winner) *
                                       padded_row_words(configured_n);
                // Step a full cyclic group.  Lane k is bank k, and the next
                // iteration writes the next word of every bank.
                update_x_row_resident:
                    for (int j0 = 0; j0 < configured_n; j0 += SCAN_LANES) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = x type = inter dependent = false
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 128
                        dt coeff[SCAN_LANES];
#pragma HLS ARRAY_PARTITION variable = coeff complete
                        resident_oct(coeff, c_base, j0);
                        ap_uint<1> lane_flag[SCAN_LANES];
#pragma HLS ARRAY_PARTITION variable = lane_flag complete
                        for (int lane = 0; lane < SCAN_LANES; ++lane) {
#pragma HLS UNROLL
                            lane_flag[lane] = 0;
                            const int j = j0 + lane;
                            if (j < configured_n) {
                                const acc_t next =
                                    static_cast<acc_t>(x[j]) -
                                    static_cast<acc_t>(step) *
                                        static_cast<acc_t>(coeff[lane]);
                                x[j] = cast_dt(next, lane_flag[lane]);
                            }
                        }
                        for (int lane = 0; lane < UF; ++lane) {
#pragma HLS UNROLL
                            lane_range_violation[lane] =
                                lane_range_violation[lane] | lane_flag[lane] |
                                lane_flag[lane + UF];
                        }
                    }
                    const std::uint32_t g_base =
                        offset_g + static_cast<std::uint32_t>(winner) *
                                       padded_row_words(configured_m);
                // m <= 8 is one packed word.  Unrolling it in the event
                // path avoids a one-trip pipeline call.  Wider rows keep
                // the pipelined loop.  Lane k is bank k in both.
                if (configured_m <= SCAN_LANES) {
                    dt grow[SCAN_LANES];
#pragma HLS ARRAY_PARTITION variable = grow complete
                    resident_oct(grow, g_base, 0);
                    ap_uint<1> lane_flag[SCAN_LANES];
#pragma HLS ARRAY_PARTITION variable = lane_flag complete
                    for (int lane = 0; lane < SCAN_LANES; ++lane) {
#pragma HLS UNROLL
                        lane_flag[lane] = 0;
                        if (lane < configured_m) {
                            const acc_t next =
                                static_cast<acc_t>(residual[lane]) -
                                static_cast<acc_t>(step) *
                                    static_cast<acc_t>(grow[lane]);
                            residual[lane] = cast_dt(next, lane_flag[lane]);
                        }
                    }
                    for (int lane = 0; lane < UF; ++lane) {
#pragma HLS UNROLL
                        lane_range_violation[lane] =
                            lane_range_violation[lane] | lane_flag[lane] |
                            lane_flag[lane + UF];
                    }
                } else {
                update_residual_row_resident:
                    for (int i0 = 0; i0 < configured_m; i0 += SCAN_LANES) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 128
                        dt grow[SCAN_LANES];
#pragma HLS ARRAY_PARTITION variable = grow complete
                        resident_oct(grow, g_base, i0);
                        ap_uint<1> lane_flag[SCAN_LANES];
#pragma HLS ARRAY_PARTITION variable = lane_flag complete
                        for (int lane = 0; lane < SCAN_LANES; ++lane) {
#pragma HLS UNROLL
                            lane_flag[lane] = 0;
                            const int i = i0 + lane;
                            if (i < configured_m) {
                                const acc_t next =
                                    static_cast<acc_t>(residual[i]) -
                                    static_cast<acc_t>(step) *
                                        static_cast<acc_t>(grow[lane]);
                                residual[i] = cast_dt(next, lane_flag[lane]);
                            }
                        }
                        for (int lane = 0; lane < UF; ++lane) {
#pragma HLS UNROLL
                            lane_range_violation[lane] =
                                lane_range_violation[lane] | lane_flag[lane] |
                                lane_flag[lane + UF];
                        }
                    }
                }
                } else {
                    load_line_buffer(matrix_rows[0], winner, configured_m,
                                     configured_n, 0, offset_c, C_ddr);
                update_x_row:
                    for (int j0 = 0; j0 < configured_n; j0 += SCAN_LANES) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = x type = inter dependent = false
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 128
                        ap_uint<1> lane_flag[SCAN_LANES];
#pragma HLS ARRAY_PARTITION variable = lane_flag complete
                        for (int lane = 0; lane < SCAN_LANES; ++lane) {
#pragma HLS UNROLL
                            lane_flag[lane] = 0;
                            const int j = j0 + lane;
                            if (j < configured_n) {
                                const acc_t next =
                                    static_cast<acc_t>(x[j]) -
                                    static_cast<acc_t>(step) *
                                        static_cast<acc_t>(matrix_rows[0][j]);
                                x[j] = cast_dt(next, lane_flag[lane]);
                            }
                        }
                        for (int lane = 0; lane < UF; ++lane) {
#pragma HLS UNROLL
                            lane_range_violation[lane] =
                                lane_range_violation[lane] | lane_flag[lane] |
                                lane_flag[lane + UF];
                        }
                    }
                    load_line_buffer(matrix_rows[0], winner, configured_m,
                                     configured_m, 0, offset_g, G_ddr);
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
                                    static_cast<acc_t>(matrix_rows[0][lane]);
                            residual[lane] = cast_dt(next, lane_flag[lane]);
                        }
                    }
                    for (int lane = 0; lane < UF; ++lane) {
#pragma HLS UNROLL
                        lane_range_violation[lane] =
                            lane_range_violation[lane] | lane_flag[lane] |
                            lane_flag[lane + UF];
                    }
                } else {
                update_residual_row:
                    for (int i0 = 0; i0 < configured_m; i0 += SCAN_LANES) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 128
                        ap_uint<1> lane_flag[SCAN_LANES];
#pragma HLS ARRAY_PARTITION variable = lane_flag complete
                        for (int lane = 0; lane < SCAN_LANES; ++lane) {
#pragma HLS UNROLL
                            lane_flag[lane] = 0;
                            const int i = i0 + lane;
                            if (i < configured_m) {
                                const acc_t next =
                                    static_cast<acc_t>(residual[i]) -
                                    static_cast<acc_t>(step) *
                                        static_cast<acc_t>(matrix_rows[0][i]);
                                residual[i] = cast_dt(next, lane_flag[lane]);
                            }
                        }
                        for (int lane = 0; lane < UF; ++lane) {
#pragma HLS UNROLL
                            lane_range_violation[lane] =
                                lane_range_violation[lane] | lane_flag[lane] |
                                lane_flag[lane + UF];
                        }
                    }
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
                // Ct is stored row-major by facet.  A resident facet reads
                // that packed row in the residual update.  STREAM still
                // fills one line buffer from the image.
                if (resident_rows) {
                    const std::uint32_t ct_base =
                        offset_ct + static_cast<std::uint32_t>(winner) *
                                        padded_row_words(configured_m);
                if (configured_m <= SCAN_LANES) {
                    dt ct_row[SCAN_LANES];
#pragma HLS ARRAY_PARTITION variable = ct_row complete
                    resident_oct(ct_row, ct_base, 0);
                    ap_uint<1> lane_flag[SCAN_LANES];
#pragma HLS ARRAY_PARTITION variable = lane_flag complete
                    for (int lane = 0; lane < SCAN_LANES; ++lane) {
#pragma HLS UNROLL
                        lane_flag[lane] = 0;
                        if (lane < configured_m) {
                            const acc_t next =
                                static_cast<acc_t>(residual[lane]) +
                                static_cast<acc_t>(delta) *
                                    static_cast<acc_t>(ct_row[lane]);
                            residual[lane] = cast_dt(next, lane_flag[lane]);
                        }
                    }
                    for (int lane = 0; lane < UF; ++lane) {
#pragma HLS UNROLL
                        lane_range_violation[lane] =
                            lane_range_violation[lane] | lane_flag[lane] |
                            lane_flag[lane + UF];
                    }
                } else {
                update_residual_facet_resident:
                    for (int i0 = 0; i0 < configured_m; i0 += SCAN_LANES) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 128
                        dt ct_row[SCAN_LANES];
#pragma HLS ARRAY_PARTITION variable = ct_row complete
                        resident_oct(ct_row, ct_base, i0);
                        ap_uint<1> lane_flag[SCAN_LANES];
#pragma HLS ARRAY_PARTITION variable = lane_flag complete
                        for (int lane = 0; lane < SCAN_LANES; ++lane) {
#pragma HLS UNROLL
                            lane_flag[lane] = 0;
                            const int i = i0 + lane;
                            if (i < configured_m) {
                                const acc_t next =
                                    static_cast<acc_t>(residual[i]) +
                                    static_cast<acc_t>(delta) *
                                        static_cast<acc_t>(ct_row[lane]);
                                residual[i] = cast_dt(next, lane_flag[lane]);
                            }
                        }
                        for (int lane = 0; lane < UF; ++lane) {
#pragma HLS UNROLL
                            lane_range_violation[lane] =
                                lane_range_violation[lane] | lane_flag[lane] |
                                lane_flag[lane + UF];
                        }
                    }
                }
                } else {
                    load_line_buffer(matrix_rows[0], winner, configured_n,
                                     configured_m, 0, offset_ct, Ct_ddr);
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
                                    static_cast<acc_t>(matrix_rows[0][lane]);
                            residual[lane] = cast_dt(next, lane_flag[lane]);
                        }
                    }
                    for (int lane = 0; lane < UF; ++lane) {
#pragma HLS UNROLL
                        lane_range_violation[lane] =
                            lane_range_violation[lane] | lane_flag[lane] |
                            lane_flag[lane + UF];
                    }
                } else {
                update_residual_facet:
                    for (int i0 = 0; i0 < configured_m; i0 += SCAN_LANES) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 128
                        ap_uint<1> lane_flag[SCAN_LANES];
#pragma HLS ARRAY_PARTITION variable = lane_flag complete
                        for (int lane = 0; lane < SCAN_LANES; ++lane) {
#pragma HLS UNROLL
                            lane_flag[lane] = 0;
                            const int i = i0 + lane;
                            if (i < configured_m) {
                                const acc_t next =
                                    static_cast<acc_t>(residual[i]) +
                                    static_cast<acc_t>(delta) *
                                        static_cast<acc_t>(
                                            matrix_rows[0][i]);
                                residual[i] = cast_dt(next, lane_flag[lane]);
                            }
                        }
                        for (int lane = 0; lane < UF; ++lane) {
#pragma HLS UNROLL
                            lane_range_violation[lane] =
                                lane_range_violation[lane] | lane_flag[lane] |
                                lane_flag[lane + UF];
                        }
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
                            configured_route != snn_v08::FORCE_STREAM,
                            offset_c, C_ddr, lane_range_violation);

            const CandidateWinner selected = scan_candidates(
                x, residual, lower, upper, configured_n, configured_m,
                configured_has_lower, configured_has_upper);
            // The scalar cap scan starts from 0 and keeps a strict maximum,
            // so a non-positive best score stays 0.
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
// Keep all configuration and refresh geometry calls at this hierarchy level.
// The allocation limit then covers A, C, Ct, G, and refresh-A uniformly.
#pragma HLS ALLOCATION function instances=load_geometry_matrix limit=1
// The geometry loader and CNS/scale loader use the same conversion pipeline.
#pragma HLS ALLOCATION function instances=binary64_to_dt limit=1
    DispatchResult result;
    result.route = configured_route;
    if (command == snn_v08::CONFIGURE) {
        result.error = configure_geometry(
            cns_cfg, row_scale_cfg, x0_in, n, m, k0_f, ctol_f, n_iters,
            projmax, has_lower, lower_f, has_upper, upper_f,
            requested_route);
        if (result.error == snn_v08::ERR_OK) {
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
    if (command == snn_v08::REFRESH_A) {
        result.error = refresh_a(n, m);
        if (result.error == snn_v08::ERR_OK) {
            load_geometry_matrix(A_cfg, A_ddr, nullptr, GEOMETRY_A, 0, n, n,
                                 configured_route, offset_a, 0,
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
            b_in, d_in, x0_in, A_ddr, C_ddr, Ct_ddr, G_ddr, x_raw_out,
            telemetry_out,
            start_mode, shift_stride, tail_policy);
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

extern "C" void snn_qp_v08(
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

#pragma HLS bind_storage variable = resident_matrix type = ram_2p impl = uram latency = 2
#pragma HLS bind_storage variable = resident_cns type = ram_1p impl = bram
#pragma HLS bind_storage variable = resident_scale type = ram_1p impl = bram
#pragma HLS bind_storage variable = resident_state type = ram_1p impl = bram

    clear_outputs(x_raw_out, telemetry_out, n);

    // ONESHOT executes exactly one scalar command.  PERSISTENT executes the
    // launch command once, then services mailbox sequence changes until STOP.
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
        // There is one sequence read per polling interval, not one read per
        // cycle.  ap_wait_n keeps the trip count visible to HLS; the volatile
        // counter also prevents native/synthesis dead-code elimination.
        std::uint32_t sequence =
            mb_in == nullptr ? seen_sequence
                             : mb_in[snn_v08::MAILBOX_SEQUENCE];
        if (sequence != seen_sequence) {
#ifndef __SYNTHESIS__
            // Native emulation uses ordinary host arrays for the payload.  A
            // compiler acquire fence models the device-side visibility point
            // that the volatile sequence load provides on the AXI path.
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
