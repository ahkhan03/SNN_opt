#pragma once

// Host-side layout for the v06 XRT ABI.  The kernel still receives typed
// pointers, but the host places the three per-solve vectors and the mailbox
// in one cache-line-aligned input BO, and places all result objects in one
// output BO.  Pointer arguments are the BO device address plus these offsets.

#include "v06_abi.hpp"

#include <cstddef>
#include <cstdint>

namespace snn_v06_host {

constexpr std::size_t CACHELINE_BYTES = 64;
constexpr std::size_t FIXED_FRACTION_BITS = 24;
constexpr std::size_t MAILBOX_BYTES =
    static_cast<std::size_t>(snn_v06::MAILBOX_WORDS) * sizeof(std::uint32_t);
constexpr std::size_t TELEMETRY_BYTES =
    snn_v06::TELEMETRY_WORDS * sizeof(std::uint64_t);

constexpr std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

struct InputLayout {
    std::size_t mailbox = 0;
    std::size_t b = 0;
    std::size_t d = 0;
    std::size_t x0 = 0;
    std::size_t bytes = 0;
};

struct OutputLayout {
    std::size_t mailbox = 0;
    std::size_t raw = 0;
    std::size_t telemetry = 0;
    std::size_t bytes = 0;
};

inline InputLayout input_layout(int n, int m) {
    InputLayout layout;
    layout.mailbox = 0;
    layout.b = align_up(MAILBOX_BYTES, CACHELINE_BYTES);
    layout.d = align_up(layout.b + static_cast<std::size_t>(n) * sizeof(double),
                        CACHELINE_BYTES);
    layout.x0 = align_up(layout.d + static_cast<std::size_t>(m) * sizeof(double),
                         CACHELINE_BYTES);
    layout.bytes = align_up(layout.x0 + static_cast<std::size_t>(n) * sizeof(double),
                            CACHELINE_BYTES);
    return layout;
}

inline OutputLayout output_layout(int n) {
    OutputLayout layout;
    layout.mailbox = 0;
    layout.raw = align_up(MAILBOX_BYTES, CACHELINE_BYTES);
    layout.telemetry = align_up(
        layout.raw + static_cast<std::size_t>(n) * sizeof(std::int64_t),
        CACHELINE_BYTES);
    layout.bytes = align_up(layout.telemetry + TELEMETRY_BYTES, CACHELINE_BYTES);
    return layout;
}

inline std::size_t matrix_image_bytes(int rows, int columns) {
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns) *
           sizeof(std::uint32_t);
}

inline std::size_t vector_bytes(int count) {
    return static_cast<std::size_t>(count) * sizeof(double);
}

}  // namespace snn_v06_host
