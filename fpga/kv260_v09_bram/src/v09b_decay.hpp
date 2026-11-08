#pragma once

#include <cmath>
#include <cstdint>

namespace snn_v09b_decay {

constexpr std::uint32_t MIN_FLOOR_QUANTA = 128;
constexpr std::uint32_t MAX_POSITIVE_QUANTA = 0x7fffffffU;

inline std::uint32_t quantize_positive(double value) {
    const double scaled = std::nearbyint(value * 16777216.0);
    if (scaled >= static_cast<double>(MAX_POSITIVE_QUANTA))
        return MAX_POSITIVE_QUANTA;
    if (scaled <= 0.0) return 0;
    return static_cast<std::uint32_t>(scaled);
}

inline int level_count(std::uint32_t start, std::uint32_t floor) {
    int levels = 1;
    std::uint64_t level = start;
    const std::uint64_t floor64 = floor;
    while (2U * level > 3U * floor64) {
        level /= 2U;
        ++levels;
    }
    return levels;
}

inline std::uint32_t next_level(std::uint32_t level, std::uint32_t floor) {
    if (level <= floor) return level;
    level /= 2U;
    return 2ULL * level <= 3ULL * floor ? floor : level;
}

inline std::uint32_t final_level(std::uint32_t start, std::uint32_t floor,
                                 int segment, int iterations) {
    std::uint32_t level = start;
    for (int outer = 0; outer + 1 < iterations; ++outer) {
        if ((outer + 1) % segment == 0)
            level = next_level(level, floor);
    }
    return level;
}

}  // namespace snn_v09b_decay
