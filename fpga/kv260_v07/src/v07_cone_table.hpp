#pragma once

#include <cstddef>
#include <cstdint>
#ifndef __SYNTHESIS__
#include <cmath>
#include <string>
#include <vector>
#include <utility>
#include <cmath>
#endif

namespace snn_v07 {

constexpr int MAX_CONES = 64;
constexpr int NORM_SQ_INTEGER_BITS = 20;
constexpr double MIN_CONE_MU = 0.000000059604644775390625;
enum ConeKind : std::uint32_t { BALL = 0, SCALED_SOC = 1 };

// A descriptor is deliberately POD so it can be copied into a fixed-size HLS
// configure image.  Coordinates [offset, offset+length) are one contiguous
// block; a scaled SOC uses coordinate offset as t and the remaining entries as z.
struct ConeDescriptor {
    ConeKind kind = BALL;
    std::uint32_t offset = 0;
    std::uint32_t length = 0;
    double radius = 0.0;
    double mu = 1.0;
    double center = 0.0;
};

inline bool cone_norm_fits(std::uint32_t kind, std::uint32_t length) {
    const std::uint64_t terms = kind == BALL ? length : length - 1;
    return terms * UINT64_C(16384) < (UINT64_C(1) << NORM_SQ_INTEGER_BITS);
}

#ifndef __SYNTHESIS__
inline bool validate_cones(const ConeDescriptor* cones, int count, int n,
                           std::string* error = nullptr) {
    auto fail = [&](const char* msg) {
        if (error) *error = msg;
        return false;
    };
    if (count < 0 || count > MAX_CONES) return fail("cone table overflow: at most 64 entries");
    if (count && cones == nullptr) return fail("cone table is null");
    std::vector<std::pair<std::uint32_t, std::uint32_t>> spans;
    spans.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        const auto& c = cones[i];
        const std::uint32_t min_len = c.kind == SCALED_SOC ? 3U : 1U;
        if (c.length < min_len) return fail("cone block is too short for its kind");
        if (c.offset > static_cast<std::uint32_t>(n) ||
            c.length > static_cast<std::uint32_t>(n) - c.offset)
            return fail("cone block is outside the state vector");
        if (c.kind != BALL && c.kind != SCALED_SOC)
            return fail("unsupported cone kind");
        if (!std::isfinite(c.radius) || c.radius < 0.0 || c.radius >= 128.0)
            return fail("ball radius must be finite, non-negative, and fit ap_fixed<32,8>");
        if (!std::isfinite(c.center) || c.center < -128.0 || c.center >= 128.0)
            return fail("ball center must fit ap_fixed<32,8>");
        if (c.kind == SCALED_SOC && (!std::isfinite(c.mu) || c.mu < MIN_CONE_MU || c.mu >= 128.0))
            return fail("SOC slope mu must be finite, at least 2^-24, and fit ap_fixed<32,8>");
        if (!cone_norm_fits(c.kind, c.length))
            return fail("cone block exceeds norm-square capacity");
        const auto end = c.offset + c.length;
        for (const auto& span : spans)
            if (c.offset < span.second && span.first < end)
                return fail("cone blocks overlap");
        spans.emplace_back(c.offset, end);
    }
    return true;
}

inline bool validate_contiguous_indices(const std::uint32_t* indices, std::size_t count,
                                        int n, std::string* error = nullptr) {
    if (count == 0) return true;
    if (indices == nullptr) { if (error) *error = "index list is null"; return false; }
    for (std::size_t i=0; i<count; ++i) {
        if (indices[i] >= static_cast<std::uint32_t>(n)) { if (error) *error = "index list is outside the state vector"; return false; }
        if (i && indices[i] != indices[i-1] + 1U) { if (error) *error = "cone indices must form one contiguous block"; return false; }
    }
    return true;
}
#endif

}  // namespace snn_v07
