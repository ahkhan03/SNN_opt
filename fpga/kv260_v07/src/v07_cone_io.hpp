#pragma once
#include "v07_cone_table.hpp"
#include <array>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace snn_v07 {
// Sidecar rows: kind offset length radius mu center. Kind is ball or soc.
// Comments start with #; all six fields are required to avoid silent defaults.
struct ConeImage {
    std::array<std::uint32_t, MAX_CONES> kinds{}, offsets{}, lengths{};
    std::array<double, MAX_CONES> radii{}, mus{}, centers{};
    int count = 0;

    static ConeImage load(const std::string& path, int n) {
        ConeImage image;
        image.mus.fill(1.0);
        std::vector<ConeDescriptor> cones;
        if (!path.empty()) {
            std::ifstream input(path);
            if (!input) throw std::runtime_error("cannot read cone table: " + path);
            std::string line;
            unsigned lineno = 0;
            while (std::getline(input, line)) {
                ++lineno;
                line = line.substr(0, line.find('#'));
                std::istringstream row(line);
                std::string kind, extra;
                if (!(row >> kind)) continue;
                ConeDescriptor c;
                long long offset = 0, length = 0;
                if (!(row >> offset >> length >> c.radius >> c.mu >> c.center) ||
                    (row >> extra) || offset < 0 || length < 0 ||
                    offset > UINT32_MAX || length > UINT32_MAX ||
                    (kind != "ball" && kind != "soc" && kind != "scaled_soc"))
                    throw std::runtime_error(path + ":" + std::to_string(lineno) +
                        ": expected kind offset length radius mu center");
                c.kind = kind == "ball" ? BALL : SCALED_SOC;
                c.offset = static_cast<std::uint32_t>(offset);
                c.length = static_cast<std::uint32_t>(length);
                cones.push_back(c);
            }
        }
        std::string error;
        if (!validate_cones(cones.data(), static_cast<int>(cones.size()), n, &error))
            throw std::runtime_error("invalid cone table: " + error);
        image.count = static_cast<int>(cones.size());
        for (int i = 0; i < image.count; ++i) {
            image.kinds[i] = cones[i].kind;
            image.offsets[i] = cones[i].offset;
            image.lengths[i] = cones[i].length;
            image.radii[i] = cones[i].radius;
            image.mus[i] = cones[i].mu;
            image.centers[i] = cones[i].center;
        }
        return image;
    }
};
} // namespace snn_v07
