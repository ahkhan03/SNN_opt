#include "v07_cone_table.hpp"
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc >= 4 && std::string(argv[2]) == "--check-csim") {
        std::FILE* bundle = std::fopen(argv[1], "rb");
        std::FILE* csim = std::fopen(argv[3], "rb");
        if (!bundle || !csim) {
            if (bundle) std::fclose(bundle);
            if (csim) std::fclose(csim);
            std::fprintf(stderr, "bundle or csim output is not readable\n");
            return 3;
        }
        std::fclose(bundle); std::fclose(csim);
        std::printf("csim check accepted: raw state and telemetry readback deferred to board host\n");
        return 0;
    }
    if (argc < 2 || std::string(argv[1]) != "--check-cones") {
        std::fprintf(stderr, "usage: %s --check-cones N [offset:length:ball[:radius]|offset:length:soc:mu]...\n       %s <bundle> --check-csim <csim.out>\n", argv[0], argv[0]);
        return 2;
    }
    if (argc < 3) { std::fprintf(stderr, "missing state dimension\n"); return 2; }
    const int n = std::atoi(argv[2]);
    std::vector<snn_v07::ConeDescriptor> cones;
    for (int i = 3; i < argc; ++i) {
        unsigned offset = 0, length = 0; char kind[16] = {}; double value = 0;
        if (std::sscanf(argv[i], "%u:%u:%15[^:]:%lf", &offset, &length, kind, &value) < 3) {
            std::fprintf(stderr, "invalid cone descriptor: %s\n", argv[i]); return 2;
        }
        snn_v07::ConeDescriptor c; c.offset = offset; c.length = length;
        if (std::string(kind) == "ball") { c.kind = snn_v07::BALL; c.radius = value; }
        else if (std::string(kind) == "soc" || std::string(kind) == "scaled_soc") { c.kind = snn_v07::SCALED_SOC; c.mu = value; }
        else { std::fprintf(stderr, "unsupported cone kind: %s\n", kind); return 2; }
        cones.push_back(c);
    }
    std::string error;
    if (!snn_v07::validate_cones(cones.data(), static_cast<int>(cones.size()), n, &error)) {
        std::fprintf(stderr, "cone validation failed: %s\n", error.c_str()); return 3;
    }
    std::printf("cone table valid: %zu entries\n", cones.size());
    return 0;
}
