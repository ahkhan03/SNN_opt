#pragma once

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <time.h>
#include <vector>

namespace msrp_v05 {

struct BenchOptions {
    int warmups = 5;
    int repetitions = 20;
    int energy_repetitions = 0;
};

inline int positive_integer(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value <= 0 ||
        value > 100000000L) {
        std::fprintf(stderr, "invalid %s: %s\n", name, text);
        std::exit(2);
    }
    return static_cast<int>(value);
}

inline BenchOptions parse_bench_options(int argc, char** argv, int first) {
    BenchOptions options;
    for (int i = first; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--warmups" && i + 1 < argc) {
            options.warmups = positive_integer(argv[++i], "warmup count");
        } else if (arg == "--reps" && i + 1 < argc) {
            options.repetitions =
                positive_integer(argv[++i], "repetition count");
        } else if (arg == "--energy-reps" && i + 1 < argc) {
            options.energy_repetitions =
                positive_integer(argv[++i], "energy repetition count");
        } else {
            std::fprintf(stderr, "unrecognized argument: %s\n", arg.c_str());
            std::exit(2);
        }
    }
    return options;
}

inline std::uint64_t monotonic_raw_ns() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    timespec stamp{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &stamp) != 0) {
        std::perror("clock_gettime(CLOCK_MONOTONIC_RAW)");
        std::exit(2);
    }
    const std::uint64_t value =
        static_cast<std::uint64_t>(stamp.tv_sec) * UINT64_C(1000000000) +
        static_cast<std::uint64_t>(stamp.tv_nsec);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return value;
}

inline std::uint64_t monotonic_raw_resolution_ns() {
    timespec resolution{};
    if (clock_getres(CLOCK_MONOTONIC_RAW, &resolution) != 0) {
        std::perror("clock_getres(CLOCK_MONOTONIC_RAW)");
        std::exit(2);
    }
    return static_cast<std::uint64_t>(resolution.tv_sec) *
               UINT64_C(1000000000) +
           static_cast<std::uint64_t>(resolution.tv_nsec);
}

inline std::uint64_t timer_overhead_median_ns() {
    constexpr std::size_t SAMPLES = 1000000;
    std::vector<std::uint64_t> deltas;
    deltas.reserve(SAMPLES);
    for (std::size_t i = 0; i < SAMPLES; ++i) {
        const std::uint64_t left = monotonic_raw_ns();
        const std::uint64_t right = monotonic_raw_ns();
        deltas.push_back(right - left);
    }
    const auto middle = deltas.begin() + deltas.size() / 2;
    std::nth_element(deltas.begin(), middle, deltas.end());
    return *middle;
}

inline void wait_for_go() {
    std::printf("READY\n");
    std::fflush(stdout);
    char line[64];
    if (!std::fgets(line, sizeof(line), stdin) ||
        std::strncmp(line, "GO", 2) != 0) {
        std::fprintf(stderr, "energy mode expected GO on stdin\n");
        std::exit(2);
    }
}

inline void print_latency_json(const char* schema, int n, int m, int threads,
                               int warmups,
                               const std::vector<double>& complete_seconds,
                               const std::vector<double>& kernel_seconds,
                               const std::uint64_t* telemetry) {
    std::printf(
        "{\"schema\":\"%s\",\"n\":%d,\"m\":%d,\"threads\":%d,"
        "\"warmups\":%d,\"clock\":\"CLOCK_MONOTONIC_RAW\","
        "\"clock_resolution_ns\":%llu,\"timer_probe_count\":1000000,"
        "\"timer_overhead_median_ns\":%llu,"
        "\"status\":%llu,\"iterations\":%llu,"
        "\"events\":%llu,\"digest\":\"%016llx\","
        "\"complete_seconds\":[",
        schema, n, m, threads, warmups,
        static_cast<unsigned long long>(monotonic_raw_resolution_ns()),
        static_cast<unsigned long long>(timer_overhead_median_ns()),
        static_cast<unsigned long long>(telemetry[1]),
        static_cast<unsigned long long>(telemetry[3]),
        static_cast<unsigned long long>(telemetry[5]),
        static_cast<unsigned long long>(telemetry[10]));
    for (std::size_t i = 0; i < complete_seconds.size(); ++i) {
        if (i) std::printf(",");
        std::printf("%.17g", complete_seconds[i]);
    }
    std::printf("],\"kernel_seconds\":[");
    for (std::size_t i = 0; i < kernel_seconds.size(); ++i) {
        if (i) std::printf(",");
        std::printf("%.17g", kernel_seconds[i]);
    }
    std::printf("]}\n");
}

}  // namespace msrp_v05
