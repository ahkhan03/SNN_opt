#include "bench_support.hpp"
#include "canonical_solve.hpp"
#include "msrp_bundle.hpp"
#include "v05_double_core.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int requested_threads() {
    const char* text = std::getenv("OMP_NUM_THREADS");
    return text ? msrp_v05::positive_integer(text, "OMP_NUM_THREADS") : 1;
}

static msrp_v05::SolveResult solve(const msrp_v05::Problem& q, int threads) {
    return threads <= 1 ? msrp_v05::solve_canonical(q)
                        : msrp_v05::solve_openmp(q, threads);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(
            stderr,
            "usage: %s <binary64-problem.bin> <output.bin> "
            "[--warmups N] [--reps N] [--energy-reps N]\n",
            argv[0]);
        return 2;
    }
    const msrp_v05::BenchOptions options =
        msrp_v05::parse_bench_options(argc, argv, 3);
    const msrp_v05::Problem q = msrp_v05::load_problem(argv[1]);
    if (q.source_is_float32) {
        std::fprintf(stderr, "v0.5 CPU baseline requires binary64 input\n");
        return 2;
    }
    const int threads = requested_threads();

    msrp_v05::SolveResult last;
    for (int i = 0; i < options.warmups; ++i) last = solve(q, threads);

    if (options.energy_repetitions > 0) {
        msrp_v05::wait_for_go();
        const std::uint64_t started = msrp_v05::monotonic_raw_ns();
        for (int i = 0; i < options.energy_repetitions; ++i)
            last = solve(q, threads);
        const std::uint64_t finished = msrp_v05::monotonic_raw_ns();
        const auto meta = msrp_v05::telemetry(q, last);
        msrp_v05::write_double_output(argv[2], q.n, last.x, meta);
        std::printf(
            "DONE REPS %d DURATION_SECONDS %.17g STATUS %d EVENTS %llu "
            "DIGEST %016llx\n",
            options.energy_repetitions,
            static_cast<double>(finished - started) * 1e-9, last.status,
            static_cast<unsigned long long>(last.observer.total),
            static_cast<unsigned long long>(last.observer.digest));
        std::fflush(stdout);
        return last.status == 0 ? 0 : 3;
    }

    std::vector<double> latency;
    latency.reserve(options.repetitions);
    for (int i = 0; i < options.repetitions; ++i) {
        const std::uint64_t started = msrp_v05::monotonic_raw_ns();
        last = solve(q, threads);
        const std::uint64_t finished = msrp_v05::monotonic_raw_ns();
        latency.push_back(static_cast<double>(finished - started) * 1e-9);
    }
    const auto meta = msrp_v05::telemetry(q, last);
    msrp_v05::write_double_output(argv[2], q.n, last.x, meta);
    msrp_v05::print_latency_json(
        "msrp-v05-a53-latency-v1", q.n, q.m, last.threads, options.warmups,
        latency, latency, meta.data());
    return last.status == 0 ? 0 : 3;
}
