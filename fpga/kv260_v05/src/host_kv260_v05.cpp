#include "bench_support.hpp"
#include "msrp_bundle.hpp"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(
            stderr,
            "usage: %s <kernel.xclbin> <binary64-problem.bin> <output.bin> "
            "[--warmups N] [--reps N] [--energy-reps N]\n",
            argv[0]);
        return 2;
    }
    const msrp_v05::BenchOptions options =
        msrp_v05::parse_bench_options(argc, argv, 4);
    const msrp_v05::Problem q = msrp_v05::load_problem(argv[2]);
    if (q.source_is_float32) {
        std::fprintf(stderr, "v0.5 KV260 host requires binary64 input\n");
        return 2;
    }
    std::vector<std::int64_t> raw(q.n, 0);
    std::array<std::uint64_t, msrp_v05::TELEMETRY_WORDS> meta{};

    auto device = xrt::device(0);
    const auto uuid = device.load_xclbin(argv[1]);
    auto kernel = xrt::kernel(device, uuid, "snn_qp_v05");
    auto make_bo = [&](int argument, std::size_t bytes) {
        return xrt::bo(device, bytes, kernel.group_id(argument));
    };
    auto bo_A = make_bo(0, q.A.size() * sizeof(double));
    auto bo_b = make_bo(1, q.b.size() * sizeof(double));
    auto bo_C = make_bo(2, q.C.size() * sizeof(double));
    auto bo_d = make_bo(3, q.d.size() * sizeof(double));
    auto bo_cns = make_bo(4, q.c_norms_sq.size() * sizeof(double));
    auto bo_scale = make_bo(5, q.row_scale.size() * sizeof(double));
    auto bo_G = make_bo(6, q.G.size() * sizeof(double));
    auto bo_x0 = make_bo(7, q.x0.size() * sizeof(double));
    auto bo_raw = make_bo(8, raw.size() * sizeof(std::int64_t));
    auto bo_meta = make_bo(9, meta.size() * sizeof(std::uint64_t));

    bo_A.write(q.A.data());
    bo_b.write(q.b.data());
    bo_C.write(q.C.data());
    bo_d.write(q.d.data());
    bo_cns.write(q.c_norms_sq.data());
    bo_scale.write(q.row_scale.data());
    bo_G.write(q.G.data());
    bo_x0.write(q.x0.data());

    auto run_once = [&](double* complete_seconds,
                        double* kernel_seconds) {
        const std::uint64_t complete_started = msrp_v05::monotonic_raw_ns();
        bo_A.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_C.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_d.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_cns.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_scale.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_G.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_x0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        const std::uint64_t kernel_started = msrp_v05::monotonic_raw_ns();
        auto run = kernel(
            bo_A, bo_b, bo_C, bo_d, bo_cns, bo_scale, bo_G, bo_x0, bo_raw,
            bo_meta, q.n, q.m, q.k0, q.constraint_tol, q.iterations,
            q.projection_cap, q.has_lower ? 1 : 0, q.lower,
            q.has_upper ? 1 : 0, q.upper);
        run.wait();
        const std::uint64_t kernel_finished = msrp_v05::monotonic_raw_ns();
        bo_raw.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_meta.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_raw.read(raw.data());
        bo_meta.read(meta.data());
        const std::uint64_t complete_finished = msrp_v05::monotonic_raw_ns();
        if (complete_seconds)
            *complete_seconds =
                static_cast<double>(complete_finished - complete_started) *
                1e-9;
        if (kernel_seconds)
            *kernel_seconds =
                static_cast<double>(kernel_finished - kernel_started) * 1e-9;
    };

    for (int i = 0; i < options.warmups; ++i) run_once(nullptr, nullptr);

    if (options.energy_repetitions > 0) {
        msrp_v05::wait_for_go();
        const std::uint64_t started = msrp_v05::monotonic_raw_ns();
        for (int i = 0; i < options.energy_repetitions; ++i)
            run_once(nullptr, nullptr);
        const std::uint64_t finished = msrp_v05::monotonic_raw_ns();
        msrp_v05::write_fixed_output(argv[3], q.n, raw, meta);
        std::printf(
            "DONE REPS %d DURATION_SECONDS %.17g STATUS %llu EVENTS %llu "
            "DIGEST %016llx\n",
            options.energy_repetitions,
            static_cast<double>(finished - started) * 1e-9,
            static_cast<unsigned long long>(meta[1]),
            static_cast<unsigned long long>(meta[5]),
            static_cast<unsigned long long>(meta[10]));
        std::fflush(stdout);
        return meta[1] == 0 ? 0 : 3;
    }

    std::vector<double> complete;
    std::vector<double> kernel_only;
    complete.reserve(options.repetitions);
    kernel_only.reserve(options.repetitions);
    for (int i = 0; i < options.repetitions; ++i) {
        double complete_s = 0.0;
        double kernel_s = 0.0;
        run_once(&complete_s, &kernel_s);
        complete.push_back(complete_s);
        kernel_only.push_back(kernel_s);
    }
    msrp_v05::write_fixed_output(argv[3], q.n, raw, meta);
    msrp_v05::print_latency_json(
        "msrp-v05-kv260-latency-v1", q.n, q.m, 0, options.warmups,
        complete, kernel_only, meta.data());
    return meta[1] == 0 ? 0 : 3;
}
