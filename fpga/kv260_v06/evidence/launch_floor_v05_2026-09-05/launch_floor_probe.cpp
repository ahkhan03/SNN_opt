// Launch-floor probe for the deployed snn_qp_v05 kernel.
// Zero-valued geometry so every projection sweep exits at ordinal 0; times
// only kernel run.start()->run.wait() over a grid of (n=m, n_iters).
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <vector>

static std::uint64_t now_ns() {
    timespec t{}; clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return std::uint64_t(t.tv_sec) * 1000000000ULL + std::uint64_t(t.tv_nsec);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: probe xclbin [reps]\n"); return 1; }
    const int reps = argc > 2 ? std::atoi(argv[2]) : 200;
    auto device = xrt::device(0);
    const auto uuid = device.load_xclbin(argv[1]);
    auto kernel = xrt::kernel(device, uuid, "snn_qp_v05");
    const int NMAX = 64;
    auto mk = [&](int arg, std::size_t bytes) { return xrt::bo(device, bytes, kernel.group_id(arg)); };
    auto bo_A = mk(0, NMAX * NMAX * 8), bo_b = mk(1, NMAX * 8), bo_C = mk(2, NMAX * NMAX * 8),
         bo_d = mk(3, NMAX * 8), bo_cns = mk(4, NMAX * 8), bo_scale = mk(5, NMAX * 8),
         bo_G = mk(6, NMAX * NMAX * 8), bo_x0 = mk(7, NMAX * 8), bo_raw = mk(8, NMAX * 8),
         bo_meta = mk(9, 16 * 8);
    std::vector<double> zeros(NMAX * NMAX, 0.0), ones(NMAX, 1.0);
    bo_A.write(zeros.data()); bo_C.write(zeros.data()); bo_G.write(zeros.data());
    bo_b.write(zeros.data()); bo_d.write(zeros.data()); bo_x0.write(zeros.data());
    bo_cns.write(ones.data()); bo_scale.write(ones.data());
    for (auto* b : {&bo_A, &bo_C, &bo_G, &bo_b, &bo_d, &bo_x0, &bo_cns, &bo_scale}) b->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    const int sizes[] = {1, 6, 20, 64};
    const int iters[] = {1, 2, 11, 101};
    std::printf("n=m,n_iters,reps,median_us,p05_us,p95_us,min_us\n");
    for (int n : sizes) for (int K : iters) {
        std::vector<double> t; t.reserve(reps);
        for (int r = 0; r < reps + 10; ++r) {
            const auto t0 = now_ns();
            auto run = kernel(bo_A, bo_b, bo_C, bo_d, bo_cns, bo_scale, bo_G, bo_x0, bo_raw, bo_meta,
                              n, n, 0.1, 1e-6, K, 64, 0, 0.0, 0, 0.0);
            run.wait();
            const auto t1 = now_ns();
            if (r >= 10) t.push_back((t1 - t0) * 1e-3);
        }
        std::sort(t.begin(), t.end());
        std::printf("%d,%d,%d,%.2f,%.2f,%.2f,%.2f\n", n, K, reps, t[t.size() / 2], t[t.size() / 20],
                    t[t.size() * 19 / 20], t.front());
        std::fflush(stdout);
    }
    bo_meta.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::uint64_t meta[16]; bo_meta.read(meta);
    std::printf("# telemetry magic ok=%d status=%llu executed=%llu events=%llu\n",
                meta[0] == 0x4d53525056303531ULL, (unsigned long long)meta[1],
                (unsigned long long)meta[3], (unsigned long long)meta[5]);
    return 0;
}
