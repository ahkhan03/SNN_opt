// Native fixed-point driver for snn_qp_v06.
//
// The persistent path deliberately uses the same publication order as the
// host ABI: payload first, mailbox words 1..5 next, and sequence last.

#include "../../kv260_v05/src/msrp_bundle.hpp"
#include "v06_abi.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

extern "C" void snn_qp_v06(
    const double* A_cfg, const double* C_cfg, const double* G_cfg,
    const double* cns_cfg, const double* row_scale_cfg, const double* b_in,
    const double* d_in, const double* x0_in, std::uint32_t* A_ddr,
    std::uint32_t* C_ddr, std::uint32_t* Ct_ddr, std::uint32_t* G_ddr,
    long long* x_raw_out, unsigned long long* telemetry_out,
    volatile const std::uint32_t* mb_in, volatile std::uint32_t* mb_out,
    int command, int launch_mode, int route_mode, int start_mode,
    int shift_stride, int tail_policy, int n, int m, double k0_f,
    double ctol_f, int n_iters, int projmax, int has_lower, double lower_f,
    int has_upper, double upper_f);

namespace {

constexpr std::uint64_t OUTPUT_MAGIC = UINT64_C(0x4d53525056303631);
constexpr int MAX_WAIT_MS = 30000;

struct Buffers {
    std::vector<std::uint32_t> a_ddr;
    std::vector<std::uint32_t> c_ddr;
    std::vector<std::uint32_t> ct_ddr;
    std::vector<std::uint32_t> g_ddr;
    std::vector<long long> raw;
    std::array<unsigned long long, snn_v06::TELEMETRY_WORDS> telemetry{};
    std::array<std::uint32_t, snn_v06::MAILBOX_WORDS> mb_in{};
    std::array<std::uint32_t, snn_v06::MAILBOX_WORDS> mb_out{};
};

void allocate_buffers(const msrp_v05::Problem& q, Buffers& b) {
    b.a_ddr.resize(static_cast<std::size_t>(q.n) * q.n);
    b.c_ddr.resize(static_cast<std::size_t>(q.m) * q.n);
    b.ct_ddr.resize(static_cast<std::size_t>(q.m) * q.n);
    b.g_ddr.resize(static_cast<std::size_t>(q.m) * q.m);
    b.raw.assign(static_cast<std::size_t>(q.n), 0);
    b.mb_in.fill(0);
    b.mb_out.fill(0);
}

void call_kernel(const msrp_v05::Problem& q, Buffers& b, int command,
                 int launch, int route, int start, int shift, int tail) {
    snn_qp_v06(
        q.A.data(), q.C.data(), q.G.data(), q.c_norms_sq.data(),
        q.row_scale.data(), q.b.data(), q.d.data(), q.x0.data(),
        b.a_ddr.data(), b.c_ddr.data(), b.ct_ddr.data(), b.g_ddr.data(),
        b.raw.data(), b.telemetry.data(), b.mb_in.data(), b.mb_out.data(),
        command, launch, route, start, shift, tail, q.n, q.m, q.k0,
        q.constraint_tol, q.iterations, q.projection_cap,
        q.has_lower ? 1 : 0, q.lower, q.has_upper ? 1 : 0, q.upper);
}

void wait_done(const Buffers& b, std::uint32_t sequence) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(MAX_WAIT_MS);
    const volatile std::uint32_t* done = b.mb_out.data();
    while (done[snn_v06::OUT_DONE_SEQUENCE] != sequence) {
        if (std::chrono::steady_clock::now() >= deadline) {
            std::fprintf(stderr, "persistent kernel timed out at sequence %u\n",
                         sequence);
            std::exit(3);
        }
        std::this_thread::yield();
    }
    std::atomic_thread_fence(std::memory_order_acquire);
}

void publish(Buffers& b, std::uint32_t sequence, int command, int start,
             int shift, int tail, int route = snn_v06::AUTO) {
    // The arrays are ordinary host memory in native emulation.  Fences retain
    // the release/acquire ordering used by the XRT BO implementation.
    b.mb_in[snn_v06::MAILBOX_COMMAND] =
        static_cast<std::uint32_t>(command);
    b.mb_in[snn_v06::MAILBOX_START_MODE] = static_cast<std::uint32_t>(start);
    b.mb_in[snn_v06::MAILBOX_SHIFT_STRIDE] = static_cast<std::uint32_t>(shift);
    b.mb_in[snn_v06::MAILBOX_TAIL_POLICY] = static_cast<std::uint32_t>(tail);
    b.mb_in[snn_v06::MAILBOX_FLAGS] =
        route == snn_v06::AUTO ? 0U : static_cast<std::uint32_t>(route);
    std::atomic_thread_fence(std::memory_order_release);
    b.mb_in[snn_v06::MAILBOX_SEQUENCE] = sequence;
}

void write_output(const char* path, const msrp_v05::Problem& q,
                  const Buffers& b) {
    std::FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::perror(path);
        std::exit(2);
    }
    const std::uint64_t magic = OUTPUT_MAGIC;
    const std::uint32_t header[3] = {1U, static_cast<std::uint32_t>(q.n),
                                     snn_v06::MAILBOX_WORDS};
    std::fwrite(&magic, sizeof(magic), 1, f);
    std::fwrite(header, sizeof(header[0]), 3, f);
    std::fwrite(b.raw.data(), sizeof(b.raw[0]), b.raw.size(), f);
    std::fwrite(b.telemetry.data(), sizeof(b.telemetry[0]), b.telemetry.size(),
                f);
    std::fwrite(b.mb_out.data(), sizeof(b.mb_out[0]), b.mb_out.size(), f);
    std::fclose(f);
}

msrp_v05::Problem generated_problem(int n, int m) {
    msrp_v05::Problem q;
    q.n = n;
    q.m = m;
    q.iterations = 3;
    q.projection_cap = 16;
    q.k0 = 0.05;
    q.constraint_tol = 1e-6;
    q.has_lower = true;
    q.lower = -1.0;
    q.has_upper = true;
    q.upper = 1.0;
    q.A.assign(static_cast<std::size_t>(n) * n, 0.0);
    q.C.assign(static_cast<std::size_t>(m) * n, 0.0);
    q.G.assign(static_cast<std::size_t>(m) * m, 0.0);
    q.b.assign(n, 0.0);
    q.d.assign(m, 0.0);
    q.c_norms_sq.assign(m, 0.0);
    q.row_scale.assign(m, 1.0);
    q.x0.assign(n, 0.0);
    for (int i = 0; i < n; ++i) {
        q.A[static_cast<std::size_t>(i) * n + i] = 1.0;
        q.b[i] = (i & 1) ? -0.1 : 0.1;
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j)
            q.C[static_cast<std::size_t>(i) * n + j] =
                ((i + j) & 1) ? -0.05 : 0.05;
        q.d[i] = -0.2;
        q.c_norms_sq[i] = 0.05 * 0.05 * n;
        for (int j = 0; j < m; ++j)
            q.G[static_cast<std::size_t>(i) * m + j] =
                (i == j) ? q.c_norms_sq[i] : 0.0;
    }
    return q;
}

int run(const msrp_v05::Problem& q, const char* output, bool persistent) {
    Buffers buffers;
    allocate_buffers(q, buffers);
    std::uint32_t sequence = 1;
    int error = 0;
    if (!persistent) {
        buffers.mb_in[snn_v06::MAILBOX_SEQUENCE] = sequence;
        call_kernel(q, buffers, snn_v06::CONFIGURE, snn_v06::ONESHOT,
                    snn_v06::AUTO, snn_v06::HOST_X0, 0, snn_v06::HOLD_TAIL);
        error = static_cast<int>(buffers.mb_out[snn_v06::OUT_ERROR_CODE]);
        ++sequence;
        buffers.mb_in[snn_v06::MAILBOX_SEQUENCE] = sequence;
        call_kernel(q, buffers, snn_v06::SOLVE, snn_v06::ONESHOT,
                    snn_v06::AUTO, snn_v06::HOST_X0, 0, snn_v06::HOLD_TAIL);
        error = static_cast<int>(buffers.mb_out[snn_v06::OUT_ERROR_CODE]);
    } else {
        publish(buffers, sequence, snn_v06::CONFIGURE, snn_v06::HOST_X0, 0,
                snn_v06::HOLD_TAIL);
        std::thread worker([&]() {
            call_kernel(q, buffers, snn_v06::CONFIGURE, snn_v06::PERSISTENT,
                        snn_v06::AUTO, snn_v06::HOST_X0, 0,
                        snn_v06::HOLD_TAIL);
        });
        wait_done(buffers, sequence);
        error = static_cast<int>(buffers.mb_out[snn_v06::OUT_ERROR_CODE]);
        ++sequence;
        publish(buffers, sequence, snn_v06::SOLVE, snn_v06::HOST_X0, 0,
                snn_v06::HOLD_TAIL);
        wait_done(buffers, sequence);
        error = static_cast<int>(buffers.mb_out[snn_v06::OUT_ERROR_CODE]);
        ++sequence;
        publish(buffers, sequence, snn_v06::STOP, snn_v06::HOST_X0, 0,
                snn_v06::HOLD_TAIL);
        wait_done(buffers, sequence);
        worker.join();
    }

    write_output(output, q, buffers);
    std::printf(
        "{\"schema\":\"msrp-v06-native-fixed-v1\",\"n\":%d,\"m\":%d,"
        "\"persistent\":%s,\"error\":%u,\"status\":%llu,"
        "\"iterations\":%llu,\"route\":%u}\n",
        q.n, q.m, persistent ? "true" : "false", error,
        static_cast<unsigned long long>(buffers.telemetry[1]),
        static_cast<unsigned long long>(buffers.telemetry[3]),
        static_cast<unsigned>(buffers.mb_out[snn_v06::OUT_SELECTED_ROUTE]));
    return error == snn_v06::ERR_OK ? 0 : 3;
}

}  // namespace

int main(int argc, char** argv) {
    bool persistent = false;
    if (argc >= 2 && std::strcmp(argv[1], "--generate") == 0) {
        if (argc < 5 || argc > 6) {
            std::fprintf(stderr,
                         "usage: %s --generate N M output.bin [--persistent]\n",
                         argv[0]);
            return 2;
        }
        const int n = std::atoi(argv[2]);
        const int m = std::atoi(argv[3]);
        persistent = argc == 6 && std::strcmp(argv[5], "--persistent") == 0;
        if (argc == 6 && !persistent) return 2;
        const auto q = generated_problem(n, m);
        return run(q, argv[4], persistent);
    }
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr,
                     "usage: %s problem.bin output.bin [--persistent]\n",
                     argv[0]);
        return 2;
    }
    persistent = argc == 4 && std::strcmp(argv[3], "--persistent") == 0;
    if (argc == 4 && !persistent) return 2;
    const auto q = msrp_v05::load_problem(argv[1]);
    return run(q, argv[2], persistent);
}
