// Byte comparison of snn_qp_v09b against the untouched v0.6 kernel.
// Both kernels are linked as their own translation units. The gate is exact
// equality of the raw state words, all 16 telemetry words, and the mailbox
// error, status, iteration count, and selected route. DDR images are not
// compared; see compare_snap.

#include "msrp_bundle.hpp"
#include "v09_abi.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
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

extern "C" void snn_qp_v09b(
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

using Problem = msrp_v05::Problem;
using KernelFn = decltype(&snn_qp_v09b);

struct Snapshot {
    std::vector<long long> raw;
    std::array<unsigned long long, snn_v08::TELEMETRY_WORDS> telemetry{};
    std::array<std::uint32_t, snn_v08::MAILBOX_WORDS> mailbox{};
    int error = 0;
};

struct RunResult {
    Snapshot snap;
    double seconds = 0.0;
};

class Session {
  public:
    KernelFn kernel;
    Problem q;
    int route;
    int launch;
    std::vector<double> payload_b;
    std::vector<double> payload_d;
    std::vector<double> payload_x0;
    std::vector<std::uint32_t> a_ddr, c_ddr, ct_ddr, g_ddr;
    std::vector<long long> raw;
    std::array<unsigned long long, snn_v08::TELEMETRY_WORDS> telemetry{};
    std::array<std::uint32_t, snn_v08::MAILBOX_WORDS> mb_in{};
    std::array<std::uint32_t, snn_v08::MAILBOX_WORDS> mb_out{};
    std::thread worker;
    std::uint32_t sequence = 0;
    bool running = false;

    Session(KernelFn fn, const Problem& in, int requested_route, int requested_launch)
        : kernel(fn), q(in), route(requested_route), launch(requested_launch),
          payload_b(in.b), payload_d(in.d), payload_x0(in.x0) {
        a_ddr.resize(static_cast<std::size_t>(q.n) * q.n);
        c_ddr.resize(static_cast<std::size_t>(q.m) * q.n);
        ct_ddr.resize(static_cast<std::size_t>(q.m) * q.n);
        g_ddr.resize(static_cast<std::size_t>(q.m) * q.m);
        raw.assign(static_cast<std::size_t>(q.n), 0);
        mb_in.fill(0);
        mb_out.fill(0);
    }

    void invoke(int command, int start, int shift, int tail) {
        kernel(q.A.data(), q.C.data(), q.G.data(), q.c_norms_sq.data(),
               q.row_scale.data(), payload_b.data(), payload_d.data(),
               payload_x0.data(), a_ddr.data(), c_ddr.data(), ct_ddr.data(),
               g_ddr.data(), raw.data(), telemetry.data(), mb_in.data(),
               mb_out.data(), command, launch, route, start, shift, tail, q.n,
               q.m, q.k0, q.constraint_tol, q.iterations, q.projection_cap,
               q.has_lower ? 1 : 0, q.lower, q.has_upper ? 1 : 0, q.upper);
    }

    void wait_done(std::uint32_t wanted) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::minutes(30);
        const volatile std::uint32_t* done = mb_out.data();
        while (done[snn_v08::OUT_DONE_SEQUENCE] != wanted) {
            if (std::chrono::steady_clock::now() >= deadline) {
                std::fprintf(stderr, "timeout waiting for sequence %u\n", wanted);
                std::exit(4);
            }
            std::this_thread::yield();
        }
        std::atomic_thread_fence(std::memory_order_acquire);
    }

    void publish(std::uint32_t seq, int command, int start, int shift, int tail) {
        mb_in[snn_v08::MAILBOX_COMMAND] = static_cast<std::uint32_t>(command);
        mb_in[snn_v08::MAILBOX_START_MODE] = static_cast<std::uint32_t>(start);
        mb_in[snn_v08::MAILBOX_SHIFT_STRIDE] = static_cast<std::uint32_t>(shift);
        mb_in[snn_v08::MAILBOX_TAIL_POLICY] = static_cast<std::uint32_t>(tail);
        mb_in[snn_v08::MAILBOX_FLAGS] = 0;
        std::atomic_thread_fence(std::memory_order_release);
        mb_in[snn_v08::MAILBOX_SEQUENCE] = seq;
    }

    void set_payload(const std::vector<double>& b, const std::vector<double>& d,
                     const std::vector<double>& x0) {
        payload_b = b;
        payload_d = d;
        payload_x0 = x0;
    }

    int configure() {
        sequence = 1;
        if (launch == snn_v08::PERSISTENT) {
            publish(sequence, snn_v08::CONFIGURE, snn_v08::HOST_X0, 0,
                    snn_v08::HOLD_TAIL);
            running = true;
            worker = std::thread([this]() {
                invoke(snn_v08::CONFIGURE, snn_v08::HOST_X0, 0, snn_v08::HOLD_TAIL);
            });
            wait_done(sequence);
        } else {
            mb_in[snn_v08::MAILBOX_SEQUENCE] = sequence;
            invoke(snn_v08::CONFIGURE, snn_v08::HOST_X0, 0, snn_v08::HOLD_TAIL);
        }
        return static_cast<int>(mb_out[snn_v08::OUT_ERROR_CODE]);
    }

    Snapshot solve(int start, int shift, int tail) {
        ++sequence;
        if (launch == snn_v08::PERSISTENT) {
            publish(sequence, snn_v08::SOLVE, start, shift, tail);
            wait_done(sequence);
        } else {
            mb_in[snn_v08::MAILBOX_SEQUENCE] = sequence;
            invoke(snn_v08::SOLVE, start, shift, tail);
        }
        Snapshot out;
        out.raw = raw;
        out.telemetry = telemetry;
        out.mailbox = mb_out;
        out.error = static_cast<int>(mb_out[snn_v08::OUT_ERROR_CODE]);
        return out;
    }

    int refresh_inplace() {
        ++sequence;
        if (launch == snn_v08::PERSISTENT) {
            publish(sequence, snn_v08::REFRESH_A, snn_v08::HOST_X0, 0,
                    snn_v08::HOLD_TAIL);
            wait_done(sequence);
        } else {
            mb_in[snn_v08::MAILBOX_SEQUENCE] = sequence;
            invoke(snn_v08::REFRESH_A, snn_v08::HOST_X0, 0, snn_v08::HOLD_TAIL);
        }
        return static_cast<int>(mb_out[snn_v08::OUT_ERROR_CODE]);
    }

    void stop() {
        if (!running) return;
        ++sequence;
        publish(sequence, snn_v08::STOP, snn_v08::HOST_X0, 0, snn_v08::HOLD_TAIL);
        wait_done(sequence);
        worker.join();
        running = false;
    }

    ~Session() { stop(); }
};

// DDR images (A_ddr, C_ddr, Ct_ddr, G_ddr) are not compared. The session
// keeps them, and a snapshot does not copy them. This gate stays on the
// host-visible words: raw state, the 16 telemetry words, and mailbox error,
// status, iterations, and selected route.
bool compare_snap(const std::string& name, const Snapshot& a, const Snapshot& b,
                  double seconds) {
    int raw_bad = 0;
    int tel_bad = 0;
    bool error_bad = a.error != b.error;
    const unsigned route_a = a.mailbox[snn_v08::OUT_SELECTED_ROUTE];
    const unsigned route_b = b.mailbox[snn_v08::OUT_SELECTED_ROUTE];
    const unsigned status_a = a.mailbox[snn_v08::OUT_STATUS];
    const unsigned status_b = b.mailbox[snn_v08::OUT_STATUS];
    const unsigned iters_a = a.mailbox[snn_v08::OUT_ITERATIONS];
    const unsigned iters_b = b.mailbox[snn_v08::OUT_ITERATIONS];
    bool route_bad = route_a != route_b;
    bool status_bad = status_a != status_b;
    bool iters_bad = iters_a != iters_b;
    if (a.raw.size() != b.raw.size()) {
        std::printf("MISMATCH fixture=%s raw_state_words length v06=%zu v09b=%zu\n",
                    name.c_str(), a.raw.size(), b.raw.size());
        raw_bad = -1;
    } else {
        for (std::size_t i = 0; i < a.raw.size(); ++i) {
            if (a.raw[i] != b.raw[i]) {
                if (raw_bad < 8) {
                    std::printf("MISMATCH fixture=%s raw[%zu] v06=%lld v09b=%lld\n",
                                name.c_str(), i,
                                static_cast<long long>(a.raw[i]),
                                static_cast<long long>(b.raw[i]));
                }
                ++raw_bad;
            }
        }
    }
    for (std::size_t i = 0; i < snn_v08::TELEMETRY_WORDS; ++i) {
        if (a.telemetry[i] != b.telemetry[i]) {
            std::printf(
                "MISMATCH fixture=%s telemetry[%zu] v06=%llu v09b=%llu\n",
                name.c_str(), i,
                static_cast<unsigned long long>(a.telemetry[i]),
                static_cast<unsigned long long>(b.telemetry[i]));
            ++tel_bad;
        }
    }
    if (error_bad) {
        std::printf("MISMATCH fixture=%s error v06=%d v09b=%d\n",
                    name.c_str(), a.error, b.error);
    }
    if (route_bad) {
        std::printf("MISMATCH fixture=%s route v06=%u v09b=%u\n",
                    name.c_str(), route_a, route_b);
    }
    if (status_bad) {
        std::printf("MISMATCH fixture=%s status v06=%u v09b=%u\n",
                    name.c_str(), status_a, status_b);
    }
    if (iters_bad) {
        std::printf("MISMATCH fixture=%s iterations v06=%u v09b=%u\n",
                    name.c_str(), iters_a, iters_b);
    }
    const bool ok = raw_bad == 0 && tel_bad == 0 && !error_bad && !route_bad &&
                    !status_bad && !iters_bad;
    if (ok) {
        std::printf(
            "MATCH fixture=%s raw_state_words=%zu all_equal telemetry[0..15] "
            "all_equal error=%d status=%u iterations=%u route=%u seconds=%.3f\n",
            name.c_str(), a.raw.size(), a.error, status_a, iters_a, route_a,
            seconds);
    } else {
        std::printf(
            "MISMATCH fixture=%s raw_state_words_differ=%d "
            "telemetry_words_differ=%d error_differ=%d status_differ=%d "
            "iterations_differ=%d route_differ=%d seconds=%.3f\n",
            name.c_str(), raw_bad, tel_bad, error_bad ? 1 : 0,
            status_bad ? 1 : 0, iters_bad ? 1 : 0, route_bad ? 1 : 0, seconds);
    }
    std::fflush(stdout);
    return ok;
}

Snapshot run_pair_kernel(KernelFn fn, const Problem& q, int route, int launch,
                         int start, int shift, int tail) {
    Session session(fn, q, route, launch);
    const int err = session.configure();
    if (err != snn_v08::ERR_OK) {
        Snapshot out;
        out.error = err;
        out.raw = session.raw;
        out.telemetry = session.telemetry;
        out.mailbox = session.mb_out;
        return out;
    }
    return session.solve(start, shift, tail);
}

bool run_solve(const std::string& name, const Problem& q, int launch, int start,
               int route = snn_v08::FORCE_FULL, int shift = 0,
               int tail = snn_v08::HOLD_TAIL) {
    const auto t0 = std::chrono::steady_clock::now();
    std::printf(
        "RUN fixture=%s n=%d m=%d iters=%d launch=%s start=%d route=%d "
        "shift=%d tail=%d\n",
        name.c_str(), q.n, q.m, q.iterations,
        launch == snn_v08::PERSISTENT ? "PERSISTENT" : "ONESHOT", start, route,
        shift, tail);
    std::fflush(stdout);
    const Snapshot a = run_pair_kernel(snn_qp_v06, q, route, launch, start, shift, tail);
    const Snapshot b = run_pair_kernel(snn_qp_v09b, q, route, launch, start, shift, tail);
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - t0)
                               .count();
    return compare_snap(name, a, b, seconds);
}

bool run_refresh(const std::string& name, const Problem& q, int launch) {
    const auto t0 = std::chrono::steady_clock::now();
    auto once = [&](KernelFn fn) {
        Session session(fn, q, snn_v08::FORCE_FULL, launch);
        struct Pair { Snapshot first; Snapshot second; int refresh_error; } pair;
        if (session.configure() != snn_v08::ERR_OK) {
            pair.refresh_error = -1;
            return pair;
        }
        pair.first = session.solve(snn_v08::HOST_X0, 0, snn_v08::HOLD_TAIL);
        for (double& value : session.q.A) value *= 0.61;
        pair.refresh_error = session.refresh_inplace();
        pair.second = session.solve(snn_v08::HOST_X0, 0, snn_v08::HOLD_TAIL);
        return pair;
    };
    const auto left = once(snn_qp_v06);
    const auto right = once(snn_qp_v09b);
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - t0)
                               .count();
    const bool refresh_ok = left.refresh_error == snn_v08::ERR_OK &&
                            right.refresh_error == snn_v08::ERR_OK;
    if (!refresh_ok) {
        std::printf("MISMATCH fixture=%s refresh_error v06=%d v09b=%d\n",
                    name.c_str(), left.refresh_error, right.refresh_error);
    }
    const bool a = compare_snap(name + ":before_refresh", left.first, right.first, seconds);
    const bool b = compare_snap(name + ":after_refresh", left.second, right.second, 0.0);
    return refresh_ok && a && b;
}

Problem random_problem(int n, int m, std::uint64_t seed, int box_mode) {
    Problem q;
    q.n = n;
    q.m = m;
    q.iterations = 3;
    q.projection_cap = 8;
    q.k0 = 0.035;
    q.constraint_tol = 1e-6;
    q.has_lower = box_mode == 1 || box_mode == 3;
    q.has_upper = box_mode == 2 || box_mode == 3;
    q.lower = -0.75;
    q.upper = 0.75;
    q.A.resize(static_cast<std::size_t>(n) * n);
    q.b.resize(static_cast<std::size_t>(n));
    q.C.resize(static_cast<std::size_t>(m) * n);
    q.d.resize(static_cast<std::size_t>(m));
    q.c_norms_sq.resize(static_cast<std::size_t>(m));
    q.row_scale.resize(static_cast<std::size_t>(m));
    q.G.resize(static_cast<std::size_t>(m) * m);
    q.x0.resize(static_cast<std::size_t>(n));
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-0.28, 0.28);
    for (int i = 0; i < n; ++i) {
        q.b[static_cast<std::size_t>(i)] = u(rng);
        q.x0[static_cast<std::size_t>(i)] = 0.1 + u(rng);
        for (int j = 0; j < n; ++j)
            q.A[static_cast<std::size_t>(i) * n + j] =
                (i == j ? 0.18 : 0.0) + u(rng) * 0.18;
    }
    for (int i = 0; i < m; ++i) {
        double norm = 0.0;
        for (int j = 0; j < n; ++j) {
            const double v = u(rng);
            q.C[static_cast<std::size_t>(i) * n + j] = v;
            norm += v * v;
        }
        q.d[static_cast<std::size_t>(i)] = (i & 1) ? -0.05 : -0.55;
        q.c_norms_sq[static_cast<std::size_t>(i)] = std::max(norm, 0.01);
        q.row_scale[static_cast<std::size_t>(i)] =
            1.0 / std::sqrt(q.c_norms_sq[static_cast<std::size_t>(i)]);
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            double dot = 0.0;
            for (int k = 0; k < n; ++k)
                dot += q.C[static_cast<std::size_t>(i) * n + k] *
                       q.C[static_cast<std::size_t>(j) * n + k];
            q.G[static_cast<std::size_t>(i) * m + j] = dot;
        }
    }
    return q;
}

Problem tie_problem() {
    Problem q = random_problem(6, 18, 0x71e5, 0);
    for (int i = 0; i < q.m; ++i) {
        const int source = i % 3;
        q.d[static_cast<std::size_t>(i)] = -0.2;
        q.row_scale[static_cast<std::size_t>(i)] = 1.0;
        for (int j = 0; j < q.n; ++j)
            q.C[static_cast<std::size_t>(i) * q.n + j] =
                q.C[static_cast<std::size_t>(source) * q.n + j];
        q.c_norms_sq[static_cast<std::size_t>(i)] =
            q.c_norms_sq[static_cast<std::size_t>(source)];
    }
    for (int i = 0; i < q.m; ++i)
        for (int j = 0; j < q.m; ++j)
            q.G[static_cast<std::size_t>(i) * q.m + j] =
                (i % 3 == j % 3) ? q.c_norms_sq[static_cast<std::size_t>(i)] : 0.0;
    q.has_lower = true;
    q.lower = -0.1;
    q.x0.assign(static_cast<std::size_t>(q.n), 0.0);
    return q;
}

Problem range_problem() {
    Problem q = random_problem(1, 1, 0x1234, 0);
    q.A[0] = 128.0;
    q.C[0] = 0.5;
    q.G[0] = 0.25;
    q.b[0] = 127.9;
    q.d[0] = -0.25;
    q.c_norms_sq[0] = 0.25;
    q.row_scale[0] = 2.0;
    q.x0[0] = 0.25;
    q.iterations = 2;
    q.projection_cap = 3;
    return q;
}

Problem cap_problem() {
    Problem q = random_problem(2, 2, 0xCAFE, 0);
    q.A.assign(4, 0.0);
    q.A[0] = q.A[3] = 0.1;
    q.C.assign(4, 0.0);
    q.C[0] = 1.0;
    q.C[3] = 1.0;
    q.d[0] = q.d[1] = 3.0;
    q.c_norms_sq[0] = q.c_norms_sq[1] = 1.0;
    q.row_scale[0] = q.row_scale[1] = 1.0;
    q.G.assign(4, 0.0);
    q.G[0] = q.G[3] = 1.0;
    q.x0.assign(2, 0.0);
    q.projection_cap = 1;
    q.iterations = 2;
    q.constraint_tol = 1e-9;
    return q;
}

constexpr int kOutputScratch = 128;
constexpr unsigned long long kLitmusRaw = UINT64_C(0x13579bdf00000000);
constexpr unsigned long long kLitmusTel = UINT64_C(0xa5a5000000000000);
constexpr std::uint32_t kLitmusInput = 0xA5A5A5A5u;

Snapshot oneshot_command(KernelFn fn, const Problem& q, int command, int route,
                         int start, int shift, int tail, int n, int m,
                         int n_iters, int projmax, int has_lower, int has_upper,
                         std::size_t raw_words, std::size_t copy_words,
                         std::uint32_t litmus_input) {
    std::vector<long long> raw(raw_words, 0);
    std::array<unsigned long long, snn_v08::TELEMETRY_WORDS> telemetry{};
    std::array<std::uint32_t, snn_v08::MAILBOX_WORDS> mb_in{};
    std::array<std::uint32_t, snn_v08::MAILBOX_WORDS> mb_out{};
    const std::size_t nn =
        static_cast<std::size_t>(q.n) * static_cast<std::size_t>(q.n);
    const std::size_t mn =
        static_cast<std::size_t>(q.m) * static_cast<std::size_t>(q.n);
    const std::size_t mm =
        static_cast<std::size_t>(q.m) * static_cast<std::size_t>(q.m);
    std::vector<std::uint32_t> a_ddr(nn), c_ddr(mn), ct_ddr(mn), g_ddr(mm);
    mb_in[snn_v08::MAILBOX_SEQUENCE] = 1;
    mb_in[snn_v08::MAILBOX_LITMUS_INPUT] = litmus_input;
    fn(q.A.data(), q.C.data(), q.G.data(), q.c_norms_sq.data(),
       q.row_scale.data(), q.b.data(), q.d.data(), q.x0.data(), a_ddr.data(),
       c_ddr.data(), ct_ddr.data(), g_ddr.data(), raw.data(), telemetry.data(),
       mb_in.data(), mb_out.data(), command, snn_v08::ONESHOT, route, start,
       shift, tail, n, m, q.k0, q.constraint_tol, n_iters, projmax, has_lower,
       q.lower, has_upper, q.upper);
    Snapshot out;
    const std::size_t kept = copy_words < raw.size() ? copy_words : raw.size();
    out.raw.assign(raw.begin(),
                   raw.begin() +
                       static_cast<std::vector<long long>::difference_type>(kept));
    out.telemetry = telemetry;
    out.mailbox = mb_out;
    out.error = static_cast<int>(mb_out[snn_v08::OUT_ERROR_CODE]);
    return out;
}

bool expect_ack(const std::string& name, const Snapshot& snap, int expected_error,
                int expected_route) {
    const unsigned route = snap.mailbox[snn_v08::OUT_SELECTED_ROUTE];
    const unsigned status = snap.mailbox[snn_v08::OUT_STATUS];
    const unsigned iterations = snap.mailbox[snn_v08::OUT_ITERATIONS];
    const bool ok = snap.error == expected_error &&
                    static_cast<int>(route) == expected_route;
    std::printf(
        "%s fixture=%s error=%d route=%u status=%u iterations=%u "
        "expected_error=%d expected_route=%d\n",
        ok ? "PASS" : "FAIL", name.c_str(), snap.error, route, status,
        iterations, expected_error, expected_route);
    std::fflush(stdout);
    return ok;
}

bool preserved_solve(const std::string& name, const Snapshot& control,
                     const Snapshot& after) {
    int raw_bad = 0;
    if (control.raw.size() != after.raw.size()) {
        raw_bad = -1;
    } else {
        for (std::size_t i = 0; i < control.raw.size(); ++i) {
            if (control.raw[i] != after.raw[i]) ++raw_bad;
        }
    }
    int tel_bad = 0;
    for (std::size_t i = 0; i < snn_v08::TELEMETRY_WORDS; ++i) {
        if (control.telemetry[i] != after.telemetry[i]) ++tel_bad;
    }
    const bool error_bad = control.error != after.error;
    const bool route_bad = control.mailbox[snn_v08::OUT_SELECTED_ROUTE] !=
                           after.mailbox[snn_v08::OUT_SELECTED_ROUTE];
    const bool status_bad =
        control.mailbox[snn_v08::OUT_STATUS] != after.mailbox[snn_v08::OUT_STATUS];
    const bool iters_bad = control.mailbox[snn_v08::OUT_ITERATIONS] !=
                           after.mailbox[snn_v08::OUT_ITERATIONS];
    const bool ok = raw_bad == 0 && tel_bad == 0 && !error_bad && !route_bad &&
                    !status_bad && !iters_bad;
    std::printf(
        "%s fixture=%s preserved raw_differ=%d telemetry_differ=%d error=%d "
        "status=%u iterations=%u route=%u\n",
        ok ? "PRESERVED" : "MISMATCH", name.c_str(), raw_bad, tel_bad, after.error,
        after.mailbox[snn_v08::OUT_STATUS], after.mailbox[snn_v08::OUT_ITERATIONS],
        after.mailbox[snn_v08::OUT_SELECTED_ROUTE]);
    std::fflush(stdout);
    return ok;
}

// First calls on each kernel in this process. A solve before any successful
// configure is rejected, and the acknowledgement route word stays AUTO.
int run_preconfigure_solve() {
    const Problem q = random_problem(6, 18, 2, 0);
    const int n = q.n;
    const std::size_t words = static_cast<std::size_t>(n);
    auto once = [&](KernelFn fn) {
        return oneshot_command(fn, q, snn_v08::SOLVE, snn_v08::FORCE_FULL,
                               snn_v08::HOST_X0, 0, snn_v08::HOLD_TAIL, n, q.m,
                               q.iterations, q.projection_cap, 0, 0, words, words,
                               0);
    };
    const Snapshot left = once(snn_qp_v06);
    const Snapshot right = once(snn_qp_v09b);
    const bool ack_left = expect_ack("preconfigure_solve:v06", left,
                                     snn_v08::ERR_NOT_CONFIGURED, snn_v08::AUTO);
    const bool ack_right = expect_ack("preconfigure_solve:v09b", right,
                                      snn_v08::ERR_NOT_CONFIGURED, snn_v08::AUTO);
    const bool match = compare_snap("preconfigure_solve", left, right, 0.0);
    return (ack_left && ack_right && match) ? 0 : 1;
}

bool litmus_pattern(const Snapshot& snap, int n, std::uint32_t marker) {
    if (static_cast<int>(snap.raw.size()) != n) return false;
    for (int i = 0; i < n; ++i) {
        const unsigned long long want =
            kLitmusRaw | static_cast<unsigned long long>(i);
        if (static_cast<unsigned long long>(snap.raw[static_cast<std::size_t>(i)]) !=
            want)
            return false;
    }
    if (snap.telemetry[0] != static_cast<unsigned long long>(marker)) return false;
    for (std::size_t i = 1; i < snn_v08::TELEMETRY_WORDS; ++i) {
        if (snap.telemetry[i] != (kLitmusTel | i)) return false;
    }
    return snap.error == static_cast<int>(snn_v08::ERR_OK) &&
           snap.mailbox[snn_v08::OUT_SELECTED_ROUTE] ==
               static_cast<std::uint32_t>(snn_v08::AUTO);
}

// Runs before any successful configure, so both kernels still report AUTO.
int run_litmus() {
    const Problem q = random_problem(6, 18, 11, 0);
    const int n = q.n;
    const std::size_t words = static_cast<std::size_t>(n);
    auto once = [&](KernelFn fn) {
        return oneshot_command(fn, q, snn_v08::LITMUS, snn_v08::FORCE_FULL,
                               snn_v08::HOST_X0, 0, snn_v08::HOLD_TAIL, n, q.m,
                               q.iterations, q.projection_cap, 0, 0, words, words,
                               kLitmusInput);
    };
    const Snapshot left = once(snn_qp_v06);
    const Snapshot right = once(snn_qp_v09b);
    const bool pattern_left = litmus_pattern(left, n, kLitmusInput);
    const bool pattern_right = litmus_pattern(right, n, kLitmusInput);
    std::printf("%s fixture=litmus_pattern v06=%d v09b=%d marker=0x%08x\n",
                (pattern_left && pattern_right) ? "PASS" : "FAIL",
                pattern_left ? 1 : 0, pattern_right ? 1 : 0, kLitmusInput);
    std::fflush(stdout);
    const bool match = compare_snap("litmus", left, right, 0.0);
    return (pattern_left && pattern_right && match) ? 0 : 1;
}

// configure_geometry checks dimensions and the scalar bounds before
// choose_route. An over-cap FORCE_CG or FORCE_STREAM is therefore
// ERR_BAD_DIMENSIONS, not ERR_CAPACITY. A rejected configure does not write
// configured_*. The following solve must match the solve taken before it.
int run_rejection_battery() {
    const Problem q = random_problem(6, 18, 2, 0);
    const int n = q.n;
    const int m = q.m;
    const int iters = q.iterations;
    const int proj = q.projection_cap;
    const std::size_t raw_n = static_cast<std::size_t>(n);
    auto solve_good = [&]() {
        return oneshot_command(
            snn_qp_v09b, q, snn_v08::SOLVE, snn_v08::FORCE_FULL, snn_v08::HOST_X0,
            0, snn_v08::HOLD_TAIL, n, m, iters, proj, 0, 0, raw_n, raw_n, 0);
    };
    auto configure = [&](int cn, int cm, int citers, int cproj, int lower,
                         int upper, int route) {
        return oneshot_command(snn_qp_v09b, q, snn_v08::CONFIGURE, route,
                               snn_v08::HOST_X0, 0, snn_v08::HOLD_TAIL, cn, cm,
                               citers, cproj, lower, upper, kOutputScratch, 0, 0);
    };

    int failures = 0;
    const Snapshot good = configure(n, m, iters, proj, 0, 0, snn_v08::FORCE_FULL);
    if (!expect_ack("reject:configure_good", good, snn_v08::ERR_OK,
                    snn_v08::FORCE_FULL))
        ++failures;
    const Snapshot control = solve_good();
    if (!expect_ack("reject:control_solve", control, snn_v08::ERR_OK,
                    snn_v08::FORCE_FULL))
        ++failures;

    struct Case {
        const char* name;
        int cn;
        int cm;
        int citers;
        int cproj;
        int lower;
        int upper;
        int route;
        int expected;
    };
    const Case cases[] = {
        {"over_cap", 129, 200, iters, proj, 0, 0, snn_v08::FORCE_FULL,
         snn_v08::ERR_BAD_DIMENSIONS},
        {"force_cg", n, m, iters, proj, 0, 0, snn_v08::FORCE_CG,
         snn_v08::ERR_CAPACITY},
        {"force_stream", n, m, iters, proj, 0, 0, snn_v08::FORCE_STREAM,
         snn_v08::ERR_CAPACITY},
        {"over_cap_force_cg", 129, 200, iters, proj, 0, 0, snn_v08::FORCE_CG,
         snn_v08::ERR_BAD_DIMENSIONS},
        {"over_cap_force_stream", 129, 200, iters, proj, 0, 0,
         snn_v08::FORCE_STREAM, snn_v08::ERR_BAD_DIMENSIONS},
        {"n_zero", 0, m, iters, proj, 0, 0, snn_v08::FORCE_FULL,
         snn_v08::ERR_BAD_DIMENSIONS},
        {"n_negative", -1, m, iters, proj, 0, 0, snn_v08::FORCE_FULL,
         snn_v08::ERR_BAD_DIMENSIONS},
        {"iters_zero", n, m, 0, proj, 0, 0, snn_v08::FORCE_FULL,
         snn_v08::ERR_BAD_DIMENSIONS},
        {"projmax_zero", n, m, iters, 0, 0, 0, snn_v08::FORCE_FULL,
         snn_v08::ERR_BAD_DIMENSIONS},
        {"has_lower_2", n, m, iters, proj, 2, 0, snn_v08::FORCE_FULL,
         snn_v08::ERR_BAD_DIMENSIONS},
    };
    for (const Case& item : cases) {
        const Snapshot rejected =
            configure(item.cn, item.cm, item.citers, item.cproj, item.lower,
                      item.upper, item.route);
        const std::string ack_name = std::string("reject:") + item.name;
        if (!expect_ack(ack_name, rejected, item.expected, snn_v08::FORCE_FULL))
            ++failures;
        if (!preserved_solve(ack_name, control, solve_good())) ++failures;
    }
    return failures;
}

int run_extras() {
    int failures = 0;
    failures += run_preconfigure_solve();
    failures += run_litmus();
    failures += run_rejection_battery();
    const Problem base = random_problem(6, 18, 2, 0);
    if (!run_solve("extra:random_6x18", base, snn_v08::ONESHOT, snn_v08::HOST_X0))
        ++failures;
    if (!run_solve("extra:random_6x18_warm", base, snn_v08::ONESHOT,
                   snn_v08::RESIDENT_WARM))
        ++failures;
    if (!run_solve("extra:random_6x18_cold", base, snn_v08::ONESHOT,
                   snn_v08::COLD_ZERO))
        ++failures;
    if (!run_solve("extra:random_6x18_persistent", base, snn_v08::PERSISTENT,
                   snn_v08::HOST_X0))
        ++failures;
    if (!run_solve("extra:random_6x18_persistent_warm", base, snn_v08::PERSISTENT,
                   snn_v08::RESIDENT_WARM))
        ++failures;
    if (!run_solve("extra:random_6x18_persistent_cold", base, snn_v08::PERSISTENT,
                   snn_v08::COLD_ZERO))
        ++failures;
    if (!run_solve("extra:shift_warm_hold", base, snn_v08::ONESHOT,
                   snn_v08::RESIDENT_WARM, snn_v08::FORCE_FULL, 2,
                   snn_v08::HOLD_TAIL))
        ++failures;
    if (!run_solve("extra:shift_warm_repeat", base, snn_v08::ONESHOT,
                   snn_v08::RESIDENT_WARM, snn_v08::FORCE_FULL, 2,
                   snn_v08::REPEAT_LAST))
        ++failures;
    if (!run_solve("extra:shift_warm_hold_persistent", base, snn_v08::PERSISTENT,
                   snn_v08::RESIDENT_WARM, snn_v08::FORCE_FULL, 2,
                   snn_v08::HOLD_TAIL))
        ++failures;
    if (!run_solve("extra:shift_warm_repeat_persistent", base,
                   snn_v08::PERSISTENT, snn_v08::RESIDENT_WARM,
                   snn_v08::FORCE_FULL, 2, snn_v08::REPEAT_LAST))
        ++failures;
    if (!run_solve("extra:box_lower", random_problem(6, 18, 7, 1),
                   snn_v08::ONESHOT, snn_v08::HOST_X0))
        ++failures;
    if (!run_solve("extra:box_upper", random_problem(6, 18, 8, 2),
                   snn_v08::ONESHOT, snn_v08::HOST_X0))
        ++failures;
    if (!run_solve("extra:box_both", random_problem(6, 18, 9, 3),
                   snn_v08::ONESHOT, snn_v08::HOST_X0))
        ++failures;
    if (!run_solve("extra:box_none", random_problem(6, 18, 10, 0),
                   snn_v08::ONESHOT, snn_v08::HOST_X0))
        ++failures;
    if (!run_solve("extra:tie", tie_problem(), snn_v08::ONESHOT, snn_v08::HOST_X0))
        ++failures;
    if (!run_solve("extra:range_edge", range_problem(), snn_v08::ONESHOT,
                   snn_v08::HOST_X0))
        ++failures;
    if (!run_solve("extra:cap_exhaustion", cap_problem(), snn_v08::ONESHOT,
                   snn_v08::HOST_X0))
        ++failures;
    if (!run_refresh("extra:refresh_6x18", base, snn_v08::ONESHOT)) ++failures;
    if (!run_refresh("extra:refresh_6x18_persistent", base, snn_v08::PERSISTENT))
        ++failures;
    return failures;
}

}  // namespace

int main(int argc, char** argv) {
    bool extras = true;
    std::vector<std::string> bundles;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-extra") == 0) {
            extras = false;
        } else {
            bundles.emplace_back(argv[i]);
        }
    }
    if (bundles.empty()) {
        std::fprintf(stderr, "usage: %s [--no-extra] bundle.bin ...\n", argv[0]);
        return 2;
    }

    int failures = 0;
    if (extras) failures += run_extras();

    for (const std::string& path : bundles) {
        const Problem q = msrp_v05::load_problem(path, false, 128, 128);
        const std::string name = "bundle:" + path.substr(path.find_last_of('/') + 1);
        if (!run_solve(name, q, snn_v08::ONESHOT, snn_v08::HOST_X0)) ++failures;
    }

    std::printf("PARITY SUMMARY failures=%d %s\n", failures,
                failures == 0 ? "PASS" : "FAIL");
    std::fflush(stdout);
    return failures == 0 ? 0 : 1;
}
