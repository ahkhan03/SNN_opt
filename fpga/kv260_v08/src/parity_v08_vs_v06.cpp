// Native byte comparison of snn_qp_v08 against the untouched v0.6 kernel.
// Fixture generators match the v0.6 parity battery. Both kernels keep their
// own file-scope state, so each translation unit is linked as-is.

#include "msrp_bundle.hpp"
#include "v08_abi.hpp"

#include <algorithm>
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
#include <utility>
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

extern "C" void snn_qp_v08(
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
using KernelFn = decltype(&snn_qp_v08);

constexpr int MAX_WAIT_MS = 120000;
constexpr int FLEET_WAIT_MS = 3 * 60 * 60 * 1000;

struct Snapshot {
    std::vector<long long> raw;
    std::array<unsigned long long, snn_v08::TELEMETRY_WORDS> telemetry{};
    std::array<std::uint32_t, snn_v08::MAILBOX_WORDS> mailbox{};
    int error = 0;
};

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
                (i % 3 == j % 3) ? q.c_norms_sq[static_cast<std::size_t>(i)]
                                 : 0.0;
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

Problem zero_problem(int n, int m) {
    Problem q;
    q.n = n;
    q.m = m;
    q.iterations = 1;
    q.projection_cap = 1;
    q.k0 = 0.01;
    q.constraint_tol = 1e-6;
    q.A.assign(static_cast<std::size_t>(n) * n, 0.0);
    q.C.assign(static_cast<std::size_t>(m) * n, 0.0);
    q.G.assign(static_cast<std::size_t>(m) * m, 0.0);
    q.b.assign(static_cast<std::size_t>(n), 0.0);
    q.d.assign(static_cast<std::size_t>(m), 0.0);
    q.c_norms_sq.assign(static_cast<std::size_t>(m), 1.0);
    q.row_scale.assign(static_cast<std::size_t>(m), 1.0);
    q.x0.assign(static_cast<std::size_t>(n), 0.0);
    return q;
}

struct Fixture {
    std::string name;
    Problem q;
};

bool route_fits(int n, int m, int route) {
    const auto padded = [](int columns) -> unsigned long long {
        const unsigned long long width =
            static_cast<unsigned long long>(columns < 0 ? 0 : columns);
        return ((width + 7ULL) / 8ULL) * 8ULL;
    };
    const unsigned long long full =
        static_cast<unsigned long long>(n) * padded(n) +
        static_cast<unsigned long long>(m) * padded(n) +
        static_cast<unsigned long long>(n) * padded(m) +
        static_cast<unsigned long long>(m) * padded(m);
    const unsigned long long cg =
        static_cast<unsigned long long>(m) * padded(n) +
        static_cast<unsigned long long>(n) * padded(m) +
        static_cast<unsigned long long>(m) * padded(m);
    const bool bram =
        16ULL * static_cast<unsigned long long>(n + m) + 4096ULL <= 129024ULL;
    if (!bram || n < 1 || m < 1 || n > 1024 || m > 1024) return false;
    if (route == snn_v08::FORCE_FULL)
        return full <= 440000ULL && (n != m || n <= 320);
    if (route == snn_v08::FORCE_CG)
        return cg <= 440000ULL && (n != m || n <= 372);
    return route == snn_v08::FORCE_STREAM;
}

const char* route_name(int route) {
    if (route == snn_v08::FORCE_FULL) return "FULL";
    if (route == snn_v08::FORCE_CG) return "CG";
    if (route == snn_v08::FORCE_STREAM) return "STREAM";
    return "AUTO";
}

const char* start_name(int start) {
    if (start == snn_v08::RESIDENT_WARM) return "WARM";
    if (start == snn_v08::HOST_X0) return "HOST_X0";
    return "COLD";
}

const char* launch_name(int launch) {
    return launch == snn_v08::PERSISTENT ? "PERSISTENT" : "ONESHOT";
}

class Session {
  public:
    KernelFn kernel;
    Problem q;
    int route;
    int launch;
    int wait_ms;
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

    Session(KernelFn fn, const Problem& in, int requested_route,
            int requested_launch, int requested_wait_ms)
        : kernel(fn), q(in), route(requested_route), launch(requested_launch),
          wait_ms(requested_wait_ms), payload_b(in.b), payload_d(in.d),
          payload_x0(in.x0) {
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
                              std::chrono::milliseconds(wait_ms);
        const volatile std::uint32_t* done = mb_out.data();
        while (done[snn_v08::OUT_DONE_SEQUENCE] != wanted) {
            if (std::chrono::steady_clock::now() >= deadline) {
                std::fprintf(stderr, "timeout waiting for sequence %u\n",
                             wanted);
                std::exit(4);
            }
            std::this_thread::yield();
        }
        std::atomic_thread_fence(std::memory_order_acquire);
    }

    void publish(std::uint32_t seq, int command, int start, int shift,
                 int tail) {
        mb_in[snn_v08::MAILBOX_COMMAND] = static_cast<std::uint32_t>(command);
        mb_in[snn_v08::MAILBOX_START_MODE] = static_cast<std::uint32_t>(start);
        mb_in[snn_v08::MAILBOX_SHIFT_STRIDE] =
            static_cast<std::uint32_t>(shift);
        mb_in[snn_v08::MAILBOX_TAIL_POLICY] = static_cast<std::uint32_t>(tail);
        mb_in[snn_v08::MAILBOX_FLAGS] = 0;
        std::atomic_thread_fence(std::memory_order_release);
        mb_in[snn_v08::MAILBOX_SEQUENCE] = seq;
    }

    void set_payload(const std::vector<double>& b, const std::vector<double>& d,
                     const std::vector<double>& x0) {
        volatile double* vb = payload_b.data();
        volatile double* vd = payload_d.data();
        volatile double* vx = payload_x0.data();
        for (std::size_t i = 0; i < b.size(); ++i) vb[i] = b[i];
        for (std::size_t i = 0; i < d.size(); ++i) vd[i] = d[i];
        for (std::size_t i = 0; i < x0.size(); ++i) vx[i] = x0[i];
        std::atomic_thread_fence(std::memory_order_release);
    }

    int configure() {
        sequence = 1;
        if (launch == snn_v08::PERSISTENT) {
            publish(sequence, snn_v08::CONFIGURE, snn_v08::HOST_X0, 0,
                    snn_v08::HOLD_TAIL);
            running = true;
            worker = std::thread([this]() {
                invoke(snn_v08::CONFIGURE, snn_v08::HOST_X0, 0,
                       snn_v08::HOLD_TAIL);
            });
            wait_done(sequence);
        } else {
            mb_in[snn_v08::MAILBOX_SEQUENCE] = sequence;
            invoke(snn_v08::CONFIGURE, snn_v08::HOST_X0, 0, snn_v08::HOLD_TAIL);
        }
        return static_cast<int>(mb_out[snn_v08::OUT_ERROR_CODE]);
    }

    Snapshot solve(const std::vector<double>& b, const std::vector<double>& d,
                   const std::vector<double>& x0, int start, int shift,
                   int tail) {
        set_payload(b, d, x0);
        ++sequence;
        Snapshot out;
        if (launch == snn_v08::PERSISTENT) {
            publish(sequence, snn_v08::SOLVE, start, shift, tail);
            wait_done(sequence);
        } else {
            mb_in[snn_v08::MAILBOX_SEQUENCE] = sequence;
            invoke(snn_v08::SOLVE, start, shift, tail);
        }
        out.raw = raw;
        out.telemetry = telemetry;
        out.mailbox = mb_out;
        out.error = static_cast<int>(mb_out[snn_v08::OUT_ERROR_CODE]);
        return out;
    }

    Snapshot refresh_only() {
        ++sequence;
        Snapshot out;
        if (launch == snn_v08::PERSISTENT) {
            publish(sequence, snn_v08::REFRESH_A, snn_v08::HOST_X0, 0,
                    snn_v08::HOLD_TAIL);
            wait_done(sequence);
        } else {
            mb_in[snn_v08::MAILBOX_SEQUENCE] = sequence;
            invoke(snn_v08::REFRESH_A, snn_v08::HOST_X0, 0, snn_v08::HOLD_TAIL);
        }
        out.raw = raw;
        out.telemetry = telemetry;
        out.mailbox = mb_out;
        out.error = static_cast<int>(mb_out[snn_v08::OUT_ERROR_CODE]);
        return out;
    }

    void stop() {
        if (!running) return;
        ++sequence;
        publish(sequence, snn_v08::STOP, snn_v08::HOST_X0, 0,
                snn_v08::HOLD_TAIL);
        wait_done(sequence);
        worker.join();
        running = false;
    }

    ~Session() { stop(); }
};

bool words_match(const Snapshot& a, const Snapshot& b, std::string& why) {
    if (a.error != b.error) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "error v06=%d v08=%d", a.error, b.error);
        why = buf;
        return false;
    }
    if (a.mailbox[snn_v08::OUT_SELECTED_ROUTE] !=
        b.mailbox[snn_v08::OUT_SELECTED_ROUTE]) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "route v06=%u v08=%u",
                      a.mailbox[snn_v08::OUT_SELECTED_ROUTE],
                      b.mailbox[snn_v08::OUT_SELECTED_ROUTE]);
        why = buf;
        return false;
    }
    if (a.raw.size() != b.raw.size()) {
        why = "raw length";
        return false;
    }
    for (std::size_t i = 0; i < a.raw.size(); ++i) {
        if (a.raw[i] != b.raw[i]) {
            char buf[160];
            std::snprintf(buf, sizeof(buf), "raw[%zu] v06=%lld v08=%lld", i,
                          static_cast<long long>(a.raw[i]),
                          static_cast<long long>(b.raw[i]));
            why = buf;
            return false;
        }
    }
    for (std::size_t i = 0; i < snn_v08::TELEMETRY_WORDS; ++i) {
        if (a.telemetry[i] != b.telemetry[i]) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "telemetry[%zu] v06=%llu v08=%llu", i,
                          static_cast<unsigned long long>(a.telemetry[i]),
                          static_cast<unsigned long long>(b.telemetry[i]));
            why = buf;
            return false;
        }
    }
    return true;
}

Snapshot capture_configure(Session& session) {
    Snapshot out;
    out.error = session.configure();
    out.raw = session.raw;
    out.telemetry = session.telemetry;
    out.mailbox = session.mb_out;
    return out;
}

bool run_cell(const Fixture& fixture, int route, int start, int launch,
              int wait_ms) {
    Session left(snn_qp_v06, fixture.q, route, launch, wait_ms);
    Session right(snn_qp_v08, fixture.q, route, launch, wait_ms);
    const Snapshot a = [&]() {
        const int err = left.configure();
        if (err != snn_v08::ERR_OK) {
            Snapshot out;
            out.error = err;
            out.raw = left.raw;
            out.telemetry = left.telemetry;
            out.mailbox = left.mb_out;
            return out;
        }
        return left.solve(fixture.q.b, fixture.q.d, fixture.q.x0, start, 0,
                          snn_v08::HOLD_TAIL);
    }();
    const Snapshot b = [&]() {
        const int err = right.configure();
        if (err != snn_v08::ERR_OK) {
            Snapshot out;
            out.error = err;
            out.raw = right.raw;
            out.telemetry = right.telemetry;
            out.mailbox = right.mb_out;
            return out;
        }
        return right.solve(fixture.q.b, fixture.q.d, fixture.q.x0, start, 0,
                           snn_v08::HOLD_TAIL);
    }();

    bool gate = a.error == snn_v08::ERR_OK;
    if (fixture.name == "range_edge") gate = gate && a.telemetry[15] == 1;
    if (fixture.name == "cap_exhaustion")
        gate = gate && a.telemetry[1] == 2 && a.telemetry[9] > 0;

    std::string why;
    const bool same = gate && words_match(a, b, why);
    if (same) {
        std::printf("PASS fixture=%s tier=%s start=%s launch=%s\n",
                    fixture.name.c_str(), route_name(route), start_name(start),
                    launch_name(launch));
    } else if (!gate) {
        std::printf(
            "FAIL fixture=%s tier=%s start=%s launch=%s gate error=%d "
            "status=%llu cap=%llu range=%llu\n",
            fixture.name.c_str(), route_name(route), start_name(start),
            launch_name(launch), a.error,
            static_cast<unsigned long long>(a.telemetry[1]),
            static_cast<unsigned long long>(a.telemetry[9]),
            static_cast<unsigned long long>(a.telemetry[15]));
    } else {
        std::printf("FAIL fixture=%s tier=%s start=%s launch=%s %s\n",
                    fixture.name.c_str(), route_name(route), start_name(start),
                    launch_name(launch), why.c_str());
    }
    std::fflush(stdout);
    return same;
}

bool run_refresh_case(const Fixture& fixture, int route, int launch) {
    auto run_one = [&](KernelFn fn) {
        Session session(fn, fixture.q, route, launch, MAX_WAIT_MS);
        struct Pair {
            Snapshot first;
            Snapshot second;
            bool configured = false;
        } pair;
        pair.configured = session.configure() == snn_v08::ERR_OK;
        pair.first = session.solve(fixture.q.b, fixture.q.d, fixture.q.x0,
                                   snn_v08::HOST_X0, 0, snn_v08::HOLD_TAIL);
        Problem changed = fixture.q;
        for (double& value : changed.A) value *= 0.61;
        session.q.A = changed.A;
        const Snapshot refresh = session.refresh_only();
        pair.configured = pair.configured && refresh.error == snn_v08::ERR_OK;
        pair.second = session.solve(changed.b, changed.d, fixture.q.x0,
                                    snn_v08::HOST_X0, 0, snn_v08::HOLD_TAIL);
        return pair;
    };
    const auto left = run_one(snn_qp_v06);
    const auto right = run_one(snn_qp_v08);
    std::string why;
    const bool ok = left.configured && right.configured &&
                    words_match(left.first, right.first, why) &&
                    words_match(left.second, right.second, why);
    std::printf("%s refresh_A tier=%s launch=%s%s%s\n", ok ? "PASS" : "FAIL",
                route_name(route), launch_name(launch), ok ? "" : " ",
                ok ? "" : why.c_str());
    std::fflush(stdout);
    return ok;
}

bool run_route_boundary_checks() {
    struct Boundary {
        const char* name;
        int n;
        int m;
        int requested;
        int expected_error;
        int expected_route;
    };
    const Boundary checks[] = {
        {"full_320", 320, 320, snn_v08::FORCE_FULL, 0, snn_v08::FORCE_FULL},
        {"full_321_refused", 321, 321, snn_v08::FORCE_FULL,
         snn_v08::ERR_CAPACITY, -1},
        {"cg_321", 321, 321, snn_v08::FORCE_CG, 0, snn_v08::FORCE_CG},
        {"cg_372", 372, 372, snn_v08::FORCE_CG, 0, snn_v08::FORCE_CG},
        {"cg_373_refused", 373, 373, snn_v08::FORCE_CG, snn_v08::ERR_CAPACITY,
         -1},
        {"auto_321", 321, 321, snn_v08::AUTO, 0, snn_v08::FORCE_CG},
        {"auto_373", 373, 373, snn_v08::AUTO, 0, snn_v08::FORCE_STREAM},
    };
    bool all = true;
    for (const Boundary& check : checks) {
        const Problem q = zero_problem(check.n, check.m);
        Session left(snn_qp_v06, q, check.requested, snn_v08::ONESHOT,
                     MAX_WAIT_MS);
        Session right(snn_qp_v08, q, check.requested, snn_v08::ONESHOT,
                      MAX_WAIT_MS);
        const Snapshot a = capture_configure(left);
        const Snapshot b = capture_configure(right);
        const int route =
            static_cast<int>(a.mailbox[snn_v08::OUT_SELECTED_ROUTE]);
        const bool expected =
            a.error == check.expected_error &&
            (check.expected_route < 0 || route == check.expected_route);
        std::string why;
        const bool ok = expected && words_match(a, b, why);
        std::printf(
            "%s route_boundary=%s n=%d m=%d requested=%d route=%d error=%d%s%s\n",
            ok ? "PASS" : "FAIL", check.name, check.n, check.m, check.requested,
            route, a.error, ok ? "" : " ", ok ? "" : why.c_str());
        std::fflush(stdout);
        all = all && ok;
    }
    return all;
}

std::size_t run_fleet_bundle(const std::string& path) {
    const Problem q = msrp_v05::load_problem(path, false, 1024, 1024);
    const std::string label = path.substr(path.find_last_of('/') + 1);
    std::size_t failed = 0;
    for (int route = snn_v08::FORCE_FULL; route <= snn_v08::FORCE_STREAM;
         ++route) {
        if (!route_fits(q.n, q.m, route)) continue;
        for (int launch = snn_v08::ONESHOT; launch <= snn_v08::PERSISTENT;
             ++launch) {
            const auto t0 = std::chrono::steady_clock::now();
            std::printf("RUN fixture=%s n=%d m=%d iters=%d tier=%s launch=%s\n",
                        label.c_str(), q.n, q.m, q.iterations, route_name(route),
                        launch_name(launch));
            std::fflush(stdout);
            const bool ok =
                run_cell({"fleet:" + label, q}, route, snn_v08::HOST_X0, launch,
                         FLEET_WAIT_MS);
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - t0)
                                .count();
            std::printf("DONE fixture=%s tier=%s launch=%s ms=%lld\n",
                        label.c_str(), route_name(route), launch_name(launch),
                        static_cast<long long>(ms));
            std::fflush(stdout);
            if (!ok) ++failed;
        }
    }
    return failed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s fleet_bundle.bin ...\n", argv[0]);
        return 2;
    }

    std::vector<Fixture> fixtures;
    fixtures.push_back({"random_1x1", random_problem(1, 1, 1, 0)});
    fixtures.push_back({"random_6x18", random_problem(6, 18, 2, 0)});
    fixtures.push_back({"random_20x60", random_problem(20, 60, 3, 0)});
    fixtures.push_back({"random_64x64", random_problem(64, 64, 4, 0)});
    fixtures.push_back({"cg_forced_small", random_problem(6, 18, 5, 0)});
    fixtures.push_back({"stream_forced_small", random_problem(6, 18, 6, 0)});
    fixtures.push_back({"tie_heavy", tie_problem()});
    fixtures.push_back({"range_edge", range_problem()});
    fixtures.push_back({"cap_exhaustion", cap_problem()});
    fixtures.push_back({"box_lower_only", random_problem(6, 18, 7, 1)});
    fixtures.push_back({"box_upper_only", random_problem(6, 18, 8, 2)});
    fixtures.push_back({"box_both", random_problem(6, 18, 9, 3)});
    fixtures.push_back({"box_none", random_problem(6, 18, 10, 0)});

    std::size_t cells = 0;
    std::size_t failures = 0;
    for (const Fixture& fixture : fixtures) {
        for (int route = snn_v08::FORCE_FULL; route <= snn_v08::FORCE_STREAM;
             ++route) {
            if (!route_fits(fixture.q.n, fixture.q.m, route)) continue;
            for (int start = snn_v08::RESIDENT_WARM;
                 start <= snn_v08::COLD_ZERO; ++start) {
                for (int launch = snn_v08::ONESHOT;
                     launch <= snn_v08::PERSISTENT; ++launch) {
                    ++cells;
                    if (!run_cell(fixture, route, start, launch, MAX_WAIT_MS))
                        ++failures;
                }
            }
        }
    }

    const Fixture& refresh_fixture = fixtures[1];
    for (int route = snn_v08::FORCE_FULL; route <= snn_v08::FORCE_STREAM;
         ++route) {
        for (int launch = snn_v08::ONESHOT; launch <= snn_v08::PERSISTENT;
             ++launch) {
            ++cells;
            if (!run_refresh_case(refresh_fixture, route, launch)) ++failures;
        }
    }

    if (!run_route_boundary_checks()) ++failures;
    cells += 7;

    if (failures == 0) {
        for (int i = 1; i < argc; ++i) {
            const std::string path(argv[i]);
            const Problem q = msrp_v05::load_problem(path, false, 1024, 1024);
            for (int route = snn_v08::FORCE_FULL;
                 route <= snn_v08::FORCE_STREAM; ++route) {
                if (!route_fits(q.n, q.m, route)) continue;
                cells += 2;
            }
            failures += run_fleet_bundle(path);
        }
    } else {
        std::printf("FLEET skipped because the battery already failed\n");
    }

    std::printf("PARITY SUMMARY cells=%zu failures=%zu %s\n", cells, failures,
                failures == 0 ? "PASS" : "FAIL");
    std::fflush(stdout);
    return failures == 0 ? 0 : 1;
}
