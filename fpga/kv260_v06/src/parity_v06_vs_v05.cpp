// Decision Gate 1: native bit-exact comparison of v06 against untouched v05.

#include "../../kv260_v05/src/msrp_bundle.hpp"
#include "v06_abi.hpp"

#if __has_include("resident_bundle.hpp")
#include "resident_bundle.hpp"
#define V06_HAVE_RESIDENT_BUNDLES 1
#else
#define V06_HAVE_RESIDENT_BUNDLES 0
#endif

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

extern "C" void snn_qp_v05(
    const double* A_in, const double* b_in, const double* C_in,
    const double* d_in, const double* cns_in, const double* row_scale_in,
    const double* G_in, const double* x0_in, long long* x_raw_out,
    unsigned long long* telemetry_out, int n, int m, double k0_f,
    double ctol_f, int n_iters, int projmax, int has_lower, double lower_f,
    int has_upper, double upper_f);

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

using Problem = msrp_v05::Problem;
constexpr int ROUTE_COUNT = 3;
constexpr int START_COUNT = 3;
constexpr int LAUNCH_COUNT = 2;
constexpr int MAX_WAIT_MS = 30000;

struct Snapshot {
    std::vector<long long> raw;
    std::array<unsigned long long, snn_v06::TELEMETRY_WORDS> telemetry{};
    std::array<std::uint32_t, snn_v06::MAILBOX_WORDS> mailbox{};
    int error = 0;
};

Snapshot run_v05(const Problem& q) {
    Snapshot out;
    out.raw.assign(static_cast<std::size_t>(q.n), 0);
    snn_qp_v05(
        q.A.data(), q.b.data(), q.C.data(), q.d.data(), q.c_norms_sq.data(),
        q.row_scale.data(), q.G.data(), q.x0.data(), out.raw.data(),
        out.telemetry.data(), q.n, q.m, q.k0, q.constraint_tol, q.iterations,
        q.projection_cap, q.has_lower ? 1 : 0, q.lower,
        q.has_upper ? 1 : 0, q.upper);
    return out;
}

std::vector<double> fixed_decode(const std::vector<long long>& raw) {
    std::vector<double> out(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i)
        out[i] = static_cast<double>(raw[i]) / static_cast<double>(UINT64_C(1) << 24);
    return out;
}

std::vector<double> shift_host(const std::vector<double>& x, int stride,
                               int tail) {
    if (stride == 0) return x;
    std::vector<double> out(x.size());
    const int n = static_cast<int>(x.size());
    for (int i = 0; i < n; ++i) {
        if (i < n - stride)
            out[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(i + stride)];
        else if (tail == snn_v06::REPEAT_LAST)
            out[static_cast<std::size_t>(i)] = x.back();
        else
            out[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(i)];
    }
    return out;
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
        // Alternate active and slack rows to exercise row and facet winners.
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
    // Exactly the first excluded value exercises the input range latch while
    // remaining close to the ap_fixed<32,8> positive limit.
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
    // Match the v06 packed resident image: each logical row is padded to an
    // eight-element (256-bit) word, and Ct has n rows by m columns.
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
    const bool bram = 16ULL * static_cast<unsigned long long>(n + m) + 4096ULL <= 129024ULL;
    if (!bram || n < 1 || m < 1 || n > 1024 || m > 1024) return false;
    if (route == snn_v06::FORCE_FULL)
        return full <= 440000ULL && (n != m || n <= 320);
    if (route == snn_v06::FORCE_CG)
        return cg <= 440000ULL && (n != m || n <= 372);
    return route == snn_v06::FORCE_STREAM;
}

class V06Session {
  public:
    Problem q;
    int route;
    int launch;
    std::vector<double> payload_b;
    std::vector<double> payload_d;
    std::vector<double> payload_x0;
    std::vector<std::uint32_t> a_ddr, c_ddr, ct_ddr, g_ddr;
    std::vector<long long> raw;
    std::array<unsigned long long, snn_v06::TELEMETRY_WORDS> telemetry{};
    std::array<std::uint32_t, snn_v06::MAILBOX_WORDS> mb_in{};
    std::array<std::uint32_t, snn_v06::MAILBOX_WORDS> mb_out{};
    std::thread worker;
    std::uint32_t sequence = 0;
    bool running = false;

    V06Session(const Problem& in, int requested_route, int requested_launch)
        : q(in), route(requested_route), launch(requested_launch),
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
        snn_qp_v06(
            q.A.data(), q.C.data(), q.G.data(), q.c_norms_sq.data(),
            q.row_scale.data(), payload_b.data(), payload_d.data(),
            payload_x0.data(), a_ddr.data(), c_ddr.data(), ct_ddr.data(),
            g_ddr.data(), raw.data(), telemetry.data(), mb_in.data(),
            mb_out.data(), command, launch, route, start, shift, tail, q.n,
            q.m, q.k0, q.constraint_tol, q.iterations, q.projection_cap,
            q.has_lower ? 1 : 0, q.lower, q.has_upper ? 1 : 0, q.upper);
    }

    void wait_done(std::uint32_t wanted) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(MAX_WAIT_MS);
        const volatile std::uint32_t* done = mb_out.data();
        while (done[snn_v06::OUT_DONE_SEQUENCE] != wanted) {
            if (std::chrono::steady_clock::now() >= deadline) {
                std::fprintf(stderr, "timeout waiting for sequence %u\n", wanted);
                std::exit(4);
            }
            std::this_thread::yield();
        }
        std::atomic_thread_fence(std::memory_order_acquire);
    }

    void publish(std::uint32_t seq, int command, int start, int shift,
                 int tail) {
        mb_in[snn_v06::MAILBOX_COMMAND] = static_cast<std::uint32_t>(command);
        mb_in[snn_v06::MAILBOX_START_MODE] = static_cast<std::uint32_t>(start);
        mb_in[snn_v06::MAILBOX_SHIFT_STRIDE] = static_cast<std::uint32_t>(shift);
        mb_in[snn_v06::MAILBOX_TAIL_POLICY] = static_cast<std::uint32_t>(tail);
        mb_in[snn_v06::MAILBOX_FLAGS] = 0;
        std::atomic_thread_fence(std::memory_order_release);
        mb_in[snn_v06::MAILBOX_SEQUENCE] = seq;
    }

    void set_payload(const std::vector<double>& b,
                     const std::vector<double>& d,
                     const std::vector<double>& x0) {
        // Volatile stores make the native mailbox test obey the same
        // release-before-doorbell rule as an XRT BO sync, without changing
        // the top-level ABI's binary64 pointer types.
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
        if (launch == snn_v06::PERSISTENT) {
            publish(sequence, snn_v06::CONFIGURE, snn_v06::HOST_X0, 0,
                    snn_v06::HOLD_TAIL);
            running = true;
            worker = std::thread([this]() {
                invoke(snn_v06::CONFIGURE, snn_v06::HOST_X0, 0,
                       snn_v06::HOLD_TAIL);
            });
            wait_done(sequence);
        } else {
            mb_in[snn_v06::MAILBOX_SEQUENCE] = sequence;
            invoke(snn_v06::CONFIGURE, snn_v06::HOST_X0, 0,
                   snn_v06::HOLD_TAIL);
        }
        return static_cast<int>(mb_out[snn_v06::OUT_ERROR_CODE]);
    }

    Snapshot solve(const std::vector<double>& b, const std::vector<double>& d,
                   const std::vector<double>& x0, int start, int shift,
                   int tail) {
        set_payload(b, d, x0);
        ++sequence;
        Snapshot out;
        if (launch == snn_v06::PERSISTENT) {
            publish(sequence, snn_v06::SOLVE, start, shift, tail);
            wait_done(sequence);
        } else {
            mb_in[snn_v06::MAILBOX_SEQUENCE] = sequence;
            invoke(snn_v06::SOLVE, start, shift, tail);
        }
        out.raw = raw;
        out.telemetry = telemetry;
        out.mailbox = mb_out;
        out.error = static_cast<int>(mb_out[snn_v06::OUT_ERROR_CODE]);
        return out;
    }

    void stop() {
        if (!running) return;
        ++sequence;
        publish(sequence, snn_v06::STOP, snn_v06::HOST_X0, 0,
                snn_v06::HOLD_TAIL);
        wait_done(sequence);
        worker.join();
        running = false;
    }

    ~V06Session() { stop(); }
};

bool compare_words(const Snapshot& a, const Snapshot& b, std::size_t& index,
                   bool& telemetry_word) {
    const std::size_t raw_count = std::min(a.raw.size(), b.raw.size());
    for (std::size_t i = 0; i < raw_count; ++i) {
        if (a.raw[i] != b.raw[i]) {
            index = i;
            telemetry_word = false;
            return false;
        }
    }
    if (a.raw.size() != b.raw.size()) {
        index = raw_count;
        telemetry_word = false;
        return false;
    }
    for (std::size_t i = 0; i < snn_v06::TELEMETRY_WORDS; ++i) {
        if (a.telemetry[i] != b.telemetry[i]) {
            index = i;
            telemetry_word = true;
            return false;
        }
    }
    return true;
}

bool run_cell(const Fixture& fixture, int route, int start, int launch,
              int shift = 0, int tail = snn_v06::HOLD_TAIL) {
    Problem expected_problem = fixture.q;
    if (start == snn_v06::COLD_ZERO)
        expected_problem.x0.assign(static_cast<std::size_t>(expected_problem.n),
                                   0.0);
    expected_problem.x0 = shift_host(expected_problem.x0, shift, tail);
    const Snapshot expected = run_v05(expected_problem);

    // Make the named edge fixtures self-checking rather than merely relying
    // on equality between two implementations.
    bool fixture_gate = true;
    if (fixture.name == "range_edge") fixture_gate = expected.telemetry[15] == 1;
    if (fixture.name == "cap_exhaustion")
        fixture_gate = expected.telemetry[1] == 2 && expected.telemetry[9] > 0;

    V06Session session(fixture.q, route, launch);
    const int configure_error = session.configure();
    Snapshot observed;
    if (configure_error == snn_v06::ERR_OK) {
        // The kernel applies shift_stride after selecting its start source.
        // Pass the unshifted source here so a nonzero shift is not applied
        // twice in HOST_X0 cells.
        observed = session.solve(fixture.q.b, fixture.q.d, fixture.q.x0,
                                 start, shift, tail);
    } else {
        observed.error = configure_error;
    }

    std::size_t index = 0;
    bool telemetry_word = false;
    const bool same = fixture_gate && observed.error == snn_v06::ERR_OK &&
                      compare_words(expected, observed, index, telemetry_word);
    const char* route_name = route == snn_v06::FORCE_FULL
                                 ? "FULL"
                                 : route == snn_v06::FORCE_CG ? "CG" : "STREAM";
    const char* start_name = start == snn_v06::RESIDENT_WARM
                                 ? "WARM"
                                 : start == snn_v06::HOST_X0 ? "HOST_X0" : "COLD";
    const char* launch_name = launch == snn_v06::PERSISTENT ? "PERSISTENT"
                                                             : "ONESHOT";
    if (same) {
        std::printf("PASS fixture=%s tier=%s start=%s launch=%s\n",
                    fixture.name.c_str(), route_name, start_name, launch_name);
    } else if (observed.error != snn_v06::ERR_OK) {
        std::printf("FAIL fixture=%s tier=%s start=%s launch=%s error=%d\n",
                    fixture.name.c_str(), route_name, start_name, launch_name,
                    observed.error);
    } else if (telemetry_word) {
        std::printf(
            "FAIL fixture=%s tier=%s start=%s launch=%s first_diff=telemetry[%zu] "
            "v05=%llu v06=%llu\n",
            fixture.name.c_str(), route_name, start_name, launch_name, index,
            static_cast<unsigned long long>(expected.telemetry[index]),
            static_cast<unsigned long long>(observed.telemetry[index]));
    } else {
        std::printf(
            "FAIL fixture=%s tier=%s start=%s launch=%s first_diff=raw[%zu] "
            "v05=%lld v06=%lld\n",
            fixture.name.c_str(), route_name, start_name, launch_name, index,
            static_cast<long long>(expected.raw[index]),
            static_cast<long long>(observed.raw[index]));
    }
    return same;
}

bool run_refresh_case(const Fixture& fixture, int route, int launch) {
    V06Session session(fixture.q, route, launch);
    bool ok = session.configure() == snn_v06::ERR_OK;
    const Snapshot first = session.solve(fixture.q.b, fixture.q.d, fixture.q.x0,
                                         snn_v06::HOST_X0, 0,
                                         snn_v06::HOLD_TAIL);
    Problem p1 = fixture.q;
    const Snapshot expected_first = run_v05(p1);
    std::size_t index = 0;
    bool tw = false;
    ok = ok && first.error == 0 && compare_words(expected_first, first, index, tw);

    Problem p2 = fixture.q;
    for (double& value : p2.A) value *= 0.61;
    // Refresh uses the changed A while all other resident geometry remains.
    session.q.A = p2.A;
    ++session.sequence;
    if (launch == snn_v06::PERSISTENT) {
        session.publish(session.sequence, snn_v06::REFRESH_A,
                        snn_v06::HOST_X0, 0, snn_v06::HOLD_TAIL);
        session.wait_done(session.sequence);
    } else {
        session.mb_in[snn_v06::MAILBOX_SEQUENCE] = session.sequence;
        session.invoke(snn_v06::REFRESH_A, snn_v06::HOST_X0, 0,
                       snn_v06::HOLD_TAIL);
    }
    p2.x0 = fixture.q.x0;
    const Snapshot second = session.solve(p2.b, p2.d, p2.x0,
                                          snn_v06::HOST_X0, 0,
                                          snn_v06::HOLD_TAIL);
    const Snapshot expected_second = run_v05(p2);
    ok = ok && second.error == 0 &&
         compare_words(expected_second, second, index, tw);
    const char* rn = route == snn_v06::FORCE_FULL
                         ? "FULL"
                         : route == snn_v06::FORCE_CG ? "CG" : "STREAM";
    const char* ln = launch == snn_v06::PERSISTENT ? "PERSISTENT" : "ONESHOT";
    std::printf("%s refresh_A tier=%s launch=%s\n", ok ? "PASS" : "FAIL", rn,
                ln);
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
        {"full_320", 320, 320, snn_v06::FORCE_FULL, 0,
         snn_v06::FORCE_FULL},
        {"full_321_refused", 321, 321, snn_v06::FORCE_FULL,
         snn_v06::ERR_CAPACITY, -1},
        {"cg_321", 321, 321, snn_v06::FORCE_CG, 0, snn_v06::FORCE_CG},
        {"cg_372", 372, 372, snn_v06::FORCE_CG, 0, snn_v06::FORCE_CG},
        {"cg_373_refused", 373, 373, snn_v06::FORCE_CG,
         snn_v06::ERR_CAPACITY, -1},
        {"auto_321", 321, 321, snn_v06::AUTO, 0, snn_v06::FORCE_CG},
        {"auto_373", 373, 373, snn_v06::AUTO, 0, snn_v06::FORCE_STREAM},
    };
    bool all = true;
    for (const Boundary& check : checks) {
        const Problem q = zero_problem(check.n, check.m);
        V06Session session(q, check.requested, snn_v06::ONESHOT);
        const int error = session.configure();
        const int route = static_cast<int>(
            session.mb_out[snn_v06::OUT_SELECTED_ROUTE]);
        const bool ok = error == check.expected_error &&
                        (check.expected_route < 0 || route == check.expected_route);
        std::printf("%s route_boundary=%s n=%d m=%d requested=%d route=%d error=%d\n",
                    ok ? "PASS" : "FAIL", check.name, check.n, check.m,
                    check.requested, route, error);
        all = all && ok;
    }
    return all;
}

#if V06_HAVE_RESIDENT_BUNDLES
bool run_resident_bundle(const char* path, const char* label) {
    const resident_v1::Bundle bundle = resident_v1::load_bundle(path);
    Problem base;
    base.n = bundle.n;
    base.m = bundle.m;
    base.iterations = bundle.iterations;
    base.projection_cap = bundle.projection_cap;
    base.has_lower = bundle.has_lower;
    base.has_upper = bundle.has_upper;
    base.k0 = bundle.k0;
    base.constraint_tol = bundle.constraint_tol;
    base.lower = bundle.lower;
    base.upper = bundle.upper;
    base.A = bundle.A;
    base.C = bundle.C;
    base.c_norms_sq = bundle.c_norms_sq;
    base.row_scale = bundle.row_scale;
    base.G = bundle.G;
    base.x0 = bundle.x0;
    if (!bundle.periods.empty()) {
        base.b = bundle.periods[0].b;
        base.d = bundle.periods[0].d;
    }
    bool all = true;
    for (int route = snn_v06::FORCE_FULL; route <= snn_v06::FORCE_STREAM;
         ++route) {
        for (int launch = snn_v06::ONESHOT;
             launch <= snn_v06::PERSISTENT; ++launch) {
            V06Session session(base, route, launch);
            // Seed the persistent payload before launching the service.  The
            // first period then has a complete payload before its configure
            // command, exactly as a host uploads the input BO before launch.
            if (bundle.period_count > 0) {
                session.set_payload(bundle.periods[0].b,
                                    bundle.periods[0].d, bundle.x0);
            }
            bool ok = session.configure() == 0;
            std::vector<double> previous = bundle.x0;
            for (int p = 0; p < bundle.period_count && ok; ++p) {
                const auto& period = bundle.periods[static_cast<std::size_t>(p)];
                // The predecessor host uses the unshifted initial x0, then
                // shifts each committed fixed state before the next period.
                const std::vector<double> host_x0 = previous;
                Problem expected_problem = base;
                expected_problem.b = period.b;
                expected_problem.d = period.d;
                expected_problem.x0 = host_x0;
                const Snapshot expected = run_v05(expected_problem);
                const int stride = p == 0 ? 0 : 2;
                const Snapshot observed = session.solve(
                    period.b, period.d, host_x0,
                    snn_v06::RESIDENT_WARM, stride, snn_v06::HOLD_TAIL);
                std::size_t index = 0;
                bool tw = false;
                ok = observed.error == 0 && compare_words(expected, observed,
                                                            index, tw);
                if (ok) previous = shift_host(fixed_decode(observed.raw), 2,
                                              snn_v06::HOLD_TAIL);
                else {
                    std::printf(
                        "FAIL fixture=%s period=%d tier=%d launch=%d first_diff=%s[%zu] "
                        "expected=%lld observed=%lld meta0=%llu meta1=%llu\n",
                        label, p, route, launch, tw ? "telemetry" : "raw",
                        index,
                        static_cast<long long>(tw ? 0 : expected.raw[index]),
                        static_cast<long long>(tw ? 0 : observed.raw[index]),
                        static_cast<unsigned long long>(expected.telemetry[1]),
                        static_cast<unsigned long long>(observed.telemetry[1]));
                }
            }
            const char* rn = route == snn_v06::FORCE_FULL
                                 ? "FULL"
                                 : route == snn_v06::FORCE_CG ? "CG" : "STREAM";
            const char* ln = launch == snn_v06::PERSISTENT ? "PERSISTENT"
                                                            : "ONESHOT";
            std::printf("%s fixture=%s tier=%s start=WARM launch=%s periods=%d\n",
                        ok ? "PASS" : "FAIL", label, rn, ln,
                        bundle.period_count);
            all = all && ok;
        }
    }
    return all;
}
#endif

}  // namespace

int main(int argc, char** argv) {
    const char* pmsm_root = argc > 1 ? argv[1] : nullptr;
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
        for (int route = snn_v06::FORCE_FULL; route <= snn_v06::FORCE_STREAM;
             ++route) {
            if (!route_fits(fixture.q.n, fixture.q.m, route)) continue;
            for (int start = snn_v06::RESIDENT_WARM;
                 start <= snn_v06::COLD_ZERO; ++start) {
                for (int launch = snn_v06::ONESHOT;
                     launch <= snn_v06::PERSISTENT; ++launch) {
                    ++cells;
                    if (!run_cell(fixture, route, start, launch)) ++failures;
                }
            }
        }
    }

    // Refresh-A is exercised for all route/launch combinations on a compact
    // fixture, including the DDR image used by STREAM.
    const Fixture& refresh_fixture = fixtures[1];
    for (int route = snn_v06::FORCE_FULL; route <= snn_v06::FORCE_STREAM;
         ++route)
        for (int launch = snn_v06::ONESHOT; launch <= snn_v06::PERSISTENT;
             ++launch) {
            ++cells;
            if (!run_refresh_case(refresh_fixture, route, launch)) ++failures;
        }

    if (!run_route_boundary_checks()) ++failures;
    cells += 7;

#if V06_HAVE_RESIDENT_BUNDLES
    if (pmsm_root != nullptr) {
        const std::string root(pmsm_root);
        const std::vector<std::pair<const char*, const char*>> bundles = {
            {"H3_N1", "/bundles/resident_H3_N1_s11.bin"},
            {"H3_N10", "/bundles/resident_H3_N10_s11.bin"},
            {"H10_N1", "/bundles/resident_H10_N1_s11.bin"},
            {"H10_N5", "/bundles/resident_H10_N5_s11.bin"},
        };
        for (const auto& item : bundles) {
            // run_resident_bundle emits one line for each route/launch pair;
            // count those six comparison cells rather than the bundle header.
            cells += ROUTE_COUNT * LAUNCH_COUNT;
            if (!run_resident_bundle((root + item.second).c_str(), item.first))
                ++failures;
        }
    }
#else
    (void)pmsm_root;
    std::printf("INFO PMSM resident headers unavailable; synthetic battery retained\n");
#endif

    std::printf("PARITY SUMMARY cells=%zu failures=%zu %s\n", cells, failures,
                failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
