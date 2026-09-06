// Gate-3 persistent mailbox stress driver.
//
// This is intentionally a thin policy layer over V06Session.  The session
// owns the XRT BO layout, publication ordering, signal cleanup, and sequence
// matching; this file only chooses a deterministic command mix and checks the
// returned records.  Define V06_STRESS_WITH_V05 when building on the board
// (or on a workstation with the HLS headers) to link the untouched v0.5
// kernel as the per-solve raw/telemetry oracle.  A mock build uses the
// session's documented deterministic mock record and still exercises all
// sequence/mix/STOP logic.

#define V06_HOST_NO_MAIN
#include "host_kv260_v06.cpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#if defined(V06_STRESS_WITH_V05)
extern "C" void snn_qp_v05(
    const double* A_in, const double* b_in, const double* C_in,
    const double* d_in, const double* cns_in, const double* row_scale_in,
    const double* G_in, const double* x0_in, long long* x_raw_out,
    unsigned long long* telemetry_out, int n, int m, double k0_f,
    double ctol_f, int n_iters, int projmax, int has_lower, double lower_f,
    int has_upper, double upper_f);
#endif

#if defined(V06_MOCK_NATIVE_HOOK)
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

void v06_mock_native_invoke(
    const msrp_v05::Problem* problem, const double* b_in, const double* d_in,
    const double* x0_in, std::uint32_t* a_ddr, std::uint32_t* c_ddr,
    std::uint32_t* ct_ddr, std::uint32_t* g_ddr, long long* raw_out,
    unsigned long long* telemetry_out, std::uint32_t* mb_in,
    std::uint32_t* mb_out, int /*command*/) {
    // The mailbox fields have already been published by V06Session.  SERVE
    // makes the native dispatcher consume exactly those fields, while the
    // ONESHOT launch keeps this hook independent of an XRT scheduler.  The
    // kernel's static resident state still persists between calls, so this
    // exercises configure/refresh/solve state transitions and wrap values.
    snn_qp_v06(
        problem->A.data(), problem->C.data(), problem->G.data(),
        problem->c_norms_sq.data(), problem->row_scale.data(), b_in, d_in,
        x0_in, a_ddr, c_ddr, ct_ddr, g_ddr, raw_out, telemetry_out, mb_in,
        mb_out, snn_v06::SERVE, snn_v06::ONESHOT, snn_v06::AUTO,
        snn_v06::HOST_X0, 0, snn_v06::HOLD_TAIL, problem->n, problem->m,
        problem->k0, problem->constraint_tol, problem->iterations,
        problem->projection_cap, problem->has_lower ? 1 : 0, problem->lower,
        problem->has_upper ? 1 : 0, problem->upper);
}
#endif

namespace {

constexpr std::uint64_t DEFAULT_STRESS_SEED = UINT64_C(0x544333360906);
constexpr std::uint64_t MOCK_TELEMETRY_MAGIC = snn_v06::TELEMETRY_MAGIC;

struct StressOptions {
    std::string xclbin;
    std::string problem;
    std::string report;
    std::uint64_t count = 10000;
    std::uint64_t seed = DEFAULT_STRESS_SEED;
    std::uint32_t sequence_seed = UINT32_C(1);
    bool force_mock = false;
};

[[noreturn]] void stress_usage(const char* program, int status) {
    std::FILE* stream = status == 0 ? stdout : stderr;
    std::fprintf(
        stream,
        "usage: %s <kernel.xclbin> <problem.bin> <report.json> "
        "[--count N] [--seed U64] [--sequence-seed UINT32] [--mock]\n",
        program);
    std::exit(status);
}

std::uint64_t parse_u64(const char* text, const char* label) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0') {
        std::fprintf(stderr, "invalid %s: %s\n", label, text);
        std::exit(2);
    }
    return static_cast<std::uint64_t>(value);
}

StressOptions parse_stress_options(int argc, char** argv) {
    if (argc < 4) stress_usage(argv[0], 2);
    StressOptions options;
    options.xclbin = argv[1];
    options.problem = argv[2];
    options.report = argv[3];
    for (int i = 4; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--count" && i + 1 < argc) {
            options.count = parse_u64(argv[++i], "stress count");
            if (options.count == 0) {
                std::fprintf(stderr, "stress count must be positive\n");
                std::exit(2);
            }
        } else if (arg == "--seed" && i + 1 < argc) {
            options.seed = parse_u64(argv[++i], "stress seed");
        } else if (arg == "--sequence-seed" && i + 1 < argc) {
            options.sequence_seed = parse_u32(argv[++i], "sequence seed");
            if (options.sequence_seed == 0) {
                std::fprintf(stderr,
                             "sequence seed must be nonzero; zero is reserved "
                             "for the first post-wrap completion\n");
                std::exit(2);
            }
        } else if (arg == "--mock") {
            options.force_mock = true;
        } else if (arg == "--help" || arg == "-h") {
            stress_usage(argv[0], 0);
        } else {
            std::fprintf(stderr, "unrecognized or incomplete argument: %s\n",
                         arg.c_str());
            stress_usage(argv[0], 2);
        }
    }
    return options;
}

struct ReferenceResult {
    std::vector<std::int64_t> raw;
    std::array<std::uint64_t, snn_v06::TELEMETRY_WORDS> telemetry{};
};

ReferenceResult reference_solve(const msrp_v05::Problem& q) {
    ReferenceResult result;
    result.raw.assign(static_cast<std::size_t>(q.n), 0);
#if defined(V06_STRESS_WITH_V05)
    std::vector<long long> raw(static_cast<std::size_t>(q.n), 0);
    std::array<unsigned long long, snn_v06::TELEMETRY_WORDS> telemetry{};
    snn_qp_v05(
        q.A.data(), q.b.data(), q.C.data(), q.d.data(), q.c_norms_sq.data(),
        q.row_scale.data(), q.G.data(), q.x0.data(), raw.data(),
        telemetry.data(), q.n, q.m, q.k0, q.constraint_tol, q.iterations,
        q.projection_cap, q.has_lower ? 1 : 0, q.lower,
        q.has_upper ? 1 : 0, q.upper);
    for (std::size_t i = 0; i < result.raw.size(); ++i)
        result.raw[i] = static_cast<std::int64_t>(raw[i]);
    for (std::size_t i = 0; i < result.telemetry.size(); ++i)
        result.telemetry[i] = static_cast<std::uint64_t>(telemetry[i]);
#else
    // v06_xrt_mock.hpp deliberately emits this compact deterministic record.
    // Keeping the fallback oracle here makes the workstation stress test
    // useful even when proprietary HLS headers are unavailable; the board
    // build switches to the independent v0.5 oracle above.
    result.telemetry[0] = MOCK_TELEMETRY_MAGIC;
    result.telemetry[2] = static_cast<std::uint64_t>(q.iterations);
    result.telemetry[3] = static_cast<std::uint64_t>(q.iterations);
    result.telemetry[4] = 1;
    result.telemetry[13] = 32;
    result.telemetry[14] = 8;
#endif
    return result;
}

bool equal_result(const ReferenceResult& expected, const ResultRecord& observed,
                  std::size_t& index, bool& telemetry_word) {
    if (expected.raw.size() != observed.raw.size()) {
        index = std::min(expected.raw.size(), observed.raw.size());
        telemetry_word = false;
        return false;
    }
    for (std::size_t i = 0; i < expected.raw.size(); ++i) {
        if (expected.raw[i] != observed.raw[i]) {
            index = i;
            telemetry_word = false;
            return false;
        }
    }
    for (std::size_t i = 0; i < expected.telemetry.size(); ++i) {
        if (expected.telemetry[i] != observed.telemetry[i]) {
            index = i;
            telemetry_word = true;
            return false;
        }
    }
    return true;
}

void mutate_vectors(msrp_v05::Problem& q, std::uint64_t ordinal,
                    std::mt19937_64& rng) {
    std::uniform_real_distribution<double> jitter(-0.015, 0.015);
    for (double& value : q.b) value = jitter(rng);
    for (double& value : q.d) value = -0.2 + jitter(rng);
    for (double& value : q.x0) value = 0.05 + jitter(rng);
    (void)ordinal;
}

void mutate_A(msrp_v05::Problem& q, std::uint64_t ordinal) {
    const double scale = 0.8 + 0.01 * static_cast<double>(ordinal % 15);
    for (std::size_t i = 0; i < q.A.size(); ++i)
        if ((i + ordinal) % 11 == 0) q.A[i] *= scale;
}

double median_us(std::vector<std::uint64_t> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() & 1U)
        return static_cast<double>(values[middle]) / 1000.0;
    return (static_cast<double>(values[middle - 1]) +
            static_cast<double>(values[middle])) /
           2000.0;
}

void write_report(const StressOptions& options, const msrp_v05::Problem& q,
                  std::uint64_t completed, std::uint64_t solve_count,
                  std::uint64_t refresh_count, std::uint64_t configure_count,
                  std::uint64_t sequence_errors, std::uint64_t parity_errors,
                  std::uint32_t first_sequence, std::uint32_t last_sequence,
                  bool wrap_crossed, bool stop_ok,
                  const std::vector<std::uint64_t>& timings,
                  const std::string& first_error, bool interrupted) {
    std::ofstream out(options.report);
    if (!out) {
        std::perror(options.report.c_str());
        std::exit(2);
    }
    const bool timing_cell = q.n == 6 && q.m == 18;
    const double p50_us = median_us(timings);
    // A workstation mock has no meaningful board latency.  Keep the target
    // visible in its report, but enforce it only on the real XRT backend and
    // the preregistered H=3,N=1 shape.
    const bool timing_checked = timing_cell && V06_HAVE_XRT;
    const bool timing_pass = !timing_checked || p50_us <= 25.0;
    out << "{\"schema\":\"snn-qp-v06-gate3-stress-v1\""
        << ",\"backend\":\""
        << (V06_HAVE_XRT ? "xrt" : "mock") << '"'
        << ",\"n\":" << q.n << ",\"m\":" << q.m
        << ",\"count_requested\":" << options.count
        << ",\"count_completed\":" << completed
        << ",\"seed\":" << options.seed
        << ",\"sequence_seed\":" << options.sequence_seed
        << ",\"first_sequence\":" << first_sequence
        << ",\"last_sequence\":" << last_sequence
        << ",\"wrap_crossed\":" << (wrap_crossed ? "true" : "false")
        << ",\"sequence_errors\":" << sequence_errors
        << ",\"parity_errors\":" << parity_errors
        << ",\"interrupted\":" << (interrupted ? "true" : "false")
        << ",\"commands\":{\"solve\":" << solve_count
        << ",\"refresh_a\":" << refresh_count
        << ",\"configure\":" << configure_count << '}'
        << ",\"timing_cell\":{\"horizon_n1\":"
        << (timing_cell ? "true" : "false")
        << ",\"target_us\":25,\"target_checked\":"
        << (timing_checked ? "true" : "false")
        << ",\"target_pass\":"
        << (timing_checked ? (timing_pass ? "true" : "false") : "null")
        << ",\"p50_us\":" << p50_us
        << ",\"samples\":" << timings.size()
        << ",\"round_trip_ns\":[";
    for (std::size_t i = 0; i < timings.size(); ++i) {
        if (i) out << ',';
        out << timings[i];
    }
    out << "]}"
        << ",\"stop_ok\":" << (stop_ok ? "true" : "false")
        << ",\"clean_stop_pass\":"
        << ((interrupted && stop_ok) ? "true" : "false")
        << ",\"pass\":"
        << ((completed == options.count && sequence_errors == 0 &&
             parity_errors == 0 && stop_ok && timing_pass)
                ? "true"
                : "false");
    if (!first_error.empty()) out << ",\"first_error\":\"" << first_error << '"';
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    SignalGuard signal_guard;
    const StressOptions stress = parse_stress_options(argc, argv);
    const msrp_v05::Problem loaded =
        msrp_v05::load_problem(stress.problem, false, 1024, 1024);
    msrp_v05::Problem problem = loaded;

    Options host;
    host.xclbin = stress.xclbin;
    host.problem = stress.problem;
    host.output = stress.report + ".last.bin";
    host.persistent = true;
    host.route = snn_v06::AUTO;
    host.start = snn_v06::HOST_X0;
    host.shift = 0;
    host.tail = snn_v06::HOLD_TAIL;
    host.force_mock = stress.force_mock;
    host.sequence_seed = stress.sequence_seed;
    host.poll = Options::Poll::Yield;
    // A stress run is intentionally a serial mailbox client.  The board
    // script uses this with the H=3,N=1 fixture, so each timing sample is one
    // complete payload -> done -> output round trip.

    V06Session session(host, problem);
    const int configure_error = session.configure();
    if (configure_error != snn_v06::ERR_OK) {
        const bool stopped = session.stop();
        const bool interrupted = signal_requested();
        if (interrupted) {
            // CONFIGURE uses the same persistent run and mailbox wait as the
            // loop.  If SIGINT lands there, preserve the same report/130
            // contract instead of returning an unreported ordinary error.
            const std::uint32_t initial_sequence =
                stress.sequence_seed == 0 ? UINT32_C(1) : stress.sequence_seed;
            const std::vector<std::uint64_t> no_timings;
            write_report(stress, problem, 0, 0, 0, 0, 0, 0, initial_sequence,
                         initial_sequence, false, stopped, no_timings,
                         "CONFIGURE interrupted", interrupted);
            return stopped ? 130 : 1;
        }
        std::fprintf(stderr, "stress CONFIGURE failed with error %d\n",
                     configure_error);
        return stopped ? 3 : 5;
    }

    std::mt19937_64 rng(stress.seed);
    std::vector<std::uint64_t> timings;
    timings.reserve(static_cast<std::size_t>(stress.count));
    std::uint64_t completed = 0;
    std::uint64_t solve_count = 0;
    std::uint64_t refresh_count = 0;
    std::uint64_t configure_count = 0;
    std::uint64_t sequence_errors = 0;
    std::uint64_t parity_errors = 0;
    std::uint32_t expected_sequence = stress.sequence_seed;
    const std::uint32_t first_sequence = expected_sequence;
    std::uint32_t last_sequence = expected_sequence;
    bool wrap_crossed = false;
    std::string first_error;

    while (completed < stress.count && !signal_requested()) {
        const std::uint64_t ordinal = completed;
        const unsigned choice =
            static_cast<unsigned>(rng() % UINT64_C(100));
        ResultRecord command_result;
        TimingRecord timing{};
        if (choice < 82U) {
            mutate_vectors(problem, ordinal, rng);
            const ReferenceResult expected = reference_solve(problem);
            if (signal_requested()) break;
            ++solve_count;
            const std::uint32_t before = session.sequence();
            command_result = session.solve(problem.b, problem.d, problem.x0,
                                           &timing, snn_v06::HOST_X0, 0,
                                           snn_v06::HOLD_TAIL);
            // An interrupted wait returns a synthetic ERR_BUSY record and
            // does not populate TimingRecord.  Go straight to STOP instead
            // of reading partial result data.
            if (signal_requested()) break;
            if (timing.sequence != before + UINT32_C(1)) ++sequence_errors;
            std::size_t index = 0;
            bool telemetry_word = false;
            if (!equal_result(expected, command_result, index, telemetry_word)) {
                ++parity_errors;
                if (first_error.empty()) {
                    char buffer[256];
                    if (telemetry_word)
                        std::snprintf(buffer, sizeof(buffer),
                                      "solve %llu telemetry[%zu] mismatch",
                                      static_cast<unsigned long long>(ordinal),
                                      index);
                    else
                        std::snprintf(buffer, sizeof(buffer),
                                      "solve %llu raw[%zu] mismatch",
                                      static_cast<unsigned long long>(ordinal),
                                      index);
                    first_error = buffer;
                }
            }
            if (command_result.mailbox[snn_v06::OUT_ERROR_CODE] !=
                snn_v06::ERR_OK)
                ++sequence_errors;
            timings.push_back(timing.complete_ns);
        } else if (choice < 92U) {
            mutate_A(problem, ordinal);
            if (signal_requested()) break;
            ++refresh_count;
            const std::uint32_t before = session.sequence();
            command_result = session.refresh_a();
            if (signal_requested()) break;
            if (session.sequence() != before + UINT32_C(1)) ++sequence_errors;
            if (command_result.mailbox[snn_v06::OUT_ERROR_CODE] !=
                snn_v06::ERR_OK)
                ++sequence_errors;
        } else {
            mutate_vectors(problem, ordinal, rng);
            mutate_A(problem, ordinal + 3);
            if (signal_requested()) break;
            ++configure_count;
            const std::uint32_t before = session.sequence();
            command_result = session.reconfigure();
            if (signal_requested()) break;
            if (session.sequence() != before + UINT32_C(1)) ++sequence_errors;
            if (command_result.mailbox[snn_v06::OUT_ERROR_CODE] !=
                snn_v06::ERR_OK)
                ++sequence_errors;
        }

        ++expected_sequence;
        last_sequence = session.last_done_sequence();
        if (last_sequence != expected_sequence) ++sequence_errors;
        if (expected_sequence < stress.sequence_seed ||
            last_sequence < stress.sequence_seed)
            wrap_crossed = true;
        ++completed;
    }

    const bool stop_ok = session.stop();
    // Sample after STOP as well as before it.  A SIGINT arriving while the
    // final drain is waiting must still produce the interrupted/130 outcome.
    const bool interrupted = signal_requested();
    const bool timing_checked = (problem.n == 6 && problem.m == 18) &&
                                V06_HAVE_XRT;
    const bool timing_pass = !timing_checked || median_us(timings) <= 25.0;
    write_report(stress, problem, completed, solve_count, refresh_count,
                 configure_count, sequence_errors, parity_errors, first_sequence,
                 last_sequence, wrap_crossed, stop_ok, timings, first_error,
                 interrupted);

    std::printf(
        "STRESS SUMMARY backend=%s completed=%llu/%llu solve=%llu "
        "refresh=%llu configure=%llu sequence_errors=%llu parity_errors=%llu "
        "wrap=%s p50_us=%.3f stop=%s interrupted=%s %s\n",
        V06_HAVE_XRT ? "xrt" : "mock",
        static_cast<unsigned long long>(completed),
        static_cast<unsigned long long>(stress.count),
        static_cast<unsigned long long>(solve_count),
        static_cast<unsigned long long>(refresh_count),
        static_cast<unsigned long long>(configure_count),
        static_cast<unsigned long long>(sequence_errors),
        static_cast<unsigned long long>(parity_errors),
        wrap_crossed ? "yes" : "no", median_us(timings), stop_ok ? "ok" : "FAIL",
        interrupted ? "yes" : "no",
        (completed == stress.count && sequence_errors == 0 &&
         parity_errors == 0 && stop_ok && timing_pass)
            ? "PASS"
            : "FAIL");
    if (interrupted) return stop_ok ? 130 : 1;
    return completed == stress.count && sequence_errors == 0 &&
                   parity_errors == 0 && stop_ok && timing_pass
               ? 0
               : 1;
}
