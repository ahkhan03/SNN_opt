// Thin PMSM stream adapter for the v06 mailbox host.
//
// The generic host owns the XRT BO layout and command ordering. Including it
// with V06_HOST_NO_MAIN keeps this client a small policy layer: the resident
// bundle is converted to the v05 Problem shape, one CONFIGURE is issued, and
// each period becomes a SOLVE mailbox transaction.
#define V06_HOST_NO_MAIN
#include "host_kv260_v06.cpp"

#include "resident_bundle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double FIXED_SCALE_D = static_cast<double>(UINT64_C(1) << 24);
constexpr double FEASIBILITY_TOLERANCE_PU =
    2.0 / static_cast<double>(UINT64_C(1) << 24);

struct StreamOptions {
    std::string xclbin;
    std::string bundle;
    std::string output;
    std::string json_out;
    int warmups = 0;
    int repetitions = 1;
    int route = snn_v06::AUTO;
    int shift = 2;
    int tail = snn_v06::HOLD_TAIL;
    bool persistent = true;
    bool host_x0 = false;
    bool force_mock = false;
    Options::PollSync poll_sync = Options::PollSync::Always;
    Options::Poll poll = Options::Poll::Yield;
    unsigned int poll_sleep_us = 0;
};

[[noreturn]] void stream_usage(const char* program, int status) {
    std::FILE* stream = status == 0 ? stdout : stderr;
    std::fprintf(
        stream,
        "usage: %s <kernel.xclbin> <resident-stream.bin> <fixed-output.bin> "
        "[--one-shot|--persistent] [--warmups N] [--reps N] "
        "[--route auto|full|cg|stream] [--host-x0] [--mock] "
        "[--poll-sync always|never] [--poll spin|yield|sleep-us=N] "
        "[--json-out PATH]\n",
        program);
    std::exit(status);
}

int stream_integer(const char* text, const char* label, bool allow_zero) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        (allow_zero ? value < 0 : value <= 0) || value > 100000000L) {
        std::fprintf(stderr, "invalid %s: %s\n", label, text);
        std::exit(2);
    }
    return static_cast<int>(value);
}

int stream_route(const char* text) {
    const std::string value(text);
    if (value == "auto") return snn_v06::AUTO;
    if (value == "full") return snn_v06::FORCE_FULL;
    if (value == "cg") return snn_v06::FORCE_CG;
    if (value == "stream") return snn_v06::FORCE_STREAM;
    std::fprintf(stderr, "invalid route: %s\n", text);
    std::exit(2);
}

StreamOptions parse_stream_options(int argc, char** argv) {
    if (argc < 4) stream_usage(argv[0], 2);
    StreamOptions options;
    options.xclbin = argv[1];
    options.bundle = argv[2];
    options.output = argv[3];
    for (int i = 4; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--one-shot") {
            options.persistent = false;
        } else if (arg == "--persistent") {
            options.persistent = true;
        } else if (arg == "--warmups" && i + 1 < argc) {
            options.warmups = stream_integer(argv[++i], "warmup count", true);
        } else if (arg == "--reps" && i + 1 < argc) {
            options.repetitions =
                stream_integer(argv[++i], "repetition count", false);
        } else if (arg == "--route" && i + 1 < argc) {
            options.route = stream_route(argv[++i]);
        } else if (arg == "--host-x0") {
            options.host_x0 = true;
        } else if (arg == "--poll-sync" && i + 1 < argc) {
            options.poll_sync = parse_poll_sync(argv[++i]);
        } else if (arg == "--poll" && i + 1 < argc) {
            options.poll = parse_poll(argv[++i], options.poll_sleep_us);
        } else if (arg == "--mock") {
            options.force_mock = true;
        } else if (arg == "--json-out" && i + 1 < argc) {
            options.json_out = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            stream_usage(argv[0], 0);
        } else {
            std::fprintf(stderr, "unrecognized or incomplete argument: %s\n",
                         arg.c_str());
            stream_usage(argv[0], 2);
        }
    }
    return options;
}

msrp_v05::Problem problem_from_bundle(const resident_v1::Bundle& bundle) {
    msrp_v05::Problem problem;
    problem.n = bundle.n;
    problem.m = bundle.m;
    problem.iterations = bundle.iterations;
    problem.projection_cap = bundle.projection_cap;
    problem.has_lower = bundle.has_lower;
    problem.has_upper = bundle.has_upper;
    problem.k0 = bundle.k0;
    problem.constraint_tol = bundle.constraint_tol;
    problem.lower = bundle.lower;
    problem.upper = bundle.upper;
    problem.A = bundle.A;
    problem.C = bundle.C;
    problem.c_norms_sq = bundle.c_norms_sq;
    problem.row_scale = bundle.row_scale;
    problem.G = bundle.G;
    problem.x0 = bundle.x0;
    problem.b = bundle.periods.front().b;
    problem.d = bundle.periods.front().d;
    return problem;
}

std::vector<double> decode_raw(const std::vector<std::int64_t>& raw) {
    std::vector<double> result(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i)
        result[i] = static_cast<double>(raw[i]) / FIXED_SCALE_D;
    return result;
}

std::vector<double> hold_tail_shift(const std::vector<double>& value,
                                    int stride) {
    if (stride <= 0 || value.empty()) return value;
    if (stride >= static_cast<int>(value.size())) return value;
    std::vector<double> result(value.size());
    for (int i = 0; i < static_cast<int>(value.size()); ++i)
        result[static_cast<std::size_t>(i)] =
            i < static_cast<int>(value.size()) - stride
                ? value[static_cast<std::size_t>(i + stride)]
                : value[static_cast<std::size_t>(i)];
    return result;
}

double max_feasibility_violation(const msrp_v05::Problem& q,
                                 const std::vector<std::int64_t>& raw) {
    const std::vector<double> x = decode_raw(raw);
    double maximum = 0.0;
    for (int i = 0; i < q.m; ++i) {
        double value = q.d[static_cast<std::size_t>(i)];
        for (int j = 0; j < q.n; ++j)
            value += q.C[static_cast<std::size_t>(i) * q.n + j] *
                     x[static_cast<std::size_t>(j)];
        maximum = std::max(maximum, value);
    }
    for (int i = 0; i < q.n; ++i) {
        if (q.has_lower)
            maximum = std::max(maximum,
                               q.lower - x[static_cast<std::size_t>(i)]);
        if (q.has_upper)
            maximum = std::max(maximum,
                               x[static_cast<std::size_t>(i)] - q.upper);
    }
    return std::max(0.0, maximum);
}

void print_double_vector(std::ostringstream& out,
                         const std::vector<double>& values) {
    out << '[' << std::setprecision(17);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

void print_i64_vector(std::ostringstream& out,
                      const std::vector<std::int64_t>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

void print_u64_vector(
    std::ostringstream& out,
    const std::array<std::uint64_t, snn_v06::TELEMETRY_WORDS>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

void print_nested_doubles(std::ostringstream& out,
                          const std::vector<std::vector<double>>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        print_double_vector(out, values[i]);
    }
    out << ']';
}

void print_nested_i64(std::ostringstream& out,
                      const std::vector<std::vector<std::int64_t>>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        print_i64_vector(out, values[i]);
    }
    out << ']';
}

void print_nested_u64(
    std::ostringstream& out,
    const std::vector<std::array<std::uint64_t, snn_v06::TELEMETRY_WORDS>>&
        values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        print_u64_vector(out, values[i]);
    }
    out << ']';
}

template <typename T>
void print_flat_u64(std::ostringstream& out, const std::vector<T>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << static_cast<std::uint64_t>(values[i]);
    }
    out << ']';
}

void print_nested_u64_scalars(
    std::ostringstream& out,
    const std::vector<std::vector<std::uint64_t>>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        print_flat_u64(out, values[i]);
    }
    out << ']';
}

void print_nested_double_scalars(
    std::ostringstream& out,
    const std::vector<std::vector<double>>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        print_double_vector(out, values[i]);
    }
    out << ']';
}

std::vector<std::vector<std::uint64_t>> group_by_period(
    const std::vector<std::uint64_t>& values, int period_count) {
    std::vector<std::vector<std::uint64_t>> grouped(
        static_cast<std::size_t>(period_count));
    if (period_count <= 0) return grouped;
    for (std::size_t i = 0; i < values.size(); ++i)
        grouped[i % static_cast<std::size_t>(period_count)].push_back(values[i]);
    return grouped;
}

std::vector<std::vector<double>> to_seconds(
    const std::vector<std::vector<std::uint64_t>>& values) {
    std::vector<std::vector<double>> result;
    result.reserve(values.size());
    for (const std::vector<std::uint64_t>& row : values) {
        std::vector<double> converted;
        converted.reserve(row.size());
        for (std::uint64_t value : row)
            converted.push_back(static_cast<double>(value) * 1e-9);
        result.push_back(std::move(converted));
    }
    return result;
}

std::vector<std::vector<double>> decoded_pass(
    const std::vector<ResultRecord>& pass) {
    std::vector<std::vector<double>> values;
    values.reserve(pass.size());
    for (const ResultRecord& result : pass)
        values.push_back(decode_raw(result.raw));
    return values;
}

std::string stream_json(
    const resident_v1::Bundle& bundle, const StreamOptions& options,
    const std::vector<TimingRecord>& timings,
    const std::vector<std::vector<ResultRecord>>& passes,
    const std::vector<double>& violation_by_period,
    const std::vector<double>& first_action_error_by_period,
    const V06Session& session) {
    const std::vector<ResultRecord>& final_pass = passes.back();
    const std::vector<std::vector<double>> decoded = decoded_pass(final_pass);
    std::vector<std::vector<std::int64_t>> raw_fixed;
    std::vector<std::array<std::uint64_t, snn_v06::TELEMETRY_WORDS>> telemetry;
    raw_fixed.reserve(final_pass.size());
    telemetry.reserve(final_pass.size());
    for (const ResultRecord& result : final_pass) {
        raw_fixed.push_back(result.raw);
        telemetry.push_back(result.telemetry);
    }

    std::vector<std::uint64_t> complete_ns, kernel_ns, payload_ns, sync_ns,
        publish_ns, wait_ns, doorbell_ns, output_ns, sequences;
    complete_ns.reserve(timings.size());
    kernel_ns.reserve(timings.size());
    payload_ns.reserve(timings.size());
    sync_ns.reserve(timings.size());
    publish_ns.reserve(timings.size());
    wait_ns.reserve(timings.size());
    doorbell_ns.reserve(timings.size());
    output_ns.reserve(timings.size());
    sequences.reserve(timings.size());
    for (const TimingRecord& item : timings) {
        complete_ns.push_back(item.complete_ns);
        kernel_ns.push_back(item.kernel_ns);
        payload_ns.push_back(item.payload_write_ns);
        sync_ns.push_back(item.sync_to_device_ns);
        publish_ns.push_back(item.mailbox_publish_ns);
        wait_ns.push_back(item.mailbox_wait_ns);
        doorbell_ns.push_back(item.doorbell_to_done_ns);
        output_ns.push_back(item.output_sync_read_ns);
        sequences.push_back(item.sequence);
    }
    const auto complete_by_period =
        group_by_period(complete_ns, bundle.period_count);
    const auto kernel_by_period = group_by_period(kernel_ns, bundle.period_count);
    const auto complete_seconds_by_period = to_seconds(complete_by_period);
    const auto kernel_seconds_by_period = to_seconds(kernel_by_period);

    std::vector<double> recorded_first_action;
    recorded_first_action.reserve(bundle.periods.size() * 2U);
    for (const resident_v1::Period& period : bundle.periods) {
        recorded_first_action.push_back(period.command_pu[0]);
        recorded_first_action.push_back(period.command_pu[1]);
    }
    std::vector<double> observed_first_action;
    observed_first_action.reserve(decoded.size() * 2U);
    for (const std::vector<double>& x : decoded) {
        observed_first_action.push_back(x.empty() ? 0.0 : x[0]);
        observed_first_action.push_back(x.size() < 2 ? 0.0 : x[1]);
    }

    const double maximum_violation = violation_by_period.empty()
                                         ? 0.0
                                         : *std::max_element(
                                               violation_by_period.begin(),
                                               violation_by_period.end());
    const double maximum_first_action_error =
        first_action_error_by_period.empty()
            ? 0.0
            : *std::max_element(first_action_error_by_period.begin(),
                                first_action_error_by_period.end());
    const bool feasibility_pass = maximum_violation <= FEASIBILITY_TOLERANCE_PU;

    std::ostringstream json;
    json << std::setprecision(17);
    json << "{\"schema\":\"resident-kv260-host-v1\""
         << ",\"kernel\":\"snn_qp_v06\""
         << ",\"v06_schema\":\"resident-kv260-v06-stream-v1\""
         << ",\"mode\":\""
         << (options.persistent ? "host_resident" : "one_shot") << '\"'
         << ",\"launch_mode\":\""
         << (options.persistent ? "persistent" : "one_shot") << '\"'
         << ",\"backend\":\"" << (session.is_mock() ? "mock" : "xrt")
         << "\",\"bundle_magic\":\"MSRPRB1\",\"bundle_version\":1"
         << ",\"xclbin\":\"" << options.xclbin << '\"'
         << ",\"n\":" << bundle.n << ",\"m\":" << bundle.m
         << ",\"periods\":" << bundle.period_count
         << ",\"warmups\":" << options.warmups
         << ",\"reps\":" << options.repetitions
         << ",\"route_requested\":\"" << route_name(options.route) << '\"'
         << ",\"route_selected\":"
         << (final_pass.empty()
                 ? 0
                 : final_pass.back().mailbox[snn_v06::OUT_SELECTED_ROUTE])
         << ",\"shift_stride\":" << options.shift
         << ",\"first_period_stride\":0"
         << ",\"tail_policy\":" << options.tail
         << ",\"poll_sync\":\"" << poll_sync_name(options.poll_sync)
         << '\"'
         << ",\"poll\":\""
         << poll_name(options.poll, options.poll_sleep_us)
         << '\"'
         << ",\"start_mode\":\""
         << (options.host_x0 ? "HOST_X0" : "RESIDENT_WARM") << '\"'
         << ",\"clock\":\"CLOCK_MONOTONIC_RAW\""
         << ",\"clock_resolution_ns\":" << clock_resolution_ns();

    json << ",\"complete_ns\":";
    print_flat_u64(json, complete_ns);
    json << ",\"complete_seconds\":[";
    for (std::size_t i = 0; i < complete_ns.size(); ++i) {
        if (i) json << ',';
        json << static_cast<double>(complete_ns[i]) * 1e-9;
    }
    json << ']';
    json << ",\"kernel_ns\":";
    print_flat_u64(json, kernel_ns);
    json << ",\"kernel_seconds\":[";
    for (std::size_t i = 0; i < kernel_ns.size(); ++i) {
        if (i) json << ',';
        json << static_cast<double>(kernel_ns[i]) * 1e-9;
    }
    json << ']';
    json << ",\"complete_seconds_by_period\":";
    print_nested_double_scalars(json, complete_seconds_by_period);
    json << ",\"complete_ns_by_period\":";
    print_nested_u64_scalars(json, complete_by_period);
    json << ",\"kernel_seconds_by_period\":";
    print_nested_double_scalars(json, kernel_seconds_by_period);
    json << ",\"kernel_ns_by_period\":";
    print_nested_u64_scalars(json, kernel_by_period);
    json << ",\"payload_write_ns\":";
    print_flat_u64(json, payload_ns);
    json << ",\"sync_to_device_ns\":";
    print_flat_u64(json, sync_ns);
    json << ",\"mailbox_publish_ns\":";
    print_flat_u64(json, publish_ns);
    json << ",\"mailbox_wait_ns\":";
    print_flat_u64(json, wait_ns);
    json << ",\"doorbell_to_done_ns\":";
    print_flat_u64(json, doorbell_ns);
    json << ",\"output_sync_read_ns\":";
    print_flat_u64(json, output_ns);
    json << ",\"total_ns\":";
    print_flat_u64(json, complete_ns);
    json << ",\"sequence\":";
    print_flat_u64(json, sequences);

    json << ",\"x_raw\":";
    print_nested_doubles(json, decoded);
    json << ",\"x_raw_pu\":";
    print_nested_doubles(json, decoded);
    json << ",\"x_raw_fixed\":";
    print_nested_i64(json, raw_fixed);
    json << ",\"telemetry_by_period\":";
    print_nested_u64(json, telemetry);
    json << ",\"committed_command_pu\":[";
    for (std::size_t i = 0; i < bundle.periods.size(); ++i) {
        if (i) json << ',';
        json << '[' << bundle.periods[i].command_pu[0] << ','
             << bundle.periods[i].command_pu[1] << ']';
    }
    json << ']';
    json << ",\"recorded_first_action_pu\":[";
    for (std::size_t i = 0; i < recorded_first_action.size(); ++i) {
        if (i) json << ',';
        json << recorded_first_action[i];
    }
    json << "]\n";
    std::string line = json.str();
    if (!line.empty() && line.back() == '\n') line.pop_back();
    std::ostringstream suffix;
    suffix << ",\"observed_first_action_pu\":[";
    for (std::size_t i = 0; i < observed_first_action.size(); ++i) {
        if (i) suffix << ',';
        suffix << observed_first_action[i];
    }
    suffix << "],\"first_action_abs_error_pu_by_period\":";
    print_double_vector(suffix, first_action_error_by_period);
    suffix << ",\"max_first_action_abs_error_pu\":"
           << maximum_first_action_error
           << ",\"max_feasibility_violation_pu_by_period\":";
    print_double_vector(suffix, violation_by_period);
    suffix << ",\"max_feasibility_violation_pu\":" << maximum_violation
           << ",\"feasibility_tolerance_pu\":" << FEASIBILITY_TOLERANCE_PU
           << ",\"feasibility_gate_pass\":"
           << (feasibility_pass ? "true" : "false")
           << ",\"status\":"
           << (final_pass.empty() ? 0 : final_pass.back().telemetry[1])
           << ",\"iterations\":"
           << (final_pass.empty() ? 0 : final_pass.back().telemetry[3])
           << ",\"events\":"
           << (final_pass.empty() ? 0 : final_pass.back().telemetry[5])
           << ",\"digest\":\"";
    if (!final_pass.empty())
        suffix << std::hex << std::setw(16) << std::setfill('0')
               << final_pass.back().telemetry[10] << std::dec
               << std::setfill(' ');
    suffix << '\"'
           << ",\"sync_counts\":{\"configure\":"
           << session.configure_sync_count() << ",\"geometry\":"
           << session.configure_sync_count() << ",\"A\":"
           << session.configure_sync_count() << ",\"C\":"
           << session.configure_sync_count() << ",\"cns\":"
           << session.configure_sync_count() << ",\"row_scale\":"
           << session.configure_sync_count() << ",\"G\":"
           << session.configure_sync_count() << ",\"input\":"
           << session.input_sync_count() << ",\"b\":"
           << session.input_sync_count() << ",\"d\":"
           << session.input_sync_count() << ",\"x0\":"
           << session.input_sync_count() << ",\"output\":"
           << session.output_sync_count() << ",\"outputs\":"
           << session.output_sync_count() << "}"
           << ",\"protocol\":{\"warm_state\":\"RESIDENT_WARM\","
              "\"shift_stride\":2,\"tail_policy\":\"HOLD_TAIL\","
              "\"first_period\":\"configured_x0_no_shift\","
              "\"one_shot_safe_variant\":\"HOST_X0\"}}";
    std::string suffix_text = suffix.str();
    suffix_text.erase(std::remove(suffix_text.begin(), suffix_text.end(), '\n'),
                      suffix_text.end());
    line += suffix_text;
    return line;
}

}  // namespace

int main(int argc, char** argv) {
    SignalGuard signal_guard;
    const StreamOptions options = parse_stream_options(argc, argv);
    const resident_v1::Bundle bundle = resident_v1::load_bundle(options.bundle);
    if (bundle.periods.empty()) {
        std::fprintf(stderr, "resident bundle has no periods\n");
        return 2;
    }
    msrp_v05::Problem problem = problem_from_bundle(bundle);
    Options generic;
    generic.xclbin = options.xclbin;
    generic.problem = options.bundle;
    generic.output = options.output;
    generic.warmups = options.warmups;
    generic.repetitions = options.repetitions;
    generic.route = options.route;
    generic.start = options.host_x0 ? snn_v06::HOST_X0 : snn_v06::RESIDENT_WARM;
    generic.shift = options.shift;
    generic.tail = options.tail;
    generic.persistent = options.persistent;
    generic.force_mock = options.force_mock;
    generic.poll_sync = options.poll_sync;
    generic.poll = options.poll;
    generic.poll_sleep_us = options.poll_sleep_us;
    V06Session session(generic, problem);
    const int configure_error = session.configure();
    if (configure_error != snn_v06::ERR_OK) {
        std::fprintf(stderr, "CONFIGURE failed with error %d\n", configure_error);
        const bool stopped = session.stop();
        return stopped ? 3 : 5;
    }

    std::vector<TimingRecord> timings;
    std::vector<std::vector<ResultRecord>> passes;
    std::vector<double> final_violation;
    std::vector<double> final_first_error;
    const int total_passes = options.warmups + options.repetitions;
    timings.reserve(static_cast<std::size_t>(options.repetitions) *
                    static_cast<std::size_t>(bundle.period_count));
    passes.reserve(static_cast<std::size_t>(options.repetitions));

    // A persistent CU retains its warm state across passes.  Keep the host
    // predecessor in lockstep so HOST_X0 control runs and HOLD_TAIL payloads
    // have the same continuous-stream semantics as RESIDENT_WARM.
    std::vector<double> resident_predecessor = bundle.x0;
    bool first_period = true;
    for (int pass = 0; pass < total_passes; ++pass) {
        std::vector<ResultRecord> pass_results;
        pass_results.reserve(static_cast<std::size_t>(bundle.period_count));
        std::vector<double> pass_violation;
        std::vector<double> pass_first_error;
        std::vector<double> previous_output = resident_predecessor;
        for (int period_index = 0; period_index < bundle.period_count;
             ++period_index) {
            if (signal_requested()) {
                std::fprintf(stderr,
                             "signal %d received before period %d; beginning "
                             "STOP cleanup\n",
                             static_cast<int>(g_signal_number), period_index);
                const bool stopped = session.stop();
                return stopped ? 130 : 5;
            }
            const resident_v1::Period& period =
                bundle.periods[static_cast<std::size_t>(period_index)];
            const bool first = first_period;
            std::vector<double> payload_x0 =
                first ? bundle.x0 : hold_tail_shift(previous_output, options.shift);
            const int start = options.host_x0 ? snn_v06::HOST_X0
                                              : snn_v06::RESIDENT_WARM;
            const int stride =
                options.host_x0 ? 0 : (first ? 0 : options.shift);
            TimingRecord timing;
            ResultRecord result = session.solve(
                period.b, period.d, payload_x0, &timing, start, stride,
                options.tail);
            if (result.mailbox[snn_v06::OUT_ERROR_CODE] != snn_v06::ERR_OK) {
                std::fprintf(stderr, "SOLVE period %d failed with error %u\n",
                             period_index,
                             result.mailbox[snn_v06::OUT_ERROR_CODE]);
                const bool stopped = session.stop();
                return stopped ? 3 : 5;
            }
            previous_output = decode_raw(result.raw);
            resident_predecessor = previous_output;
            first_period = false;
            if (signal_requested()) {
                std::fprintf(stderr,
                             "signal %d received after period %d; beginning "
                             "STOP cleanup\n",
                             static_cast<int>(g_signal_number), period_index);
                const bool stopped = session.stop();
                return stopped ? 130 : 5;
            }
            const double observed0 = previous_output.empty() ? 0.0 : previous_output[0];
            const double observed1 =
                previous_output.size() < 2 ? 0.0 : previous_output[1];
            const double first_error = std::max(
                std::abs(observed0 - period.command_pu[0]),
                std::abs(observed1 - period.command_pu[1]));
            pass_results.push_back(std::move(result));
            msrp_v05::Problem period_problem = problem;
            period_problem.d = period.d;
            pass_violation.push_back(max_feasibility_violation(
                period_problem, pass_results.back().raw));
            pass_first_error.push_back(first_error);
            if (pass >= options.warmups) timings.push_back(timing);
        }
        if (pass >= options.warmups) {
            final_violation = pass_violation;
            final_first_error = pass_first_error;
            passes.push_back(std::move(pass_results));
        }
    }
    const bool stopped = session.stop();
    if (!stopped) return 5;
    if (passes.empty()) {
        std::fprintf(stderr, "no timed repetitions were requested\n");
        return 2;
    }
    const std::vector<ResultRecord>& final_pass = passes.back();
    resident_v1::write_fixed_output(options.output, bundle.n,
                                    final_pass.back().raw,
                                    final_pass.back().telemetry);
    const std::string line = stream_json(
        bundle, options, timings, passes, final_violation, final_first_error,
        session);
    if (!options.json_out.empty()) {
        std::FILE* file = std::fopen(options.json_out.c_str(), "w");
        if (!file) {
            std::perror("--json-out");
            return 2;
        }
        std::fwrite(line.data(), 1, line.size(), file);
        std::fputc('\n', file);
        std::fclose(file);
    }
    std::printf("%s\n", line.c_str());
    const double maximum_violation = final_violation.empty()
                                         ? 0.0
                                         : *std::max_element(
                                               final_violation.begin(),
                                               final_violation.end());
    return (session.is_mock() || maximum_violation <= FEASIBILITY_TOLERANCE_PU)
               ? 0
               : 4;
}
