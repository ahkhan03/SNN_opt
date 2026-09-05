// Generic XRT 2.13 host for the v06 resident kernel.
//
// The board path uses one input BO (mailbox, b, d, x0) and one output BO
// (mailbox, raw state, telemetry).  Geometry configure inputs and the four
// fixed-point backing images remain separate BOs.  When XRT is unavailable,
// the same control path is built against v06_xrt_mock.hpp so mailbox ordering,
// offsets, JSON, and the litmus procedure can be checked on a workstation.

#include "v06_abi.hpp"
#include "v06_host_protocol.hpp"
#include "v06_xrt_compat.hpp"

#include "../../kv260_v05/src/msrp_bundle.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <exception>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <time.h>
#include <utility>
#include <vector>

namespace {

constexpr std::uint64_t FIXED_SCALE = UINT64_C(1) << 24;
constexpr std::uint64_t LITMUS_RAW0 = UINT64_C(0x13579bdf00000000);
constexpr std::uint64_t LITMUS_META0 = UINT64_C(0xa5a5000000000000);
constexpr std::uint32_t LITMUS_INPUT = UINT32_C(0xdecafbad);
constexpr int WAIT_TIMEOUT_MS = 30000;
constexpr int STOP_TIMEOUT_MS = 30000;

// A signal handler may only publish a flag.  The owning session notices the
// flag in its mailbox wait and then performs the ordinary STOP handshake from
// process context, where XRT calls are legal.
volatile std::sig_atomic_t g_signal_number = 0;

void v06_signal_handler(int signal_number) {
    g_signal_number = signal_number;
}

bool signal_requested() { return g_signal_number != 0; }

class SignalGuard {
  public:
    SignalGuard()
        : old_int_(std::signal(SIGINT, v06_signal_handler)),
          old_term_(std::signal(SIGTERM, v06_signal_handler)) {}

    ~SignalGuard() {
        std::signal(SIGINT, old_int_);
        std::signal(SIGTERM, old_term_);
    }

    SignalGuard(const SignalGuard&) = delete;
    SignalGuard& operator=(const SignalGuard&) = delete;

  private:
    using Handler = void (*)(int);
    Handler old_int_;
    Handler old_term_;
};

struct Options {
    std::string xclbin;
    std::string problem;
    std::string output;
    std::string json_out;
    int warmups = 0;
    int repetitions = 1;
    int route = snn_v06::AUTO;
    int start = snn_v06::HOST_X0;
    int shift = 0;
    int tail = snn_v06::HOLD_TAIL;
    bool persistent = true;
    bool litmus = false;
    bool force_mock = false;
    enum class PollSync { Always, Never };
    enum class Poll { Spin, Yield, SleepUs };
    PollSync poll_sync = PollSync::Always;
    Poll poll = Poll::Yield;
    unsigned int poll_sleep_us = 0;
};

[[noreturn]] void usage(const char* program, int status) {
    std::FILE* stream = status == 0 ? stdout : stderr;
    std::fprintf(
        stream,
        "usage: %s <kernel.xclbin> <problem.bin> <fixed-output.bin> "
        "[--one-shot|--persistent] [--warmups N] [--reps N] "
        "[--route auto|full|cg|stream] [--start host_x0|resident_warm|cold_zero] "
        "[--shift S] [--tail hold|repeat] [--litmus] [--mock] "
        "[--poll-sync always|never] [--poll spin|yield|sleep-us=N] "
        "[--json-out PATH]\n",
        program);
    std::exit(status);
}

int parse_integer(const char* text, const char* label, bool allow_zero) {
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

int parse_route(const char* text) {
    const std::string value(text);
    if (value == "auto") return snn_v06::AUTO;
    if (value == "full") return snn_v06::FORCE_FULL;
    if (value == "cg") return snn_v06::FORCE_CG;
    if (value == "stream") return snn_v06::FORCE_STREAM;
    std::fprintf(stderr, "invalid route: %s\n", text);
    std::exit(2);
}

int parse_start(const char* text) {
    const std::string value(text);
    if (value == "host_x0" || value == "host") return snn_v06::HOST_X0;
    if (value == "resident_warm" || value == "warm")
        return snn_v06::RESIDENT_WARM;
    if (value == "cold_zero" || value == "cold") return snn_v06::COLD_ZERO;
    std::fprintf(stderr, "invalid start mode: %s\n", text);
    std::exit(2);
}

int parse_tail(const char* text) {
    const std::string value(text);
    if (value == "hold" || value == "hold_tail") return snn_v06::HOLD_TAIL;
    if (value == "repeat" || value == "repeat_last")
        return snn_v06::REPEAT_LAST;
    std::fprintf(stderr, "invalid tail policy: %s\n", text);
    std::exit(2);
}

Options::PollSync parse_poll_sync(const char* text) {
    const std::string value(text);
    if (value == "always") return Options::PollSync::Always;
    if (value == "never") return Options::PollSync::Never;
    std::fprintf(stderr, "invalid poll-sync policy: %s\n", text);
    std::exit(2);
}

Options::Poll parse_poll(const char* text, unsigned int& sleep_us) {
    const std::string value(text);
    if (value == "spin") return Options::Poll::Spin;
    if (value == "yield") return Options::Poll::Yield;
    constexpr const char* PREFIX = "sleep-us=";
    if (value.rfind(PREFIX, 0) == 0 && value.size() > std::strlen(PREFIX)) {
        sleep_us = static_cast<unsigned int>(parse_integer(
            value.c_str() + std::strlen(PREFIX), "poll sleep microseconds",
            true));
        return Options::Poll::SleepUs;
    }
    std::fprintf(stderr, "invalid poll policy: %s\n", text);
    std::exit(2);
}

const char* poll_sync_name(Options::PollSync policy) {
    return policy == Options::PollSync::Always ? "always" : "never";
}

std::string poll_name(Options::Poll policy, unsigned int sleep_us) {
    if (policy == Options::Poll::Spin) return "spin";
    if (policy == Options::Poll::Yield) return "yield";
    return std::string("sleep-us=") + std::to_string(sleep_us);
}

std::string poll_name(const Options& options) {
    return poll_name(options.poll, options.poll_sleep_us);
}

#ifndef V06_HOST_NO_MAIN
Options parse_options(int argc, char** argv) {
    if (argc < 4) usage(argv[0], 2);
    Options options;
    options.xclbin = argv[1];
    options.problem = argv[2];
    options.output = argv[3];
    for (int i = 4; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--one-shot") {
            options.persistent = false;
        } else if (arg == "--persistent") {
            options.persistent = true;
        } else if (arg == "--warmups" && i + 1 < argc) {
            options.warmups = parse_integer(argv[++i], "warmup count", true);
        } else if (arg == "--reps" && i + 1 < argc) {
            options.repetitions = parse_integer(argv[++i], "repetition count", false);
        } else if (arg == "--route" && i + 1 < argc) {
            options.route = parse_route(argv[++i]);
        } else if (arg == "--start" && i + 1 < argc) {
            options.start = parse_start(argv[++i]);
        } else if (arg == "--shift" && i + 1 < argc) {
            options.shift = parse_integer(argv[++i], "shift stride", true);
        } else if (arg == "--tail" && i + 1 < argc) {
            options.tail = parse_tail(argv[++i]);
        } else if (arg == "--poll-sync" && i + 1 < argc) {
            options.poll_sync = parse_poll_sync(argv[++i]);
        } else if (arg == "--poll" && i + 1 < argc) {
            options.poll = parse_poll(argv[++i], options.poll_sleep_us);
        } else if (arg == "--litmus") {
            options.litmus = true;
        } else if (arg == "--mock") {
            options.force_mock = true;
        } else if (arg == "--json-out" && i + 1 < argc) {
            options.json_out = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0], 0);
        } else {
            std::fprintf(stderr, "unrecognized or incomplete argument: %s\n",
                         arg.c_str());
            usage(argv[0], 2);
        }
    }
    return options;
}
#endif  // V06_HOST_NO_MAIN

std::uint64_t monotonic_raw_ns() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    timespec stamp{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &stamp) != 0) {
        std::perror("clock_gettime(CLOCK_MONOTONIC_RAW)");
        // Keep cleanup paths unwindable if the clock is temporarily
        // unavailable.  The fallback is only a last-resort diagnostic clock;
        // normal board measurements always use CLOCK_MONOTONIC_RAW.
        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
    }
    const std::uint64_t result = static_cast<std::uint64_t>(stamp.tv_sec) *
                                     UINT64_C(1000000000) +
                                 static_cast<std::uint64_t>(stamp.tv_nsec);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return result;
}

std::uint64_t clock_resolution_ns() {
    timespec stamp{};
    if (clock_getres(CLOCK_MONOTONIC_RAW, &stamp) != 0) {
        std::perror("clock_getres(CLOCK_MONOTONIC_RAW)");
        return 0;
    }
    return static_cast<std::uint64_t>(stamp.tv_sec) * UINT64_C(1000000000) +
           static_cast<std::uint64_t>(stamp.tv_nsec);
}

template <typename Bo, typename Direction>
auto sync_bo_impl(Bo& buffer, Direction direction, std::size_t bytes,
                  std::size_t offset, int)
    -> decltype(buffer.sync(direction, bytes, offset), void()) {
    buffer.sync(direction, bytes, offset);
}

template <typename Bo, typename Direction>
auto sync_bo_impl(Bo& buffer, Direction direction, std::size_t, std::size_t,
                  long) -> decltype(buffer.sync(direction), void()) {
    buffer.sync(direction);
}

template <typename Bo, typename Direction>
void sync_bo(Bo& buffer, Direction direction, std::size_t bytes,
             std::size_t offset) {
    sync_bo_impl(buffer, direction, bytes, offset, 0);
}

void append_u64_array(std::ostringstream& json,
                      const std::vector<std::uint64_t>& values) {
    json << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) json << ',';
        json << values[i];
    }
    json << ']';
}

void append_i64_array(std::ostringstream& json,
                      const std::vector<std::int64_t>& values) {
    json << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) json << ',';
        json << values[i];
    }
    json << ']';
}

void append_double_array(std::ostringstream& json,
                         const std::vector<double>& values) {
    json << '[' << std::setprecision(17);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) json << ',';
        json << values[i];
    }
    json << ']';
}

const char* route_name(int route) {
    if (route == snn_v06::FORCE_FULL) return "FULL";
    if (route == snn_v06::FORCE_CG) return "CG";
    if (route == snn_v06::FORCE_STREAM) return "STREAM";
    return "AUTO";
}

const char* start_name(int start) {
    if (start == snn_v06::RESIDENT_WARM) return "RESIDENT_WARM";
    if (start == snn_v06::COLD_ZERO) return "COLD_ZERO";
    return "HOST_X0";
}

std::size_t padded_matrix_row(int columns) {
    const std::size_t width = columns < 0 ? 0U : static_cast<std::size_t>(columns);
    return ((width + 7U) / 8U) * 8U;
}

bool host_route_fits(const msrp_v05::Problem& q, int route) {
    if (q.n < 1 || q.m < 1 || q.n > 1024 || q.m > 1024) return false;
    const std::size_t pn = padded_matrix_row(q.n);
    const std::size_t pm = padded_matrix_row(q.m);
    const std::size_t full = static_cast<std::size_t>(q.n) * pn +
                             static_cast<std::size_t>(q.m) * pn +
                             static_cast<std::size_t>(q.n) * pm +
                             static_cast<std::size_t>(q.m) * pm;
    const std::size_t cg = static_cast<std::size_t>(q.m) * pn +
                           static_cast<std::size_t>(q.n) * pm +
                           static_cast<std::size_t>(q.m) * pm;
    const bool bram = 16U * static_cast<std::size_t>(q.n + q.m) + 4096U <=
                      129024U;
    if (!bram) return false;
    if (route == snn_v06::FORCE_FULL)
        return full <= 440000U && (q.n != q.m || q.n <= 320);
    if (route == snn_v06::FORCE_CG)
        return cg <= 440000U && (q.n != q.m || q.n <= 372);
    return route == snn_v06::FORCE_STREAM;
}

int host_selected_route(const msrp_v05::Problem& q, int requested) {
    if (requested == snn_v06::AUTO) {
        for (int candidate = snn_v06::FORCE_FULL;
             candidate <= snn_v06::FORCE_STREAM; ++candidate)
            if (host_route_fits(q, candidate)) return candidate;
        return snn_v06::AUTO;
    }
    return host_route_fits(q, requested) ? requested : snn_v06::AUTO;
}

struct TimingRecord {
    std::uint64_t payload_write_ns = 0;
    std::uint64_t sync_to_device_ns = 0;
    std::uint64_t mailbox_publish_ns = 0;
    std::uint64_t mailbox_wait_ns = 0;
    std::uint64_t doorbell_to_done_ns = 0;
    std::uint64_t kernel_ns = 0;
    std::uint64_t output_sync_read_ns = 0;
    std::uint64_t complete_ns = 0;
    std::uint32_t sequence = 0;
};

struct ResultRecord {
    std::vector<std::int64_t> raw;
    std::array<std::uint64_t, snn_v06::TELEMETRY_WORDS> telemetry{};
    std::array<std::uint32_t, snn_v06::MAILBOX_WORDS> mailbox{};
};

class V06Session {
  public:
    explicit V06Session(const Options& options, const msrp_v05::Problem& problem)
        : options_(options), q_(problem), input_layout_(snn_v06_host::input_layout(
                                      problem.n, problem.m)),
          output_layout_(snn_v06_host::output_layout(problem.n)),
          device_(0), uuid_(device_.load_xclbin(options.xclbin.c_str())),
          kernel_(device_, uuid_, "snn_qp_v06") {
        // All pointers in each group share the same HLS m_axi bundle.  The
        // group selected from the first pointer therefore places the whole
        // packed BO on the intended memory bank.
        input_bo_ = std::make_unique<xrt::bo>(
            device_, input_layout_.bytes, kernel_.group_id(5));
        output_bo_ = std::make_unique<xrt::bo>(
            device_, output_layout_.bytes, kernel_.group_id(12));
        a_cfg_bo_ = make_bo(0, q_.A.size() * sizeof(double));
        c_cfg_bo_ = make_bo(1, q_.C.size() * sizeof(double));
        g_cfg_bo_ = make_bo(2, q_.G.size() * sizeof(double));
        cns_cfg_bo_ = make_bo(3, q_.c_norms_sq.size() * sizeof(double));
        scale_cfg_bo_ = make_bo(4, q_.row_scale.size() * sizeof(double));
        a_ddr_bo_ = make_bo(8, snn_v06_host::matrix_image_bytes(q_.n, q_.n));
        c_ddr_bo_ = make_bo(9, snn_v06_host::matrix_image_bytes(q_.m, q_.n));
        ct_ddr_bo_ = make_bo(10, snn_v06_host::matrix_image_bytes(q_.n, q_.m));
        g_ddr_bo_ = make_bo(11, snn_v06_host::matrix_image_bytes(q_.m, q_.m));

        input_map_ = input_bo_->map<std::uint8_t*>();
        output_map_ = output_bo_->map<std::uint8_t*>();
        if (input_map_ == nullptr || output_map_ == nullptr) {
            std::fprintf(stderr, "XRT BO mapping returned null\n");
            std::exit(3);
        }
        std::memset(input_map_, 0, input_layout_.bytes);
        std::memset(output_map_, 0, output_layout_.bytes);
        upload_configure_inputs();
    }

    ~V06Session() noexcept {
        // Destruction is the last safety net for exceptions and early
        // returns.  Never let an XRT exception escape a destructor while a
        // persistent CU may still own an outstanding command.
        try {
            (void)stop();
        } catch (const std::exception& error) {
            std::fprintf(stderr,
                         "ERROR: STOP cleanup raised an exception: %s; "
                         "the only known recovery from an orphaned persistent "
                         "run is a board reboot.\n",
                         error.what());
        } catch (...) {
            std::fprintf(stderr,
                         "ERROR: STOP cleanup raised an unknown exception; "
                         "the only known recovery from an orphaned persistent "
                         "run is a board reboot.\n");
        }
    }

    int configure() {
        sequence_ = 1;
        publish(sequence_, snn_v06::CONFIGURE, options_.start, 0,
                snn_v06::HOLD_TAIL, false);
        if (options_.persistent) {
            persistent_run_ = std::make_unique<xrt::run>(kernel_);
            set_run_arguments(*persistent_run_, snn_v06::CONFIGURE,
                              snn_v06::PERSISTENT);
            // Arm the cleanup guard before start().  XRT can report an error
            // after submitting a CU command; treating that path as live
            // ensures the destructor still publishes STOP instead of
            // abandoning an outstanding persistent run.
            persistent_started_ = true;
            running_ = true;
            persistent_run_->start();
            mock_complete(sequence_, snn_v06::CONFIGURE);
            if (!wait_done(sequence_, nullptr)) return snn_v06::ERR_BUSY;
        } else {
            auto run = std::make_unique<xrt::run>(kernel_);
            set_run_arguments(*run, snn_v06::SERVE, snn_v06::ONESHOT);
            run->start();
            run->wait();
            mock_complete(sequence_, snn_v06::CONFIGURE);
        }
        configured_ = true;
        const ResultRecord result = read_output(nullptr);
        return static_cast<int>(result.mailbox[snn_v06::OUT_ERROR_CODE]);
    }

    ResultRecord solve(const std::vector<double>& b, const std::vector<double>& d,
                       const std::vector<double>& x0, TimingRecord* timing,
                       int start_override = -1, int shift_override = -1,
                       int tail_override = -1) {
        if (!configured_) {
            ResultRecord result;
            result.mailbox[snn_v06::OUT_ERROR_CODE] = snn_v06::ERR_NOT_CONFIGURED;
            return result;
        }
        if (static_cast<int>(b.size()) != q_.n || static_cast<int>(d.size()) != q_.m ||
            static_cast<int>(x0.size()) != q_.n) {
            std::fprintf(stderr, "solve payload dimensions do not match problem\n");
            ResultRecord result;
            result.mailbox[snn_v06::OUT_ERROR_CODE] = snn_v06::ERR_BAD_DIMENSIONS;
            return result;
        }
        const std::uint64_t total_start = monotonic_raw_ns();
        const std::uint64_t write_start = total_start;
        std::memcpy(input_map_ + input_layout_.b, b.data(), b.size() * sizeof(double));
        std::memcpy(input_map_ + input_layout_.d, d.data(), d.size() * sizeof(double));
        std::memcpy(input_map_ + input_layout_.x0, x0.data(), x0.size() * sizeof(double));
        const std::uint64_t write_done = monotonic_raw_ns();

        const int solve_start =
            start_override < 0 ? options_.start : start_override;
        const int solve_shift =
            shift_override < 0 ? options_.shift : shift_override;
        const int solve_tail =
            tail_override < 0 ? options_.tail : tail_override;
        ++sequence_;
        write_mailbox_fields(snn_v06::SOLVE, solve_start, solve_shift,
                             solve_tail, false);
        std::atomic_thread_fence(std::memory_order_release);
        mailbox_in()[snn_v06::MAILBOX_SEQUENCE] = sequence_;
        const std::uint64_t publish_done = monotonic_raw_ns();
        const std::uint64_t sync_start = publish_done;
        sync_to_device(*input_bo_, input_layout_.bytes, 0);
        const std::uint64_t sync_done = monotonic_raw_ns();

        const std::uint64_t doorbell_start = sync_done;
        if (options_.persistent) {
            mock_complete(sequence_, snn_v06::SOLVE);
            if (!wait_done(sequence_, timing)) {
                ResultRecord failed;
                failed.mailbox[0] = sequence_;
                failed.mailbox[1] = snn_v06::ERR_BUSY;
                return failed;
            }
        } else {
            auto run = std::make_unique<xrt::run>(kernel_);
            set_run_arguments(*run, snn_v06::SERVE, snn_v06::ONESHOT,
                              -1, solve_start, solve_shift, solve_tail);
            run->start();
            run->wait();
            mock_complete(sequence_, snn_v06::SOLVE);
        }
        const std::uint64_t doorbell_done = monotonic_raw_ns();
        const std::uint64_t output_start = doorbell_done;
        const ResultRecord result = read_output(timing);
        const std::uint64_t complete_done = monotonic_raw_ns();

        if (timing != nullptr) {
            timing->payload_write_ns = write_done - write_start;
            timing->sync_to_device_ns = sync_done - sync_start;
            timing->mailbox_publish_ns = publish_done - write_done;
            timing->mailbox_wait_ns = doorbell_done - doorbell_start;
            timing->doorbell_to_done_ns = doorbell_done - doorbell_start;
            timing->kernel_ns = timing->doorbell_to_done_ns;
            timing->output_sync_read_ns = complete_done - output_start;
            timing->complete_ns = complete_done - total_start;
            timing->sequence = sequence_;
        }
        return result;
    }

    bool run_litmus(std::ostringstream& report) {
        // The marker is deliberately outside the normal command fields.  The
        // kernel echoes it into telemetry[0] after consuming the input BO.
        mailbox_in()[snn_v06::MAILBOX_LITMUS_INPUT] = LITMUS_INPUT;
        ++sequence_;
        publish(sequence_, snn_v06::LITMUS, snn_v06::HOST_X0, 0,
                snn_v06::HOLD_TAIL, true);
        if (options_.persistent) {
            mock_complete(sequence_, snn_v06::LITMUS);
            if (!wait_done(sequence_, nullptr)) return false;
        } else {
            auto run = std::make_unique<xrt::run>(kernel_);
            set_run_arguments(*run, snn_v06::SERVE, snn_v06::ONESHOT);
            run->start();
            run->wait();
            mock_complete(sequence_, snn_v06::LITMUS);
        }

        const std::uint64_t raw_without_sync =
            *reinterpret_cast<const std::uint64_t*>(output_map_ + output_layout_.raw);
        const std::uint64_t meta_without_sync =
            *reinterpret_cast<const std::uint64_t*>(output_map_ + output_layout_.telemetry);
        sync_from_device(*output_bo_, output_layout_.bytes, 0);
        const std::uint64_t raw_with_sync =
            *reinterpret_cast<const std::uint64_t*>(output_map_ + output_layout_.raw);
        const std::uint64_t meta_with_sync =
            *reinterpret_cast<const std::uint64_t*>(output_map_ + output_layout_.telemetry);
        const std::uint64_t echoed =
            *reinterpret_cast<const std::uint64_t*>(output_map_ + output_layout_.telemetry);
        const bool pattern_without = raw_without_sync == LITMUS_RAW0;
        const bool pattern_with = raw_with_sync == LITMUS_RAW0;
        const bool input_seen = meta_with_sync == LITMUS_INPUT;
        report << "{\"output_pattern_without_sync\":"
               << (pattern_without ? "true" : "false")
               << ",\"output_pattern_with_sync\":"
               << (pattern_with ? "true" : "false")
               << ",\"input_marker\":" << LITMUS_INPUT
               << ",\"input_marker_seen\":" << (input_seen ? "true" : "false")
               << ",\"meta_without_sync\":" << meta_without_sync
               << ",\"meta_with_sync\":" << meta_with_sync
               << ",\"backend\":\"" << (is_mock() ? "mock" : "xrt")
               << "\",\"echoed\":" << echoed << '}';
        return pattern_with && input_seen;
    }

    bool stop() noexcept {
        if (!persistent_started_) return true;
        if (stop_sent_) return stop_ok_;
        ++sequence_;
        try {
            publish(sequence_, snn_v06::STOP, snn_v06::HOST_X0, 0,
                    snn_v06::HOLD_TAIL, false);
            // Do not suppress a retry if the publish itself failed.  Once the
            // command is on the wire, stop_sent_ makes cleanup idempotent.
            stop_sent_ = true;
            mock_complete(sequence_, snn_v06::STOP);
            const bool acknowledged =
                wait_done(sequence_, nullptr, STOP_TIMEOUT_MS, false);
            if (!acknowledged) {
                std::fprintf(
                    stderr,
                    "ERROR: STOP sequence %u was not acknowledged within %d ms; "
                    "the persistent CU may remain outstanding.\n",
                    sequence_, STOP_TIMEOUT_MS);
                std::fprintf(stderr,
                             "ERROR: the only known recovery from an orphaned "
                             "persistent run is a board reboot.\n");
                running_ = false;
                stop_ok_ = false;
                return false;
            }

            const bool completed = wait_run_complete(STOP_TIMEOUT_MS);
            if (!completed) {
                std::fprintf(
                    stderr,
                    "ERROR: STOP was acknowledged but the XRT run did not "
                    "complete within %d ms; reboot the board before relaunch.\n",
                    STOP_TIMEOUT_MS);
            }
            running_ = false;
            persistent_started_ = false;
            stop_ok_ = completed;
            return stop_ok_;
        } catch (const std::exception& error) {
            std::fprintf(stderr,
                         "ERROR: STOP cleanup failed: %s; the persistent CU "
                         "may remain outstanding.\n",
                         error.what());
            std::fprintf(stderr,
                         "ERROR: the only known recovery from an orphaned "
                         "persistent run is a board reboot.\n");
            // Permit the destructor's safety-net call to retry a failed
            // publish or transient XRT operation, but never hide the failure
            // from the caller.
            stop_sent_ = false;
            stop_ok_ = false;
            return false;
        } catch (...) {
            std::fprintf(stderr,
                         "ERROR: STOP cleanup failed with an unknown error; "
                         "the persistent CU may remain outstanding.\n");
            std::fprintf(stderr,
                         "ERROR: the only known recovery from an orphaned "
                         "persistent run is a board reboot.\n");
            stop_sent_ = false;
            stop_ok_ = false;
            return false;
        }
    }

    std::uint64_t input_sync_count() const { return input_syncs_; }
    std::uint64_t output_sync_count() const { return output_syncs_; }
    std::uint64_t configure_sync_count() const { return configure_syncs_; }
    // Backend selection is compile-time: --mock is useful with the fallback
    // build, but must never make a real XRT run look like a simulated result.
    bool is_mock() const { return V06_HAVE_XRT == 0; }

  private:
    std::unique_ptr<xrt::bo> make_bo(int argument, std::size_t bytes) {
        return std::make_unique<xrt::bo>(device_, bytes, kernel_.group_id(argument));
    }

    std::uint32_t* mailbox_in() {
        return reinterpret_cast<std::uint32_t*>(input_map_ + input_layout_.mailbox);
    }
    const std::uint32_t* mailbox_in() const {
        return reinterpret_cast<const std::uint32_t*>(input_map_ + input_layout_.mailbox);
    }
    std::uint32_t* mailbox_out() {
        return reinterpret_cast<std::uint32_t*>(output_map_ + output_layout_.mailbox);
    }
    const std::uint32_t* mailbox_out() const {
        return reinterpret_cast<const std::uint32_t*>(output_map_ + output_layout_.mailbox);
    }

    void sync_to_device(xrt::bo& buffer, std::size_t bytes, std::size_t offset) {
        sync_bo(buffer, XCL_BO_SYNC_BO_TO_DEVICE, bytes, offset);
        ++input_syncs_;
    }
    void sync_from_device(xrt::bo& buffer, std::size_t bytes, std::size_t offset) {
        sync_bo(buffer, XCL_BO_SYNC_BO_FROM_DEVICE, bytes, offset);
        ++output_syncs_;
    }

    void upload_configure_inputs() {
        a_cfg_bo_->write(q_.A.data(), q_.A.size() * sizeof(double));
        c_cfg_bo_->write(q_.C.data(), q_.C.size() * sizeof(double));
        g_cfg_bo_->write(q_.G.data(), q_.G.size() * sizeof(double));
        cns_cfg_bo_->write(q_.c_norms_sq.data(), q_.c_norms_sq.size() * sizeof(double));
        scale_cfg_bo_->write(q_.row_scale.data(), q_.row_scale.size() * sizeof(double));
        sync_bo(*a_cfg_bo_, XCL_BO_SYNC_BO_TO_DEVICE, q_.A.size() * sizeof(double), 0);
        sync_bo(*c_cfg_bo_, XCL_BO_SYNC_BO_TO_DEVICE, q_.C.size() * sizeof(double), 0);
        sync_bo(*g_cfg_bo_, XCL_BO_SYNC_BO_TO_DEVICE, q_.G.size() * sizeof(double), 0);
        sync_bo(*cns_cfg_bo_, XCL_BO_SYNC_BO_TO_DEVICE,
                q_.c_norms_sq.size() * sizeof(double), 0);
        sync_bo(*scale_cfg_bo_, XCL_BO_SYNC_BO_TO_DEVICE,
                q_.row_scale.size() * sizeof(double), 0);
        // The four fixed images are output targets at CONFIGURE.  Initializing
        // and syncing them avoids stale data on one-shot persistence tests.
        sync_bo(*a_ddr_bo_, XCL_BO_SYNC_BO_TO_DEVICE,
                snn_v06_host::matrix_image_bytes(q_.n, q_.n), 0);
        sync_bo(*c_ddr_bo_, XCL_BO_SYNC_BO_TO_DEVICE,
                snn_v06_host::matrix_image_bytes(q_.m, q_.n), 0);
        sync_bo(*ct_ddr_bo_, XCL_BO_SYNC_BO_TO_DEVICE,
                snn_v06_host::matrix_image_bytes(q_.n, q_.m), 0);
        sync_bo(*g_ddr_bo_, XCL_BO_SYNC_BO_TO_DEVICE,
                snn_v06_host::matrix_image_bytes(q_.m, q_.m), 0);
        // CONFIGURE consumes x0 through the shared input BO.  Seed all three
        // vectors before publishing sequence one so a resident configure is
        // deterministic on real hardware as well as in the mock.
        std::memcpy(input_map_ + input_layout_.b, q_.b.data(),
                    q_.b.size() * sizeof(double));
        std::memcpy(input_map_ + input_layout_.d, q_.d.data(),
                    q_.d.size() * sizeof(double));
        std::memcpy(input_map_ + input_layout_.x0, q_.x0.data(),
                    q_.x0.size() * sizeof(double));
        ++configure_syncs_;
    }

    void write_mailbox_fields(int command, int start, int shift, int tail,
                              bool litmus) {
        std::uint32_t* mb = mailbox_in();
        mb[snn_v06::MAILBOX_COMMAND] = static_cast<std::uint32_t>(command);
        mb[snn_v06::MAILBOX_START_MODE] = static_cast<std::uint32_t>(start);
        mb[snn_v06::MAILBOX_SHIFT_STRIDE] = static_cast<std::uint32_t>(shift);
        mb[snn_v06::MAILBOX_TAIL_POLICY] = static_cast<std::uint32_t>(tail);
        mb[snn_v06::MAILBOX_FLAGS] = litmus ? UINT32_C(0x80000000) :
                                                  (options_.route == snn_v06::AUTO
                                                       ? 0U
                                                       : static_cast<std::uint32_t>(options_.route));
    }

    void publish(std::uint32_t sequence, int command, int start, int shift,
                 int tail, bool litmus) {
        write_mailbox_fields(command, start, shift, tail, litmus);
        std::atomic_thread_fence(std::memory_order_release);
        mailbox_in()[snn_v06::MAILBOX_SEQUENCE] = sequence;
        // One transfer publishes payload, command words, and the sequence in
        // their release order.  This is the timed doorbell sync on HP/HPC.
        sync_to_device(*input_bo_, input_layout_.bytes, 0);
    }

    void set_run_arguments(xrt::run& run, int command, int launch,
                           int route_override = -1,
                           int start_override = -1,
                           int shift_override = -1,
                           int tail_override = -1) {
        run.set_arg(0, *a_cfg_bo_);
        run.set_arg(1, *c_cfg_bo_);
        run.set_arg(2, *g_cfg_bo_);
        run.set_arg(3, *cns_cfg_bo_);
        run.set_arg(4, *scale_cfg_bo_);
        run.set_arg(5, input_bo_->address() + input_layout_.b);
        run.set_arg(6, input_bo_->address() + input_layout_.d);
        run.set_arg(7, input_bo_->address() + input_layout_.x0);
        run.set_arg(8, *a_ddr_bo_);
        run.set_arg(9, *c_ddr_bo_);
        run.set_arg(10, *ct_ddr_bo_);
        run.set_arg(11, *g_ddr_bo_);
        run.set_arg(12, output_bo_->address() + output_layout_.raw);
        run.set_arg(13, output_bo_->address() + output_layout_.telemetry);
        run.set_arg(14, input_bo_->address() + input_layout_.mailbox);
        run.set_arg(15, output_bo_->address() + output_layout_.mailbox);
        run.set_arg(16, command);
        run.set_arg(17, launch);
        run.set_arg(18, route_override < 0 ? options_.route : route_override);
        run.set_arg(19, start_override < 0 ? options_.start : start_override);
        run.set_arg(20, shift_override < 0 ? options_.shift : shift_override);
        run.set_arg(21, tail_override < 0 ? options_.tail : tail_override);
        run.set_arg(22, q_.n);
        run.set_arg(23, q_.m);
        run.set_arg(24, q_.k0);
        run.set_arg(25, q_.constraint_tol);
        run.set_arg(26, q_.iterations);
        run.set_arg(27, q_.projection_cap);
        run.set_arg(28, q_.has_lower ? 1 : 0);
        run.set_arg(29, q_.lower);
        run.set_arg(30, q_.has_upper ? 1 : 0);
        run.set_arg(31, q_.upper);
    }

    bool wait_done(std::uint32_t wanted, TimingRecord* timing,
                   int timeout_ms = WAIT_TIMEOUT_MS,
                   bool honor_signal = true) {
        const std::uint64_t started = monotonic_raw_ns();
        const std::uint64_t deadline = started +
                                       static_cast<std::uint64_t>(timeout_ms) *
                                           UINT64_C(1000000);
        while (true) {
            if (!is_mock() && options_.poll_sync == Options::PollSync::Always)
                sync_from_device(*output_bo_, snn_v06_host::MAILBOX_BYTES, 0);
            const volatile std::uint32_t* mailbox =
                reinterpret_cast<const volatile std::uint32_t*>(
                    output_map_ + output_layout_.mailbox);
            const std::uint32_t done = mailbox[snn_v06::OUT_DONE_SEQUENCE];
            if (done == wanted) break;
            if (honor_signal && signal_requested()) {
                std::fprintf(stderr,
                             "signal %d received while waiting for sequence "
                             "%u; beginning STOP cleanup\n",
                             static_cast<int>(g_signal_number), wanted);
                return false;
            }
            if (monotonic_raw_ns() >= deadline) {
                std::fprintf(stderr, "timeout waiting for mailbox sequence %u\n",
                             wanted);
                return false;
            }
            if (options_.poll == Options::Poll::Yield) {
                std::this_thread::yield();
            } else if (options_.poll == Options::Poll::SleepUs) {
                std::this_thread::sleep_for(
                    std::chrono::microseconds(options_.poll_sleep_us));
            }
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (timing != nullptr) {
            timing->mailbox_wait_ns = monotonic_raw_ns() - started;
        }
        return true;
    }

    bool wait_run_complete(int timeout_ms) {
        if (!persistent_run_) return true;
        // The mock has no scheduler, so waiting transitions its state to the
        // same terminal value used by ERT and returns immediately.
        if (is_mock()) {
            persistent_run_->wait();
            return true;
        }
        const std::uint64_t deadline =
            monotonic_raw_ns() + static_cast<std::uint64_t>(timeout_ms) *
                                     UINT64_C(1000000);
        while (true) {
            try {
                if (v06_run_terminal(static_cast<int>(persistent_run_->state()))) {
                    persistent_run_->wait();
                    return true;
                }
            } catch (const std::exception& error) {
                std::fprintf(stderr, "ERROR: unable to query XRT run state: %s\n",
                             error.what());
                return false;
            } catch (...) {
                std::fprintf(stderr,
                             "ERROR: unable to query XRT run state (unknown error)\n");
                return false;
            }
            if (monotonic_raw_ns() >= deadline) return false;
            std::this_thread::yield();
        }
    }

    ResultRecord read_output(TimingRecord* timing) {
        const std::uint64_t started = monotonic_raw_ns();
        sync_from_device(*output_bo_, output_layout_.bytes, 0);
        ResultRecord result;
        result.raw.resize(static_cast<std::size_t>(q_.n));
        std::memcpy(result.raw.data(), output_map_ + output_layout_.raw,
                    result.raw.size() * sizeof(std::int64_t));
        std::memcpy(result.telemetry.data(), output_map_ + output_layout_.telemetry,
                    snn_v06_host::TELEMETRY_BYTES);
        std::memcpy(result.mailbox.data(), output_map_ + output_layout_.mailbox,
                    snn_v06_host::MAILBOX_BYTES);
        if (timing != nullptr) timing->output_sync_read_ns = monotonic_raw_ns() - started;
        return result;
    }

    void mock_complete(std::uint32_t sequence, int command) {
        if (!is_mock()) return;
        std::uint32_t* mb = mailbox_out();
        mb[snn_v06::OUT_ERROR_CODE] = command == snn_v06::STOP
                                          ? snn_v06::ERR_STOPPED
                                          : snn_v06::ERR_OK;
        mb[snn_v06::OUT_STATUS] = 0;
        mb[snn_v06::OUT_SELECTED_ROUTE] =
            static_cast<std::uint32_t>(host_selected_route(q_, options_.route));
        mb[snn_v06::OUT_ITERATIONS] = static_cast<std::uint32_t>(q_.iterations);
        if (command == snn_v06::LITMUS) {
            std::int64_t* raw = reinterpret_cast<std::int64_t*>(
                output_map_ + output_layout_.raw);
            for (int i = 0; i < q_.n; ++i)
                raw[i] = static_cast<std::int64_t>(LITMUS_RAW0 |
                                                   static_cast<std::uint64_t>(i));
            std::uint64_t* telemetry = reinterpret_cast<std::uint64_t*>(
                output_map_ + output_layout_.telemetry);
            telemetry[0] = mailbox_in()[snn_v06::MAILBOX_LITMUS_INPUT];
            for (std::size_t i = 1; i < snn_v06::TELEMETRY_WORDS; ++i)
                telemetry[i] = LITMUS_META0 | i;
        } else {
            std::int64_t* raw = reinterpret_cast<std::int64_t*>(
                output_map_ + output_layout_.raw);
            std::fill(raw, raw + q_.n, 0);
            std::uint64_t* telemetry = reinterpret_cast<std::uint64_t*>(
                output_map_ + output_layout_.telemetry);
            std::fill(telemetry, telemetry + snn_v06::TELEMETRY_WORDS, 0);
            telemetry[0] = snn_v06::TELEMETRY_MAGIC;
            telemetry[2] = static_cast<std::uint64_t>(q_.iterations);
            telemetry[3] = static_cast<std::uint64_t>(q_.iterations);
            telemetry[4] = 1;
            telemetry[13] = 32;
            telemetry[14] = 8;
        }
        std::atomic_thread_fence(std::memory_order_release);
        mb[snn_v06::OUT_DONE_SEQUENCE] = sequence;
    }

    const Options& options_;
    const msrp_v05::Problem& q_;
    const snn_v06_host::InputLayout input_layout_;
    const snn_v06_host::OutputLayout output_layout_;
    xrt::device device_;
    xrt::uuid uuid_;
    xrt::kernel kernel_;
    std::unique_ptr<xrt::bo> input_bo_;
    std::unique_ptr<xrt::bo> output_bo_;
    std::unique_ptr<xrt::bo> a_cfg_bo_;
    std::unique_ptr<xrt::bo> c_cfg_bo_;
    std::unique_ptr<xrt::bo> g_cfg_bo_;
    std::unique_ptr<xrt::bo> cns_cfg_bo_;
    std::unique_ptr<xrt::bo> scale_cfg_bo_;
    std::unique_ptr<xrt::bo> a_ddr_bo_;
    std::unique_ptr<xrt::bo> c_ddr_bo_;
    std::unique_ptr<xrt::bo> ct_ddr_bo_;
    std::unique_ptr<xrt::bo> g_ddr_bo_;
    std::unique_ptr<xrt::run> persistent_run_;
    std::uint8_t* input_map_ = nullptr;
    std::uint8_t* output_map_ = nullptr;
    std::uint32_t sequence_ = 0;
    bool configured_ = false;
    bool running_ = false;
    bool persistent_started_ = false;
    bool stop_sent_ = false;
    bool stop_ok_ = true;
    std::uint64_t input_syncs_ = 0;
    std::uint64_t output_syncs_ = 0;
    std::uint64_t configure_syncs_ = 0;
};

#ifndef V06_HOST_NO_MAIN
std::string make_json(const Options& options, const msrp_v05::Problem& q,
                      int selected_route, const std::vector<TimingRecord>& timing,
                      const std::vector<ResultRecord>& results,
                      const V06Session& session, const std::string& litmus = {}) {
    std::ostringstream json;
    json << std::setprecision(17);
    json << "{\"schema\":\"snn-qp-v06-xrt-v1\""
         << ",\"mode\":\"" << (options.persistent ? "persistent" : "one_shot")
         << "\",\"backend\":\"" << (session.is_mock() ? "mock" : "xrt")
         << "\",\"xclbin\":\"" << options.xclbin << "\""
         << ",\"n\":" << q.n << ",\"m\":" << q.m
         << ",\"route_requested\":\"" << route_name(options.route) << '\"'
         << ",\"route_selected\":\"" << route_name(selected_route) << '\"'
         << ",\"start_mode\":\"" << start_name(options.start) << '\"'
         << ",\"shift_stride\":" << options.shift
         << ",\"tail_policy\":" << options.tail
         << ",\"poll_sync\":\"" << poll_sync_name(options.poll_sync)
         << '\"'
         << ",\"poll\":\"" << poll_name(options) << '\"'
         << ",\"warmups\":" << options.warmups
         << ",\"reps\":" << options.repetitions
         << ",\"clock\":\"CLOCK_MONOTONIC_RAW\""
         << ",\"clock_resolution_ns\":" << clock_resolution_ns();

    std::vector<std::uint64_t> payload, sync, publish, wait, doorbell, kernel,
        output, complete, sequences;
    for (const TimingRecord& item : timing) {
        payload.push_back(item.payload_write_ns);
        sync.push_back(item.sync_to_device_ns);
        publish.push_back(item.mailbox_publish_ns);
        wait.push_back(item.mailbox_wait_ns);
        doorbell.push_back(item.doorbell_to_done_ns);
        kernel.push_back(item.kernel_ns);
        output.push_back(item.output_sync_read_ns);
        complete.push_back(item.complete_ns);
        sequences.push_back(item.sequence);
    }
    json << ",\"payload_write_ns\":"; append_u64_array(json, payload);
    json << ",\"sync_to_device_ns\":"; append_u64_array(json, sync);
    json << ",\"mailbox_publish_ns\":"; append_u64_array(json, publish);
    json << ",\"mailbox_wait_ns\":"; append_u64_array(json, wait);
    json << ",\"doorbell_to_done_ns\":"; append_u64_array(json, doorbell);
    json << ",\"kernel_ns\":"; append_u64_array(json, kernel);
    json << ",\"output_sync_read_ns\":"; append_u64_array(json, output);
    json << ",\"complete_ns\":"; append_u64_array(json, complete);
    // complete_ns is the protocol total round-trip interval. Keep an
    // explicit alias for consumers that use the measurement-plan name.
    json << ",\"total_ns\":"; append_u64_array(json, complete);
    json << ",\"sequence\":"; append_u64_array(json, sequences);

    json << ",\"x_raw\":[";
    for (std::size_t r = 0; r < results.size(); ++r) {
        if (r) json << ',';
        std::vector<double> decoded(results[r].raw.size());
        for (std::size_t i = 0; i < decoded.size(); ++i)
            decoded[i] = static_cast<double>(results[r].raw[i]) /
                        static_cast<double>(FIXED_SCALE);
        append_double_array(json, decoded);
    }
    json << "]\n";
    // Remove the newline inserted above while keeping the construction easy
    // to inspect in a debugger.
    std::string line = json.str();
    if (!line.empty() && line.back() == '\n') line.pop_back();
    std::ostringstream tail;
    tail << ",\"x_raw_fixed\":[";
    for (std::size_t r = 0; r < results.size(); ++r) {
        if (r) tail << ',';
        append_i64_array(tail, results[r].raw);
    }
    tail << "]\n,\"telemetry_by_call\":[";
    for (std::size_t r = 0; r < results.size(); ++r) {
        if (r) tail << ',';
        tail << '[';
        for (std::size_t i = 0; i < snn_v06::TELEMETRY_WORDS; ++i) {
            if (i) tail << ',';
            tail << results[r].telemetry[i];
        }
        tail << ']';
    }
    tail << "]\n,\"sync_counts\":{\"configure\":"
         << session.configure_sync_count() << ",\"input\":"
         << session.input_sync_count() << ",\"output\":"
         << session.output_sync_count() << '}';
    if (!litmus.empty()) tail << ",\"litmus\":" << litmus;
    tail << '}';
    std::string suffix = tail.str();
    suffix.erase(std::remove(suffix.begin(), suffix.end(), '\n'), suffix.end());
    line += suffix;
    return line;
}
#endif  // V06_HOST_NO_MAIN

}  // namespace

#ifndef V06_HOST_NO_MAIN
int main(int argc, char** argv) {
    SignalGuard signal_guard;
    const Options options = parse_options(argc, argv);
    const msrp_v05::Problem problem = msrp_v05::load_problem(options.problem);
    if (problem.n <= 0 || problem.m <= 0) {
        std::fprintf(stderr, "v06 requires positive n and m\n");
        return 2;
    }

    V06Session session(options, problem);
    const int configure_error = session.configure();
    if (configure_error != snn_v06::ERR_OK) {
        std::fprintf(stderr, "CONFIGURE failed with error %d\n", configure_error);
        const bool stopped = session.stop();
        return stopped ? 3 : 5;
    }
    if (signal_requested()) {
        std::fprintf(stderr,
                     "signal %d received after CONFIGURE; beginning STOP "
                     "cleanup\n",
                     static_cast<int>(g_signal_number));
        const bool stopped = session.stop();
        return stopped ? 130 : 5;
    }

    std::string litmus_json;
    if (options.litmus) {
        std::ostringstream litmus;
        const bool passed = session.run_litmus(litmus);
        litmus_json = litmus.str();
        const bool stopped = session.stop();
        std::ostringstream wrapped;
        wrapped << "{\"schema\":\"snn-qp-v06-coherency-litmus-v1\"," 
                << "\"mode\":\""
                << (options.persistent ? "persistent" : "one_shot")
                << "\",\"poll_sync\":\""
                << poll_sync_name(options.poll_sync)
                << "\",\"poll\":\"" << poll_name(options)
                << "\",\"result\":" << litmus_json << '}';
        const std::string line = wrapped.str();
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
        return passed && stopped ? 0 : (passed ? 5 : 4);
    }

    std::vector<TimingRecord> timing;
    std::vector<ResultRecord> results;
    for (int i = 0; i < options.warmups; ++i) {
        TimingRecord ignored;
        const ResultRecord warmup =
            session.solve(problem.b, problem.d, problem.x0, &ignored);
        if (warmup.mailbox[snn_v06::OUT_ERROR_CODE] != snn_v06::ERR_OK) {
            std::fprintf(stderr, "warmup sequence %u failed with error %u\n",
                         ignored.sequence,
                         warmup.mailbox[snn_v06::OUT_ERROR_CODE]);
            const bool stopped = session.stop();
            return stopped ? 3 : 5;
        }
        if (signal_requested()) {
            const bool stopped = session.stop();
            return stopped ? 130 : 5;
        }
    }
    timing.reserve(static_cast<std::size_t>(options.repetitions));
    results.reserve(static_cast<std::size_t>(options.repetitions));
    for (int i = 0; i < options.repetitions; ++i) {
        TimingRecord item;
        ResultRecord result = session.solve(problem.b, problem.d, problem.x0, &item);
        if (result.mailbox[snn_v06::OUT_ERROR_CODE] != snn_v06::ERR_OK) {
            std::fprintf(stderr, "SOLVE sequence %u failed with error %u\n",
                         item.sequence,
                         result.mailbox[snn_v06::OUT_ERROR_CODE]);
            const bool stopped_now = session.stop();
            return stopped_now ? 3 : 5;
        }
        if (signal_requested()) {
            const bool stopped_now = session.stop();
            return stopped_now ? 130 : 5;
        }
        timing.push_back(item);
        results.push_back(std::move(result));
    }
    const int selected_route = results.empty()
                                   ? options.route
                                   : static_cast<int>(results.back().mailbox[
                                         snn_v06::OUT_SELECTED_ROUTE]);
    const bool stopped = session.stop();

    if (!results.empty()) {
        std::array<std::uint64_t, snn_v06::TELEMETRY_WORDS> telemetry =
            results.back().telemetry;
        std::vector<std::int64_t> raw = results.back().raw;
        msrp_v05::write_fixed_output(options.output, problem.n, raw, telemetry);
    }
    const std::string line = make_json(options, problem, selected_route, timing,
                                       results, session);
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
    return stopped ? 0 : 5;
}
#endif  // V06_HOST_NO_MAIN
