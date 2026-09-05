#pragma once

#include <cstdint>

namespace snn_v06 {

enum Command : int {
    SERVE = 0,
    CONFIGURE = 1,
    SOLVE = 2,
    REFRESH_A = 3,
    STOP = 4,
    // Diagnostic command used only by the host coherency litmus.  It does
    // not enter the recurrence and is intentionally outside the normative
    // configure/solve/refresh/stop sequence.
    LITMUS = 5,
};

enum LaunchMode : int {
    ONESHOT = 0,
    PERSISTENT = 1,
};

enum RouteMode : int {
    AUTO = 0,
    FORCE_FULL = 1,
    FORCE_CG = 2,
    FORCE_STREAM = 3,
};

// Short tier names are useful in host code and mirror the plan's wording.
constexpr int FULL = FORCE_FULL;
constexpr int CG = FORCE_CG;
constexpr int STREAM = FORCE_STREAM;

enum StartMode : int {
    RESIDENT_WARM = 0,
    HOST_X0 = 1,
    COLD_ZERO = 2,
};

enum TailPolicy : int {
    HOLD_TAIL = 0,
    REPEAT_LAST = 1,
};

enum ErrorCode : std::uint32_t {
    ERR_OK = 0,
    ERR_BAD_COMMAND = 1,
    ERR_BAD_DIMENSIONS = 2,
    ERR_CAPACITY = 3,
    ERR_NOT_CONFIGURED = 4,
    ERR_BUSY = 5,
    ERR_STOPPED = 6,
};

constexpr std::uint32_t MAILBOX_WORDS = 64;
constexpr std::uint32_t MAILBOX_SEQUENCE = 0;
constexpr std::uint32_t MAILBOX_COMMAND = 1;
constexpr std::uint32_t MAILBOX_START_MODE = 2;
constexpr std::uint32_t MAILBOX_SHIFT_STRIDE = 3;
constexpr std::uint32_t MAILBOX_TAIL_POLICY = 4;
constexpr std::uint32_t MAILBOX_FLAGS = 5;
constexpr std::uint32_t MAILBOX_LITMUS_INPUT = 6;

constexpr std::uint32_t OUT_DONE_SEQUENCE = 0;
constexpr std::uint32_t OUT_ERROR_CODE = 1;
constexpr std::uint32_t OUT_STATUS = 2;
constexpr std::uint32_t OUT_SELECTED_ROUTE = 3;
constexpr std::uint32_t OUT_ITERATIONS = 4;

constexpr std::uint32_t FLAG_ROUTE_MASK = 0x3U;

constexpr std::uint64_t TELEMETRY_MAGIC = UINT64_C(0x4d53525056303531);
constexpr std::uint64_t NO_CANDIDATE = UINT64_MAX;
constexpr std::size_t TELEMETRY_WORDS = 16;

}  // namespace snn_v06
