#pragma once

#if defined(V06_FORCE_MOCK)
#define V06_HAVE_XRT 0
#include "v06_xrt_mock.hpp"
#elif defined(__has_include)
#if __has_include(<xrt/xrt_bo.h>) && __has_include(<xrt/xrt_device.h>) && \
    __has_include(<xrt/xrt_kernel.h>)
#define V06_HAVE_XRT 1
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <ert.h>
#else
#define V06_HAVE_XRT 0
#include "v06_xrt_mock.hpp"
#endif

#else
#define V06_HAVE_XRT 0
#include "v06_xrt_mock.hpp"
#endif

// XRT's ERT command-state enum is ordered NEW/QUEUED/RUNNING followed by
// COMPLETED and terminal error states.  Keeping this tiny predicate numeric
// avoids depending on enum spelling across the 2.13 board packaging variants.
// ERT assigns 4 to COMPLETED and values above it to error/abort/timeout
// states (and, on some XRT releases, SUBMITTED).  STOP is clean only when the
// persistent run reached COMPLETED, so do not collapse failures into success.
inline bool v06_run_terminal(int state) { return state == 4; }
