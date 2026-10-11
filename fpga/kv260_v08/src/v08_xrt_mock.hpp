#pragma once

// Small in-process XRT-shaped facade used when the workstation does not have
// XRT headers or libxrt_coreutil.  It deliberately models BO offsets, sync
// counters, and run lifecycle, but does not pretend to execute an FPGA kernel.
// The host's --mock path fills a deterministic telemetry record after each
// completed run so packing and mailbox ordering can still be tested.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#ifndef XCL_BO_SYNC_BO_TO_DEVICE
#define XCL_BO_SYNC_BO_TO_DEVICE 0
#endif
#ifndef XCL_BO_SYNC_BO_FROM_DEVICE
#define XCL_BO_SYNC_BO_FROM_DEVICE 1
#endif

namespace xrt {

struct uuid {
    std::uint64_t value = 0;
};

class device {
  public:
    explicit device(unsigned int = 0) {}
    uuid load_xclbin(const char*) { return uuid{}; }
};

class bo {
    struct Storage {
        explicit Storage(std::size_t n) : bytes(n, 0) {}
        std::vector<std::uint8_t> bytes;
    };
    std::shared_ptr<Storage> storage_;
    std::uint64_t address_ = 0;

    static std::uint64_t next_address() {
        static std::uint64_t next = UINT64_C(0x100000000);
        const std::uint64_t result = next;
        next += UINT64_C(0x10000);
        return result;
    }

  public:
    bo() = default;
    bo(device&, std::size_t bytes, int) : storage_(std::make_shared<Storage>(bytes)),
                                           address_(next_address()) {}

    template <typename T>
    T map() {
        return reinterpret_cast<T>(storage_ ? storage_->bytes.data() : nullptr);
    }

    void write(const void* source, std::size_t bytes, std::size_t offset) {
        if (!storage_ || offset > storage_->bytes.size() ||
            bytes > storage_->bytes.size() - offset)
            return;
        std::memcpy(storage_->bytes.data() + offset, source, bytes);
    }

    void read(void* destination, std::size_t bytes,
              std::size_t offset = 0) const {
        if (!storage_ || offset > storage_->bytes.size() ||
            bytes > storage_->bytes.size() - offset)
            return;
        std::memcpy(destination, storage_->bytes.data() + offset, bytes);
    }

    void sync(int, std::size_t = 0, std::size_t = 0) const {}
    std::uint64_t address() const { return address_; }
    std::size_t size() const { return storage_ ? storage_->bytes.size() : 0; }
};

class kernel;

class run {
    // Keep the numeric state values compatible with the ERT command states
    // used by the real XRT path: 1 means running and 4 means completed.
    int state_ = 0;

  public:
    run() = default;
    explicit run(const kernel&) {}

    template <typename T>
    void set_arg(int, const T&) {}
    void start() { state_ = 1; }
    int wait() {
        state_ = 4;
        return state_;
    }
    int state() const { return state_; }
};

class kernel {
  public:
    kernel() = default;
    kernel(device&, const uuid&, const char*) {}
    int group_id(int argument) const { return argument; }
    run operator()() const { return run(*this); }
};

}  // namespace xrt
