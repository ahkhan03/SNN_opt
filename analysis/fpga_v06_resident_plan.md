# SNN-QP KV260 v06 resident framework: preregistered plan

## Goal and claim

The goal is one XCK26 bitstream that exposes a configure, solve, and refresh-A command ABI, retains invariant geometry and the warm state on the device when capacity permits, and still solves larger configured problems by streaming fixed-point geometry images from DDR. The primary implementation is a persistent kernel that accepts mailbox commands after one launch. A one-shot launch mode using the same dispatcher is retained as a fallback and as the launch-floor control.

The claim is deliberately two-part. First, for every supported `(n,m)`, route, command sequence, and start variant, v06 returns the same raw `ap_fixed<32,8,AP_RND_CONV,AP_SAT>` state and the same 16-word v05 telemetry, including event digest, as v05 on identical inputs. Second, for a PMSM-sized solve, persistent service removes the approximately 68 microsecond XRT launch floor without changing the recurrence. Resource, timing, latency, and energy claims are made only at the appropriate synthesis, implementation, or measured-on-board rung.

## Measured motivation

The deployed-v05 launch-floor probe (`fpga/kv260_v06/evidence/launch_floor_v05_2026-09-05/`, 300 samples per cell) gives medians of 68.0 microseconds for `n=m=1,N=1`, 72.3 microseconds for `6,1`, 90.2 microseconds for `20,1`, and 203.9 microseconds for `64,1`. At `N=101` the corresponding medians are 190.6, 252.9, 502.4, and 1747.1 microseconds. The probe attributes about 4, 22, and 135 microseconds to per-launch binary64 geometry reads at sizes 6, 20, and 64, while event-free outer iterations cost about 1.8, 4.1, and 15.4 microseconds at the latter sizes. The fixed-trip-count lower and upper scans account for the 1.2 microsecond tiny-`n` floor.

The rung-1 PMSM rows (the PMSM paper's rung-1 result block, kernel-only medians with geometry kept in device DDR by the host) measured kernel-only medians of 89.581 microseconds at `H=3,N=1` (`n=6,m=18`) and 144.541 microseconds at `H=10,N=1` (`n=20,m=60`) with geometry resident in device DDR. The measured geometry-only improvement is about 5 microseconds at H=3 and 30 microseconds at H=10. Thus geometry residency is useful but leaves the launch floor; only a persistent command service addresses the stated 10 to 20 microsecond arithmetic-scale target.

## Design

### ABI and command semantics

The v06 top-level argument order is fixed as follows. `*_cfg` and the per-solve vectors are binary64, preserving the v05 transport and device-side cast point. The four `*_ddr` buffers hold 32-bit fixed-point words after configure and are used only as backing images for streamed tiers.

```cpp
extern "C" void snn_qp_v06(
    const double* A_cfg, const double* C_cfg, const double* G_cfg,
    const double* cns_cfg, const double* row_scale_cfg,
    const double* b_in, const double* d_in, const double* x0_in,
    std::uint32_t* A_ddr, std::uint32_t* C_ddr,
    std::uint32_t* Ct_ddr, std::uint32_t* G_ddr,
    long long* x_raw_out, unsigned long long* telemetry_out,
    volatile const std::uint32_t* mb_in,
    volatile std::uint32_t* mb_out,
    int command, int launch_mode, int route_mode, int start_mode,
    int shift_stride, int tail_policy, int n, int m,
    double k0_f, double ctol_f, int n_iters, int projmax,
    int has_lower, double lower_f, int has_upper, double upper_f);
```

The scalar enumerations are `command=0 SERVE`, `1 CONFIGURE`, `2 SOLVE`, `3 REFRESH_A`, `4 STOP`; `launch_mode=0 ONESHOT`, `1 PERSISTENT`; `route_mode=0 AUTO`, `1 FORCE_FULL`, `2 FORCE_CG`, `3 FORCE_STREAM`; `start_mode=0 RESIDENT_WARM`, `1 HOST_X0`, `2 COLD_ZERO`; and `tail_policy=0 HOLD_TAIL`, `1 REPEAT_LAST`. Commands 1 through 3 are normative. All scalar registers are 32-bit except the four binary64 values, which occupy pairs of words. Set `PSU__MAXIGP0__DATA_WIDTH` to 32.

CONFIGURE validates dimensions, casts `A,C,G,cns,row_scale` once, builds `C^T` and fixed-point DDR images, latches scalars, and initializes state from `x0_in` (or zero). REFRESH_A updates only A and its image. SOLVE casts `b,d`, selects and shifts its start vector, runs the fixed horizon, writes raw state plus unchanged v05 telemetry, and commits the result. HOST_X0 and COLD_ZERO are labelled variants; resident warm is the default.

The cache-line-aligned 64-word input mailbox is `[sequence, command, start_mode, shift_stride, tail_policy, flags]`; output words are `[done_sequence, error_code, status, selected_route, iterations_executed]`. Payload and command fields are synced before sequence, which is written last; the kernel writes done_sequence last. Error codes are 0 success, 1 bad command, 2 bad dimensions, 3 capacity refusal, 4 not configured, 5 sequencing/busy, and 6 stopped.

For `shift_stride=s>0`, `new[i]=old[i+s]` for `i<n-s`; HOLD_TAIL keeps the final `s` entries in place, matching PMSM `shift_warm_start(s=2)`, while REPEAT_LAST fills them with `old[n-1]`. `s=0` is plain warm start, and `s>n` is rejected. Shift is applied before the gradient update.

### Memory plan and capacity ladder

The physical budgets are 144 BRAM36 and 64 URAM. In logical 32-bit words these are `144*36864/32 = 165,888` BRAM words and `64*294912/32 = 589,824` URAM words. Reserve 32 BRAM36 for control, FIFOs, and partitioned scratch, and 12 URAM for shell and routing margin. Thus at most 112 BRAM36 (129,024 words) and 52 URAM (479,232 words) are allocated; a guarded matrix limit of 440,000 logical URAM words is the route cap. HLS reports may lower this cap, never raise it silently. Matrices are dual-port URAM; vectors, mailbox staging, and line buffers are BRAM.

`C` has row-major and transposed fixed-point copies. The extra `mn` words make row events, facet events, and residual scans burstable. The baseline keeps full `A` and `G`, rather than assuming symmetry, because supplied fixed-point words define parity. Let `S(n,m)=16(n+m)+4096` BRAM words. The route checks `S <= 129024` separately from the URAM matrix predicate. AUTO selects the first fitting tier:

* FULL: `A,C,C^T,G` resident when `n^2+2mn+m^2 <= 440000`; a square is conservatively capped at 320.
* CG: `C,C^T,G` resident and `A` streamed when `2mn+m^2 <= 440000`; a square is capped at 372.
* STREAM: no matrix resident, with `1 <= n,m <= 1024`.

For one outer iteration, FULL adds no geometry DDR words. CG adds `n^2` A words. STREAM adds `n^2+mn+E_r*(n+m)+E_f*m`, where `E_r` and `E_f` are row and facet event counts. In the STREAM tier `C^T` turns a facet update into a sequential `m`-word burst. At H=3 (`6,18`) and H=10 (`20,60`) the FULL footprints are 576 and 6,400 words. A 512-square or 1024-square case is STREAM. Configure and refresh are one-time or amortized, and fixed-point images halve geometry word width relative to v05 binary64 reads.

### Runtime routing rule

AUTO evaluates FULL, then CG, then STREAM against the exact integer predicates above, using the configured `n,m` and the built capacities. FORCE modes are diagnostic: if the requested tier does not fit, configure returns error 3 and leaves the previous valid configuration intact. Dimensions are immutable until the next configure. Any dimension outside `1 <= n,m <= 1024` is rejected; `m=0` remains outside this v06 contract because the v05 winner scan requires a first explicit row.

### Per-solve overhead mechanism and fallback

The persistent path launches once with `launch_mode=PERSISTENT` and `command=CONFIGURE`, then waits. It polls sequence every 256 cycles at 200 MHz (1.28 microseconds); the wait loop touches no matrix memory, so its dynamic power is the polling read alone. A solve is payload write/sync, doorbell, bounded detection, recurrence, and output read. The H=3 resident rung leaves about 17 microseconds for arithmetic after the 68 microsecond floor, so the preregistered projection is a 19 to 23 microsecond median round trip and a 25 microsecond p50 target, pending measurement.

HP ports are non-coherent, and every XRT `bo.sync` is an ioctl costing several microseconds on the A53, so the per-solve payload is packed into exactly two BOs: one input BO holding the mailbox-in words, `b`, `d` and the optional `x0` (at most one sync per solve), and one output BO holding the mailbox-out words, `x_raw_out` and the telemetry (at most one sync per completion). The portable baseline calls `XCL_BO_SYNC_BO_TO_DEVICE` on the input BO before publishing sequence and `XCL_BO_SYNC_BO_FROM_DEVICE` on the output BO after `done_sequence` matches, with release/acquire fences around the commit words. On this board image the default (non-`cacheable`) zocl BO may already be an uncached mapping for which sync is a no-op and host polling of `done_sequence` is a plain load; the first board step is therefore a coherency litmus test (kernel writes a pattern, host reads it with and without sync, and the reverse) that decides whether the syncs stay in the timed path. Sync time is recorded separately either way. STOP waits for an in-flight solve, acknowledges, and exits. A killed host leaves the kernel waiting harmlessly; recovery is `xmutil unloadapp` then `xmutil loadapp`, and the interrupted run is invalidated.

ONESHOT uses the same top function and command values, executes one command per XRT launch, and retains the approximately 68 microsecond floor. It is the fallback when persistent polling or BO coherency is unavailable, and the control arm for decomposition. A board smoke test must first demonstrate that configured static state survives separate one-shot launches; otherwise one-shot remains a debug path using HOST_X0, not a claimed resident service.

### Recurrence changes and byte-identical regions

The v05 arithmetic, cast boundaries, accumulator types, event order, candidate IDs, strict `>` tie rule, residual propagation through G, cap recheck, status 2 behavior, no terminal clip, and fixed horizon are copied byte-identically in intent. Accessors select resident banks or fixed-point DDR images but never perform a new arithmetic transformation. Geometry range violations are latched per current configure/refresh image and combined with per-solve flags exactly as v05 telemetry word 15 requires. The only planned loop change is replacing the lower and upper `MAXN` scans with loops bounded by runtime `n`, retaining ascending order; this removes the measured tiny-n waste without changing winners. `RG=2`, `UF=4`, row-interleave matvec, 200 MHz clock, and the serialized projection sweep stay unchanged.

## Decision gates

1. **Native storage parity.** Run random, tie-heavy, range-edge, cap-exhaustion, and PMSM fixtures through v06 native modes and `native_fixed_v05`. Pass requires identical raw words and all 16 telemetry words for every tier and start variant. On pass, proceed to HLS. On any mismatch, freeze the design, bisect accessor/image/loop changes, and do not build a board claim.
2. **Capacity and timing.** Controller runs csynth and implementation with the guarded caps. Pass requires inferred memories within 112 BRAM36 and 52 URAM, II=1 in row-interleave loops, and Fmax at least 200 MHz. On pass, retain the four predicates. On fail, lower caps or remove the highest tier while retaining STREAM correctness; 250 MHz is not a rescue target.
3. **Mailbox correctness and latency.** Board stress uses at least 10,000 mixed configure/solve/refresh sequences, sequence wrap tests, STOP, and killed-host recovery. Pass requires no lost or duplicated sequence, exact parity, and H=3 p50 round trip <=25 microseconds. On pass, persistent is the headline. On fail, retain the one-shot arm and report persistent as unsupported or above-target, with the measured reason.
4. **Route boundary parity.** Exercise each side of every predicate, including 320/321 and 372/373 square cases plus asymmetric cases, at small `N` so native parity stays cheap. Pass requires AUTO selecting the documented tier and bit-exact outputs. On fail, reduce the route threshold to the last passing boundary and record the rejected capacity.
5. **PMSM application gate.** Run all selected H=3 and H=10 periods with the thin client adapter. Pass requires every committed state feasible within two fixed-point LSBs (`2^-24` per unit, tolerance `1.1920928955078125e-7`) and exact event digest parity. First-action deviation from the float64 command is diagnostic, as in the predecessor, not a substitute for raw parity. On fail, the framework may remain qualified only for synthetic fixtures; no PMSM device-resident claim is made.
6. **Energy protocol gate.** Five independent batches must have sensor conversions independent of the polling rate and a stable idle bracket. On pass, report total and idle-subtracted dynamic energy per solve. On fail, report total module energy with the unstable subtraction flagged and make no dynamic-energy headline claim.

## Implementation steps

1. **Seat work:** copy the v05 build skeleton into `fpga/kv260_v06/`, add `env.yaml`, README, ABI constants, mailbox schema, and guarded capacity calculator. Check that v05 is untouched and the predicates reproduce the table.
2. **Seat work:** implement fixed-image generation, resident/CG/G/STREAM accessors, and the v06 native command dispatcher. Artifact: native executable and unit fixtures. Check against v05 native for all gates in Decision Gate 1.
3. **Seat work:** implement the HLS kernel wrapper and directives, including URAM binding, C transpose, dynamic scan bounds, range-flag latching, and persistent WAIT/STOP state. Check with g++ using the Vitis HLS headers and static argument-order review.
4. **Seat work:** implement the generic host, BO/mailbox handshake, one-shot fallback, and a thin adaptation of `host_kv260_hostres.cpp`. Artifact: host smoke driver and mock-mailbox test. Check sequence ordering, shift semantics, and output schemas without a board.
5. **Seat work:** add native-v05, v06, and CPU-double golden comparison harnesses and serialized parity fixtures. Check that the double core is used only as an algorithmic reference, never as a replacement for fixed-point parity.
6. **Controller-only VM:** run csim, csynth, cosim, link, and implementation with Vitis 2022.1, 32-bit AXI-Lite, and 200 MHz. Check reports against Decision Gates 2 and 4, then return xclbin and hashes.
7. **Controller-only board:** deploy the xclbin, run configure/refresh/solve/shift smoke tests, route-boundary tests, persistent STOP, and abnormal-exit recovery. Check raw/telemetry parity before timing.
8. **Controller-only board:** run the latency and energy protocols below for synthetic cells and PMSM bundles. Check the 25 microsecond H=3 target, two-LSB band, sensor independence, and artifact provenance. The controller then folds the evidence into the branch record.

## Verification ladder

The lowest rung is v06 native fixed emulation, compared directly with the copied v05 native emulation and with `v05_double_core.hpp` only for explanatory numerical diagnostics. The next rung is HLS csim from the exact kernel source; it must match v06 native raw words and telemetry. Csynth supplies estimates of II, memory, and resource use, not measured latency. RTL cosim checks command transitions, fixed-image reads, and cycle-level outputs. Post-implementation reports establish routed resources and 200 MHz timing. The final rung is KV260 board parity for one-shot and persistent modes, every route tier, refresh-A, warm/host/cold starts, shift policies, and cap/range fixtures. The PMSM client then checks every period against the existing two-LSB feasibility band, while preserving the predecessor's distinction between hard raw-word parity and diagnostic float64 first-action deviation.

## Measurement protocol

For each synthetic size (`n=m` in 1, 6, 20, 64, 320, 372, and 1024), the two route boundaries, and PMSM H=3 and H=10, use 20 warmups and at least 1,000 solves per measured cell. Include `N=1` and `N=101`, zero-event fixtures matching the probe, and event-rich fixtures. Randomize cell order. Use `CLOCK_MONOTONIC_RAW`; in one-shot mode retain the probe interval exactly around `run.start()` through `run.wait()` and do not include buffer syncs. In persistent mode record separate medians for payload write, BO sync, doorbell-to-done, device cycle counter, output sync/read, and total round trip. Configure and refresh are reported separately, with refresh amortized over its declared cadence `K`; no overlap is credited because `b,d` are feedback-dependent.

Energy uses the INA260-class sensor at no faster than its conversion period, five fresh batches per configuration, and a calibrated active window of at least ten seconds. The primary estimand is total whole-module energy per solve over the active batch. For persistent mode, the idle bracket is the same bitstream and kernel loaded in its WAIT state with the same 256-cycle polling policy, sampled immediately before and after the active batch. Report `E_total/solve` and, separately, `(E_active - P_idle*T_active)/solve`; report the idle floor itself. For one-shot, the bracket is the loaded bitstream with no command in flight. A noisy or negative idle-subtracted value remains visible as a failed secondary gate, never silently clipped.

## Risks and unknowns

URAM width packing, banking, and routing may make the logical 440,000-word budget unattainable; the implementation gate controls the published cap. A static HLS memory may not retain contents across separate XRT launches, which is why one-shot persistence is explicitly tested. XRT 2.13 board images differ in non-cacheable allocation support, so explicit BO sync is the portability baseline and polling cost may exceed the target. Fixed-image endianness, C transpose indexing, and range-flag latching are parity-sensitive. Large STREAM cases may be correct but DDR-bandwidth dominated, especially with many facet events. The 1024 stream limit, `m=0` exclusion, mailbox lease timeout, and tail-policy extension are framework choices made because the brief does not define them; they must remain visible in the README and evidence. Finally, sensor conversion latency, host polling energy, and a killed persistent process can bias measurements unless the bracket and recovery records are complete.

## Design decisions on the first draft (2026-09-05)

The first draft was revised with these changes: the G-only residency tier and its FORCE mode were removed (three tiers, FULL / CG / STREAM, cover every measured user, and the asymmetric split is the CG tier); the in-kernel heartbeat and host lease were removed (the kernel cannot act on a dead host, recovery is `xmutil unloadapp`/`loadapp`, and the threat model is accident, not tampering); the per-solve payload was fixed to one input BO and one output BO with a coherency litmus test as the first board step, because each `bo.sync` is an ioctl and six of them would eat the gain the mailbox exists to deliver; route-boundary tests were bounded to small `N`. Everything else in the draft, including the parity chain, the mailbox sequence protocol, the capacity-ladder arithmetic and the six decision gates, is adopted as written. The launch-floor probe source and CSV are committed under `fpga/kv260_v06/evidence/launch_floor_v05_2026-09-05/`.

