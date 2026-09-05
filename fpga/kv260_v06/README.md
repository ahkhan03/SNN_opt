# SNN-QP v0.6 resident KV260 kernel

This is the round B implementation of the resident framework. The
v0.5 source under fpga/kv260_v05/ is the parity reference and is not
modified. v0.6 keeps the v0.5 fixed-point recurrence and adds configure,
solve, refresh-A, and persistent mailbox commands.

## Hardware shape and memory plan

The resident image stores eight 32-bit fixed-point values in one 256-bit
ap_uint word. Every matrix row is padded to a multiple of eight logical
values. A is n rows by n columns, C is m by n, C-transpose is n by m, and G
is m by m. The logical footprints are:

    FULL = n*pad8(n) + m*pad8(n) + n*pad8(m) + m*pad8(m)
    CG   =             m*pad8(n) + n*pad8(m) + m*pad8(m)

The default logical cap remains 440000 words. It is allocated as
ceil(440000/8)=55000 256-bit words, which maps to 14 URAM rows by four URAMs
per row, or 56 URAM total. Eight URAMs remain outside this allocation. The
current square limits are FULL 320, CG 372, and STREAM 1024 dimensions. The
guarded MATRIX_WORD_CAP macro may be lowered for implementation, never raised
silently; if it is lowered, update env.yaml and these limits together.

FULL and CG load one aligned wide word per matrix row into BRAM line buffers
before their arithmetic loops. The buffers are cyclic-four partitioned in the
column dimension and the two row lanes are completely partitioned. Thus the
Hessian matvec, residual matvec, cap residual matvec, and the three event
update loops contain only BRAM/vector arithmetic and retain PIPELINE II=1.
CG streams A and STREAM streams every matrix from its fixed-point DDR image.
Row bases are formed once per loaded row, not once per lane. The 96-bit
gradient product and gradient-next arrays are bound to single-port BRAM; b,
d, x, and residual are four-way cyclic BRAM vectors.

There are three AXI bundles. g0 carries binary64 configure inputs and maps to
HP0. g1 carries the four fixed-point images and maps to HP1. g2 carries the
input BO (mailbox-in, b, d, x0) and output BO (mailbox-out, x_raw_out,
telemetry), and maps to HPC0. This keeps each packed BO in one memory group
and reduces adapter count.

The persistent wait loop reads the mailbox sequence once, then executes a
visible 256-trip ap_wait_n(1) interval with a volatile cycle counter. It
acknowledges output fields before writing done_sequence.

## ABI and mailbox

The top function argument order is:

    extern "C" void snn_qp_v06(
        const double* A_cfg, const double* C_cfg, const double* G_cfg,
        const double* cns_cfg, const double* row_scale_cfg,
        const double* b_in, const double* d_in, const double* x0_in,
        uint32_t* A_ddr, uint32_t* C_ddr, uint32_t* Ct_ddr, uint32_t* G_ddr,
        long long* x_raw_out, unsigned long long* telemetry_out,
        volatile const uint32_t* mb_in, volatile uint32_t* mb_out,
        int command, int launch_mode, int route_mode, int start_mode,
        int shift_stride, int tail_policy, int n, int m,
        double k0_f, double ctol_f, int n_iters, int projmax,
        int has_lower, double lower_f, int has_upper, double upper_f);

Commands are SERVE=0, CONFIGURE=1, SOLVE=2, REFRESH_A=3, STOP=4, and
diagnostic LITMUS=5. Launch modes are ONESHOT=0 and PERSISTENT=1. Routes are
AUTO=0, FULL=1, CG=2, STREAM=3. Starts are RESIDENT_WARM=0, HOST_X0=1,
COLD_ZERO=2; tails are HOLD_TAIL=0 and REPEAT_LAST=1. The input mailbox words
are [sequence, command, start_mode, shift_stride, tail_policy, flags].
Output words are [done_sequence, error_code, status, selected_route,
iterations_executed]. The low two flag bits request a route.

The host publishes payload and command fields, executes a release fence, then
writes sequence and synchronizes the input BO. It waits for matching
done_sequence, synchronizes the output BO, and reads raw state plus all
16 telemetry words. The packed input layout is mailbox, cache-line aligned b,
d, and x0; the output layout is mailbox, aligned raw state, then telemetry.
src/v06_host_protocol.hpp is the single offset definition.

## Generic XRT host

src/host_kv260_v06.cpp is an XRT 2.13 driver. It allocates one input BO, one
output BO, five configure BOs, and four fixed-image BOs. Persistent mode
launches once with CONFIGURE, services mailbox SOLVE commands, and sends STOP
at exit. One-shot mode launches once per command and is the fallback. HOST_X0
is the safe one-shot variant; RESIDENT_WARM in one-shot mode is the explicit
test that static state survives separate launches on the board. A persistent
launch owns an outstanding CU command until STOP is acknowledged and the XRT
run reaches a terminal state. The host sends STOP on normal return, every
handled error path, and SIGINT/SIGTERM, with a bounded wait and a loud failure
message. If that handshake is lost, the only known recovery from an orphaned
persistent run is a board reboot. xmutil unloadapp/loadapp did not clear the
outstanding-command state in the observed incident, so they are not a recovery
claim here. The acknowledgement and terminal-run waits are each bounded at
30 seconds in the host; a timeout returns a failure status and prints the
reboot instruction.

The board smoke sequence must include a polled busy-wait run followed by an
interrupt-mode launch. That ordering is a suspected trigger for the
outstanding-command failure and remains an explicit regression probe. Exercise
normal STOP, an injected solve error, SIGINT, and SIGTERM; each case must show
the STOP acknowledgement and a completed run before the next launch.

The --litmus option runs the coherency test. The kernel writes
0x13579bdf00000000|i to raw output and echoes input marker 0xdecafbad in
telemetry. The JSON reports output reads before and after sync, the input
marker result, backend, and both directions' observations. Keep syncs in the
timed path unless this test proves the board mapping is uncached.

Timing JSON uses CLOCK_MONOTONIC_RAW and records payload write,
sync_to_device_ns, mailbox publish/wait, doorbell-to-done, kernel, output
sync/read, complete_ns, and the total_ns alias. Arrays are flattened by call,
with raw fixed-point and decoded state retained.

## PMSM stream client

src/host_kv260_v06_stream.cpp reads the read-only resident_bundle.hpp format
used by the PMSM experiment. It configures static arrays once, then submits
each period's b and d through the generic mailbox session. The default is
resident warm state with HOLD_TAIL and shift_stride=2 after the first period.
The first period uses the configured x0 without a shift, matching the
predecessor's recorded first action; --host-x0 performs the same shift on the
host and is the safe one-shot control.

The stream JSON preserves the predecessor keys kernel_ns, complete_ns, x_raw,
x_raw_pu, x_raw_fixed, telemetry_by_period, and sync_counts, and adds mailbox
timing arrays, sequence numbers, recorded and observed first actions, and
per-period feasibility. The hard application gate is every committed state
feasible within two fractional LSBs, 2*2^-24 =
1.1920928955078125e-7 per unit. Mock runs report the gate but do not claim
board feasibility.

## Build and parity

Build both hosts with:

    fpga/kv260_v06/build/build_host.sh

On a board with XRT 2.13 the script uses:

    g++ -std=c++17 -O2 -I/usr/include/xrt ... -lxrt_coreutil

With no XRT headers it defines V06_FORCE_MOCK and builds the in-process
facade in src/v06_xrt_mock.hpp. Native recurrence and parity use the HLS
headers:

    HLS_INCLUDE=/home/ameer/RD/dev_projects/platforms/kria/hls_include/2022.1 fpga/kv260_v06/build/run_parity.sh

The parity battery compares raw state and all 16 telemetry words against
v0.5 across routes, starts, launch modes, refresh-A, range/cap fixtures, and
PMSM bundles. A passing run ends with
PARITY SUMMARY cells=271 failures=0 PASS.

## Qualification boundary and deviations

This seat cannot run Vitis HLS, link, implementation, or a board. The
controller must check the loop table, inferred URAM/BRAM/LUT, timing, BO
coherency, one-shot persistence, and the two-LSB PMSM gate. The only shape
deviation from the original raw image is the explicit 256-bit row packing and
the corrected asymmetric C-transpose row extent; fixed-point values and
recurrence operation order are unchanged. The mock backend checks host
ordering and JSON only.
