# SNN-QP v09b resident BRAM kernel

v09b keeps the v0.8 command ABI, mailbox, 16 telemetry words, fixed-point
formats, and the gradient / scan / event / digest operation order. The
resident store is the 2026-06-16 WINNER row-interleave BRAM image, capped at
n, m <= 128. The top function is `snn_qp_v09b`. fpga/kv260_v06/ is the
byte-level reference and is not modified.

This package is the qualified resident BRAM kernel. It has an x86 parity
gate, an HLS script, a routed bitstream, and a board measurement. Clock,
latency, and resource counts are in the qualification section below. CG
and STREAM are not in this image, and n or m above 128 is rejected.

## Memory plan

Static 2D arrays, `ap_fixed<32,8>` elements, bound to BRAM. No URAM, no
packed 256-bit words, no `resident_half`.

| array | shape | partition |
| --- | --- | --- |
| A | n x n, max 128 x 128 | cyclic 4 on dim 1, cyclic 8 on dim 2 |
| C | m x n | cyclic 4 on dim 1, cyclic 8 on dim 2 |
| Ct | n x m, Ct[j][i] = C[i][j] | cyclic 8 on dim 2 |
| G | m x m | cyclic 8 on dim 2 |
| x, residual, row scale | vectors | cyclic 8 |
| b, d, Ax | vectors | cyclic 4 |

Ct is the facet image v0.8 kept as C transpose. A facet event reads
Ct[winner][:]. A row event reads C[winner][:] into x and G[winner][:] into
the residual.

The logical footprint is n*n + 2*m*n + m*m fixed words (A, C, Ct, G), plus
the vectors. At the cap that is 4 * 128 * 128 = 65536 matrix words. The
factors in the partition pragmas are literals. This HLS version does not
expand a macro in an ARRAY_PARTITION factor.

## Cap and routes

MAXN and MAXM are 128. The only implemented route is FULL.
`route_selected` is FULL on every acknowledgement, including a configure
that is rejected and a solve that arrives before any successful configure.

A geometry outside 1..128, or a non-positive iteration count or projection
cap, or a bound flag other than 0 or 1, returns ERR_BAD_DIMENSIONS (2).
FORCE_CG and FORCE_STREAM are still in the ABI and return ERR_CAPACITY (3)
at a legal size. AUTO and FORCE_FULL inside the cap select FULL. The status
vocabulary is unchanged. CG and STREAM arithmetic is not in this kernel.

## Arithmetic

The Hessian and the residual matvec still issue RG = 4 rows by UF = 8
columns in one pipeline step (32 products). The accumulation inside that
step is two mac_four partials. `part_lo` folds columns j+0..j+3 from zero,
in ascending order. `part_hi` folds j+4..j+7 from zero. Then
`acc[g] += part_lo` and `acc[g] += part_hi`. Partial boundaries sit on
absolute columns 0, 4, 8, and so on, the same step as v0.8. Lanes past n or
m are skipped. The store is v0.8's `cast_dt`. The residual adds d in
`acc_t` before that cast.

A product of two `dt` values has 48 fraction bits. Adding it into an
`acc_t` partial rounds against the bits already in that partial. An
eight-long chain rounds the upper four columns against the lower four, and
that does not match v0.6. Two separate `acc_t` adds match mac_four when a
partial saturates. One add of `(part_lo + part_hi)` matches only while that
sum does not saturate. A tree over the four products would not match either.

The gradient is one fused pass with the v0.8 product type and casts. The
scan is the v0.8 fused eight-lane scan: explicit rows (`residual * scale`),
then lower facets (`lower - x`), then upper facets (`x - upper`). Family
priority is row, then lower, then upper. Within a family the first maximum
wins. The step is the true division `residual[j] / cns[j]`.

Event updates step UF = 8. x is updated from C[jmax][:], the residual from
G[jmax][:], and a facet from the Ct row. A residual or facet row with m <= 8
is unrolled; a wider row is pipelined. Per-lane arithmetic is the v0.8
sequence. Lane range flags are kept and are not part of telemetry word 15,
which matches v0.6 and v0.8.

## ABI

The argument list is the v0.8 list. Commands are SERVE=0, CONFIGURE=1,
SOLVE=2, REFRESH_A=3, STOP=4, LITMUS=5. Launch modes are ONESHOT=0 and
PERSISTENT=1. Routes are AUTO=0, FULL=1, CG=2, STREAM=3. Starts are
RESIDENT_WARM=0, HOST_X0=1, COLD_ZERO=2. Tails are HOLD_TAIL=0 and
REPEAT_LAST=1. Mailbox input words are [sequence, command, start_mode,
shift_stride, tail_policy, flags]. Output words are [done_sequence,
error_code, status, selected_route, iterations_executed]. Telemetry word 0
is the v0.5 magic `0x4d53525056303531`. `src/v09_abi.hpp` is that vocabulary
under the `snn_v08` namespace, so a host include is a rename of the header
file and of the top function.

AXI bundles match v0.8: g0 binary64 configure (HP0), g1 fixed-point images
(HP1), g2 mailbox, b, d, x0, raw state, telemetry (HPC0). Interface depths
are the cap (16384 matrix words, 128 vector words), not the old 1024 cap.
Depth is an HLS hint. The C types are unchanged.

`build/run_hls.tcl` is the v0.8 script with the top set to `snn_qp_v09b`.
Part xck26-sfvc784-2LV-c, 5.0 ns clock, 2.5 ns uncertainty. It has not been
run.

## x86 gate

`build/build_native.sh` builds `native_fixed_v09b`. The CLI is
`native_fixed_v09b problem.bin output.bin`. Optional flags match the v0.8
driver (`--persistent`, `--route auto|full|cg|stream`). The dump magic is
`MSRP V091`. Telemetry inside the dump is still the kernel's 16 words.

`build/run_parity.sh` compiles v09b and v06 into one binary and compares, per
fixture, every raw state word and telemetry[0..15], plus the mailbox error
and the selected route. It runs the three v06 fixtures, fleet n=10 and n=40
at 2000 iterations, and the square T bundles at K=2000. A mismatch exits
non-zero. Set `HLS_INCLUDE` to the Vitis HLS 2022.1 include tree. Set
`FLEET_DIR` to the directory that holds the two fleet bundles. The script
names those two files. It does not pick up larger fleet bundles in the same
directory.

Both x86 sets match v06 after that split. `build/run_parity.sh` (the
K=2000 set) exits 0 with `failures=0`. Wall time was 10.33 s, including the
compile. The four K=30000 squares, run with `--no-extra` on that same
binary, also exit 0 with `failures=0`. Wall time was 46.21 s. T=80 at
K=30000 matches every raw state word and telemetry[0..15]. Those K=30000
files are still not in `run_parity.sh`. The earlier mismatch, 23 raw words
on T=80 with the first differing final state at K=2777, was the eight-wide
fold. It is gone.

## Qualified implementation (2026-10-24)

Bitstream `snn_qp_v09b_200m_3a352d326c0a`, built from kernel commit `bf208c2`
with Vitis 2022.1 at 200 MHz (HLS uncertainty 2.5 ns,
`Performance_ExplorePostRoutePhysOpt`): routed WNS +0.006 ns, TNS 0,
WHS +0.010 ns; 72,204 CLB LUT (61.65 %), 83,745 FF, 86 BRAM tiles (59.72 %),
0 URAM, 374 DSP (29.97 %). Board app of the same name; `kria-clock load`
verdict OK, kernel_hz_measured 199,999,998 (pl0_ref 249,999,998, ratio 0.8;
quote the kernel hertz, never pl0_ref). Byte identity on the board: 13
fixtures x {ONESHOT, PERSISTENT} against the x86 oracle records,
`identity failures=0`; fleet n=160/320 SKIP by cap. Latency (square
bundles, K=30000, 5 warm-ups + 50 reps, `--poll-sync never`), median
complete/30000 at the measured clock:

| T | us/iter | cyc/iter | v08 cyc/iter | legacy WINNER cyc/iter |
|---:|---:|---:|---:|---:|
| 16 | 4.392 | 878.5 | 1205.6 | 573.8 |
| 32 | 5.939 | 1187.8 | 1875.7 | 947.0 |
| 64 | 9.827 | 1965.4 | 3668.5 | 1773.3 |
| 80 | 11.813 | 2362.6 | 4713.8 | 2239.8 |

Full records: `evidence/board_2026-10-24_3a352d326c0a/`.

## What is not claimed

Energy per solve (no INA260 campaign), CG/STREAM routes (removed;
rejected with ERR_CAPACITY), sizes above the cap 128, PMSM/stream clients,
and a lean event body: the row/bound event bodies are the v08 ones, which is
why small-T cyc/iter sits above the legacy WINNER image (whose ~88-cycle
event body pairs with retired box semantics). A lean-event-FSM pass is the
recorded next lever for this lineage.

## Host and board kit

`src/host_kv260_v09b.cpp` is the v0.8 XRT host with the top function renamed
to `snn_qp_v09b`. The command-line flags match that host.
`build/build_host.sh` compiles it. When `/usr/include/xrt/xrt_bo.h` or
`/usr/include/xrt/xrt/xrt_bo.h` exists, the script links `-lxrt_coreutil`
and adds `-I/usr/include/xrt -I/usr/include`. Otherwise it builds the
in-process mock with `-DV09B_FORCE_MOCK`. The mock writes a deterministic
telemetry record and does not run the kernel. The `--mock` flag does not
change the backend. The JSON field `backend` is `xrt` or `mock`.

The host process still loads a bundle up to n,m = 1024. The kernel rejects
n or m above 128 with `ERR_BAD_DIMENSIONS`. `evidence/native_records/` holds
oneshot dumps from `native_fixed_v09b` for each fixture that loader accepts.
`fleet_n160_it2000.bin` and `fleet_n320_it2000.bin` produce no record,
because `load_problem` is called with a cap of 128 and exits 2. The
directory README records the generating binary sha256 and the commands.

`tools/board/run.sh` and `tools/board/cmp_out.py` compare a board dump with
those records: raw state, all 16 telemetry words, error, and selected route.
Set `XCLBIN`, `HOST`, `FLEET_DIR`, and `CLOCK_JSON`. The runner prints SKIP
for a fleet row above n=128. CG and STREAM are not identity cells. The
kernel returns `ERR_CAPACITY` for those routes at a legal size.
