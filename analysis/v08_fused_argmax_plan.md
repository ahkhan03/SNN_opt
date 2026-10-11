# SNN-QP KV260 v08 fused-argmax kernel: preregistered plan

## Decision question

Does a bespoke kernel that keeps the v06 canonical event sweep bit for bit, but
evaluates the winner-take-all candidate scan L lanes per cycle, beat the best
same-semantics Cortex-A53 implementation on event-bound polyhedral problems by
a material factor, and can it be qualified to replace v06 as the polyhedral
package named by `fpga/CURRENT`?

## Measured motivation

On the economic-dispatch band fleet (m = 2 band rows, box [0, 1], 2000
iterations) the qualified v06 kernel at 200 MHz measured 77.5, 177, 1310 and
4310 us per iteration at n = 10, 40, 160, 320, which is 0.25, 0.50, 0.92 and
1.09 times one A53 core running the library's canonical C core; four A53 cores
gave the same time as one. The workload is the serial projection sweep: 61 to
856 events per iteration, about 84 % of them upper-limit clips. Every event in
v06 scans m + 2n candidates one per cycle (`scan_rows`, `scan_lower`,
`scan_upper`, three pipelines with their own fill and drain), so at n = 320 the
scan alone is about 640 cycles per event. The v06 synthesis report also shows
`load_line_words` at II = 8 because the resident URAM read and the DDR image
read share one pipelined loop.

The calibrated cycle model of the programmable-datapath design study
(calibration v3, csynth-pinned loop depths) predicts a bespoke canonical-sweep
kernel with an 8-lane fused argmax at 2.4 to 8.6 times one A53 core running the
library core at n = 40 to 320. That reference is the generic library core, not
a lean harness, so the gate below is evaluated against the lean harness instead.

## Design

The v08 package is `fpga/kv260_v08/`, derived from `fpga/kv260_v06/`. Top
function `snn_qp_v08`. Unchanged: the argument list and command ABI, mailbox,
16 telemetry words and their semantics (including the event digest), routes and
capacity ladder, fixed-point formats (`ap_fixed<32,8,AP_RND_CONV,AP_SAT>`
state, `ap_fixed<48,16,AP_RND_CONV,AP_SAT>` accumulation), and the recurrence's
operation order.

1. **Fused L-lane candidate scan.** The candidate set and order stay those of
   v06: explicit rows (score `residual_i * scale_i`), then lower facets
   (`lower - x_i`), then upper facets (`x_i - upper`), strict first maximum in
   that order wins, and the sweep leaves when the best score is `<= ctol`.
   v08 evaluates L candidates per cycle and reduces them with a tree whose
   tie rule reproduces the first-maximal winner exactly. Scores are computed
   in the same `acc_t` arithmetic as v06, so the winner, kind and best value
   are identical. The cap recheck uses the same scan. L is a compile-time
   constant, default 8.
2. **Line-buffer load at II = 1 on the resident path.** The resident URAM read
   and the DDR image read become separate loops (or functions) so the
   resident path is not scheduled at the image path's memory latency.
3. **Other cycle reductions that keep outputs byte-identical** (for example
   fusing the three gradient loops into one pass) are permitted when the
   synthesis report shows they pay; each is listed in the package README.

Out of scope: any change to which candidate wins, to how the winner is found
(every candidate is evaluated at every event, as in v06), or to the update
arithmetic; new commands or telemetry; cone support (v07 remains the conic
package).

## CPU reference for the gates

The A53 reference is the lean same-semantics canonical-sweep harness that
replaces `cpu_baseline_v05` as the polyhedral `cpu_baseline:` in
`fpga/CURRENT`: binary64, `-O3 -march=native -funroll-loops` (no
semantics-changing flags), serial at one thread and a persistent OpenMP team
otherwise, same candidate order and tie rule, per-iteration work counters. Its
best measured time at one or four cores is the bar.

**Amendment (before any G3 number was computed).** "Same semantics" includes
how the winner is found: both substrates evaluate every candidate at every
event (the reference algorithm's linear winner-take-all scan). Within that
structure the CPU harness is tuned without restriction (blocked Hessian,
interleaved row dots, the fastest exact scan the A53 runs).

## Acceptance gates (fixed before any v08 number exists)

* **G1 native parity.** The v08 native emulation returns the same raw state
  and all 16 telemetry words, byte for byte, as the v06 native emulation on the
  full v06 parity battery (every route, launch mode, start mode, refresh-A,
  range and cap fixture) and on the band fleet bundles at n = 10, 40, 160, 320.
  Any difference fails the gate.
* **G2 synthesis.** Vitis HLS 2022.1 at the 200 MHz target: the fused scan
  loop at II = 1 for L = 8, the resident line load at II = 1, estimated clock
  within the 5.0 ns budget, and no loop whose II is worse than in the v06
  report.
* **G3 model gate (kill criterion).** The v3 cycle model with the v08
  synthesis loop depths substituted predicts `t_A53_best / t_v08 >= 2.0` at
  both n = 160 and n = 320 on the band fleet, where `t_A53_best` is the
  measured lean harness above. If G3 fails, no bitstream is built and this
  file records the negative result.
* **G4 implementation.** The linked design closes timing at 200 MHz (WNS >= 0)
  and fits the device with the v06 URAM plan.
* **G5 on board.** `kria-clock` verifies the 200 MHz kernel clock; raw state
  and telemetry are byte-identical to the native emulation on the v06 board
  fixture set (6x18, 20x60, 64x64; one-shot and persistent; forced STREAM and
  CG) and on the band fleet.

The measured PL and A53 times on the band fleet are then reported in whichever
direction they come out; they are a result, not a gate.

## Exit rule

G1 to G5 pass: v08 becomes the polyhedral package in `fpga/CURRENT` (boundary
README, `env.yaml` with the qualified kernel commit, evidence directory, and an
`evidence/v08-fused-argmax-kv260-<date>` tag), and v06 stays in the tree as the
superseded package. A failed gate is recorded here as a negative result and the
branch follows the experiment cadence.

## Gate record

* G1 (2026-10-11): PASS. 271 cells byte-identical (v06 battery plus the band
  fleet at n = 10, 40, 160, 320).
* G2 (2026-10-11): PASS. Every changed loop at II = 1 within the 3.65 ns
  budget; the remaining II misses are the v06 set.
* G3 (2026-10-11, design point de4d460): FAIL. The v3 model (c_call = 9.3,
  c_fn = 72 cycles; it reproduces the measured v06 band fleet within 1 %)
  with the v08 loop depths predicts 68.5 / 122.7 / 554.8 / 1348.1 us per
  iteration at n = 10 / 40 / 160 / 320 at 200 MHz. The lean linear-scan A53
  harness measured 12.8 / 35.8 / 324.3 / 1160.5 us (best of 1 and 4 cores),
  so `t_A53_best / t_v08` = 0.58 at n = 160 and 0.86 at n = 320 against the
  2.0 gate. The fused scan removed the n-per-event term, but each event still
  pays three scan-pass entries and, for a bound event, an out-of-line
  line-buffer call (about 250 cycles per event in total), and the Hessian
  copies every row pair through the line buffer before its column loop. No
  bitstream is built at this design point.
* G3 (2026-10-11, design point after the overhead round): PASS on the
  model. With the synthesized depths (fused scan II = 1 depth 8, resident
  Hessian column loop depth 11, x updates depth 7-8, straight-line residual
  update taken as its 8-cycle body latency) the v3 model predicts 147.6 us at
  n = 160 and 412.7 us at n = 320, so `t_A53_best / t_v08` = 2.20 and 2.81.
  The n = 160 margin depends on the straight-line update costing no loop-call
  overhead; at the old 17.3 cycles per bound event it would be about 1.95.
  G4 and G5 decide.
* G4 (2026-10-11): PASS at commit 4e0a9e3 (HLS clock uncertainty 2.5 ns):
  routed WNS +0.006 ns at 200 MHz, 73.6 % LUT, 56 URAM. The 1.35 ns default
  missed by 1.32 ns and 2.0 ns by 0.065 ns; physical optimization alone did
  not close the latter.
* G5 (2026-10-11): PASS. Byte-identical raw state and telemetry on the v06
  board fixture set and on the band fleet (against the v06 bitstream's board
  outputs); kernel clock verified at 200.000 MHz.

## Measured result

Band fleet, median per outer iteration on the board: v08 26.6 / 43.9 / 209.2 /
536.0 us at n = 10 / 40 / 160 / 320; the lean linear-scan A53 harness 12.8 /
35.8 / 324.3 / 1160.5 (best of one and four cores). `t_A53_best / t_v08` =
0.48 / 0.82 / 1.55 / 2.17. The model under-predicted the kernel by 22-30 %
(160.1 and 437.8 us predicted at n = 160 and 320), so the measured ratio at
n = 160 sits below the 2.0 the model promised; the crossover is near n = 100.
Exit: v08 enters `fpga/CURRENT` as the polyhedral package.

