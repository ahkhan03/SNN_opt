# Lever C on the board: the fused event pipeline is a measured dead end (2026-10-24)

`remedy_options.md` §3 recorded the continuous event pipeline (forward the
changed coordinate, start the next scan inside the update, "~85 -> ~50
cycles/event") as the next v08 change for the event-heavy regime, worth
28-42% of an ED iteration. It was built, qualified bit-identical, routed
with margin, and measured on the same board and clock as every anchor in
`cycle_diagnosis.md`. It is 1.7-2.0x SLOWER per outer iteration on the ED
fleet. This file is the record so the lever is not re-attempted blind.

## What was built

The v08 kernel with the post-event scan fused into the event's update loops
on the resident FULL route: a row event scores the bound families inside the
x-update loop and the row family inside the residual-update loop (one
combine after both); a facet event runs one span-wide loop that updates the
residual, pokes x, and scores every family, reading the poked coordinate
from the forwarded stored value. The opening scan and the cap-recheck scan
stay separate. The arithmetic per coordinate, the winner-selection order
(row > lower > upper, first maximum in family), the true division, the
digest input sequence, and all casts are v08's. The HLS image folds the
route to a compile-time FULL constant (`-DV09U_HLS_FULL_ONLY`), deleting the
CG/STREAM datapaths and the post-event scan from the netlist; a configure
that resolves to a removed route is rejected with ERR_CAPACITY before any
state changes. Source: branch `experiment/v09-polyhedral`,
`fpga/kv260_v09_uram` (not promoted to main; the resolver never routes to
it).

## What was proven before the latency run

- x86: bit-identical to v06 on all 15 fixtures (v06 trio, ED fleet
  n=10/40/160/320 at 2000 iterations, square T=16..80 at K=2000 and
  K=30000), plus a rejection battery (forced CG/STREAM and AUTO-on-373
  return ERR_CAPACITY; a prior good FULL configuration survives every
  rejection bit-identically; the x86 binary keeps all three routes).
- Routed: WNS +0.049 ns, TNS 0, WHS +0.010 ns at 200 MHz; 64,881 LUT
  (55.40%), 81,522 FF, 29 BRAM tiles, 56 URAM, 201 DSP. The fused loops
  report achieved II 1 in csynth.
- Board: byte identity on all 15 fixtures x {ONESHOT, PERSISTENT} against
  the x86 oracle records, `identity failures=0`; the rejection cells
  return error 3 on the board with a matching control solve afterwards;
  clock verdict OK, kernel_hz_measured 199,999,998.

## What the board measured

ED fleet, K=2000, PERSISTENT, HOST_X0, 5 warm-ups, 50 timed solves,
`--poll-sync never`, median complete/2000 at the measured 200 MHz kernel
clock. v08 column: the qualified anchors (`README.md` of the 2026-10-11
board evidence; identical in `cycle_diagnosis.md` §6).

| n | v09u us/iter | v08 us/iter | v09u cyc/iter | v08 cyc/iter | v09u/v08 | events/iter | excess cyc/event |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 47.165 | 26.580 | 9433.0 | 5316.0 | 1.77x slower | 60.9 | +67.6 |
| 40 | 88.348 | 43.940 | 17669.5 | 8788.0 | 2.01x slower | 105.6 | +84.1 |
| 160 | 402.562 | 209.232 | 80512.5 | 41846.3 | 1.92x slower | 424.0 | +91.2 |
| 320 | 932.665 | 535.969 | 186532.9 | 107193.7 | 1.74x slower | 856.5 | +92.6 |

Square T=80 K=30000 PERSISTENT (informational; not a routed arm): 23.846
us/iter = 4769.2 cyc/iter against v08's 4713.8 and the BRAM lineage's
2362.6.

## Reading

The regression is entirely in the event path: telemetry is bit-identical,
so both kernels execute the same events in the same order, and the excess
is a flat +67 to +93 cycles per event, growing slightly with n. The
csynth static schedule predicted the opposite sign: every fused loop at
achieved II 1 with iteration latencies 15-17, and a per-iteration event
skeleton 43-81 cycles CHEAPER than v08's update-plus-scan at these sizes.
The static report did not capture whatever costs ~90 cycles per event at
runtime. Untested suspects, in order of plausibility: dynamic stalls on
the single shared score units (`ALLOCATION ... limit=1` for
`score_row_group`/`score_bound_group`, latency 5 and 3) when the fused
loops call them every iteration alongside their widened register/memory
fan-in; URAM/BRAM port pressure inside the fused facet loop (packed word +
Ct row + x/residual banks + scale in one II=1 body); or an unreported
handshake chain between the fused loops and the shared units. A
re-attempt needs a verbose-schedule root cause of the +90 first.

## Verdict

Lever C, in this form, is a dead end on this part: v08 remains the
event-heavy polyhedral kernel, and the ~85->50 cycles/event figure in
`remedy_options.md` §3 is superseded by this measurement. Method lesson
for the campaign class: csynth trip-latency arithmetic predicted the WRONG
SIGN for a restructured event path; the verified ±6% model in
`cycle_diagnosis.md` fits EXISTING schedules, not redesigns. Event-path
claims gate on board cycles only. The bit-identity machinery (dual-oracle
harness, rejection battery, board byte-identity) worked exactly as
intended and is reusable for any successor attempt.
