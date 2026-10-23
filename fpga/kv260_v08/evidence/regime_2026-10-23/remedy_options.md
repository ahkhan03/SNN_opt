# Remedy options

Promoted as-written from the 2026-10-23 diagnosis, with one status note: the
routing rule ADOPTED into `fpga/CURRENT` and `../../README.md` extends
Option A's draft below with the box-semantics matrix (the legacy
row-interleave image carries the retired pre-v0.5.0 box route and is
reproduction-only, never a new paper's headline arm), and the Option C event
pipeline plus the widened-matvec lever are tracked together as the v09
campaign, with B2 folded into its requalification.

Gains below use the cycle model in `cycle_diagnosis.md`. Paper points are n = m = T, K = 30000, route FULL. ED points are m = 2, K = 2000, route FULL. v08 is qualified bit-identical to v06, so any change to rounding or to MAC order needs a new qualification.

## 1. Option A: regime boundary

v08 is the event-heavy polyhedral kernel. On the ED fleet it is 2.92x, 4.03x, 6.28x, 8.03x faster than v06, and the model puts 84 to 96% of each iteration in the fused scan and the short m = 2 event path. The alt-investment QP is the other regime: two square matvecs and about 4.5 to 5.0 events per iteration, where the same kernel is 1.98x to 2.10x slower than WINNER because it issues 8 products per cycle against 32.

WINNER does not cover the ED fleet: `MAXN` is 128, and its sweep clips the box after the row scan instead of scoring facets in the same pass. The split is by shape.

Draft text for the polyhedral comment in `fpga/CURRENT`, in place of the sentence that currently names v08 as the polyhedral package with no shape limit:

```
# polyhedral = rows plus box bounds, no cones.
# Two qualified images, picked by shape:
#   event-heavy (m small, tens to hundreds of facet events per outer
#   iteration; the economic-dispatch band and the v06 parity fixtures):
#   fpga/kv260_v08. Qualified 2026-10-11, bit-identical to v06 on that set.
#   matvec-heavy square box QPs (n = m, a handful of projections per outer
#   iteration; the alt-investment horizons T = 16..80): the WINNER
#   row-interleave image, snn_qp_solve_hw, 32 multipliers/cycle, post-sweep
#   clip. v08 is about 2.1x slower per outer iteration on that shape at the
#   same 200 MHz kernel clock. Do not resolve that shape through v08.
polyhedral:
  package: fpga/kv260_v08
  also_qualified_for_matvec_square: <WINNER package path>
  boundary: fpga/kv260_v08/README.md
  cpu_baseline: fpga/cpu_baseline/cpu_baseline.cpp
  datapath_width_source: fpga/kv260_v08/src/dt.h
  qualified: 2026-10-11
  shape_rule: events_per_iter
```

`also_qualified_for_matvec_square` is a placeholder. The WINNER source is in the paper tree (`analysis/offdevice/kv260/src/snn_qp_solve_hw.cpp`); SNN_opt does not carry it as a package yet. `tests/test_fpga_current.py` checks every path, so the field waits until a package path exists. Until then the comment carries the rule and `package` stays v08.

Protocol note. An arm that asks for the current polyhedral package records n, m, and events per outer iteration beside the package name. Under about 10 events/iteration with n = m, use the WINNER image. At m = 2 and tens or hundreds of events/iteration, use v08.

## 2. Option B: cheap v08 tweaks

None of these close the T = 80 gap of 2474 measured cycles. The matvec term there is 2804 modeled cycles of excess by itself.

B1. Delete the operation-free states after the m > 8 residual updates. The FULL FSM has 58 such states on a row event and 64 on a bound event, fixed, after `update_residual_row_resident` and `update_residual_facet_resident`. They are real states in the 650-state graph. If each one is a board cycle, the saving on this workload is 273, 277, 304, 305 cycles/iteration at T = 16, 32, 64, 80 (23%, 15%, 8%, 6% of measured v08). The m = 2 path has none of these states, so the ED fleet does not move.

The model that counts those states already sits 5.1% to 5.8% above the board (+62 to +255 cycles). Part of the chain may be the scheduler's view of the pipeline stall, which is also inside the module latency. The board-supported saving is therefore at most that over-prediction, not the full state count. A source change can only ask HLS for a shorter schedule. It cannot name those states.

Numerical semantics stay if the MAC order and the saturation points stay. `mac_four` (kernel.cpp:447-449) records that a tree over the four products would not match the accumulator's saturation boundary. Risk: the refill moves that boundary. Cost: a schedule experiment, then a requalification if any op moves. If nothing numerical moves, the existing v06 parity still applies.

B2. Replace the row-event divider with a multiply by a precomputed reciprocal of `resident_cns`. The divide is source line 1189. This build schedules it as 60 parent states (39 through 98), core `sdiv_56ns_32s_56_60_seq_1`. `resident_cns` is constant for the solve. ESTIMATE: the replacement multiply costs about 4 cycles, in line with the product already inside the depth-9 x update, so the saving is about 56 cycles per row event.

Paper, row events only: 140, 144, 171, 171 cycles/iteration. That is 7% of the T = 80 measured gap. ED, where row events are the smaller kind: about 1060 cycles at n = 10 (20% of measured) and 2200 at n = 320 (2%). Bound events do not take this divider.

Semantics change. `ap_fixed<32,8>` division and a multiply by a rounded reciprocal do not match bit for bit. Requalification against v06 is required. Cost is small (one extra resident vector, one change at line 1189). Risk is a silent fixed-point drift on every row correction.

Widening `UF` to 8 and `RG` to 4 is the change that would attack the 4x MAC-slot gap. This report has no schedule for it, so it has no cited cycle gain, and `mac_four` (kernel.cpp:447-449) makes the product order part of the qualified rounding. Clearing `has_lower` does not shorten the scan while `has_upper` is set (span rule, kernel.cpp:908-909).

## 3. Option C: overlap the next scan

The recorded next step forwards the one changed coordinate so the next scan starts before the rest of the event finishes, cutting the event from about 85 cycles to about 50. That is a flat -35 cycles per event. Applied to this model:

| point | events/iter | -35 cyc/iter | share of measured v08 | measured gap vs WINNER |
|---|---:|---:|---:|---:|
| T = 16 | 4.49 | 157 | 13% | 632 |
| T = 32 | 4.56 | 160 | 9% | 929 |
| T = 64 | 5.04 | 176 | 5% | 1895 |
| T = 80 | 5.05 | 177 | 4% | 2474 |
| ED n = 10 | 60.9 | 2132 | 40% | v08 already faster |
| ED n = 40 | 105.6 | 3698 | 42% | v08 already faster |
| ED n = 160 | 424.0 | 14839 | 35% | v08 already faster |
| ED n = 320 | 856.5 | 29978 | 28% | v08 already faster |

On the paper that cut is 157 to 177 cycles/iteration, against a gap of 632 to 2474. The T = 80 ratio would move from 2.10 to about 2.03.

The work this schedule can actually move is the next scan: module `ceil(span/8)+9` plus the 3-state handshake, which is 14, 16, 20, 22 cycles on the paper and 63, 73, 101, 111 cycles/iteration once multiplied by the event count. The recorded 35 is larger than that scan at every paper T, so 35 also hides part of the divider or the digest. On ED the scan is 14, 17, 32, 52 cycles per event, so at n = 160 and 320 hiding the scan (about 32% and 42% of measured) beats the flat 35.

Either reading leaves the matvec untouched. C is a large change on the ED fleet and a small one on alt-investment. It does not need a semantics change if the forwarded value is the same coordinate the scan would have read, and the MAC order is untouched. It does need a new schedule, and a requalification if the overlap moves a saturation boundary.

## 4. Recommendation

Adopt A in `fpga/CURRENT` now, and treat C as the next v08 change for the event-heavy regime. The alt-investment regression is the 8-versus-32 matvec on a square problem with about five events per iteration. C removes at most a scan per event there (about 110 cycles at T = 80, or 177 if the recorded 35 is real). B1 and B2 are a few hundred cycles and a requalification, and neither touches the 2804-cycle matvec excess at T = 80. v08 stays the qualified image for the ED shape, where it is several times faster than v06 and where C's gain is 28 to 42% of an iteration that is already mostly events. B2 belongs inside that requalification. B1 is a schedule experiment: book the saving from the new report, not from the state count.

Applied outside its regime, A would say the WINNER image is the right polyhedral kernel on the ED fleet. It is not: v08 is 2.9x to 8.0x faster there, and WINNER does not fit n = 320 or the in-sweep facet semantics. B and C would say a per-event tweak closes the alt-investment gap. At T = 80 the event term is 9% of the modeled excess and the matvec term is all of it. Leaving v08 as the single `polyhedral.package` with no shape rule is the claim that put this kernel on the alt-investment workload.
