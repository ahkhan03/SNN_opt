# Lever C root cause: the fused facet arm pays 118 empty parent states (2026-10-24)

The board record in `lever_c_board_result_2026-10-24.md` measured the fused
event pipeline 1.7-2.0x slower per iteration than v08 and left the mechanism
open. This note closes it from the verbose schedule of `solve_current` on
that image (383 FSM states). The two reports in `csynth_v09u/` beside this
file are a regeneration, not the shipped build's own csynth: kernel bytes
sha256 958d9438d424ff3e8b64df10f3fe3e8a61bf6aeb8783f18cf7b625b4f69641a4,
commit 0d80871, `-DV09U_HLS_FULL_ONLY`, part xck26-sfvc784-2LV-c, 5.0 ns
clock, 2.5 ns uncertainty, csynth elapsed 144 s. The shipped build's
`solve_current` csynth and the regenerated
`p_anonymous_namespace_solve_current_csynth.rpt` match after line-numbered
identifiers (`_lnN`, `VITIS_LOOP_N_M`, `_fu_N`) and the Date and Project
headers are normalized: zero remaining differences, so the schedule
describes the measured image. The shipped report predates its PROVENANCE
guard. It prints `VITIS_LOOP_1204_2`, the loop's line at ea28d7d, while
the shipped da99298 bytes and this regeneration print line 1212. The
8-line difference is the FULL-only configure guard, outside the event
path, which is why the line-numbered identifiers differ: source-line
tokens that differ are 8 higher here, and some `_fu_` instance numbers
differ by 16. One normalization was applied to the verbose schedule: the
build host's absolute source prefix was removed, so kernel citations read
`src/snn_qp_v09u_kernel.cpp`. No schedule number was edited. State names below are the FSM states in
`csynth_v09u/p_anonymous_namespace_solve_current.verbose.sched.rpt`. Module
latencies are the instance rows in
`csynth_v09u/p_anonymous_namespace_solve_current_csynth.rpt`.

`Tx` is `ceil(n / 8)`. A bound event is a lower or upper facet. v08's m <= 8
formulas are from `cycle_diagnosis.md`: a row is `130 + 2*Tx`, a bound is
`61 + Tx`.

## Board measurement

The excess over v08 on the four economic-dispatch anchors is a bound-facet
cost. It has no term proportional to `Tx`, and the row path saves `Tx`.

Kind splits are the telemetry in `cycle_diagnosis.md` section 6 (the same
events: the board record's telemetry is bit-identical). Cycle columns are
the board record. `N_bound` is lower plus upper.

| n | Tx | N_row | N_bound | v09u cyc/iter | v08 cyc/iter | excess |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2 | 18.974 | 41.938 | 9433.0 | 5316.0 | 4117.0 |
| 40 | 5 | 15.199 | 90.446 | 17669.5 | 8788.0 | 8881.5 |
| 160 | 20 | 27.843 | 396.139 | 80512.5 | 41846.3 | 38666.2 |
| 320 | 40 | 39.296 | 817.209 | 186532.9 | 107193.7 | 79339.2 |

Least squares of

```
excess = a * N_bound + (b - Tx) * N_row
```

gives `a = 99.000` and `b = 0.183`. Residuals are -0.39, +0.54, +0.09, and
-0.08 cycles per iteration. A free coefficient on `N_bound * Tx` comes out
at -0.00005, so there is no per-trip term. Forcing the row term flat, with
no `Tx` credit, leaves residuals up to 119 cycles at n=160, and up to
1.4% of a v08 iteration at n=10. A free slope on `Tx` per row event is -1.002.

Read against v08's formulas, the fit is a facet of `160.00 + Tx` and a row
of `130.18 + Tx`. The facet `Tx` cancels v08's `61 + Tx`, which is why the
bound coefficient is 99 cycles at every width in this table. The row saves
one `Tx` because the fused row does not run a separate post-event scan.
This is the m <= 8 comparison. The fused facet arm below does not get
shorter when m <= 8.

## Static walk

ST_30 is the projection-loop header. Its successors are ST_31, ST_33,
ST_230, ST_379, and ST_371. The switch at source line 1300 sends kind 0 to
ST_33. The two other kinds, the two arms of the line 1532 delta cast, go to
ST_31 and ST_230 and rejoin at ST_32. ST_379 is the break. ST_371 is the
normal sweep exit. Both facet entries then take one path to the back-edge
at ST_370.

**Facet arm.** From ST_31 to ST_370, and separately from ST_230 to ST_370,
the transition list is a single chain of 143 states. Every state on either
chain has exactly one successor. ST_32 does contain `br_ln348` and
`br_ln392`, and both targets collapse to ST_231, so they are not a split.
There is no compare on `configured_m` anywhere on the chain. The m <= 8
straight-line facet in the source is not in this netlist.

Of the 143 states, 118 are operation-free: ST_240 through ST_357, each with
zero operations, Delay 0.00, and a state id that steps by one (SV 33
through 150). They sit immediately after the `update_facet_forward` call.
ST_238 issues the call and ST_239 waits. The instance row for that module
is 18 to 145 cycles. The endpoints differ by 127, so a one-trip minimum
makes the latency `trip + 17`. Counting the wait as `Tx + 17`, the header
once, and every other state on the chain once:

| piece | cycles | where |
|---|---:|---|
| header | 1 | ST_30 |
| delta cast, either kind | 1 | ST_31 or ST_230, line 1532 |
| ap_fixed normalize | 1 | ST_32, line 181 |
| x loads | 1 | ST_231 |
| poke of `x[winner]` | 2 | ST_232, ST_233, line 1541 |
| `ct_base` multiply | 4 | ST_234 to ST_237, line 1547 |
| call issue | 1 | ST_238 |
| `update_facet_forward` | Tx + 17 | ST_239, instance latency 18 to 145 |
| operation-free chain | 118 | ST_240 to ST_357 |
| loads and `combine_families` | 2 | ST_358, ST_359, line 1628 |
| event counters | 1 | ST_229, lines 1736 to 1742 |
| digest and back-edge | 11 | ST_360 to ST_370, line 1286 |
| total | 160 + Tx | |

The total matches the board fit. Unaccounted: 0. The same sum at one trip
is 161 cycles. Nothing on this arm is near the printed projection-loop
minimum of 95.

**Divider, and why a hoisted divide is not the facet cost.** ST_34 through
ST_93 are 60 states, each carrying `sdiv_ln1349` on the `DivnS_SEQ` core
(latency 59). ST_34's only predecessor is ST_33, and ST_33's only
predecessor is ST_30. A walk that starts at ST_31 or at ST_230 and stops on
returning to ST_30 reaches exactly the 143 facet states and none of ST_34
through ST_93. The divide operands are prepared speculatively in ST_30
under the kind==0 predicate. The 60-cycle core runs only on the row arm.

**Row arm, m <= 8, at the fleet width m = 2.** Kind 0 continues ST_33, the
divider, the `update_x_row_resident` call at ST_94 and ST_95 (line 1318,
instance latency 16 to 143, which is `Tx + 15`), and ST_96. ST_96 branches
on `configured_m < 9` (line 1362) and on `configured_m > 0`. The true side
of `m < 9` is the unrolled body ST_163 through ST_227. That body has eight
lane branches (ST_165, 172, 179, 186, 193, 200, 207, 214), each at line
1374. The live side of each branch is 5 states and the other side is a
direct edge. At m = 2 the compares keep lanes 0 and 1 (`m > 0`, `m >= 2`)
and bypass the other six, so the body is `65 - 6*5 = 35` states. The whole
function has only two operation-free blocks, ST_98 to ST_161 and ST_240 to
ST_357, and neither is on this path.

| piece | cycles |
|---|---:|
| header ST_30 | 1 |
| operand loads ST_33 | 1 |
| divider ST_34 to ST_93 | 60 |
| ST_94 | 1 |
| x-update wait ST_95 | Tx + 15 |
| branch ST_96 | 1 |
| unrolled residual, m = 2 | 35 |
| ST_162, ST_228, ST_229 | 3 |
| digest and back-edge ST_360 to ST_370 | 11 |
| total | 128 + Tx |

Zero operation-free states. The board fit is `130.18 + Tx`, about 2 cycles
above the walk. The earlier static row cell was `132 + Tx`, about 2 cycles
above the fit. Each extra live lane, up to m = 8, adds the 5-state body;
m = 8 is `158 + Tx` on this same arm. The m > 8 side is the other successor
of ST_96: ST_97 calls `update_residual_row_resident` (line 1402), and ST_98
through ST_161 are 64 operation-free states.

**The printed minimum of 95 is not a floor.** The solve_current loop table
gives `projection_loop` iteration latency `95 ~ 432`. The shortest path
from ST_30 to ST_370 is 78 edges (79 states): the kind==0 divider, the
x-update call, then ST_96's direct edge to ST_162, then the shared
epilogue. That direct edge is the remaining combination of the two
compares, `configured_m >= 9` and `configured_m == 0`. No configuration
satisfies both. A real m = 0 row stays on the m <= 8 arm and only skips the
eight lane bodies, which is a longer path. The x-update module minimum on
the short path is 16 cycles. 78 edges plus 16 is 94, one cycle under the
printed 95. The facet arm is 161 cycles at one trip, and the m = 2 row arm
is `128 + Tx`. The 95 is this infeasible path, not a floor under either
real arm, and it was the wrong place to read a facet fixed cost.

## What the earlier budget got right and wrong

The Reading section of the board record says the static schedule predicted
the opposite sign: fused loops at achieved initiation interval 1, iteration
latencies 15 to 17, and a skeleton 43 to 81 cycles cheaper than v08. The
facet cell behind that prediction is `Tx + 17`. The row cell is `132 + Tx`
(`112 + (Tx + 15) + 5`).

The row cell was right. It sits about 2 cycles above the board fit and
about 4 above the walk. The facet cell was the child module alone. The
parent arm is `160 + Tx`, so `Tx + 17` undercounts by 143 cycles at every
`Tx`. The chain from the facet entry to ST_370 is 143 states, with the
wait counted as one of them. Charging that wait at `Tx + 17` and adding
the header replaces one cycle with `Tx + 17` and adds one cycle, so the
cycles the cell missed equal the chain length, 143. The skeleton deltas
of 43 to 81 were that undercount, with the sign flipped.

The solve_current loop table is why the undercount was easy to miss. It
prints `projection_loop` at `95 ~ 432` and leaves both initiation-interval
columns blank. The instance table prints each child call as not a pipeline,
with `update_facet_forward` and `update_residual_row_resident` both at 18
to 145 cycles. Neither table has a line for ST_240 to ST_357 or for ST_98
to ST_161. The per-pipeline initiation interval and the iteration latency
of 17 describe the child. They do not describe the parent states that
follow the call. Those states are cycles (`cycle_diagnosis.md` records the
same fact for v08: operation-free states are cycles and do not grow with
trip count), and they are visible only in the verbose schedule.

## What is still unexplained

The length of the empty chains. `update_facet_forward` and
`update_residual_row_resident` have the same instance latency, 18 to 145.
The facet arm then waits through 118 empty states, and the row m > 8 arm
waits through 64. Neither length equals the child's maximum of 145, and
the two children do not explain why the chains differ. This note measures
both lengths and does not derive them. A successor that wants the cycles
back has to remove the states, not retune a latency that the report already
prints.

## Verdict

Lever C in this form is a dead end. The board result stands: v08 remains
the event-heavy kernel, and the ~85 to ~50 cycles/event figure is
superseded by the measurement. The schedule now says why. Fusing the facet
scan into the update loop deleted the m <= 8 straight-line facet and left
every bound event on one parent chain of 143 states, 118 of them empty, at
every m. The excess against v08's `61 + Tx` is 99 cycles. 118 of the fused
arm's 160 fixed cycles are those empty states; v08's m = 2 facet walk has
none (`cycle_diagnosis.md`). The other fixed terms differ by 19 cycles in
v08's favor, and `118 - 19 = 99`.

A successor must not fuse the facet scan into the update loop. The
promising direction is an m <= 8 facet fast path on the BRAM lineage
(`fpga/kv260_v09_bram`): keep that lineage's wide matvec, and for m <= 8
take a straight-line facet of the kind v08 already walks in 49 states with
no operation-free chain, instead of the span-wide fused loop whose parent
carries ST_240 to ST_357.
