# Cycle accounting: kv260_v08 against the WINNER kernel

Path check: every source path this diagnosis cites was verified on disk.

Two data gaps shaped the method. `22_fpga_v08.yaml` records `csynth: null`, and the checked-in depths miss the gate (section 5). v08 was C-synthesised for that reason: Vitis HLS 2022.1, solution `solution_200mhz`, reports dated Fri Oct 23 20:17:07 2026, produced on the build host by reusing `fpga/kv260_v08/build/run_hls.tcl` unchanged; the load-bearing reports are mirrored under `csynth/` beside this file. Separately, the paper tree's vendored `analysis/SNN_opt/src/snn_opt/solver.py` still clips the box after the sweep. The paper venv's installed `snn_opt` is the box-in-sweep solver, and that one reproduces `08_kkt_trace.yaml`. The kernel mapping is from the two kernel sources. v08 YAML rows store `m: null`; read as n = m = T, matching the WINNER row `m: 16` at T = 16.

## 1. Verdict

Both images ran at kernel clock 199999998 Hz. Cycles/iteration = `median_us * 200 / 30000`. The ratio is 2.10, 1.98, 2.07, 2.10 at T = 16, 32, 64, 80.

The dominant term is the resident matvec. v08 issues 8 products/cycle (`RG` 2, `UF` 4, `snn_qp_v08_kernel.cpp:50`, `:54`). WINNER issues 32 (`UF` 8, `RG` 4, `snn_qp_solve_hw.cpp:15`, `:18`): four times the MAC slots (`n^2/8` vs `n^2/32`), and heavier row-group glue, so the two matvecs cost 3.21x to 3.42x the WINNER pair. Share of the modeled excess: 53%, 80%, 97%, 102%.

The per-event path is second. A row event carries a latency-60 divider, an x update, a residual update, and 58 operation-free FSM states. A bound event carries 64 such states and no divider. At 4.49 to 5.05 events/iteration this is 55%, 32%, 15%, 9% of the modeled excess.

The fused scan is the cheaper scan. WINNER's post-sweep clip narrows the ratio by 7 to 9% of the excess. Mailbox, telemetry, CG, and padding are each under 1 cycle/iteration. The ratio stays near 2.1 because the matvec gap and the WINNER baseline both grow with `n^2`, while the event excess stays near 250 to 390 cycles.

## 2. Anchor table

v08 uses `kernel_latency_us.median` (`22_fpga_v08.yaml` lines 234, 320, 406, 492). WINNER uses `latency_us.median` (`20_fpga_protocol.yaml` lines 205, 866, 1297, 1568). Host `latency_us` on v08 is 14 to 28 us higher and is not the estimand. Both `clock_record` blocks give `kernel_hz_measured: 199999998` (22 yaml line 916, 20 yaml line 3937). v08's `measured_pl0_ref_hz` is 249999998 with `ratio: 0.8` (lines 921, 925). WINNER's platform reference is already 199999998 with `ratio: 1.0` (lines 3941, 3946). The 250 MHz figure is that platform reference. The accounting is in cycles at 200 MHz.

| T | WINNER us | WINNER cyc/iter | v08 us | v08 cyc/iter | ratio |
|---:|---:|---:|---:|---:|---:|
| 16 | 86062.75 | 573.75 | 180839.338 | 1205.60 | 2.101 |
| 32 | 142045.55 | 946.97 | 281349.414 | 1875.66 | 1.981 |
| 64 | 265993.35 | 1773.29 | 550270.239 | 3668.47 | 2.069 |
| 80 | 335976.5 | 2239.84 | 707068.182 | 4713.79 | 2.105 |

## 3. Per-iteration accounting

Parent module latency, checked wherever the report prints both ends. If iteration latency `PD` exceeds `II`, the parent pays `(trips-1)*II + PD + 1`. If `PD == II`, the parent pays `(trips-1)*II + PD + 2`. For `II = 1` and `PD > 1` that is `trips + PD`.

WINNER, from `snn_qp_solve_csynth.rpt` and `csynth.xml`. `gstep` II 2, PD 4, trip 1..255, loop 3..511, parent 5..513 (rpt lines 47, 54). `clip` II 3, PD 3, loop 3..765, parent 5..767 (rpt line 54). `amac`/`cmac` II 1, PD 7 (pipeline rpts, loop table line 44). `xupd`/`rupd` II 1, PD 4; `rupd` parent minimum 5 matches trip 1. Init and store around `amac` are parent latency 6 (`VITIS_LOOP_77_9`, `VITIS_LOOP_90_12`). The `cmac` pair matches (`VITIS_LOOP_100_13` latency 6; `VITIS_LOOP_113_16` II 1, PD 2, trip 4, parent 6).

v08, all II 1, new report. `hessian_columns_resident` iteration 14, loop 13..268, parent 15..270 (trips + 14). `residual_columns_resident` iteration 15, parent 16..271. `scan_fused` and the three resident updates iteration 9, loop 8..135, parent 10..137 (trips + 9). `hessian_rows_resident` iteration 29..284 = 9 FSM states + column module + store module (store parent 5, `VITIS_LOOP_1106_2`). Loop latency = trips times that iteration (512 * 284 = 145408). `residual_rows_resident` iteration 32..287 = 9 + column module + store (store parent 7, `VITIS_LOOP_521_2`); 512 * 287 = 146944. The +9 equals the state count of one row-group body, so the called module is added in full on top of those states.

A replay of the pinned design-study model constants uses a different schedule (`c_call` 9.31819, `D_div` 40, `c_event` 8, column depths 15 and 16, scan/update depth 10, straight-line residual 11). On these event counts it gives -25.4%, -17.2%, -10.3%, -8.2%, outside the gate at T = 16 and 32. The plan's narrative depths (scan 8, Hessian column 11, `v08_fused_argmax_plan.md` lines 127-134) are a third set. The intervals below replace both. Both dividers in these reports are latency-60 cores; `D_div` 40 is the v06 fit.

Top estimated period is 3.981 ns (`csynth.xml`) on a 5.00 ns target, uncertainty 2.50 ns (`run_hls.tcl`). The board clock is clk_wiz ratio 0.8 of the 250 MHz `pl0_ref` (`build_xclbin.sh`). Cycles below are at 200 MHz.

### WINNER

`snn_qp_solve_hw.cpp`, loop `iters` at line 72. n = m = T, T divisible by 8.

- A matvec. Groups `n/4`. Each group is init 6 + `amac` `(n/8 + 7)` + store 6 = `n/8 + 19`. Total `(n/4)*(n/8+19)`.
- C matvec, same in `m` and `n`. Square, so equal to A. The `+ d` store (line 113) is inside this term.
- Gradient. No PIPELINE pragma (line 92). HLS scheduled II 2, PD 4. Module `2*n + 3`.
- Scan. `VITIS_LOOP_119_17`, source line 119 (`for i = 1; i < m`), II 1, PD 2, trip `m-1`. Module `m+1`. Scans/iteration = `E + 1`, including the scan that takes `gmax <= ctol` (line 120).
- Per event. Divide core `sdiv_56ns_33ns_32_60_seq_1` (rpt line 115), source line 122, 60 cycles. `xupd` module `n/8 + 4`, `rupd` module `m/8 + 4`. Square: `68 + n/4` per event. No added `c_call`. Stacking 9.32 on the module latency double-counts the wrapper; zero extra parent FSM is what meets the board.
- Clip, lines 141-147. II 3, PD 3, module `3*n + 2`. Once per outer iteration, after the sweep.
- Per-SOLVE loads of A, C, G (lines 65-67) and the store at line 149. ESTIMATE: one cycle per element is `3*n*n/30000` = 0.64 at T = 80, less below.
- Padding. Trips `n/8` and `n/4` have no remainder. Booked 0.

`E` is `n_projections/30000` from section 4.

| T | A | C | grad | scans | events | clip | model | measured | residual |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 84 | 84 | 35 | 73.6 | 239.8 | 50 | 566.5 | 573.75 | -1.27% |
| 32 | 184 | 184 | 67 | 145.8 | 259.7 | 98 | 938.5 | 946.97 | -0.89% |
| 64 | 432 | 432 | 131 | 286.6 | 286.3 | 194 | 1761.9 | 1773.29 | -0.64% |
| 80 | 580 | 580 | 163 | 359.6 | 302.7 | 242 | 2227.3 | 2239.84 | -0.56% |

### v08, route FULL

`snn_qp_v08_kernel.cpp`. FULL reads the resident matrix at the use site. `load_line_image` is II 8, iteration latency 17, and it is the other edge of state 99, predicated on STREAM. A line-buffer event of that depth at n = 320 does not fit in the measured 107194 cycles/iteration.

- Header, state 11: 1.
- A matvec, `hessian_rows_resident` (1079-1111). `ceil(n/2) * (ceil(n/4) + 28)`.
- Gradient, `gradient_fused_step` (1150-1163). Report prints II 1 and iteration latency 9, interval `?`. Sibling identity (parent = trips + iteration latency), trips = n, plus handshake states 19 and 29. Total `n + 11`.
- Residual (call 1165, function 488-565). Loop `ceil(m/2)*(ceil(n/4)+31)`, the printed `residual_rows_resident` cost. Plus states 30-31. Plus ESTIMATE +2 for the function entry/return, which the parent instance row leaves as `?`. Wrap +4.
- Scans. `scan_fused` parent `ceil(span/8)+9`, plus a 3-state handshake (states 32-34). Span is `max(n, m)` when either box side is on (lines 908-909). `lower_bound` 0 is configured, so lower facets are scanned even though they never win. Scans/iteration = `E + 1`. The cap-recheck scan (1512) is not taken: cap rechecks are 0.
- Event bodies, walked from state 35 through the digest back-edge 637, scan module and scan handshake removed. Row, m > 8: 145 states (58 operation-free) plus x-update `ceil(n/8)+9` plus residual-update `ceil(m/8)+9`. States 39-98 are the 60-cycle divider, core `sdiv_56ns_32s_56_60_seq_1` (rpt line 171), source line 1189. The HLS name `sdiv_ln1349` is not source line 1349. Bound, m > 8: 91 states (64 operation-free) plus facet update `ceil(m/8)+9`. The x poke (1366-1369) is inside the 91. Digest states 626-637 (12) are inside both bodies. The break adds states 35 and 646 (2 cycles) on top of its scan.
- No post-sweep clip. The box is inside the sweep (1188-1430).
- Per-SOLVE. States 1-10 run once. `write_telemetry` once after the outer loop (1535-1538), report minimum 2 cycles. Under 0.01 cycle/iteration at K = 30000.
- Padding 0. Every paper T is divisible by 8. The column bound is the live width, not `MAXN` 1024. The 256-bit word is read twice (`resident_half`, 409-431) because `UF` is 4: that is the 8-multiplier rate, not padding.

The operation-free states are cycles and do not grow with trip count. Counting them plus the module latency pins the Hessian iteration. Dropping them misses T = 16.

m <= 8 (the ED fleet) takes the straight-line residual (1231-1253). At m = 2 the walked body is 109 states (row) and 49 (bound), with zero operation-free states. Row = `112 + scan_module + x_module`. Bound = `52 + scan_module`. No residual-pipeline module. The first m lanes are taken; the rest are skipped.

| T | hess | resid | grad | header | scans | event bodies | model | measured | residual |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 256 | 284 | 27 | 1 | 76.9 | 622.2 | 1267.0 | 1205.60 | +5.10% |
| 32 | 576 | 628 | 43 | 1 | 89.0 | 648.3 | 1985.2 | 1875.66 | +5.84% |
| 64 | 1408 | 1508 | 75 | 1 | 120.8 | 762.7 | 3875.5 | 3668.47 | +5.64% |
| 80 | 1920 | 2044 | 91 | 1 | 133.0 | 779.8 | 4968.8 | 4713.79 | +5.41% |

At T = 80 a row event is 205 cycles and a bound event is 132, scan included. The break scan is 24. Matvecs are 3964 of 4969 modeled cycles.

Modeled excess over WINNER: 701, 1047, 2114, 2742 cycles.

| T | matvec gap | event gap | gradient | WINNER clip | header |
|---:|---:|---:|---:|---:|---:|
| 16 | +372 (53%) | +386 (55%) | -8 | -50 | +1 |
| 32 | +836 (80%) | +332 (32%) | -24 | -98 | +1 |
| 64 | +2052 (97%) | +311 (15%) | -56 | -194 | +1 |
| 80 | +2804 (102%) | +250 (9%) | -72 | -242 | +1 |

A negative gradient entry means v08's gradient is the smaller one. The clip is entirely on the WINNER side. Shares are of the modeled excess.

## 4. Event-count grounding

WINNER scans constraint rows only and breaks on `gmax <= ctol` (`snn_qp_solve_hw.cpp:118-120`), then clips (`:141-147`). v08 `scan_candidates` (`:895-958`) scores rows, then lower facets, then upper facets, and the box correction is inside the projection loop (`:1188-1430`).

`05_computational_realization.yaml` is the rows-only family (backend `c`, the clip-after port): 99928, 102528, 102261, 103194 over 30000 iterations. Per iteration: 3.331, 3.418, 3.409, 3.440.

`08_kkt_trace.yaml` is the box-in-sweep family: 134733, 136827, 151202, 151352. The kind split is not in the yaml. Recomputed in the paper venv (`observe_projection_events=True`, adaptive, lower 0, upper `c_max`, 30000 iterations, early stop off) it matches those totals, cap rechecks 0:

| T | row | lower | upper | per iteration |
|---:|---:|---:|---:|---|
| 16 | 74802 | 0 | 59931 | 2.493 row, 1.998 bound |
| 32 | 76985 | 0 | 59842 | 2.566 row, 1.995 bound |
| 64 | 91430 | 0 | 59772 | 3.048 row, 1.992 bound |
| 80 | 91594 | 0 | 59758 | 3.053 row, 1.992 bound |

Float64 Python, not FPGA telemetry. Lower never wins here. The kernel still walks lower lanes because `has_lower` is set, so the scan span stays T.

## 5. Residual analysis

WINNER: -1.27%, -0.89%, -0.64%, -0.56%. All four inside ±15%. The shortfall is under 8 cycles/iteration, an unprinted parent cycle around the inlined divide or the argmax call. Nothing was added to close it.

v08: +5.10%, +5.84%, +5.64%, +5.41%. All four inside ±15%. The absolute gap is +62, +110, +207, +255 cycles and grows with T, in step with row-group calls and per-event modules. That is the shape of a small per-call overlap: the model counts every operation-free state and also the full module latency. The Hessian identity says that sum is what HLS reported for the matvec. On the event path the same sum sits a few percent high, so some of the 58 and 64 empty states may overlap the residual-update stall they follow. That overlap is not subtracted. A fitted per-event constant would absorb it and would no longer be a cited term.

The composition that misses the gate is the pre-csynth one in section 3 (-25% at T = 16). Its missing pieces are the intervals this report prints: column iteration 14 and 15, scan and update iteration 9, a 60-state divider, a 12-state digest, and the empty-state chains.

## 6. Second regime: the ED fleet

Same v08 constants. m = 2 straight-line bodies, span = n. Logs `fleet_n{10,40,160,320}.log`: persistent, `route_requested=AUTO`, `route_selected=FULL`, 2000 iterations, 50 calls, one telemetry record. Cycles = `median kernel_ns * 0.2 / 2000`. Telemetry words: magic, status, n_iters, executed, flag, total_events, row, lower, upper, cap_rechecks. Cap rechecks are 0. Kind sums equal `total_events`.

| n | events/iter (row, lower, upper) | measured | model | residual | event share |
|---:|---|---:|---:|---:|---:|
| 10 | 60.912 (18.974, 1.994, 39.944) | 5316.0 | 5415.5 | +1.87% | 96% |
| 40 | 105.645 (15.199, 46.768, 43.678) | 8788.0 | 8973.2 | +2.11% | 90% |
| 160 | 423.982 (27.843, 48.294, 347.845) | 41846.3 | 42541.5 | +1.66% | 87% |
| 320 | 856.504 (39.296, 95.639, 721.570) | 107193.7 | 108571.1 | +1.28% | 84% |

All four inside ±30%, and inside ±3%. Median us/iteration: 26.580, 43.940, 209.232, 535.969, against the plan's rounded 26.6 / 43.9 / 209.2 / 536.0 (lines 145-150).

v08 is the faster kernel here. Published v06 times at these sizes are 77.5, 176.9, 1313.6, 4305.9 us/iteration, so v08 is 2.92x, 4.03x, 6.28x, 8.03x faster. Same constants: the iteration is 84 to 96% events, the fused scan replaces an O(n) serial scan, and the m = 2 residual has no empty-state chain. The matvec gap is about 17700 cycles at n = 320 and is a small slice of that iteration. The plan's "under-predicted by 22-30%" (lines 147-149) is the older depths, which replay to -29% through -18% here.

## 7. Hypotheses

H1, per-event overhead dominates at small m. Holds on ED: events are 84 to 96% of the iteration at m = 2. On alt-investment it is 386 of 701 excess cycles at T = 16 and 250 of 2742 at T = 80. It does not hold the ratio at 2.1 as T grows.

H2, the fused scan fails to amortise at m = 16..80. The scan is the cheaper one. At T = 80 the v08 module is `80/8+9 = 19`, paid 6.05 times (133 cycles with the handshake). WINNER's serial module is `m+1 = 81`, paid 4.44 times (360 cycles). The scan is a credit to v08.

H3, cyclic-8 banking or `MAXN` padding. Trip counts at these T have no remainder, and every hot loop achieved II 1. The double read of each 256-bit word is the `UF` 4 rate (H4). Contribution: 0.

H4, 8 multipliers/cycle against 32. Holds, and it is the term that grows. MAC slots are 4.0x at every T. Cycles are 3.21x, 3.27x, 3.38x, 3.42x after glue (`+28` and `+31` per group, and twice as many groups because `RG` is 2). At T = 80 the gap is 2804 cycles, larger than the measured excess of 2474, with the event path, the gradient, and WINNER's clip netting the difference.

H5, mailbox or telemetry inside the iteration. `write_telemetry` is one call after `outer_loop` (1535-1538). Mailbox dispatch (1549-1613) and `persistent_wait` (1772-1831, `POLL_CYCLES` 256 at line 57) sit outside the iteration. Host minus kernel, 14 to 28 us, is 0.09 to 0.18 cycles/iteration.

H6, CG overhead. Paper arms are `route: full`. ED logs say `route_selected=FULL`. `choose_route` (289-304) picks FULL under AUTO when `route_fits` passes. `FULL_SQUARE_CAP` is 320 (line 38), so T <= 80 stays on FULL. Cap rechecks are 0 in the venv replay and in every ED record, so the second residual-plus-scan (1505-1524) is not paid.

The latency-60 divider is about 180 cycles/iteration at T = 80, inside the per-event term. State 236's edge names are ambiguous (merge vs the `update_x_row_resident` call at states 237-238). The walk takes the call. That is the path whose module meets the board.

## 8. Missing evidence

The paper kind split is float64. A persistent SOLVE of the alt-investment payloads at T = 16, 32, 64, 80, with the same host that wrote `fleet_n10.log`, would record `kernel_ns` and `telemetry_by_call` for those sizes. Board runs were out of scope for this diagnosis. On the ED fleet the fixed-point mix does differ from a float replay (lower facets fire there). Swapping the paper mix all the way to "every event is a row" moves T = 80 by about `1.99 * (205 - 132) = 145` cycles, 3%, still inside the gate.

The gradient interval is `?`. A tripcount on `gradient_fused_step` would print it. The transferred identity matches every sibling that prints both ends. State 236's predicate is the remaining soft spot: grep `p_anonymous_namespace_solve_current.verbose.rpt` for that branch. The fit includes the x-update module either way. Another csynth is not required.
