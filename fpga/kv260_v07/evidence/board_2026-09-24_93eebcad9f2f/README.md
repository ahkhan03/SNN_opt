# Board qualification of the v07 resident conic kernel (2026-09-24)

Bitstream `snn_qp_v07_200m_93eebcad9f2f` (xclbin SHA-256 prefix `93eebcad9f2f`),
kernel source at commit `6f16cc4`, Vitis 2022.1 on the `kr260_min` platform,
linked with `--clock.freqHz 200000000` and
`run.impl_1.STRATEGY=Performance_ExtraTimingOpt` (the default
`Performance_ExplorePostRoutePhysOpt` link missed by 6 ps on one v06-inherited
configure path, `s_axi int_m -> load_geometry_matrix` bound).
Routed timing: WNS +0.083 ns, TNS 0, WHS +0.010 ns, 0 failing endpoints of
206,335. Routed utilization: 66,922 LUT (57.1%), 50 BRAM tiles, 56 URAM,
318 DSP (`implementation/`). Board: KV260, XRT 2.13.

## Clock

The link inserts a clock wizard (clk_in1 250 MHz -> clk_out1 200 MHz,
ratio 0.8), so the kernel clock is 0.8 x `pl0_ref`. The v07 dtbo sets
`pl0_ref` to 250 MHz; measured on the board with v07 loaded:
`/sys/kernel/debug/clk/pl0_ref/clk_rate = 249999998`, i.e. the kernel runs at
200 MHz. (With the v06 app loaded the same register read 199999998, so the
v06 kernel, whose dtbo sets 200 MHz, runs at 160 MHz.)

## Parity

`host_kv260_v07_xrt` wrote board records (`results/*.bin|*.out`,
`MSRPFX1` format); `tools/compare_v07.py` compares raw fixed-point state and
all 16 telemetry words against the native emulation records (`...VPRSM`
format). The byte `cmp` lines printed as "DIFFERS" in `results/board_run.log`
compare the two different file formats and are not a verdict.

| fixture | cones | mode | raw state | 16 telemetry words |
|---|---|---|---|---|
| n=8 Euclidean ball | 1 ball | one-shot | equal | equal |
| n=8 Euclidean ball | 1 ball | persistent | equal | equal |
| friction q=1, mu=0.4, seed 4903 (1024 events) | 1 scaled SOC | one-shot | equal | equal |
| mixed row + upper bound + ball + scaled SOC | 2 | one-shot | equal | equal |
| cones-off v06 parity bundle | 0 | one-shot | equal | equal |
| v06 fixture 6x18 (vs v06 native record) | 0 | one-shot, persistent | equal | equal |
| v06 fixture 20x60 (vs v06 native record) | 0 | one-shot | equal | equal |
| v06 fixture 64x64 (vs v06 native record) | 0 | one-shot | equal | equal |

The v06 fixtures and their native records are
`fpga/kv260_v06/evidence/board_2026-09-06_37f33fcfe2ca/fixtures/`. With cones
off, v07 on the board is bit-identical to the v06 native emulation. dmesg
showed no CU hang or outstanding-command messages; the v06 app was restored to
slot 0 afterwards (`tools/run_board.sh`).

## Not claimed here

Latency and energy comparisons (the JSON timing fields are single
`poll_sync=always` verification runs, not the tuned path), the CG and STREAM
tiers with cones, and cone tables beyond the fixtures above.
