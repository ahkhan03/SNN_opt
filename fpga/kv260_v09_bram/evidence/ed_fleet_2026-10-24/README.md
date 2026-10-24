# ED fleet latency, v09b against v08 (2026-10-24)

One board session, both images interleaved, on the economic-dispatch
bundles `fleet_n10_it2000.bin` and `fleet_n40_it2000.bin` (m=2, 2000
iterations). The comparison uses median `kernel_ns`. `complete_ns` is
recorded beside it and is not the ratio.

## Protocol

KV260, XRT 2.13. Apps `snn_qp_v09b_200m_3a352d326c0a` and
`snn_qp_v08_k200_c2321d6f0833`, one unload and one load per cell.
`kria-clock check` on the board after each load. Host flags, both
binaries: `--persistent --warmups 5 --reps 50 --route auto --start
host_x0 --poll-sync never`. The v09b host is the qualified board
binary. The v08 host was built on the board from the v08 sources
against `libxrt_coreutil` (sha256
`368ba77992ed5c914ae0271ddb1d91071667de15c6d6821986c3ae7d951fc95d`).

Order: v09b, v08, v09b, v08 at n=10, then the same four cells at n=40.
Before a latency number was kept, each JSON had to show backend `xrt`,
persistent mode, poll-sync never, start `HOST_X0`, `route_selected`
FULL, 50 reps, telemetry status 0, 2000 iterations executed, and cap
rechecks 0. All 50 reps in a cell carry the same 16 telemetry words,
and the two kernels carry the same event counts on a given bundle.

The incumbent app at session start was
`snn_qp_v08_k200_c2321d6f0833`. It was loaded back into the active slot
after the eighth cell. A one-shot control solve of `prob_6x18` on that
restored image exited 0 with route FULL and telemetry status 0.

## Clock

Every cell, the pre-session check, and the restore check: verdict OK,
`pl0_ref` 249999998, **kernel_hz_measured 199999998**. Quote the kernel
clock. `pl0_ref` is the clk_wiz input (250 MHz nominal, ratio 0.8).

## Cells

p90 is linear interpolation at `0.9 * 49` on the sorted 50 samples.
events/iter is total events / 2000. us/iter is median `kernel_ns` /
2000 / 1000.

| cell | kernel | n | median kernel_ns | p90 kernel_ns | median complete_ns | us/iter | events/iter | reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c1 | v09b | 10 | 46548460.5 | 46549477.1 | 46559075.5 | 23.274230 | 60.9115 | 50 |
| c2 | v08 | 10 | 53159927.0 | 53161521.0 | 53169967.0 | 26.579964 | 60.9115 | 50 |
| c3 | v09b | 10 | 46548605.5 | 46549510.0 | 46558310.0 | 23.274303 | 60.9115 | 50 |
| c4 | v08 | 10 | 53159846.5 | 53160951.9 | 53170026.5 | 26.579923 | 60.9115 | 50 |
| c5 | v09b | 40 | 74535865.5 | 74537151.0 | 74548261.0 | 37.267933 | 105.6445 | 50 |
| c6 | v08 | 40 | 87879459.0 | 87880383.0 | 87891954.0 | 43.939729 | 105.6445 | 50 |
| c7 | v09b | 40 | 74535835.5 | 74536867.9 | 74548066.0 | 37.267918 | 105.6445 | 50 |
| c8 | v08 | 40 | 87879589.0 | 87880390.0 | 87891724.0 | 43.939795 | 105.6445 | 50 |

Raw files: `c1_v09b_n10.json` through `c8_v08_n40.json`. The `xclbin`
field in each file is the on-board firmware path.

## Event counts

n=10, every cell and every rep: total 121823, row 37947, lower 3988,
upper 79888, cap rechecks 0. n=40: total 211289, row 30397, lower
93536, upper 87356, cap rechecks 0.

## Ratios

`r = (median kernel_ns of v09b - median kernel_ns of v08) / v08`. The
size ratio uses the mean of the two cell medians. The pair ratios
follow the interleaved order.

| size | pair ratios | size ratio | v09b us/iter | v08 us/iter |
| --- | --- | --- | --- | --- |
| n=10 | -0.124369, -0.124365 | -0.124367 | 23.2743 | 26.5799 |
| n=40 | -0.151840, -0.151841 | -0.151841 | 37.2679 | 43.9398 |

v09b is 12.4 percent faster at n=10 and 15.2 percent faster at n=40.
A quiet cell's relative SD of `kernel_ns` is about 0.002 percent or
smaller. The two pairs at each size agree in the fourth significant
digit of r.

The event-heavy band between n=41 and n=128 is unmeasured; the measured
advantage grows with n (12.4 percent at n=10, 15.2 percent at n=40) and
the square points at n=m=16..80 bracket it at 1.37-2.00x, so the
extrapolation runs in v09b's favour.

## Control gate

The 2026-10-11 v08 anchors are 26.580 us/iter at n=10 and 43.940
us/iter at n=40.

| v08 cell | us/iter | deviation |
| --- | --- | --- |
| c2 | 26.579964 | -0.000135% |
| c4 | 26.579923 | -0.000290% |
| c6 | 43.939729 | -0.000617% |
| c8 | 43.939795 | -0.000467% |

The largest absolute deviation is 0.000617 percent, inside 0.5 percent.

## Anomaly that did not drop a cell

Cell c2, rep 50, is 53366773 ns, 0.389 percent above that cell's
median. Its telemetry matches the other 49 reps and the other n=10
cells. The median, which is what r uses, does not move. Cell c4 has no
comparable rep.
