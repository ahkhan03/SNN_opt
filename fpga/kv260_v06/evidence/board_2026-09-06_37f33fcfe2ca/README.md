# Board qualification of the v06 resident kernel (2026-09-06)

Bitstream `snn_qp_v06_200m_37f33fcfe2ca` (xclbin SHA-256 prefix `37f33fcfe2ca`),
built from kernel source at commit 958c440 with Vitis 2022.1 on the recorded
K26 platform at 200 MHz: WNS +0.075 ns, TNS 0, WHS +0.005 ns; routed
utilization 55,666 LUT (47.5%), 61,450 FF, 46.5 BRAM tiles, 56 URAM, 225 DSP
(`implementation/`). Board: KV260, XRT 2.13, governor `userspace`.

## Parity (framework fixtures, `fixtures/`)

Three binary64 problem bundles (`make_fixture.py`: 6x18, 20x60, 64x64, box
both sides, 50 iterations, projection cap 512) were solved on the board by
`host_kv260_v06` in ONESHOT and PERSISTENT mode (AUTO route = FULL), and
6x18 additionally under FORCE_STREAM and FORCE_CG. In every case the raw
fixed-point state and all 16 telemetry words are byte-identical to the
native emulation records `*.native_oneshot.out` (`tools/compare_records.py`;
verdicts in `results/slot.log`). A three-solve RESIDENT_WARM sequence ran
without error.

## Coherency litmus (`results/litmus.json`)

With the default zocl buffer objects on this image, the kernel's mailbox and
output writes were observed by the host both without and with a
`sync` call, and the host's input marker reached the kernel. `--poll-sync
never` is therefore valid here and is the measured fast path below.

## PMSM resident streams (`results/*.lat.*.json.gz`, 1000 passes x 12 periods)

Committed feasibility gate (two fixed-point LSB) passed on every run; smoke
maximum violation 0.

| run | poll_sync | complete med / p99 / min (us) | doorbell-or-launch to done med / p99 / min (us) | gate |
|---|---|---|---|---|
| H10_N1.lat.oneshot | always | 178.2 / 223.1 / 170.5 | 166.6 / 211.4 / 159.3 | True |
| H10_N1.lat.persistent.always | always | 98.4 / 104.9 / 90.8 | 88.4 / 95.5 / 74.0 | True |
| H10_N1.lat.persistent.never | never | 96.7 / 103.7 / 91.1 | 86.8 / 93.7 / 37.8 | True |
| H3_N1.lat.oneshot | always | 111.8 / 123.2 / 109.2 | 101.8 / 112.4 / 99.6 | True |
| H3_N1.lat.persistent.always | always | 34.3 / 36.0 / 30.5 | 26.2 / 28.0 / 15.8 | True |
| H3_N1.lat.persistent.never | never | 33.0 / 35.3 / 30.9 | 25.1 / 27.0 / 15.3 | True |

The one-shot rows launch the kernel through XRT per solve (the plan's
control arm); the persistent rows publish a mailbox sequence and poll with
`--poll yield`. Reference points measured by the PMSM paper on the v05
bitstream the same night: host-resident 89.6 us kernel / 105.4 us complete
at H3_N1, tuned host (persistent run handle, busy-wait) 61 / 72.3 us.

## Not claimed here

Energy per solve (the PMSM paper's INA260 protocol runs the device-resident
arm separately), the CG and STREAM tiers at sizes where they are selected
automatically, and behaviour above n, m = 64 on the board.
