# Launch-floor probe on the deployed v05 bitstream (2026-09-05)

`launch_floor_probe.cpp` launches the qualified `snn_qp_v05` kernel
(`msrp_v05_200m_d1cf83225e7d`, 200 MHz, XRT 2.13, Kria KV260, governor
`userspace`) with zero-valued geometry so every projection sweep exits at
ordinal 0, and times only `run.start()` to `run.wait()` with
`CLOCK_MONOTONIC_RAW` over a grid of `n = m` and `n_iters`, 300 timed
launches per cell after 10 warmups. `launch_floor_2026-09-05.csv` is the
verbatim output. Compile on the board with

```bash
g++ -std=c++17 -O2 -I/usr/include/xrt -o launch_floor_probe launch_floor_probe.cpp -lxrt_coreutil
```

Reading: the XRT launch floor alone is about 68 us; the per-launch binary64
geometry read costs about 4 us at `n = m = 6`, 22 us at 20 and 135 us at 64;
an event-free outer iteration costs 1.2 us at `n = 1` (the facet scans run
over `MAXN`), 1.8 us at 6, 4.1 us at 20 and 15.4 us at 64.
