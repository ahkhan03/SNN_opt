# KV260 v0.7 native-reset resident kernel

This package is a v06 resident-ABI derivative. `src/snn_qp_v07_kernel.cpp` and `build/run_parity.sh` are the frozen cones-off reference and reproduce the v06 emulation on the three qualification fixtures. `src/snn_qp_v07_kernel.cpp` is the single v07 resident top with native resets: rows and bounds compete with configure-time contiguous Euclidean-ball and scaled-SOC blocks (`mu=1` is the ordinary SOC), capped at 64 descriptors.

The descriptor ABI is `src/v07_cone_table.hpp`. Host validation runs before launch and rejects table overflow, unsupported kinds, non-contiguous/out-of-range or overlapping blocks, non-finite parameters, and values outside `ap_fixed<32,8>`. The fixed reset unit uses `ap_ufixed<49,25>`, a 32-entry midpoint seed table, two rounded Newton updates, and the `2^-48` norm guard. Lifts, affine subspaces, Dykstra, and callbacks remain software-side.

## Local checks

```text
bash fpga/kv260_v07/build/build_native.sh
bash fpga/kv260_v07/build/build_host.sh
bash fpga/kv260_v07/build/run_parity.sh
bash fpga/kv260_v07/build/run_cone_parity.sh
.venv/bin/python -m pytest -q tests/test_fpga_v07_assets.py
```

The v06 parity battery is expected to report `cells=247 failures=0 PASS`. The cone parity wrapper drives the same resident top for an n=8 ball, a mixed row/bound/ball/SOC case, and the q=1/q=4 friction seeds (4903, 4914, 4925). It records raw integer equality, event agreement and state gaps against the binary64 reference, plus Clarabel objective values. It does not claim HLS or board timing; any raw mismatch remains visible in `anchors.json`.

## VM and board handoff

On the Vitis VM, run `build/run_hls.tcl` for C-simulation/csynth and `build/build_xclbin.sh` for the 200 MHz link. If 200 MHz does not close, rerun with the v13 documented 125 MHz link fallback and record the actual clock and routed WNS in `env.yaml`. The resident cones-off bitstream uses `build/build_xclbin.sh`; board parity remains `build/run_parity.sh`. The XRT host source is `src/host_kv260_v07_xrt.cpp`; the descriptor checker is `src/host_kv260_v07.cpp`.

`build/build_xclbin.sh` runs `kria-clock pack` after the link, so the dtbo is generated from the xclbin and `assigned-clock-rates` is the platform clock, not the kernel request. Every board run loads with `kria-clock load <app> --json-out <results>/clock_measured.json` and aborts if that check fails.
