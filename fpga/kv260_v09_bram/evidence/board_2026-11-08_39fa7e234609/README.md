# v09b corrected k0-decay qualification, 2026-11-08

Application and bitstream: `snn_qp_v09b_k0d_200m_39fa7e234609`.
Base commit: `6b1056383c010cba7c747917ff7e0dba3e1cdc30` with uncommitted
source changes. Kernel source sha256:
`39fa7e234609fb2f761f12840c6db2ec375e52d81aac25854da617748b00f0b3`.
`SOURCE_HASH.md` gives the exact file set, order, framing, and reproduction
command.

This qualification supersedes the earlier decay image. After each exact
`k0 = dt(k0 / 2)` halving, the kernel compares raw quanta against a
configure-time `3*floor`. A result in `(floor, 1.5*floor]` goes directly to
the floor and never executes as a level. `decay_levels` and
`k0_final_effective` describe the executed schedule.

## Gates

- Native floor 0: 13/13 legacy records byte-identical, including raw state,
  error, route, and all 16 telemetry words. Official v09b/v06 parity:
  `failures=0`.
- Native and HLS C-sim chain assertion: healthy and stiff both print
  `DECAY_CHAIN ... match=true`, and both output files MATCH the native oracle
  byte-for-byte. The actual quanta come from native-only kernel tracing; the
  expected quanta independently implement the software level rule.
- Routed 200 MHz implementation: WNS +0.040 ns, TNS 0.000 ns, WHS +0.010 ns,
  THS 0.000 ns; all constraints met. Resources: 71809 CLB LUT (61.31%),
  84355 FF (36.01%), 86 BRAM tiles (59.72%), 378 DSP (30.29%), 0 URAM.
- Board floor 0: 26/26 MATCH. Board decay: 4/4 MATCH. Total 30 MATCH lines,
  fleet n=160/320 SKIP, warm3_ok, `identity failures=0`.
- Every one-shot and persistent decay JSON is checked for floor, segment,
  executed level count, and final effective k0. All 31 host JSON files,
  including warm3, have `backend:"xrt"`.
- `kria-clock load`: verdict OK, `kernel_hz_measured` 199999998.
- The superseded app was moved out of the firmware directory and preserved
  with a hash manifest. The separately stashed app was restored with every
  file hash OK.

## Executed decay chains

Healthy, `prob_6x18`, floor 0.375, segment 10:

```
expected=actual=[50641056,25320528,12660264,6291456]
```

Four levels execute. The old extra 0.377305 level is absent; the floor gets
the final 20 iterations. Status 0, 50/50 iterations, 250 events. Native
record sha256:
`09eaecbffe33622aa6a52d9adeaf52f22f33c5373435956a770ef92d94918265`.

Stiff, `square_T16_K2000`, floor 0.002, segment 400:

```
expected=actual=[317556,158778,79389,33554]
```

Four levels execute. The old extra 0.002366 level is absent; the floor gets
the final 800 iterations. Status 0, 2000/2000 iterations, 8912 events.
Native record sha256:
`b6b08dd686d1ddeaca75a46a756688547cf1d0fdef74a4fb196c35208d454cfe`.

The final effective values are 0.375 and 0.001999974250793457 in both launch
modes. Telemetry remains 16 words.

## Artifact hashes

- xo: `3f7ead02754d6d5bfb1701b1924399eeb8a9c03d2fc6de6617a86c5d4129c661`
- xclbin: `fb4f0519a47e6fd37131cb9b7221d6ec5d1974beab1e9d9d616b9031a8e63354`
- bit.bin: `42055c43498c429398897ee88f69e7bd964045d1349820382e2e9ba0ab99326b`
- dtbo: `ba7bee41178c6a8b763d2266a90248fa9d23fd821731d34e8f0e59d4cc94336c`
- native oracle: `183c2322ee0c1a26f8d4a31e9be4d930b1b99ffd787225f46b7121f1e28422fc`
- board XRT host: `0dc22d5f557f3f63e79b070b518a8572dc1d9a230af26fa9085204d864bfe825`

## Commands and records

Build used Vitis 2022.1, 200 MHz, 2.5 ns HLS uncertainty, and
`Performance_ExplorePostRoutePhysOpt`:

```
bash fpga/kv260_v09_bram/build/run_parity.sh
vitis_hls -f fpga/kv260_v09_bram/build/run_hls.tcl
bash fpga/kv260_v09_bram/build/build_xclbin.sh /tools/build/<work>/fix-implementation
bash fpga/kv260_v09_bram/build/build_host.sh /home/ubuntu/<work>/bin
kria-clock load snn_qp_v09b_k0d_200m_39fa7e234609 --json-out clock_measured.json
XCLBIN=/lib/firmware/xilinx/snn_qp_v09b_k0d_200m_39fa7e234609/snn_qp_v09b.xclbin \
HOST=/home/ubuntu/<work>/bin/host_kv260_v09b \
FLEET_DIR=/home/ubuntu/<work>/fleet CLOCK_JSON=clock_measured.json \
FIX_DIR=fpga/kv260_v06/evidence/board_2026-09-06_37f33fcfe2ca/fixtures \
SQUARE_DIR=fpga/kv260_v09_bram/evidence/fixtures \
RECORDS=/home/ubuntu/<work>/fix_records RESULTS=results/identity \
bash fpga/kv260_v09_bram/tools/board/run.sh
```

`results/identity.log` is the 30-cell identity log. `results/identity/`
contains each dump, JSON, and host log. `results/clock_measured.json`,
`results/build_host.log`, `results/app_artifact_sha256.txt`, and
`results/stash_restore.log` preserve supporting records. Routed reports are
in `implementation/`; C-sim logs and the synthesis report are in `csynth/`.
The external build archive contains all artifacts and raw records.
