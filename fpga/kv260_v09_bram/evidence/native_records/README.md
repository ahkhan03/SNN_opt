# Native oneshot records

These files are the byte-identity targets for the board host. Each one is
the default oneshot dump from `native_fixed_v09b`: route AUTO, no `--persistent`.
On every file below the oracle printed error 0 and route 1 (FULL).

The dump is a little-endian uint64 magic, three uint32 header words
(version, n, mailbox word count 64), n int64 raw state words, 16 uint64
telemetry words, and 64 uint32 mailbox-out words. Error is mailbox word 1.
Route is mailbox word 3.

Records derived from paper-payload bundles (`fleet_*`, `square_*`) are
kept beside the build archive on the build host and are deliberately not
committed to the public tree; regenerate them with the package oracle from
the bundles (see `../fixtures/`). The committed `prob_*` records pair with
the v06 synthetic fixtures, which are in the repository.

## Generating binary

sha256 `e6dd5bcfa67509a2bbdff2ec913eb386c4b773c6965c1c897d9bf46fd97b1628`

Built with g++ 15.2.0 by:

```
ROOT=<worktree root>
HLS_INCLUDE=<Vitis HLS 2022.1 include directory>
HLS_INCLUDE="$HLS_INCLUDE" bash "$ROOT/fpga/kv260_v09_bram/build/build_native.sh" \
    "$ROOT/fpga/kv260_v09_bram/build/work/native"
```

`build/work/` is gitignored. The binary sha256 above is a build of the
kernel source in which a pre-configure acknowledgement reports AUTO: the
route word starts at AUTO and a successful in-cap configure selects FULL.
Configure-first oneshot bytes are the files below. Re-running the build and
the invocations on g++ 15.2.0 reproduces this binary sha256 and these record
sha256 values. A different compiler may change the binary sha256 and still
write the same records.

## Commands

```
ROOT=<worktree root>
FLEET_DIR=<directory containing the fleet bundles named below>
BIN="$ROOT/fpga/kv260_v09_bram/build/work/native/native_fixed_v09b"
OUT="$ROOT/fpga/kv260_v09_bram/evidence/native_records"
"${BIN}" "${ROOT}/fpga/kv260_v06/evidence/board_2026-09-06_37f33fcfe2ca/fixtures/prob_6x18.bin" "${OUT}/prob_6x18.native_oneshot.out"
"${BIN}" "${ROOT}/fpga/kv260_v06/evidence/board_2026-09-06_37f33fcfe2ca/fixtures/prob_20x60.bin" "${OUT}/prob_20x60.native_oneshot.out"
"${BIN}" "${ROOT}/fpga/kv260_v06/evidence/board_2026-09-06_37f33fcfe2ca/fixtures/prob_64x64.bin" "${OUT}/prob_64x64.native_oneshot.out"
"${BIN}" "${FLEET_DIR}/fleet_n10_it2000.bin" "${OUT}/fleet_n10_it2000.native_oneshot.out"
"${BIN}" "${FLEET_DIR}/fleet_n40_it2000.bin" "${OUT}/fleet_n40_it2000.native_oneshot.out"
"${BIN}" "${ROOT}/fpga/kv260_v09_bram/evidence/fixtures/square_T16_K2000.bin" "${OUT}/square_T16_K2000.native_oneshot.out"
"${BIN}" "${ROOT}/fpga/kv260_v09_bram/evidence/fixtures/square_T32_K2000.bin" "${OUT}/square_T32_K2000.native_oneshot.out"
"${BIN}" "${ROOT}/fpga/kv260_v09_bram/evidence/fixtures/square_T64_K2000.bin" "${OUT}/square_T64_K2000.native_oneshot.out"
"${BIN}" "${ROOT}/fpga/kv260_v09_bram/evidence/fixtures/square_T80_K2000.bin" "${OUT}/square_T80_K2000.native_oneshot.out"
"${BIN}" "${ROOT}/fpga/kv260_v09_bram/evidence/fixtures/square_T16_K30000.bin" "${OUT}/square_T16_K30000.native_oneshot.out"
"${BIN}" "${ROOT}/fpga/kv260_v09_bram/evidence/fixtures/square_T32_K30000.bin" "${OUT}/square_T32_K30000.native_oneshot.out"
"${BIN}" "${ROOT}/fpga/kv260_v09_bram/evidence/fixtures/square_T64_K30000.bin" "${OUT}/square_T64_K30000.native_oneshot.out"
"${BIN}" "${ROOT}/fpga/kv260_v09_bram/evidence/fixtures/square_T80_K30000.bin" "${OUT}/square_T80_K30000.native_oneshot.out"
```

The v09b loader calls `load_problem` with a cap of 128. The two commands
below exit 2 and write no record. stderr is `problem dimensions or horizons are out of range`.

```
"${BIN}" "${FLEET_DIR}/fleet_n160_it2000.bin" "${OUT}/fleet_n160_it2000.native_oneshot.out"
"${BIN}" "${FLEET_DIR}/fleet_n320_it2000.bin" "${OUT}/fleet_n320_it2000.native_oneshot.out"
```

`FLEET_DIR` for the successful rows needs `fleet_n10_it2000.bin` and
`fleet_n40_it2000.bin`. The refused names are how the cap was checked.


## Record sha256

- `prob_6x18.native_oneshot.out`  `d7cad695c093ac36305ed7d36e33b7c30637659de0662e47f4d8bc584446a234`
- `prob_20x60.native_oneshot.out`  `5db9f0f8c9f4c62b4e9b37c8633fc320808069a2436f882c9033cedf2c76baa8`
- `prob_64x64.native_oneshot.out`  `57b50c212d0152594b0ab095dbd77b3cdd01e95ea132be884de96ce24549e74c`
- `fleet_n10_it2000.native_oneshot.out`  `8dd6183fcf2a9f5ebdbb69ffd982bab286580e925faa8159c3ba078b7fa039e9`
- `fleet_n40_it2000.native_oneshot.out`  `e1314f0b72c3d5a49e8127d56c517370a3711b48f6e4d8c9373fccf13ab0c037`
- `square_T16_K2000.native_oneshot.out`  `c0d7de2db31e798315d5a4f984a8d507887c6121740e85f25e398f5ec9b4e02b`
- `square_T32_K2000.native_oneshot.out`  `9f76e1dd58f9c15307eb0fc750a1470afdc8665c688202963fef8cd7b05ff458`
- `square_T64_K2000.native_oneshot.out`  `6682c989b944df384cb4be297d350c763348d2b7b91669b2a8e037ad341322c1`
- `square_T80_K2000.native_oneshot.out`  `fbb880ea5ec9628211468b12072b277f3aa44f369241c62bfdbd3b1a97e9848c`
- `square_T16_K30000.native_oneshot.out`  `cbc8936a022816aa867bedf3b4360ffbcc961dcd7a7fba08ca9881d777a0db8a`
- `square_T32_K30000.native_oneshot.out`  `c73bb4eb76c7f21f9097ab04a117e7c3d142a3b4fe5c211ed4e124bb3a3eaba3`
- `square_T64_K30000.native_oneshot.out`  `751745fd6503c333092c72d45d1c152cc975f76d174cc4a10ade427d15c572af`
- `square_T80_K30000.native_oneshot.out`  `6ae8ecd9af5820ea0fbc924e50ede12a31e07c353df9d94ccff54358a2cb84fc`
