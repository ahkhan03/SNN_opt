#!/usr/bin/env python3
"""Gate-4 route-boundary parity harness.

The harness deliberately keeps the fixtures tiny in horizon (one fixed
iteration) while making the configured geometry dimensions sit on both sides
of the v06 capacity ladder.  Native v06 and the MAXN=1024 v0.5 reference are
run on every generated bundle; a mock generic host run independently checks
the host-side AUTO selection.  The companion ``remote_gate4.sh`` consumes the
same fixture directory on the KV260.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path


CAP = 440_000
BRAM_CAP = 129_024
FULL = 1
CG = 2
STREAM = 3


FIXTURES = (
    # Square caps are explicit conservative guards in the v06 plan.
    ("full_320", 320, 320, FULL),
    ("full_321", 321, 321, CG),
    ("cg_372", 372, 372, CG),
    ("cg_373", 373, 373, STREAM),
    # Multiples of eight make the physical row padding neutral, so these
    # asymmetric pairs straddle the plan's stated integer predicates exactly.
    ("full_asym_fit", 32, 624, FULL),
    ("full_asym_next", 32, 632, CG),
    ("cg_asym_fit", 40, 624, CG),
    ("cg_asym_next", 48, 624, STREAM),
)


def pad8(value: int) -> int:
    return (value + 7) // 8 * 8


def packed_full(n: int, m: int) -> int:
    return n * pad8(n) + m * pad8(n) + n * pad8(m) + m * pad8(m)


def packed_cg(n: int, m: int) -> int:
    return m * pad8(n) + n * pad8(m) + m * pad8(m)


def route_prediction(n: int, m: int) -> int:
    if n < 1 or m < 1 or n > 1024 or m > 1024:
        raise ValueError(f"dimensions out of v06 range: {n}x{m}")
    if 16 * (n + m) + 4096 > BRAM_CAP:
        raise ValueError(f"BRAM shape does not fit: {n}x{m}")
    if packed_full(n, m) <= CAP and (n != m or n <= 320):
        return FULL
    if packed_cg(n, m) <= CAP and (n != m or n <= 372):
        return CG
    return STREAM


def route_name(route: int) -> str:
    return {FULL: "FULL", CG: "CG", STREAM: "STREAM"}.get(
        route, f"UNKNOWN({route})"
    )


def write_problem(path: Path, n: int, m: int) -> None:
    """Write a deterministic, event-free binary64 problem bundle.

    The values are intentionally simple and deterministic at the v06
    fixed-point cast boundary.  One outer iteration and a projection cap of
    one keep the 373x373 native comparison inexpensive while still traversing
    configure, image construction, route selection, and solve.
    """

    with path.open("wb") as stream:
        stream.write(b"MSRPDL1\0")
        stream.write(struct.pack("<7I", 1, n, m, 1, 1, 0, 0))
        stream.write(struct.pack("<4d", 0.05, 1.0e-6, -1.0, 1.0))

        # A: diagonal 1/4, all other entries zero.
        zero_row = struct.pack("<" + "d" * n, *([0.0] * n))
        diag_row = [0.0] * n
        for i in range(n):
            diag_row[i] = 0.25
            stream.write(struct.pack("<" + "d" * n, *diag_row))
            diag_row[i] = 0.0

        stream.write(struct.pack("<" + "d" * n, *([0.1] * n)))  # b

        # C and G are zero.  d is zero, cns/row_scale are one.  Write in
        # chunks so the script does not retain multi-megabyte Python lists.
        for _ in range(m):
            stream.write(zero_row)
        stream.write(struct.pack("<" + "d" * m, *([0.0] * m)))  # d
        stream.write(struct.pack("<" + "d" * m, *([1.0] * m)))  # cns
        stream.write(struct.pack("<" + "d" * m, *([1.0] * m)))  # row_scale
        for _ in range(m):
            stream.write(struct.pack("<" + "d" * m, *([0.0] * m)))
        stream.write(struct.pack("<" + "d" * n, *([0.0] * n)))  # x0


def run_checked(command: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True)
    log_path.write_text(
        "$ " + " ".join(command) + "\n" + result.stdout + result.stderr,
        encoding="utf-8",
    )
    return result


def read_v05(path: Path, n: int) -> tuple[list[int], list[int]]:
    data = path.read_bytes()
    if data[:8] != b"MSRPFX1\0":
        raise ValueError(f"unexpected v05 output magic for {path}")
    header = 8 + 2 * 4
    version, header_n = struct.unpack("<2I", data[8:header])
    if version != 1 or header_n != n:
        raise ValueError(f"unexpected v05 output header for {path}")
    raw_end = header + n * 8
    raw = list(struct.unpack("<" + "q" * n, data[header:raw_end]))
    meta = list(struct.unpack("<16Q", data[raw_end : raw_end + 16 * 8]))
    if len(data) != raw_end + 16 * 8:
        raise ValueError(f"unexpected v05 output length for {path}: {len(data)}")
    return raw, meta


def read_v06(path: Path, n: int) -> tuple[list[int], list[int], list[int]]:
    data = path.read_bytes()
    # native_fixed_v06 writes the uint64 magic in host little-endian order.
    if data[:8] != struct.pack("<Q", 0x4D53525056303631):
        raise ValueError(f"unexpected v06 output magic for {path}")
    header = 8 + 3 * 4
    version, header_n, mailbox_words = struct.unpack("<3I", data[8:header])
    if version != 1 or header_n != n or mailbox_words != 64:
        raise ValueError(f"unexpected v06 output header for {path}")
    raw_end = header + n * 8
    raw = list(struct.unpack("<" + "q" * n, data[header:raw_end]))
    meta_end = raw_end + 16 * 8
    meta = list(struct.unpack("<16Q", data[raw_end:meta_end]))
    mailbox = list(struct.unpack("<64I", data[meta_end : meta_end + 64 * 4]))
    if len(data) != meta_end + 64 * 4:
        raise ValueError(f"unexpected v06 output length for {path}: {len(data)}")
    return raw, meta, mailbox


def parse_host_route(json_path: Path) -> int:
    # The generic host emits one JSON object on stdout and --json-out.  Read
    # the file so incidental diagnostic lines never affect the assertion.
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    value = payload.get("route_selected")
    if isinstance(value, str):
        names = {"FULL": FULL, "CG": CG, "STREAM": STREAM}
        if value not in names:
            raise ValueError(f"unknown host route {value!r}")
        return names[value]
    return int(value)


def maybe_build(args: argparse.Namespace, work: Path) -> None:
    if args.skip_build or (args.v06 and args.v05 and args.host):
        return
    build = Path(args.build_script)
    if not build.exists():
        raise SystemExit(
            "gate-4 binaries are missing; pass --v06/--v05/--host or use "
            f"--build-script {build}"
        )
    result = subprocess.run([str(build), str(work / "bin")], text=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    args.v06 = args.v06 or str(work / "bin" / "native_fixed_v06")
    args.v05 = args.v05 or str(work / "bin" / "native_fixed_v05")
    args.host = args.host or str(work / "bin" / "host_kv260_v06")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", default="build/work/gate4", help="artifact directory")
    parser.add_argument("--v06", help="native_fixed_v06 executable")
    parser.add_argument("--v05", help="MAXN=1024 native_fixed_v05 executable")
    parser.add_argument("--host", help="mock host executable for AUTO check")
    parser.add_argument(
        "--build-script",
        default=str(Path(__file__).resolve().parent.parent / "build" / "build_gate4.sh"),
    )
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args(argv)
    work = Path(args.work).resolve()
    fixture_dir = work / "fixtures"
    output_dir = work / "outputs"
    log_dir = work / "logs"
    for directory in (fixture_dir, output_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        maybe_build(args, work)
    if not args.v06 or not args.v05 or not args.host:
        raise SystemExit(
            "--v06, --v05, and --host are required (or allow --build-script)"
        )

    summary: list[dict[str, object]] = []
    all_pass = True
    for name, n, m, documented in FIXTURES:
        fixture = fixture_dir / f"{name}.bin"
        write_problem(fixture, n, m)
        predicted = route_prediction(n, m)
        # Keep the plan's unpadded expressions visible in the artifact.  The
        # implementation packs rows to eight lanes, so route_prediction uses
        # the corresponding padded realization used by the host and kernel.
        logical_full = n * n + 2 * n * m + m * m
        logical_cg = 2 * n * m + m * m

        v06_out = output_dir / f"{name}.v06.out"
        v05_out = output_dir / f"{name}.v05.out"
        v06_run = run_checked(
            [args.v06, str(fixture), str(v06_out), "--route", "auto"],
            log_dir / f"{name}.v06.log",
        )
        v05_run = run_checked(
            [args.v05, str(fixture), str(v05_out)],
            log_dir / f"{name}.v05.log",
        )

        host_route = None
        host_ok = True
        if args.host:
            host_json = output_dir / f"{name}.host.json"
            host_out = output_dir / f"{name}.host.out"
            host_run = run_checked(
                [
                    args.host,
                    "mock.xclbin",
                    str(fixture),
                    str(host_out),
                    "--mock",
                    "--one-shot",
                    "--route",
                    "auto",
                    "--reps",
                    "1",
                    "--json-out",
                    str(host_json),
                ],
                log_dir / f"{name}.host.log",
            )
            host_ok = host_run.returncode == 0 and host_json.exists()
            if host_ok:
                try:
                    host_route = parse_host_route(host_json)
                    host_ok = host_route == predicted
                except (TypeError, ValueError, KeyError) as error:
                    host_ok = False
                    (log_dir / f"{name}.host-parse-error.txt").write_text(
                        str(error), encoding="utf-8"
                    )

        native_ok = False
        raw_equal = meta_equal = False
        selected = None
        if v06_run.returncode == 0 and v05_run.returncode == 0 and v06_out.exists() and v05_out.exists():
            try:
                raw06, meta06, mailbox = read_v06(v06_out, n)
                raw05, meta05 = read_v05(v05_out, n)
                raw_equal = raw06 == raw05
                meta_equal = meta06 == meta05
                selected = mailbox[3]
                native_ok = raw_equal and meta_equal and selected == predicted
            except (OSError, struct.error, ValueError) as error:
                (log_dir / f"{name}.parse-error.txt").write_text(str(error))
        ok = native_ok and host_ok and predicted == documented
        all_pass = all_pass and ok
        record = {
            "name": name,
            "n": n,
            "m": m,
            "documented_route": route_name(documented),
            "predicted_route": route_name(predicted),
            "selected_route": None if selected is None else route_name(selected),
            "host_route": None if host_route is None else route_name(host_route),
            "logical_full_words": logical_full,
            "logical_cg_words": logical_cg,
            "packed_full_words": packed_full(n, m),
            "packed_cg_words": packed_cg(n, m),
            "raw_equal": raw_equal,
            "telemetry_equal": meta_equal,
            "native_pass": native_ok,
            "host_route_pass": host_ok,
            "pass": ok,
        }
        summary.append(record)
        print(
            f"{'PASS' if ok else 'FAIL'} {name} n={n} m={m} "
            f"predicted={route_name(predicted)} selected="
            f"{route_name(selected) if selected else 'NA'} "
            f"raw={'yes' if raw_equal else 'no'} telemetry={'yes' if meta_equal else 'no'} "
            f"host={'yes' if host_ok else 'no'}"
        )

    report = {"schema": "snn-qp-v06-gate4-route-boundary-v1", "pass": all_pass, "fixtures": summary}
    report_path = work / "route_boundary_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"ROUTE BOUNDARY SUMMARY fixtures={len(summary)} report={report_path} "
          f"{'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
