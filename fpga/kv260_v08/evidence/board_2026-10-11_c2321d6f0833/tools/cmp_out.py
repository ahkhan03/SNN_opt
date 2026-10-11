#!/usr/bin/env python3
"""Compare raw fixed-point state and the 16 telemetry words of two records.

Accepts board outputs (magic ``MSRPFX1``) and native-emulation records.
Usage: cmp_out.py <reference> <candidate>; exit status 0 on a byte match.
"""
import struct
import sys

TELEMETRY_WORDS = 16


def load(path):
    data = open(path, "rb").read()
    if data[:8] == b"MSRPFX1\0":
        _, n = struct.unpack_from("<2I", data, 8)
        offset = 16
    else:  # native driver record: magic, then version, n, flags
        _, n, _ = struct.unpack_from("<3I", data, 8)
        offset = 20
    raw = struct.unpack_from(f"<{n}q", data, offset)
    telemetry = struct.unpack_from(f"<{TELEMETRY_WORDS}Q", data, offset + 8 * n)
    return n, raw, telemetry


def main():
    ref, cand = load(sys.argv[1]), load(sys.argv[2])
    if ref == cand:
        print(f"MATCH {sys.argv[2]}")
        return 0
    words = [i for i in range(TELEMETRY_WORDS) if ref[2][i] != cand[2][i]]
    print(f"DIFF  {sys.argv[2]}  n {ref[0]} vs {cand[0]}, raw_equal={ref[1] == cand[1]}, "
          f"telemetry words differing={words}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
