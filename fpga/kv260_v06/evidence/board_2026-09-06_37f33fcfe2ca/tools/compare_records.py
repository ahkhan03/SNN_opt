#!/usr/bin/env python3
"""Compare raw state and telemetry between a native v06 record and a board fixed-output record."""

import struct
import sys

TELEMETRY_WORDS = 16


def native(path):
    data = open(path, "rb").read()
    assert data[:8] == b"160VPRSM", data[:8]
    _, n, _ = struct.unpack_from("<3I", data, 8)
    offset = 20
    raw = struct.unpack_from(f"<{n}q", data, offset)
    offset += 8 * n
    telemetry = struct.unpack_from(f"<{TELEMETRY_WORDS}Q", data, offset)
    return n, raw, telemetry


def board(path):
    data = open(path, "rb").read()
    assert data[:8] == b"MSRPFX1\0", data[:8]
    _, n = struct.unpack_from("<2I", data, 8)
    offset = 16
    raw = struct.unpack_from(f"<{n}q", data, offset)
    offset += 8 * n
    telemetry = struct.unpack_from(f"<{TELEMETRY_WORDS}Q", data, offset)
    return n, raw, telemetry


def main():
    n1, raw1, tel1 = native(sys.argv[1])
    n2, raw2, tel2 = board(sys.argv[2])
    ok = n1 == n2 and raw1 == raw2 and tel1 == tel2
    verdict = "MATCH" if ok else "MISMATCH"
    print(verdict, "n", n1, n2, "raw_equal", raw1 == raw2, "telemetry_equal", tel1 == tel2)
    if not ok:
        for i, (a, c) in enumerate(zip(raw1, raw2)):
            if a != c:
                print("  raw", i, a, c)
                break
        for i, (a, c) in enumerate(zip(tel1, tel2)):
            if a != c:
                print("  tel", i, a, c)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
