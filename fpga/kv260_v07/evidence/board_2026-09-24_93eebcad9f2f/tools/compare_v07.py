#!/usr/bin/env python3
"""Compare raw state + 16 telemetry words: native emulation record (v06/v07 '...VPRSM') vs board MSRPFX1 record."""

import struct
import sys

T = 16


def native(p):
    d = open(p, "rb").read()
    assert d[4:8] == b"VPRSM"[1:] or d[3:8] in (b"VPRSM",) or b"VPRSM" in d[:8], d[:8]
    _, n, _ = struct.unpack_from("<3I", d, 8)
    o = 20
    raw = struct.unpack_from(f"<{n}q", d, o)
    o += 8 * n
    return n, raw, struct.unpack_from(f"<{T}Q", d, o)


def board(p):
    d = open(p, "rb").read()
    assert d[:8] == b"MSRPFX1\0", d[:8]
    _, n = struct.unpack_from("<2I", d, 8)
    o = 16
    raw = struct.unpack_from(f"<{n}q", d, o)
    o += 8 * n
    return n, raw, struct.unpack_from(f"<{T}Q", d, o)


for nat, brd in zip(sys.argv[1::2], sys.argv[2::2]):
    n1, r1, t1 = native(nat)
    n2, r2, t2 = board(brd)
    diff = [i for i, (a, b) in enumerate(zip(t1, t2)) if a != b]
    print(
        f"{'MATCH' if (n1 == n2 and r1 == r2 and not diff) else 'MISMATCH'}  {brd.split('/')[-1]:36s} n={n1}/{n2} raw_equal={r1 == r2} telemetry_diff_words={diff}"
    )
