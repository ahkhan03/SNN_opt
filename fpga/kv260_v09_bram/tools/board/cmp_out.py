#!/usr/bin/env python3
"""Compare raw state, 16 telemetry words, error, and selected route.

Usage: cmp_out.py <reference> <candidate> [--json board.json] [--decision decision.json]

A native record starts with an 8-byte magic other than MSRPFX1\\0, then
three little-endian uint32 values (version, n, mailbox word count), then
n int64 raw words, 16 uint64 telemetry words, then that many uint32
mailbox-out words. Error is mailbox word 1. Route is mailbox word 3.

A board dump starts with MSRPFX1\\0, then two uint32 values (version, n),
then n int64 raw words and 16 uint64 telemetry words. It has no mailbox.
Pass --json with the host JSON so route_selected can be checked. The
native error must be 0. Omitting --json for an MSRPFX1 candidate fails
the compare, so a board run always checks the route.

Two native records compare error and route from the mailbox. Two MSRPFX1
dumps compare raw state and telemetry only.

Route integers: 0 AUTO, 1 FULL, 2 CG, 3 STREAM. The host prints those
names in route_selected.

Exit status 0 prints MATCH. Exit status 1 prints DIFF. Exit status 2 means
a record file was missing, or a --json or --decision document could not be
read or parsed; the reason goes to stderr. A short or truncated record
keeps its own message.
"""
import json
import struct
import sys

TELEMETRY_WORDS = 16
ROUTE_NAMES = {0: "AUTO", 1: "FULL", 2: "CG", 3: "STREAM"}
MSRPFX1 = b"MSRPFX1\0"


def load(path):
    data = open(path, "rb").read()
    if len(data) < 16:
        raise SystemExit(f"short record {path}")
    if data[:8] == MSRPFX1:
        _version, n = struct.unpack_from("<2I", data, 8)
        offset = 16
        mailbox_words = 0
        kind = "board"
    else:
        _version, n, mailbox_words = struct.unpack_from("<3I", data, 8)
        offset = 20
        kind = "native"
    if n > 1_000_000 or len(data) < offset + 8 * n + 8 * TELEMETRY_WORDS:
        raise SystemExit(f"truncated payload in {path} (n={n}, {len(data)} bytes)")
    raw = struct.unpack_from(f"<{n}q", data, offset)
    tel_off = offset + 8 * n
    telemetry = struct.unpack_from(f"<{TELEMETRY_WORDS}Q", data, tel_off)
    error = None
    route = None
    if mailbox_words:
        mb_off = tel_off + 8 * TELEMETRY_WORDS
        if len(data) < mb_off + 4 * mailbox_words:
            raise SystemExit(f"truncated mailbox in {path}")
        if mailbox_words <= 3:
            raise SystemExit(f"mailbox in {path} has no error/route words")
        mailbox = struct.unpack_from(f"<{mailbox_words}I", data, mb_off)
        error = mailbox[1]
        route = mailbox[3]
    return {
        "kind": kind,
        "n": n,
        "raw": raw,
        "telemetry": telemetry,
        "error": error,
        "route": route,
    }


def route_name(value):
    if value is None:
        return None
    return ROUTE_NAMES.get(value, f"ROUTE_{value}")


def _flag_path(args, index, flag):
    if index + 1 >= len(args):
        print(f"cmp_out.py: {flag} needs a path", file=sys.stderr)
        return None, 2
    return args[index + 1], 0


def parse_args(argv):
    if len(argv) < 3:
        print(
            "usage: cmp_out.py <reference> <candidate> "
            "[--json board.json] [--decision decision.json]",
            file=sys.stderr,
        )
        return None
    json_path = None
    decision_path = None
    positional = []
    args = argv[1:]
    index = 0
    while index < len(args):
        if args[index] in ("--json", "--decision"):
            path, status = _flag_path(args, index, args[index])
            if status != 0:
                return None
            if args[index] == "--json":
                json_path = path
            else:
                decision_path = path
            index += 2
            continue
        positional.append(args[index])
        index += 1
    # --json and --decision with no dumps checks the schedule only.
    if not positional and json_path and decision_path:
        return None, None, json_path, decision_path
    if len(positional) != 2:
        print(
            "usage: cmp_out.py <reference> <candidate> "
            "[--json board.json] [--decision decision.json]",
            file=sys.stderr,
        )
        return None
    return positional[0], positional[1], json_path, decision_path


_MISSING = object()


def decision_problems(host, decision):
    """Exact equality of the quantized schedule against the decision record.

    Every field must be present and non-null on both sides: a missing field is
    a failure, never an accidental None == None match.
    """
    if not isinstance(host, dict):
        return ["host JSON is not an object"]
    if not isinstance(decision, dict):
        return ["decision JSON is not an object"]
    board = decision.get("board")
    bundle = decision.get("bundle")
    if not isinstance(board, dict):
        return ["decision has no board object"]
    if not isinstance(bundle, dict):
        return ["decision has no bundle object"]
    chain = board.get("chain_quanta")
    if not isinstance(chain, list) or not chain:
        return ["decision has no chain_quanta list"]
    expected = {
        "k0_start_quanta": bundle.get("k0_quanta", _MISSING),
        "k0_floor_quanta": board.get("floor_quanta", _MISSING),
        "decay_segment": board.get("segment", _MISSING),
        "decay_levels": len(chain),
        "k0_final_effective": board.get("final_effective", _MISSING),
        "decay_chain_quanta": chain,
    }
    problems = []
    for field, want in expected.items():
        if want is _MISSING or want is None:
            problems.append(f"{field} missing in decision")
            continue
        got = host.get(field, _MISSING)
        if got is _MISSING or got is None:
            problems.append(f"{field} missing in host")
            continue
        if isinstance(want, list):
            if (not isinstance(got, list) or len(got) != len(want)
                    or any(int(a) != int(b) for a, b in zip(got, want))):
                problems.append(f"{field} {got} != {want}")
        elif got != want:
            problems.append(f"{field} {got} != {want}")
    return problems


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def compare_full(ref, cand, json_path, decision_path):
    """Compare two loaded records and the optional JSON dumps.

    Returns the problem list. Raises ``OSError``/``ValueError`` when a
    ``--json`` or ``--decision`` file cannot be read; ``main`` turns that
    into exit 2, as the schedule-only mode already did.
    """
    problems = []
    if ref["n"] != cand["n"]:
        problems.append(f"n {ref['n']} vs {cand['n']}")
    if ref["raw"] != cand["raw"]:
        problems.append("raw")
    tel_words = [
        i
        for i in range(TELEMETRY_WORDS)
        if ref["telemetry"][i] != cand["telemetry"][i]
    ]
    if tel_words:
        problems.append(f"telemetry {tel_words}")

    if ref["kind"] == "native" and cand["kind"] == "native":
        if ref["error"] != cand["error"]:
            problems.append(f"error {ref['error']} vs {cand['error']}")
        if ref["route"] != cand["route"]:
            problems.append(
                f"route {route_name(ref['route'])} vs {route_name(cand['route'])}"
            )
    elif cand["kind"] == "board":
        if ref["error"] not in (None, 0):
            problems.append(
                f"error {ref['error']} (board dump has no mailbox; native error must be 0)"
            )
        if json_path is None:
            problems.append("route unchecked (MSRPFX1 candidate requires --json)")
        else:
            doc = read_json(json_path)
            selected = doc.get("route_selected")
            if selected != route_name(ref["route"]):
                problems.append(
                    f"route {route_name(ref['route'])} vs json {selected}"
                )
    elif ref["kind"] == "board" and cand["kind"] == "native":
        if cand["error"] not in (None, 0):
            problems.append(f"error {cand['error']}")
        if json_path is None:
            problems.append("route unchecked (MSRPFX1 side requires --json)")
        else:
            doc = read_json(json_path)
            selected = doc.get("route_selected")
            if selected != route_name(cand["route"]):
                problems.append(
                    f"route json {selected} vs {route_name(cand['route'])}"
                )

    if decision_path is not None:
        if json_path is None:
            problems.append("decision compare needs --json")
        else:
            host = read_json(json_path)
            decision = read_json(decision_path)
            problems.extend(decision_problems(host, decision))

    return problems


def main(argv):
    parsed = parse_args(argv)
    if parsed is None:
        return 2
    reference, candidate, json_path, decision_path = parsed
    if reference is None:
        try:
            host = read_json(json_path)
            decision = read_json(decision_path)
        except (OSError, ValueError) as exc:
            print(f"cmp_out.py: {exc}", file=sys.stderr)
            return 2
        problems = decision_problems(host, decision)
        if not problems:
            print(f"MATCH {json_path}")
            return 0
        print(f"DIFF  {json_path}  " + ", ".join(problems))
        return 1
    try:
        ref = load(reference)
        cand = load(candidate)
        problems = compare_full(ref, cand, json_path, decision_path)
    except (OSError, ValueError) as exc:
        print(f"cmp_out.py: {exc}", file=sys.stderr)
        return 2

    label = candidate
    if not problems:
        if ref["error"] is not None and cand["kind"] != "board":
            extra = f" error={ref['error']} route={route_name(ref['route'])}"
        elif cand["kind"] == "board" and ref["error"] is not None:
            extra = f" error=0 route={route_name(ref['route'])}"
        else:
            extra = ""
        print(f"MATCH {label}{extra}")
        return 0
    print(f"DIFF  {label}  " + ", ".join(problems))
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
