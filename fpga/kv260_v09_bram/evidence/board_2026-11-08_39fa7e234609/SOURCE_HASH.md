# Qualified kernel source hash

`qualified_kernel_source_sha256` is the SHA-256 of an ordered framing of four
files, not a Git commit hash and not the hash of any one file. The file set and
order are:

1. `dt.h`
2. `v09_abi.hpp`
3. `v09b_decay.hpp`
4. `snn_qp_v09b_kernel.cpp`

For each file, in that order, the command hashes the UTF-8 basename, a zero
byte, the exact file bytes, and another zero byte. Run from the worktree root:

```
python3 - "$PWD" <<'PY'
from pathlib import Path
import hashlib
import sys

root = Path(sys.argv[1]) / "fpga/kv260_v09_bram/src"
h = hashlib.sha256()
for name in ["dt.h", "v09_abi.hpp", "v09b_decay.hpp",
             "snn_qp_v09b_kernel.cpp"]:
    path = root / name
    h.update(name.encode())
    h.update(b"\0")
    h.update(path.read_bytes())
    h.update(b"\0")
print(h.hexdigest())
PY
```

Expected output for this qualification:

```
39fa7e234609fb2f761f12840c6db2ec375e52d81aac25854da617748b00f0b3
```

The base commit is `6b1056383c010cba7c747917ff7e0dba3e1cdc30`; the qualified
source is that base plus the uncommitted working-tree changes.
