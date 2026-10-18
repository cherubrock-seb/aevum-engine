#!/usr/bin/env python3
"""GF61 csqq / csqaq must not call themselves.

Their fallback for large M61 counts used to reduce the input and recurse (csqq(modM61(a), 2, 2)).  The CUDA compiler refuses to inline a
self-recursive function, so any caller that reached it (tailSquare's onePairSq through csqa(ab, addin, 6)) spilled registers to local memory.
The fallback must reassign the count parameters instead, which means they cannot be const.
"""
import pathlib
import re
import sys

MATH = pathlib.Path(__file__).resolve().parent.parent / "src" / "cl" / "math.cl"
text = MATH.read_text()
fail = []

for name in ("csqq", "csqaq"):
    # The four-argument GF61 definition; the body runs to the first closing brace in column 0.
    m = re.search(r"^GF61 OVERLOAD %s\(GF61 a,[^)]*u32 x_m61_count, [^)]*u32 y_m61_count\) \{\n(.*?)^\}" % name, text, re.M | re.S)
    if not m:
        fail.append(f"{name}: four-argument GF61 definition not found")
        continue
    header = m.group(0).split("\n", 1)[0]
    body = m.group(1)
    if re.search(r"\b%s\s*\(" % name, body):
        fail.append(f"{name}: body calls {name} (self-recursive fallback)")
    if "const u32" in header:
        fail.append(f"{name}: count parameters are const, the fallback cannot reassign them")
    if not re.search(r"x_m61_count = 2, y_m61_count = 2", body):
        fail.append(f"{name}: fallback no longer resets the counts to 2, 2")

if fail:
    print("\n".join(fail))
    sys.exit(1)
print("GF61 csqq/csqaq: non-recursive fallback ok")
