#!/usr/bin/env python3
"""Exhaustive/source contract for PFA7 fftP initial n1 fixed-delta mapping."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "src" / "cl" / "fftp.cl").read_text()

guard = "#if PFA_RADIX == 7 && NWORDS == 3670016u && PFA_BINARY_LENGTH == 524288u && PFA_L_INV == 4u"
for token in (
    guard,
    "u32 n1 = n0 + (3u * PFA_BINARY_LENGTH + 1u);",
    "if (n1 >= NWORDS) n1 -= NWORDS;",
):
    assert token in src, token

# Only primary fftP is specialized: fftPCarryB and the FP32/PFA path retain
# the canonical pfaLogicalIndex call for n1.
assert src.count("u32 n1 = pfaLogicalIndex(row, first_binary_pair * 2u + 1u);") >= 2

NWORDS = 3670016
RADIX = 7
BINARY = 524288
LINV = 4
DELTA = 3 * BINARY + 1
assert DELTA == 1572865

def logical(row, b):
    d = (row + RADIX - (b % RADIX)) % RADIX
    t = (d * LINV) % RADIX
    return b + BINARY * t

checks = 0
wraps = 0
for row in range(RADIX):
    for pair in range(BINARY // 2):
        b0 = pair * 2
        n0 = logical(row, b0)
        direct = logical(row, b0 + 1)
        derived = n0 + DELTA
        if derived >= NWORDS:
            derived -= NWORDS
            wraps += 1
        assert derived == direct
        checks += 1

print(f"ok: PFA7 fftP initial n1 fixed delta {checks} exhaustive binary-pair checks; wraps={wraps}")
