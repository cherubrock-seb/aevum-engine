#!/usr/bin/env python3
from pathlib import Path

RADIX = 7
WIDTH = 512
BIG_HEIGHT = 3584
NWORDS = 3670016
BINARY = 524288
L_INV = 4
LOGICAL_STEP = 3211264
PAIR_COUNT = NWORDS // 2

def logical(row: int, binary_index: int) -> int:
    delta = (row + RADIX - binary_index % RADIX) % RADIX
    t = (delta * L_INV) % RADIX
    return binary_index + BINARY * t

checks = 0
for row in range(RADIX):
    for binary_pair in range(BINARY // 2):
        n_even = logical(row, binary_pair * 2)
        n_odd = logical(row, binary_pair * 2 + 1)

        pair_even = n_even >> 1
        pair_odd = n_odd >> 1

        x_even = pair_even // BIG_HEIGHT
        line_even = pair_even - x_even * BIG_HEIGHT

        line_odd = line_even + 1536
        x_odd = x_even + 219
        if line_odd >= BIG_HEIGHT:
            line_odd -= BIG_HEIGHT
            x_odd += 1
        if x_odd >= WIDTH:
            x_odd -= WIDTH

        expected_x_odd = pair_odd // BIG_HEIGHT
        expected_line_odd = pair_odd - expected_x_odd * BIG_HEIGHT
        assert line_odd == expected_line_odd
        assert x_odd == expected_x_odd
        assert line_odd * WIDTH + x_odd == expected_line_odd * WIDTH + expected_x_odd
        checks += 1

assert checks == 1835008
assert LOGICAL_STEP // (2 * BIG_HEIGHT) == 448

source = Path("src/cl/fftw.cl").read_text()
guard = "#if PFA_RADIX == 7 && PFA_BINARY_LENGTH == 524288u && PFA_L_INV == 4u && PFA_LOGICAL_STEP == 3211264u && BIG_HEIGHT == 3584u && WIDTH == 512u"
assert source.count(guard) == 1
assert "u32 lineOdd = lineEven + 1536u;" in source
assert "u32 xOdd = xEven + 219u;" in source
assert source.count("const u32 nOdd  = pfaWLogicalIndex(row, firstBinaryPair * 2u + 1u);") >= 2

print(f"ok: PFA7 fftW GF61 odd-coordinate derivation {checks} exhaustive checks")
