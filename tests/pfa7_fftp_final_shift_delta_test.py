#!/usr/bin/env python3
"""Model/source contract for exact PFA7 fftP final-shift delta experiment."""
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "src" / "cl" / "fftp.cl").read_text()

guard = "#if PFA_RADIX == 7 && EXP == 125000003u && NWORDS == 3670016u && WIDTH == 512u && SMALL_HEIGHT == 512u && PFA_BINARY_LENGTH == 524288u && PFA_L_INV == 4u && PFA_LOGICAL_STEP == 3211264u"
for token in (
    guard,
    "const u32 weightPairDelta = 3u * PFA_BINARY_LENGTH + 1u;",
    "const u32 roundingBit =",
    "const uint2 shifts1 = U2(shift1_31, shift1_61);",
    "const uint2 shifts1 = pfaWeightShifts3161(n1, m31_step, m61_step);",
):
    assert token in src, token

# Only the primary fftP path is specialized; fftPCarryB keeps the canonical helper.
assert src.count("const uint2 shifts1 = pfaWeightShifts3161(n1, m31_step, m61_step);") >= 1

NWORDS = 3670016
BINARY = 524288
PAIR = 3 * BINARY + 1
WRAP = NWORDS - PAIR
HI = 256829732
LO = 2454267025
MASK = (1 << 32) - 1

def combo(n):
    return n * HI + ((n * LO) >> 32) - 1

def shift(n, p, step):
    c = combo(n) + ((((n % p) * step) % p) << 32) + MASK
    return (c >> 32) % p

pair_product = PAIR * LO
pair_combo = PAIR * HI + (pair_product >> 32)
wrap_product = WRAP * LO
wrap_combo = WRAP * HI + (wrap_product >> 32)

assert (pair_combo >> 32) == 94053
assert (wrap_combo >> 32) == 125405
assert (pair_product & MASK) == 0xffe12491
assert (wrap_product & MASK) == 0xffd6db6f

def derived(n0, p, step):
    s0 = shift(n0, p, step)
    frac0 = combo(n0)
    product0 = (n0 * LO) & MASK
    frac0lo = frac0 & MASK

    if n0 < WRAP:
        product1 = (product0 + (pair_product & MASK)) & MASK
        pcarry = int(product1 < product0)
        effective = ((pair_combo & MASK) + pcarry) & MASK
        frac1lo = (frac0lo + effective) & MASK
        q = int(frac1lo < frac0lo) + int(frac1lo != 0) - int(frac0lo != 0)
        base = ((pair_combo >> 32) + ((PAIR % p) * step) % p) % p
        return (s0 + base + q) % p
    pborrow = int(product0 < (wrap_product & MASK))
    effective = ((wrap_combo & MASK) + pborrow) & MASK
    frac1lo = (frac0lo - effective) & MASK
    q = int(frac0lo < effective) + int(frac0lo != 0) - int(frac1lo != 0)
    base = (-(wrap_combo >> 32) - ((WRAP % p) * step) % p) % p
    return (s0 + base - q) % p

starts = {0,1,WRAP-1,WRAP,WRAP+1,NWORDS-2,NWORDS-1}
rng=random.Random(0x3161F17)
starts.update(rng.randrange(NWORDS) for _ in range(75000))
checks=0
for n0 in starts:
    n1 = n0 + PAIR if n0 < WRAP else n0 - WRAP
    for p,step in ((31,6),(61,27)):
        assert derived(n0,p,step) == shift(n1,p,step)
        checks += 1

print(f"ok: PFA7 fftP final-shift delta {checks} sampled/boundary plane checks")
