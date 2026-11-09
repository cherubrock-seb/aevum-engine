#!/usr/bin/env python3
"""Host model/source contract for exact PFA7 fftP paired weight derivation."""
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src" / "cl" / "fftp.cl").read_text()

for token in (
    "pfaWeightShifts3161State",
    "pfaWeightShifts3161PairSecond",
    "#if PFA_RADIX == 7 && NWORDS == 3670016u && PFA_BINARY_LENGTH == 524288u && PFA_L_INV == 4u",
    "const u32 pairDelta = 3u * PFA_BINARY_LENGTH + 1u;",
    "const u32 wrapDistance = NWORDS - pairDelta;",
    "frac1 = frac0 + deltaCombo + (low1 < low0);",
    "frac1 = frac0 - wrapCombo - (low0 < wrapLo);",
):
    assert token in SOURCE, token

NWORDS = 3670016
BINARY = 524288
PAIR_DELTA = 3 * BINARY + 1
WRAP_DISTANCE = NWORDS - PAIR_DELTA
EXP = 125000003
FRAC_HI = 256829732
FRAC_LO = 2454267025
MASK = (1 << 32) - 1

exp_mod = EXP % NWORDS
step31 = ((((NWORDS - exp_mod) * 18) % 31) + 30) % 31
step61 = ((((NWORDS - exp_mod) * 56) % 61) + 60) % 61
assert (step31, step61) == (6, 27)
assert PAIR_DELTA == 1572865
assert WRAP_DISTANCE == 2097151

def combo(n):
    return n * FRAC_HI + ((n * FRAC_LO) >> 32) - 1

def residue(n, p, step):
    return ((n % p) * step) % p

def shift(c, r, p):
    return ((c + (r << 32) + MASK) >> 32) % p

delta_lo = (PAIR_DELTA * FRAC_LO) & MASK
delta_combo = PAIR_DELTA * FRAC_HI + ((PAIR_DELTA * FRAC_LO) >> 32)
wrap_lo = (WRAP_DISTANCE * FRAC_LO) & MASK
wrap_combo = WRAP_DISTANCE * FRAC_HI + ((WRAP_DISTANCE * FRAC_LO) >> 32)

delta31 = ((PAIR_DELTA % 31) * step31) % 31
delta61 = ((PAIR_DELTA % 61) * step61) % 61
wrap31 = ((WRAP_DISTANCE % 31) * step31) % 31
wrap61 = ((WRAP_DISTANCE % 61) * step61) % 61
assert (delta31, delta61, wrap31, wrap61) == (15, 9, 6, 10)

starts = {0, 1, WRAP_DISTANCE - 1, WRAP_DISTANCE, WRAP_DISTANCE + 1, NWORDS - 2, NWORDS - 1}
rng = random.Random(8573161)
starts.update(rng.randrange(NWORDS) for _ in range(50000))

checks = 0
for n0 in starts:
    frac0 = combo(n0)
    r310 = residue(n0, 31, step31)
    r610 = residue(n0, 61, step61)
    low0 = (n0 * FRAC_LO) & MASK

    if n0 < WRAP_DISTANCE:
        n1 = n0 + PAIR_DELTA
        low1 = (low0 + delta_lo) & MASK
        frac1 = frac0 + delta_combo + int(low1 < low0)
        r311 = (r310 + delta31) % 31
        r611 = (r610 + delta61) % 61
    else:
        n1 = n0 - WRAP_DISTANCE
        frac1 = frac0 - wrap_combo - int(low0 < wrap_lo)
        r311 = (r310 - wrap31) % 31
        r611 = (r610 - wrap61) % 61

    assert frac1 == combo(n1)
    assert r311 == residue(n1, 31, step31)
    assert r611 == residue(n1, 61, step61)
    assert shift(frac1, r311, 31) == shift(combo(n1), residue(n1, 31, step31), 31)
    assert shift(frac1, r611, 61) == shift(combo(n1), residue(n1, 61, step61), 61)
    checks += 1

print(f"ok: PFA7 fftP paired weight derivation {checks} sampled/boundary states")
