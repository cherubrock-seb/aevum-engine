#!/usr/bin/env python3
"""Host model and source contract for the exact PFA7 fftP weight recurrence."""
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src" / "cl" / "fftp.cl").read_text()

for token in (
    "pfaWeightShifts3161State",
    "#if PFA_RADIX == 7 && NWORDS == 3670016u && PFA_LOGICAL_STEP == 3211264u",
    "weightForwardFracLo",
    "weightForwardCombo",
    "weightWrapDistance",
    "weightWrapFracLo",
    "weightWrapCombo",
    "weightForward31",
    "weightForward61",
    "weightWrap31",
    "weightWrap61",
):
    assert token in SOURCE, token

NWORDS = 3670016
LOGICAL_STEP = 3211264
WRAP_DISTANCE = NWORDS - LOGICAL_STEP
EXP = 125000003
FRAC_HI = 256829732
FRAC_LO = 2454267025
MASK = (1 << 32) - 1

exp_mod = EXP % NWORDS
big31 = ((NWORDS - exp_mod) * 18) % 31
big61 = ((NWORDS - exp_mod) * 56) % 61
step31 = (big31 + 30) % 31
step61 = (big61 + 60) % 61

def combo(n):
    return n * FRAC_HI + ((n * FRAC_LO) >> 32) - 1

def residue(n, p, step):
    return ((n % p) * step) % p

def shift(c, r, p):
    return ((c + (r << 32) + MASK) >> 32) % p

forward_lo = (LOGICAL_STEP * FRAC_LO) & MASK
forward_combo = LOGICAL_STEP * FRAC_HI + ((LOGICAL_STEP * FRAC_LO) >> 32)
wrap_lo = (WRAP_DISTANCE * FRAC_LO) & MASK
wrap_combo = WRAP_DISTANCE * FRAC_HI + ((WRAP_DISTANCE * FRAC_LO) >> 32)
forward31 = ((LOGICAL_STEP % 31) * step31) % 31
forward61 = ((LOGICAL_STEP % 61) * step61) % 61
wrap31 = ((WRAP_DISTANCE % 31) * step31) % 31
wrap61 = ((WRAP_DISTANCE % 61) * step61) % 61

assert (step31, step61) == (6, 27)
assert (forward31, forward61) == (30, 9)
assert (wrap31, wrap61) == (22, 10)

starts = {
    0, 1, WRAP_DISTANCE - 1, WRAP_DISTANCE,
    WRAP_DISTANCE + 1, NWORDS - 2, NWORDS - 1,
}
rng = random.Random(73161)
starts.update(rng.randrange(NWORDS) for _ in range(20000))

checks = 0
wraps = 0
for start in starts:
    n = start
    low = (n * FRAC_LO) & MASK
    c = combo(n)
    r31 = residue(n, 31, step31)
    r61 = residue(n, 61, step61)

    for _ in range(8):
        assert c == combo(n)
        assert r31 == residue(n, 31, step31)
        assert r61 == residue(n, 61, step61)
        assert shift(c, r31, 31) == shift(combo(n), residue(n, 31, step31), 31)
        assert shift(c, r61, 61) == shift(combo(n), residue(n, 61, step61), 61)
        checks += 1

        if n < WRAP_DISTANCE:
            new_low = (low + forward_lo) & MASK
            carry = int(new_low < low)
            low = new_low
            c += forward_combo + carry
            r31 = (r31 + forward31) % 31
            r61 = (r61 + forward61) % 61
            n += LOGICAL_STEP
        else:
            borrow = int(low < wrap_lo)
            low = (low - wrap_lo) & MASK
            c -= wrap_combo + borrow
            r31 = (r31 - wrap31) % 31
            r61 = (r61 - wrap61) % 61
            n -= WRAP_DISTANCE
            wraps += 1

print(f"ok: PFA7 fftP weight recurrence {checks} states, {wraps} wrap transitions")
