#!/usr/bin/env python3
"""Host-only exact model for canonical carryB predecessor, PFA7:1:512:7:512:202.

The GPU correctness check still needs the independent dense Type2 differential.
PRPLL's native PFA layout is *different*; this only removes divisions from
Aevum's existing canonical carryB predecessor mapping.
"""
from pathlib import Path

WIDTH = 512
BIG_HEIGHT = 7 * 512
CARRY_LEN = 8
HB = BIG_HEIGHT // CARRY_LEN
assert HB == 448

for x in range(WIDTH):
    for gy in range(HB):
        prev = (gy + HB * x + (HB * WIDTH - 1)) % (HB * WIDTH)
        original_line = prev % HB
        original_col = prev // HB
        direct_line = HB - 1 if gy == 0 else gy - 1
        direct_col = (WIDTH - 1 if x == 0 else x - 1) if gy == 0 else x
        assert (original_line, original_col) == (direct_line, direct_col), (x, gy)

gpu = Path("src/Gpu.cpp").read_text()
cl = Path("src/cl/carryb.cl").read_text()
options = Path("src/UseOptions.h").read_text()
assert 'std::getenv("AEVUM_PFA7_CARRYB_FAST")' in gpu
assert 'config["PFA7_CARRYB_FAST"] = value;' in gpu
assert 'if (key == "PFA7_CARRYB_FAST") return true;' in options
assert '#if PFA_RADIX == 7 && PFA7_CARRYB_FAST >= 1' in cl
assert '#if PFA_RADIX == 7 && PFA7_CARRYB_FAST >= 2' in cl
print("PFA7_CARRYB_PREDECESSOR_MODEL=PASS", WIDTH * HB, "addresses")
