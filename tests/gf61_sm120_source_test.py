#!/usr/bin/env python3
from pathlib import Path

gpu = Path("src/Gpu.cpp").read_text(encoding="utf-8")
use = Path("src/UseOptions.h").read_text(encoding="utf-8")
math = Path("src/cl/math.cl").read_text(encoding="utf-8")

for token in [
    'std::getenv("AEVUM_GF61_SM120")',
    'AEVUM_GF61_SM120 must be 0 or 1',
    'AEVUM_GF61_SM120 requires CUDA GF61 and sm_20+',
    'AEVUM_GF61_SM120 conflicts with NO_ASM or limb32',
]:
    assert token in gpu, token

assert 'key == "AEVUM_GF61_SM120"' in use

for token in [
    '#ifndef AEVUM_GF61_SM120',
    '#if AEVUM_GF61_SM120 && HAS_PTX >= 200',
    'shr.b32    top',
    'add.cc.u32 %0, %0, top',
    'addc.u32   %1, %1, 0',
    '#if HAS_PTX >= 320 && !AEVUM_GF61_SM120',
]:
    assert token in math, token

# The experiment must remain paired: enabling the option changes both
# modM61 and weakModM61 selection together.
assert math.count("AEVUM_GF61_SM120") >= 5

print("GF61_SM120_SOURCE_GATE=PASS")
