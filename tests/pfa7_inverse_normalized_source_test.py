#!/usr/bin/env python3
"""Preprocess the actual option policy and guard inverse-only dispatch."""
from pathlib import Path
import os
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "src/cl/fft-middle.cl").read_text()
start = source.index("#ifndef AEVUM_PFA7_FOLD_INV7")
end = source.index("#if NTT_GF31", start)
policy = source[start:end]
compiler = shlex.split(os.environ.get("CXX", "c++"))
prefix = """
#define FFT3161 1
#define FFT_TYPE 1
#define PFA_RADIX 7
#define AEVUM_PFA7_NUSS8 1
#define NTT_GF31 0
#define NTT_GF61 0
"""
def preprocess(overrides):
    definitions = "".join(f"#undef {key}\n#define {key} {value}\n"
                          for key, value in overrides.items())
    return subprocess.run(compiler + ["-E", "-P", "-x", "c++", "-",
                                     "-I", str(ROOT / "src/cl")],
                          input=prefix + definitions + policy, text=True,
                          capture_output=True).returncode

assert preprocess({}) == 0
assert preprocess({"AEVUM_PFA7_FOLD_INV7": "0", "PFA_RADIX": "3"}) == 0
assert preprocess({"AEVUM_PFA7_FOLD_INV7": "1"}) == 0
assert preprocess({"AEVUM_PFA7_FOLD_INV7": "1", "AEVUM_PFA_LAZY61": "0"}) == 0
for value in ("2", "-1", "invalid"):
    assert preprocess({"AEVUM_PFA7_FOLD_INV7": value}) != 0
for key, value in (("PFA_RADIX", "0"), ("PFA_RADIX", "3"), ("PFA_RADIX", "9"),
                   ("FFT_TYPE", "3"), ("AEVUM_PFA7_NUSS8", "0"),
                   ("AEVUM_PFA_GF31_COSBASE", "1"), ("AEVUM_GF61_LIMB32", "1"),
                   ("AEVUM_APPLE_OPENCL12", "1"), ("AEVUM_PFA_LAZY61", "1")):
    assert preprocess({"AEVUM_PFA7_FOLD_INV7": "1", key: value}) != 0, (key, value)

def function_body(signature):
    begin = source.index("{", source.index(signature))
    depth, end = 1, begin + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]

for field in ("GF31", "GF61"):
    assert "aevumPfa7InverseNormalized" not in function_body(
        f"void OVERLOAD pfaForwardMiddle({field} *u)")
    assert "aevumPfa7InverseNormalized(u);" in function_body(
        f"void OVERLOAD pfaInverseMiddle({field} *u)")

# A prescaled row must never enter the normalized helper a second time.
inverse61 = function_body("void OVERLOAD pfaInverseMiddle(GF61 *u)")
assert """#ifdef AEVUM_PFA7_MIDOUT_TWIDDLE_PRESCALE
  pfaDft7(u, (Z61)PFA7_INVROOT61);
#elif AEVUM_PFA7_FOLD_INV7
  aevumPfa7InverseNormalized(u);""" in inverse61
print("PFA7 normalized inverse policy and dispatch: PASS")
