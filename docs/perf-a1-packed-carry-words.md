# Packed carry words: owner validation handoff

IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED

Baseline: `71e1b9819a7041c7f83889dd51e36ba2f1b3fa58`.
Source implementation: `dc8e3abe7480e1f74010934dfff3fca4e1a0de40`.
Independent candidate A1; no dependency on directional fusion PR #7.

## Scope and use

Use the existing manual profile with `AEVUM_CARRY_PACKED_WORDS_V1=1`
through `-use` or `AEVUM_PRP_USE`; default is 0. Compare the identical
profile with this one key set to 0. Disable automatic tuning for the comparison.

The active source gate requires FFT3161, WordSize=8, no PFA, integer
`EXP/NWORDS` from 1 through 31, OpenCL C 2.0+, and AMD or NVIDIA.
CUDA and OpenCL 1.2 (including Apple) retain the original source path.
WMUL, INPLACE, MUL3, LL, ROE and STATS retain their original handling.
Primary shapes: `1:512:16:512:101` and `1:1K:8:512:202`.
Confirm the resolved shape actually uses carryFused before measuring.

## Representation and lifetime argument

The first i96 sloppy carry step returns an unsigned low word: below 22
bits/word it masks to at most 22 bits; from 22 through 31 it returns
the low uint32 limb. Thus the first component lies in [0, 2^32-1].
The second step either uses the signed i96 carryStep or directly returns
an int32 limb. Under the gate both produce a sign-extended int32.
This applies after the unchanged LL/MUL3 arithmetic and on both sides
of the SLOPPY_MAXBPW threshold.

After weightAndCarryPairSloppy reads u31[i] and u61[i], neither input slot
is read again as an NTT value. Store the two integer components in u31[i],
with conversion to uint32 preserving the signed component modulo 2^32.
Before carryFinal, zero-extend x and bitcast/sign-extend y into Word2.
Use the original 64-bit carryFinal, then the original make_Z31/make_Z61
and weight shifts. Do not keep the result compact after carryFinal:
adding the final carry may enlarge the second component.

All slots have been reconstructed before fft_WIDTH2 touches either
array. No pointer reinterpretation, union alias, change to GF61 quick
arithmetic, modular reduction, CRT construction, or carry synchronization
is introduced.

## Observable hypothesis

The active source removes the separate 16*NW-byte Word2 array and holds
the handoff pair in 8*NW bytes of already consumed GF31 slots. The width
arrays themselves and LDS allocation remain. Source storage is not a
register-count prediction: an optimizing compiler may already infer the
word ranges and coalesce the original arrays, or may scalarize the
destructive representation less effectively.

Reject if the JIT output is unchanged or full PRP performance regresses.
Acceptance requires fewer live registers/spills or a useful occupancy
change, followed by a reproducible end-to-end gain.

## Measurements required

1. Build/JIT on RTX 3080 and Radeon VII; also build CUDA/Apple fallback.
2. Dense all-word differential against knob=0 on both primary shapes,
   low/central/high legal exponents, signed and unsigned 32-bit boundary
   patterns, and around floor(bpw)=21/22 and 31/32. Explicitly exercise
   the excluded floor(bpw)>=32 path.
3. Cover WMUL=1/2, INPLACE=0/1/2 where supported, LL and MUL3, ROE/STATS,
   both CARRY64 profiles where legal, PFA and other-type fallback.
   Check repeated iterations and checkpoints, not only RES64.
4. Record registers/VGPRs, spills, LDS, occupancy and generated
   sign/zero-extension operations for the exact candidate SHA.
5. Alternate baseline/candidate kernel and complete PRP runs on each GPU;
   retain fixed plans, flags and iteration counts. Report raw paired
   timings, dispersion and whether any resource reduction helps throughput.

No test or performance result is asserted here.
