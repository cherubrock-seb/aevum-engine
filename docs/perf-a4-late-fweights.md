# Late FP64 forward weights in carryFused

IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED

Baseline: `71e1b9819a7041c7f83889dd51e36ba2f1b3fa58`.
Source implementation: `d378d2e9053d2d5a20364cd045eb46f43ed7c986`.
Sequence A4. Independent of A1/A2/C1/A3 and PR #7.

## Selection

Opt in with `AEVUM_CARRY_LATE_FWEIGHTS_V1=1` through the existing
`-use` / `AEVUM_PRP_USE` interface. The default is 0.
The active gate requires FFT64 (type 0), AMD/NVIDIA OpenCL C 2.0+,
NW=4/8, width 256/512/1024, WMUL=1/2, no PFA, no CUDA backend and
FUSE_WEIGHT_BUTTERFLY=0.

Normal NVIDIA FP64 configurations use an unfused source path. Most AMD
FP64 configurations use the fused-weight butterfly and therefore fall
back unchanged; the existing width-256/NW-8 exception can be active.
Do not disable fusion in only one side of a comparison.

FFT6431 has analogous source, but Aevum's normal explicit-plan selector
rejects that type. It is excluded from the final candidate. No host
selector changes, new FFT modes or FFT323161/CRT edits are included.

## Hypothesis and live state

The baseline computes all NW forward weight pairs into u[NW] before
shuffling carries and waiting for the previous group. Those 16*NW bytes
coexist with wu[NW] and carry[NW+1] across the carry exchange.

The candidate computes the common base after the incoming carry loads,
then constructs each pair directly at the final word conversion. Only
weights.y (one FP64 scalar) is required across the exchange for this
calculation. The full forward transform output array still exists, and
its values accumulate as the final loop consumes wu/carry entries.

This removes the source requirement for 16*NW bytes of precomputed
weights during the exchange, replacing it with an 8-byte base input:
56 bytes at NW=4 or 120 bytes at NW=8 before scalar temporaries.
These are logical payload differences, not measured register savings.
The inverse FFT or the forward FFT can still determine the kernel's
maximum allocation.

There is no new LDS/global traffic and no additional weight arithmetic.
The cost is giving up independent FP64 work before the carry wait:
less latency hiding may outweigh a shorter register lifetime. An
unrolling compiler may already sink the baseline or hoist the candidate,
making generated code identical.

## Arithmetic and fallback argument

For every i in [0,NW), both paths use the same base=optionalHalve(weights.y),
the same i==0 special case, and the same fweightStep(i), fancyMul and
optionalHalve calls. fancyMul is the original explicit FMA. The two
multiplications by the final integer words use the same operand types.
Only the time of materialization changes; no recurrence, reassociation,
weight table access or rounding primitive is introduced.

The helpers depend only on private values and constants, not on ready
flags or shuttle state. carryFinal, big/little counters, outgoing carry
publication, incoming rotation, memory fences and INPLACE coordination
remain untouched. Group zero still returns at the original point.

FUSE_WEIGHT_BUTTERFLY is excluded because that path retains high-half
weights for a later FMA butterfly. The fallback contains the original
precompute loop and final multiply statement verbatim. Default zero,
CUDA, Apple/OpenCL 1.2, PFA, other types, fused butterflies and excluded
shapes all retain that original path.

## Owner validation

- Select a legal explicit type-0 plan with identical exponent, WMUL,
  NW, radix and compiler options in both sides. Confirm the active
  preprocessor gate; a knob=1 fallback is not candidate evidence.
- Compare exact full-word results for low/central/high legal exponents,
  both NW values, WMUL=1/2, carry modes, LL/MUL3, ROE/STATS, INPLACE=0/1/2,
  and every lane/slot including the cyclic seam. Record ROE separately.
- On RTX 3080 and eligible Radeon VII plans, inspect the final ISA for
  weight calculation placement, registers, occupancy and spills.
  Unchanged resource allocation is a plausible outcome.
- Alternate carryFused timing and full PRP timing with identical plans.
  Record carry-wait stalls and FP64 utilization where available.
- Check default zero and excluded fused/CUDA/Apple/PFA/type-1/type-4
  fallback separately. Reject any dense differential mismatch.
- Reject the performance hypothesis if moving work behind the wait loses
  overall time or fails to provide useful resource relief. No speedup or
  exactness result is claimed by this source-only change.

## Campaign handoff

Published order: A1 (#8), A2 (#9), C1 (#10), A3 (#11), A4.
All remain uncompiled/unmeasured candidates, not accepted optimizations.
Do not repeat generic GF61-limb32, carry_batch1/4, carry epoch,
cache-buffer-args, L2 persistence or directional-fusion-V1.

Next source question: can the very first LDS transpose in carryFused
avoid its reuse barrier when no work-item has touched that allocation?
Inspect only the first inverse GF31 phase and preserve all post-store,
inter-phase and carry barriers; constrain away from PR #7's NVIDIA 1K
directional second-transpose work. This question is not implemented here.
