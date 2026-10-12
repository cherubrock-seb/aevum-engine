# PFA / Woltman implementation checkpoint

Status: experimental source candidate; no build, test, GPU validation or timing run.
No KEEP decision and no measured speedup.

## Pinned state

- Aevum main baseline: 02bbffaee03480cf9d2cf2fe530c2a333c01930d.
- PrMers main reference: 0e788c3a1f47dfa64f2933ded59eb6d0bc8a3161 (no integration change needed).
- Woltman gpuowl master reference: e44e0976584bc3d1dd9c155c86c4d7c1c814b05a.
- Last confirmed remote implementation commit at checkpoint creation: baseline above.
  The first candidate is the source commit containing this checkpoint; its remote
  SHA will be recorded with the next implementation milestone and in the Draft PR.
- Dedicated branch: exp/pfa-woltman-performance. PR #22 and its FP32/Gold64 branch
  are separate. No AUTO policy, capacity, overflow/carry guard or selector change.

## Focused comparison and opportunities

1. **Selected: GF61 PFA7 wide dot products.**
   Upstream src/cl/fft-middle.cl pfaDftR(GF61*) executes through
   pfaMiddleIn/pfaMiddleOut in src/cl/fftmiddlein.cl and fftmiddleout.cl.
   Aevum uses pfaDft7 via pfaForwardMiddle/pfaInverseMiddle and fft_MIDDLE/
   ifft_MIDDLE in the same files. The similarly named fft7.cl is a floating-point
   butterfly, not this NTT path. Aevum already has paired sums/differences;
   this change accumulates three products in u128 before each weakModM61,
   removing per-product reductions and intermediate modular additions.
   Keep the base-field roots, pair order, ordinary GF61 output range and inverse
   scale. Dependency: existing mul64/mad64/weakModM61/modM61q in math.cl.
   Principal risks: range assumptions and compiler unrolling/register spills.
2. **Next: normalize the shared inverse row twiddle.**
   Upstream pfaMiddleOut(GF31*/GF61*) scales its row twiddle by 1/PFA.
   Aevum pfaInverseMiddle/pfaMiddleTwiddle still scale each GF31 output and
   each radix-3 GF61 output separately; NVIDIA GF61 radix 7/9 already fuse this.
   Remove repeated scalar normalization without changing the transform layout.
   Dependency: modify inverse butterfly and its following twiddle together.
   Principal risk: missing/double normalization or inadvertently scaling forward
   transforms. Inspect actual upstream radix-3/radix-9 paths before extending.

The current PrMers docs/PRMERS_AUTO_VS_PRPLL.md comparison was read; its historical
devices, commits, types and timings are not evidence for this candidate.
Do not repeat already-present paired butterflies or rejected tuning experiments.

## Implemented milestone: PFA7 GF61 DOT128

- src/cl/fft-middle.cl: pfaDft7(GF61*, Z61), with attribution to Woltman.
  Three-term cosine/sine dot products use existing unsigned 128-bit helpers.
  Explicit unroll directives keep coefficient indices constant.
- src/Gpu.cpp: clDefines reads and validates AEVUM_PFA7_DOT128 and adds it to
  the compilation configuration. Requested non-PFA7/non-GF61 plans, invalid values,
  conflicting selections and Apple staged kernels reject with an error.
- src/UseOptions.h: registers the compiler option.
- docs/pfa-woltman-implementation-status.md: this checkpoint.

Selection: set AEVUM_PFA7_DOT128=1 and explicitly request an existing legal
pfa7:1:<width>:7:<height>:<variant> Engine API FFT spec. Existing PFA7 plans
with other GF61-containing types also use the candidate GF61 plane; their FP32
and GF31 planes are unchanged. The existing capacity checks still apply.
Unset the variable or set it to 0 for the equivalent PFA baseline. Direct
-use AEVUM_PFA7_DOT128=1 is also recognized. This option does not select a shape
or enable any otherwise-disallowed arithmetic type. Verbose engine creation
prints the actual plan and candidate.

Execution: fftP -> fftMiddleInGF61 -> pfaMiddleTwiddle -> pfaDft7(candidate)
-> existing tailSquare/tailMul -> fftMiddleOutGF61 -> inverse pfaDft7(candidate)
-> existing inverse scale/twiddle -> fftW -> carryA -> carryB.
Prepared operands enter through regPrepare -> fftMidIn and are consumed by the
existing prepared multiply. Kernel compilation receives the selection through
the common compiler definitions. Buffer sizes, roots/twiddles/weights, memory
layout, carry, prepared representation and invalidation lifecycle are unchanged.
Factor-3 and square-minus-two retain their existing Engine API implementation
and support limits. This is an arithmetic change in the real middle pipeline;
it removes no global passes or launches.

Source invariants: inputs use the existing [0, M61 + epsilon] representation.
Pair sums <= 2*M61+epsilon, positive differences <= 3*M61+epsilon. Three products
fit below 2^126. The full 128-bit weak reduction returns <= 2*M61+epsilon;
pc+qs <= 5*M61+epsilon and pc-qs+3*M61 is positive and < 8*M61.
Output is reduced back to ordinary GF61 range. Root direction, odd-axis scale,
Good-Thomas ordering and binary normalization are preserved.

## Owner validation handoff (not executed)

Baseline SHA is above; candidate SHA is the confirmed Draft PR head.
Use existing tests/native_pfa_dense_compare.cpp, native_pfa_engine_compare.cpp,
engine_small_factor_gpu_test.cpp and the Engine API. Inspect each harness before
invocation: native_pfa_engine_compare constructs a stock plan first, so globally
enabling this strict PFA7 option would correctly reject that stock engine.
Scope selection to candidate engine creation (unset for baseline/production).
Do not weaken the guard to accommodate a harness.

First compile OpenCL/CUDA variants with option off/on, inspect the logged actual
plan and compiler output. Then compare every exported residue word across
equivalent PFA baseline/candidate and the best legal production plan. Include
dense signed and boundary inputs, cyclic carry, repeated square, factor-3,
square-minus-two where supported, prepared multiply, source mutation/reprepare,
cache invalidation and independent engine instances. Check existing carry/accuracy
errors. Invalid flag values and incompatible requested plans must fail clearly.

Only after exact comparisons pass, measure end-to-end operations under identical
GPU/backend/compiler, exponent, plan, carry, tuning and workload; record actual
register/spill behavior. Compare against both the equivalent PFA plan and the best
legal production plan. Instruction counts are not performance measurements.

## Remaining work / exact next step

PFA7: compile and differential validation pending; GF31 and FP32 are unchanged.
PFA3/PFA9: no new candidate yet. Their 3x3 decomposition and quick arithmetic
must not be assumed faster from source alone.
Next implementation: examine upstream inverse scale and radix-3/radix-9 call
contracts, then connect an opt-in fused inverse twiddle to both affected stages
if it preserves the existing layout and normalization. Commit separately.

