# PFA / Woltman implementation checkpoint

Status: experimental source candidate; no build, test, GPU validation or timing run.
No KEEP decision and no measured speedup.

## Pinned state

- Aevum main baseline: 02bbffaee03480cf9d2cf2fe530c2a333c01930d.
- PrMers main reference: 0e788c3a1f47dfa64f2933ded59eb6d0bc8a3161 (no integration change needed).
- Woltman gpuowl master reference: e44e0976584bc3d1dd9c155c86c4d7c1c814b05a.
- Last confirmed remote implementation commit when writing this milestone:
  c76b53ff529adbc5f524b9aaf9fe18a5dbcfe776 (PFA7 DOT128).
  This source commit adds the PFA3 milestone; the Draft PR records its remote SHA.
- Draft PR: https://github.com/cherubrock-seb/aevum-engine/pull/23.
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
2. **Selected next: normalize the shared inverse row twiddle.**
   Upstream pfaMiddleOut(GF31*/GF61*) scales its row twiddle by 1/PFA.
   Aevum pfaInverseMiddle/pfaMiddleTwiddle still scale each GF31 output and
   each radix-3 GF61 output separately; NVIDIA GF61 radix 7/9 already fuse this.
   Remove repeated scalar normalization without changing the transform layout.
   Dependency: modify inverse butterfly and its following twiddle together.
   Principal risk: missing/double normalization or inadvertently scaling forward
   transforms. Upstream pfaDftR's radix-3 and 3x3 radix-9 paths were inspected in
   both GF31/GF61; normalization is outside those butterflies in pfaMiddleOut.

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

## Implemented milestone: PFA3 inverse twiddle normalization

Selection: AEVUM_PFA_FUSED_INVERSE=1 with a legal explicit
pfa3:1:<width>:3:<height>:<variant> plan. Set AEVUM_PFA7_DOT128=0 or unset it.
The Gpu.cpp gate accepts only PFA3 GF31+GF61 at this milestone and rejects Apple.
UseOptions.h also accepts the same name through -use. Off/unset is the baseline.

src/cl/fftmiddleout.cl defines inverse-only scaling constants before including
fft-middle.cl. pfaInverseMiddle returns the unscaled radix-3 outputs, and
pfaMiddleTwiddle applies 1/3 once to the common row twiddle in each NTT plane.
Thus three complex-by-scalar normalizations become one per plane. The general
cmul used by WADD does not require a unit-circle twiddle. The base-field scalar
commutes with SWAP_XY; binary carry normalization stays unchanged. Middle-in
does not define these constants and keeps its forward transform unchanged.

The existing square and prepared multiply schedules both invoke middle-out;
no buffer, API, tail, root table or prepared-cache representation changes.
KernelCompiler.cpp includes common definitions and kernel source in its cache
hash. Remaining risks are missing/double scale across compile variants and
backend code generation; only targeted source review was performed.


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
PFA3: inverse normalization candidate integrated; compilation/validation pending.
PFA9: no new candidate yet. Its existing 3x3 decomposition is retained; wide
PFA7 dots do not transfer to the different radix-3 identity. Quick arithmetic
is not replayed as a new experiment.
Next implementation: extend inverse-only normalization to GF31 PFA9 and to
GF61 PFA9 where NVIDIA does not already fuse it. Preserve that existing NVIDIA
path and commit the connected stages together.

