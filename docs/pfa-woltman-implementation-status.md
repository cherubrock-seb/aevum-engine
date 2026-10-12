# PFA / Woltman implementation checkpoint

Status: experimental source candidate; no build, test, GPU validation or timing run.
No KEEP decision and no measured speedup.

## Pinned state

- Aevum main baseline: 02bbffaee03480cf9d2cf2fe530c2a333c01930d.
- PrMers main reference: 0e788c3a1f47dfa64f2933ded59eb6d0bc8a3161 (no integration change needed).
- Woltman gpuowl master reference: e44e0976584bc3d1dd9c155c86c4d7c1c814b05a.
- Last confirmed remote implementation commit when writing this milestone:
  c58cbb203ec1bb2648c913486018d4a19e15e3e1 (PFA3 fused inverse).
  PFA7 DOT128 was first committed as c76b53ff529adbc5f524b9aaf9fe18a5dbcfe776.
  This source commit adds the PFA7/PFA9 inverse extension; the Draft PR records
  the final confirmed remote head (a commit cannot embed its own SHA).
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
with FFT323161 also use the candidate GF61 plane; DOT128 alone leaves FP32
and GF31 unchanged. The existing capacity checks still apply.
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
The original PFA3 milestone gated this to PFA3 GF31+GF61; the extension below
adds PFA7/PFA9. Apple staged kernels still reject.
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


## Implemented milestone: extend inverse fusion to PFA7 / PFA9

AEVUM_PFA_FUSED_INVERSE=1 now supports explicit legal
pfa3:1:<width>:3:<height>:<variant>, pfa7:1:<width>:7:<height>:<variant> and
pfa9:1:<width>:9:<height>:<variant> plans, plus existing full/adaptive type-4
PFA7/PFA9 plans. Inspect the logged resolved plan, since type-4 PFA9 may elide FP32.
Use AEVUM_PFA7_DOT128=1 as an independent, combinable option only on PFA7.

fftmiddleout.cl supplies the existing 1/7 and 1/9 constants. In fft-middle.cl,
pfaInverseMiddle omits the matching per-output scale and pfaMiddleTwiddle applies
it once. GF31 is covered for all three radices. GF61 is covered for PFA3 and
non-NVIDIA PFA7/PFA9; NVIDIA's existing PFA7 prescale/PFA9 fused normalization
continues unchanged and is not counted as new work. The host gate requires a
matching radix/middle and both NTT planes, so every accepted plan has a changed
GF31 inverse path. Apple staged kernels are rejected.

This removes seven or nine scalar output normalizations in favor of one common
twiddle normalization per newly covered plane. Both inverse stages are changed
in the same commit. The natural output order of the radix-3 and 3x3 radix-9
decompositions, Aevum's CRT/Good-Thomas permutation and normalized carry input
are preserved. No root/weight table, buffer or prepared-operand format changes.
Only fftmiddleout.cl defines the scale macros; forward kernels cannot see them.

Changed files for this milestone: src/Gpu.cpp, src/cl/fftmiddleout.cl,
src/cl/fft-middle.cl and this checkpoint. The complete campaign additionally
changes src/UseOptions.h. No PrMers companion change is required.


## Owner validation handoff (not executed)

Baseline SHA is above; candidate SHA is the confirmed Draft PR head.
Use existing tests/native_pfa_dense_compare.cpp, native_pfa_engine_compare.cpp,
engine_small_factor_gpu_test.cpp and the Engine API. Inspect each harness before
invocation: native_pfa_engine_compare constructs a stock plan first, so globally
enabling either strict candidate option would reject an incompatible stock engine.
Scope selection to candidate engine creation (unset for baseline/production).
Do not weaken the guard to accommodate a harness.

First compile OpenCL/CUDA variants with option off/on, inspect the logged actual
plan and compiler output. Then compare every exported residue word across
equivalent PFA baseline/candidate and the best legal production plan. Include
dense signed and boundary inputs, cyclic carry, repeated square, factor-3,
square-minus-two where supported, prepared multiply, source mutation/reprepare,
cache invalidation and independent engine instances. Check existing carry/accuracy
errors. Invalid flag values and incompatible requested plans must fail clearly.
For PFA7, validate DOT128 alone, fused inverse alone, and both together against
both options off. For PFA3/PFA9, leave DOT128 off. On NVIDIA PFA7/PFA9, confirm
the existing GF61 normalization remains single; the new fusion affects GF31.
Include forced full FFT323161 PFA7/PFA9 where legal (pfa9full:4 for PFA9),
as well as the resolved paired FFT3161 plans. FP32 arithmetic is not changed.

Only after exact comparisons pass, measure end-to-end operations under identical
GPU/backend/compiler, exponent, plan, carry, tuning and workload; record actual
register/spill behavior. Compare against both the equivalent PFA plan and the best
legal production plan. Instruction counts are not performance measurements.

## Remaining work / exact next step

PFA7: DOT128 GF61 plus fused inverse GF31/non-NVIDIA GF61 are integrated.
PFA3/PFA9: fused inverse is integrated; the existing radix-3/3x3 butterflies
are retained. PFA7's wide dots do not transfer to the different radix-3 identity;
quick arithmetic is not replayed as a new experiment. FP32 is unchanged for
every radix. No new retained-state, tail, carry or global-pass optimization.
All candidates remain uncompiled and unvalidated; no end-to-end gain is established.
Exact next step: owner compiles the option-off/on OpenCL and CUDA variants, then
runs complete-residue comparisons at the logged legal plans using the matrix
above. Scope flags to candidate engine construction. Record candidate head,
compiler, backend and validation outcome before any timing or KEEP decision.

