# PFA11 / Woltman implementation checkpoint

Experimental source candidate. No build, test, device discovery, benchmark or
autotune was executed. No KEEP decision or measured speedup.

## Pinned state and prior work
- Aevum main/base: 02bbffaee03480cf9d2cf2fe530c2a333c01930d.
- Woltman gpuowl master: e44e0976584bc3d1dd9c155c86c4d7c1c814b05a.
- PrMers main inspected: 0e788c3a1f47dfa64f2933ded59eb6d0bc8a3161.
- Last confirmed remote implementation at this checkpoint:
  642ba5ee8f0a22c7e31ab964c2c5d387d3941d2b (GF31 batching integrated).
  Initial source commit: 2c956221460db6357726e5400f672b56e7f25eda.
  Follow-up source review moved the PFA11 compile guard after the FFT3161 macro
  definition; the initial ordering would reject a valid Type1 compile.
  Final source review also makes root selection independent of the assumption
  that generator 5 works for the new radix. Final SHA is recorded in the Draft
  PR, since a commit cannot embed its own SHA.
- Draft PR: https://github.com/cherubrock-seb/aevum-engine/pull/24.
- Dedicated branch: exp/pfa11-woltman-ntt. Independent of PR #23 and #22.
- Previous PFA11 work was reported exact but slower and rejected. Recovered
  history mentions pfa11-r58/64/67/69/70/76/77/78/79/80 and disposable r56
  worktrees; no reproducible previous source SHA was available in current main.
  Preserve that negative result. Do not claim this source matches those trials.
  The new mechanism is Woltman's five-term GF61 wide-dot accumulation plus
  inverse twiddle normalization, not another register-count or LDS experiment.

## Focused source comparison (three opportunities)
1. Upstream src/cl/fft-middle.cl pfaDftR(GF61*) is called through pfaMiddleIn/
   pfaMiddleOut by fftMiddleInGF61/fftMiddleOutGF61. For PFA=11 it pairs five
   inputs, uses mul64/mad64 and one full-width weakModM61 per dot component.
   Aevum previously had no dispatched PFA11. Adapt this with the existing
   math.cl wide helpers, retaining Aevum's gathered natural odd-row order.
   Remove per-product reductions; principal risk: loose GF61 ranges and spills.
2. Upstream pfaMiddleOut(GF31*/GF61*) multiplies the shared binary row twiddle
   by 1/PFA. Connect unnormalized inverse DFT and scaled twiddle together.
   Remove eleven output normalizations in favor of one scalar normalization
   per plane; risk: double/missing normalization and layout mismatch.
3. Beyond upstream: GF31 pfaDftR still reduces every product and partial sum.
   Aevum pfaDft11 can accumulate five reduced-input products in 3+2 bounded
   u64 batches using math.cl mad32. Keep [0,M31] pair values; never use the
   GF61 unreduced pairing here. Risk: u64 accumulation and u32 fold overflow.

The upstream layout has pfaRotate/row permutations and PFA_M2 row subtransforms.
Aevum's fftP CRT gather already forms ordered odd rows with MIDDLE=11; importing
those upstream permutations would be wrong. The existing inverse fftW scatter
returns canonical carry order without a pfaUnpack pass. This adaptation keeps
that layout and only supports the single odd middle, not composite middles.

## Implemented milestone
- src/FFTConfig.h: isPfa recognizes 11.
- src/FFTConfig.cpp: explicit pfa11:1:W:11:H:V parser, mandatory
  AEVUM_EXPERIMENT_PFA11=1, Type1 only. Inherit stock middle-16 BPW minus 0.20
  and cap the validation envelope at 32 BPW. This is not validated capacity.
  PFA11 rejects overcapacity even with -od/AEVUM_ALLOW_OVERCAPACITY_PLAN.
  AUTO enumeration and existing radices' limits/policies are unchanged.
- src/Gpu.cpp: independently guard direct construction, generate order-11
  roots and cosine/sine constants for both fields, trying generator 5 and
  deterministically finding a nontrivial root if needed; check exact root
  order; inject constants and candidate modes into compiler configuration.
- src/UseOptions.h: AEVUM_PFA11_DOT128 and AEVUM_PFA11_GF31_DOT64 options.
- src/cl/base.cl: reject unsupported PFA11 type/middle/missing mode at compile.
- src/cl/fft-middle.cl: pfaDft11 GF31 paired reference and GF61 DOT128/reference,
  actual forward/inverse dispatch, and normalized inverse row twiddle.
- src/cl/fftmiddleout.cl: scope inverse normalization to inverse kernels.
- docs/pfa11-woltman-implementation-status.md: this independent checkpoint.

Selection:
  AEVUM_EXPERIMENT_PFA11=1
  AEVUM_PFA11_DOT128=1
  explicit Engine API FFT spec pfa11:1:<width>:11:<height>:<variant>
For example pfa11:1:512:11:512:202, only at an exponent admitted by its resolver.
DOT128 defaults to 1 only for an explicitly enabled PFA11 plan. Select 0 for
the equivalent paired GF61 reference using the same layout, inverse fusion,
tail and carry. It is a local comparison reference, not the historical rejected
binary. Actual mode/plan are logged; invalid/conflicting modes reject.
Apple, FP32/Type4 and other arithmetic types are excluded. No PrMers companion
is needed for the Engine API path; no new PrMers command-line alias is provided.

## Additional milestone: opt-in GF31 bounded dot batching
Select AEVUM_PFA11_GF31_DOT64=1 in addition to the experiment gate and explicit
PFA11 plan. This flag defaults to 0 and is independent of GF61 DOT128.
Gpu.cpp/UseOptions.h validate/register it; base.cl requires an explicit compiler
mode; pfaDft11(GF31*) dispatches through pfa11Dot5_31 in both directions.
The original per-product GF31 path remains when the flag is 0.

The five products use two u64 accumulators (three terms and two terms), replacing
five product reductions and their intermediate modular sums. Bounds:
0 <= a_i <= M31, 0 <= c_i < M31, so three products <3*2^62<2^64.
pfa11Fold64_31 reduces (low31+middle31) before adding the top two bits, then
combines the two reduced subtotals. This prevents a u32 overflow in a naive
low31+middle31+high2 fold. Every output stays in [0,M31]. No field limit,
normalization, data representation or general math.cl helper is changed.
The grouped reductions are a new source candidate, not a measured gain.

## Execution and preserved contracts
Engine square/prepare/multiply -> fftP CRT gather/width NTT -> middle-in
binary twiddle + odd DFT -> existing row-local NTT tailSquare/tailMul ->
middle-out inverse odd DFT + common twiddle scaled by 1/11 -> fftW canonical
scatter -> carryA + cyclic carryB. Prepared source mutation/invalidation,
factor-3 and square-minus-two retain existing Engine API support and errors.
PFA11 does not enable the PFA9 lead bridge, fused LL or prepared lead cache.

isPfa connects existing buffer sizing, launch counts (two exceptional lines
per odd row), TrigBufCache middle/tail binary-row tables, weights and carry
binary normalization. No new global-memory pass, buffer representation or
carry relaxation. GF31/GF61 roots stay in the base fields: NTT tails pair
inside each row, unlike FP32 cross-row conjugate pairing.

GF61 input components retain [0,M61+epsilon]. Sums <=2*M61+epsilon and
positive differences <=3*M61+epsilon; five products fit below 2^126.
Use weakModM61(...,128), never the 125-bit variant for this dot. Accumulate DC
in two groups so u64 never sums all eleven loose components. Fold outputs back
to ordinary GF61 range. Inverse = forward with output indices reversed.
All inputs are captured before output writes; no separate y[11] array is
needed. Actual register allocation and unrolling remain compiler-dependent.

## Validation handoff (owner only; not executed)
Baseline production SHA is above. Candidate is the confirmed Draft PR head.
Use existing tests/native_pfa_dense_compare.cpp, native_pfa_engine_compare.cpp
and engine_small_factor_gpu_test.cpp plus Engine API entry points. Scope the
DOT128/GF31_DOT64 environment variables to PFA11 candidate engine construction:
a harness creating a stock engine first must leave them unset/0 for that engine.
For equivalent PFA11 A/B, keep the explicit plan and experiment gate, switch
DOT128 0/1, and record the actual instantiated plan and all effective options.
Validate all four (DOT128, GF31_DOT64) combinations: (0,0), (1,0), (0,1), (1,1).
For GF31, include residues around 0, M31 and the folded-limb boundaries, all
supported MODM31 variants, and both native-ASM and NO_ASM code generation.

Compile OpenCL/CUDA variants, then compare complete exported residues,
including dense signed/boundary inputs, cyclic carry, repeated square,
factor-3, square-minus-two where supported, prepared multiplication, mutation/
reprepare/invalidation and independent engines. Keep all existing numerical
guards; verify rejection of disabled experiment, unsupported type and excess
exponent. No successful compilation or exactness is claimed here.
Only after differential success, compare end-to-end iteration cost against
both equivalent PFA11 reference and best legal production plan with matched
GPU/backend/compiler, exponent, carry, workload and tuning. Inspect spills.

## Remaining work and next exact step
First integrated PFA11 GF31+GF61 candidate is ready for owner compilation and
differential validation; neither is done. Type4/FP32, composite middles,
retained carry state and PrMers CLI exposure remain out of scope.
Exact next step: owner compiles the four candidate/reference combinations on
OpenCL and CUDA, then compares complete residues with the existing Engine API
harnesses under the stated flag scoping. Record the actual head, resolved plan,
compiler and backend before timing. Do not widen the envelope to chase a win.
