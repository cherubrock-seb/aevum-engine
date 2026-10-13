# Experimental FP32 + Gold64 backend

**Status: source implementation, incomplete PRP backend, unvalidated.**
The current interruption checkpoint is
[`fp32-gold64-implementation-status.md`](fp32-gold64-implementation-status.md).

## Implemented components

The original coefficient oracle in `src/cl/fp32_gold64_experiment.cl` and
`tests/fp32_gold64_experiment_test.py` is retained unchanged. Its O(N^2)
convolution is only a small-input reference; it does not exercise the new path.

The new implementation uses 4,194,304 **real words**, with two different
transform geometries representing the same IBDWT convolution:

| Component | Source | Contract |
|---|---|---|
| Arithmetic/tables | `Gold64Plan.h`, `gold64_arith.cl` | Canonical residues, Marin roots/weights, O(N) weight recurrence |
| Exact modular plane | `Gold64Gpu.h`, `gold64_transform.cl` | Marin radix-64 + fused radix-1024 square/prepared multiply, natural-order modular output |
| Approximate plane | `Fp32Gold64Gpu.h`, `fp32_gold64_bridge.cl` | Existing Aevum FP32 kernels at 512:8:512/202, signed 64-bit input, tiled layout bridge |
| Candidate lift/carry | `Fp32Gold64Reconstruction.h`, `fp32_gold64_reconstruct.cl` | Signed 128-bit coefficients, factor 1/3, square-minus-two, full cyclic normalization and global diagnostics |

The supervised Engine API owns the two-plane OpenCL dispatch, buffers and
register lifecycle through `Fp32Gold64Engine.h`. Production workloads, the old
create API, standalone stock Gpu execution and native CUDA reject the new type.
AUTO and validated selectors are unchanged. The added helper in Gpu.cpp reuses
the existing FP32 configuration/weight builders and optimized transform kernels.

## Common mathematical contract

For exponent q and length N, digit j begins at `ceil(q*j/N)` and has width
`ceil(q*(j+1)/N)-ceil(q*j/N)`. Its forward weight is
`2^(ceil(q*j/N)-q*j/N)`. Gold64 realizes this with Marin's N-th root of two;
the Aevum FP32 path uses its existing floating weights and input scale 2^-24.
Root legality requires `N | (p-1)/192`, for `p=2^64-2^32+1`.

The Gold64 packed transform returns N/2 times the weighted convolution, so
its inverse weights include 1/(N/2). Aevum's fftMiddleOut already applies
1/(2*N); its bridge only removes weights, undoes the two 2^-24 input scales,
swaps the inverse pair components and transposes to natural order. Applying
Gold64 normalization to the FP32 plane would be incorrect.

After both planes, `residue[j] = X_j mod p` and `estimate[j]` approximates the
same signed coefficient X_j. Reconstruction forms `X = residue + k*p` using
the floating quotient hint. One 64-bit product with `2^32-1` implements the
signed 128-bit lift; optional multiplication by 3 and subtraction of 2 are
exact integer operations on that candidate.

## Hard limitations

- **Every candidate is marked UnprovenPrecision.** A near-integer quotient is
  never accepted as proof; nonfinite values, unrepresentable quotient hints,
  invalid residues, unsupported operations and carry overflow are flagged.
- The 33-bit digit-format gate and the 4M root geometry do **not** establish
  M136279841 capacity. A rigorous FP32 reconstruction-error bound is missing.
- Full cyclic carry is implemented in source, with explicit overflow and
  consistency flags. Its output remains a normalized reconstruction candidate.
- All compilation/device validation and a rigorous error certificate remain
  missing. Production PRP/LL/Gerbicz acceptance, standalone execution and
  production timing counters are disabled. No performance claim is made.

## Validation still required

No compilation, CPU/GPU tests, benchmarks, SSH, device discovery or CI polling
was performed during this continuation. Commits request CI skipping. Historical
oracle results from the original PR do not validate the new transform code.

Owner-led validation must first compile the headers/kernels, compare host and
device arithmetic/root/weight tables, and compare the actual two-plane square
and prepared multiplication against exact weighted coefficients (including
signed inputs, impulses, boundary values and hostile patterns). Then establish
the quotient bound, validate full carry, and validate the resulting engine
against trusted residues, factor-3/LL and Gerbicz cases on RTX 3080 and Radeon VII.
Paired performance measurements come after correctness; no speedup is claimed.

## Provenance

Selected Gold64 arithmetic and packed transform functions derive from Marin
(Copyright 2025 Yves Gallot), as maintained in `cherubrock-seb/PrMers`:
`kernels/marin.cl`, `include/marin/arith.h`, `include/marin/ibdwt.h`, with the
4M dispatch sequence from `include/marin/engine_gpu.h`.
The MIT notice is retained in `licenses/Marin-MIT.txt`. Existing Aevum FP32
kernels and repository licensing are retained.
## Engine entry point

The actual Engine API now supports the explicit plan
`gold64-validation:512:8:512:202` (numeric FFT type 54). Its certified maxExp
remains zero. Only the new supervised workload enables execution:

```cpp
aevum_engine_handle h = aevum_engine_create_ex(
    136279841, 3, device_index, 1,
    "gold64-validation:512:8:512:202", nullptr,
    AEVUM_WORKLOAD_GOLD64_VALIDATION);
// Check h and every API return; failures use aevum_engine_last_error().
// Existing set/get_words, prepare, square_mul, mul, add/sub/copy/equal apply.
// aevum_engine_gold64_diagnostics(h, &report) always reports unproven precision.
// aevum_engine_gold64_get_digits reads 4194304 balanced natural-order int64s.
// aevum_engine_gold64_square_minus_two performs a fused diagnostic candidate step.
// Destroy h after validation; these outputs are never PRP/LL certificates.
```

Factors 1 and 3 are supported. The prepared RHS cache holds one register and
rebuilds after invalidation. Input and result buffers remain distinct even for
in-place API operations; the register is replaced only after global fatal-error
rejection. Compact upload/readback/equality use the existing Aevum host packing.
Arithmetic, reconstruction and carry stay on the GPU, with a 16-byte diagnostic
readback before accepting each candidate operation. No PrMers repository change
is needed for a C API validation harness; existing PrMers production workloads
are explicitly rejected for this plan.

## Cyclic carry and precision policy

The normalized candidate APIs now use 256-digit segments and a two-level
three-state map prefix scan. They solve the cyclic carry fixed point in five
bounded GPU launches, followed by two small diagnostic reductions. Negative
wrap, signed-wide addition overflow, signed-64 local carry overflow and final
block-exit consistency are explicit. This is exact normalization of the lifted
coefficients, not evidence that FP32 selected the correct coefficients.

For exponent 136279841, N=4194304, balanced inputs have 32/33-bit widths. FP32
conversion already rounds large digits. The reused kernels then introduce
weight/twiddle approximation, FMA/butterfly roundoff and inverse/export error.
The quotient calculation additionally rounds the modular residue conversion,
reciprocal correction and subtraction. A certificate would need a conservative
combined error strictly below half a Gold64 modulus (including quotient-stage
error), for every supported input and actual compiler arithmetic mode. No such
bound has been established here; a small fractional quotient distance is not
that bound. The integer range check abs(quotient)<2^23 is only representability.
It bounds lifted candidates below 2^87 and factor-3 candidates below 2^89;
it cannot certify the true coefficient or copy a Type2 exponent limit.

`PrecisionPolicy::RequireProven` rejects before dispatch. Only an explicit
`SupervisedValidation` request can execute the checked normalized APIs, and
every fatal flag anywhere in the cyclic result rejects it. The report preserves
UnprovenPrecision even when all operational checks pass. Factor 3 is exact after
lifting, so it triples any wrong-prime-multiple error and supplies no FP32
headroom. There is no production PRP/LL acceptance path. This source has not
been compiled, tested or GPU-validated.

