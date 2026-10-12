# Experimental FP32 + Gold64 convolution

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
| Candidate lift | `Fp32Gold64Reconstruction.h`, `fp32_gold64_reconstruct.cl` | Signed 128-bit coefficients, factor 1/3, square-minus-two, individual carry split |

The headers expose source-level OpenCL experiments using Aevum's Queue,
Buffer and KernelCompiler. No FFT type, CLI plan, Engine API operation or
production AUTO policy exposes this as an operational hybrid PRP backend.
The native CUDA backend is not wired. No existing transform/carry kernel was
rewritten; the added helper in Gpu.cpp reuses its configuration/weight builders.

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
- The carry primitive splits each coefficient independently into a balanced
  digit and signed 64-bit outgoing carry. Inter-digit and cyclic long-carry
  propagation are **not implemented**. Its output is not a normalized residue.
- Gerbicz/register operations, persistence and a dedicated engine/type path
  remain unwired. Full milestones C, D and E remain incomplete.

## Validation still required

No compilation, CPU/GPU tests, benchmarks, SSH, device discovery or CI polling
was performed during this continuation. Commits request CI skipping. Historical
oracle results from the original PR do not validate the new transform code.

Owner-led validation must first compile the headers/kernels, compare host and
device arithmetic/root/weight tables, and compare the actual two-plane square
and prepared multiplication against exact weighted coefficients (including
signed inputs, impulses, boundary values and hostile patterns). Then establish
the quotient bound, implement full carry, and validate the resulting engine
against trusted residues, factor-3/LL and Gerbicz cases on RTX 3080 and Radeon VII.
Paired performance measurements come after correctness; no speedup is claimed.

## Provenance

Selected Gold64 arithmetic and packed transform functions derive from Marin
(Copyright 2025 Yves Gallot), as maintained in `cherubrock-seb/PrMers`:
`kernels/marin.cl`, `include/marin/arith.h`, `include/marin/ibdwt.h`, with the
4M dispatch sequence from `include/marin/engine_gpu.h`.
The MIT notice is retained in `licenses/Marin-MIT.txt`. Existing Aevum FP32
kernels and repository licensing are retained.
