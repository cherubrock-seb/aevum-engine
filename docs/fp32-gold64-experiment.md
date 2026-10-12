# Experimental FP32 + Gold64: coefficient-level oracle

**Status:** isolated mathematical/GPU experiment, NOT a production Aevum FFT/IBDWT backend.

## Goal

Verify that a floating-point approximation can lift coefficients from one exact
Goldilocks residue to a signed 128-bit integer. Compare the exact Gold64 path and
FP32 approximation on a GPU before designing an expensive NTT/IBDWT integration.

The source is `src/cl/fp32_gold64_experiment.cl`; no existing FFT type, selector,
kernel bundle, carry or PrMers run path is modified. The reference convolution is
O(n^2), deliberately bounded to n <= 64 in validation. It is NOT a GPU performance
candidate for a 4M-word transform.

## Mathematical contract

Let p = 2^64 - 2^32 + 1, let c be a signed convolution coefficient, and let
r = c mod p (0 <= r < p). For some signed integer k,

    c = r + k*p.

The FP32 lane supplies an approximation to `(c - r) / p`. If that estimate is
strictly within 0.5 of k, round-to-nearest recovers k. The kernel returns
little-endian signed two's-complement 128-bit coefficients in `ulong2`.

**Caveat:** the warning flag is merely a diagnostic; a hint close to *some*
integer is not proof that it is close to the *correct* integer. Dense comparisons
against arbitrary precision GMP/Python are mandatory, as are rigorous bounds on
FP32 propagation error and on k overflow. The reference's input restriction
`a_i,b_i < 2^32`, `n <= 64` is deliberately MUCH narrower than M136279841.

## Tests

From the Aevum repository root:

    python3 tests/fp32_gold64_experiment_test.py

Runs 30,000 randomized modular arithmetic tests, 20,000 signed reconstruction
cases, boundary cases and small dense cyclic-convolution comparisons; with
`clang` installed it also syntax-checks the OpenCL 1.2 kernel.

On Ubuntu with a GPU and an OpenCL ICD:

    python3 -m pip install numpy pyopencl
    python3 tests/fp32_gold64_experiment_test.py --gpu

This compiles and executes both kernels, checks dense convolution output against
unbounded Python integers and tests Gold64 arithmetic on near-p input values.

No GPU benchmark, throughput improvement or M136279841 correctness claim is made.

## Required work before a usable PrMers/Aevum production FFT type

1. Implement the **Goldilocks real-length IBDWT** using Marin's existing
   proven kernels and root/weight layout. GF(M61^2) half-length kernels cannot
   be reused by replacing a modulus constant.
2. Derive and implement a compatible complex FP32 transform/weight layout
   matched coefficient-for-coefficient to Gold64; integrate both plane
   buffers and kernel scheduling. Verify inverse pairing and odd radix cases.
3. Integrate a Gold+FP32 `weightAndCarryOne` into `carryutil.cl` and
   `carryfused.cl`, with a mathematically justified error bound and safe
   carry representation. Do not accept coefficients on the diagnostic warning
   flag alone.
4. Expose an **explicit-only** FFT type with capacity and size checks. Do not
   alter native AUTO until long-run PRP/LL and dense GMP validation on both
   Radeon VII and RTX 3080.
5. Measure each component and end-to-end iterations against *both* Type2 and
   Marin using paired timings; include `square_mul(...,3)` and Gerbicz.

At present NONE of these production steps is implemented. The isolated oracle
is a first correctness gate, not a partial substitute for an exact FFT.

## Provenance

Gold64 modular add/sub/reduce/mul was adapted from `kernels/marin.cl` in
`cherubrock-seb/PrMers`, originally Marin (MIT license), copyright 2025 Yves
Gallot. The new oracle and cross-checks preserve that attribution. No other
production files are changed.
