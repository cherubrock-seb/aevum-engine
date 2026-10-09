# Directional fusion experiment

Track: `OPENAI-MATH-DIRECTIONAL-FUSION`.
Baseline: `0769889c9dfffa7638d76e7c5f671de00d2334f6` (main).

The local architectural motivation is to consume permuted loads directly in
the adjacent butterfly, as suggested by the directional-operation viewpoint
of OpenAI Math Family 130. This is a fixed radix-16 entry experiment, not an
implementation of an asymptotic Fourier algorithm.

## Current-source inventory

`fftbase.cl` contains four typed `shufl_and_fft2` overloads. The 1K/radix-8
path uses `f=8`, `WG=128`, then selects `fft8_16a` for lanes 0..63 and
`fft8_16b` for lanes 64..127. FP64 has both variant-0 broadcast and tabulated
call sites; FP32, GF31 and GF61 have the tabulated call site. Width and height
instantiate this shared code. The preceding `f=1` shuffle is separate.

The existing FFT2 combines LDS locations `i*64+lane` and `512+i*64+lane`
in the unpadded layout, where `lane=lowMe%64`. The next independent pairs
are `(0,4)`, `(1,5)`, `(2,6)`, `(3,7)`. GF31 vector and GF61 scalar padded
paths use their own existing address mapping and must retain it.

GF61 `fft8_16a` starts with two `X2q` and two `X2q_mul_t4` operations;
`fft8_16b` starts with four `X2qt4` operations. These are the candidate
operations to consume while a pair is read, leaving the subsequent
normalization, nontrivial roots, radix-4 cores and final swaps unchanged.

## Invariants

- GF61 shuffle additions are unreduced `addq`, yielding 0..2*M61+epsilon;
  subtraction results represent -M61-epsilon..M61+epsilon in unsigned storage.
- The upper-half entry produces sums in 0..4*M61+epsilon and differences
  in -2*M61-epsilon..2*M61+epsilon. The lower-half rotated entry remains
  within -2*M61-epsilon..2*M61+epsilon. Preserve the exact original operation
  order within each pair, including the fused quarter-root operations.
- Keep every `modM61q` count, `optional_add`, delayed `mul_t8q` and
  `mul_3t8q` unchanged. A field identity alone does not justify changing the
  quick representation. GF31 `addq`/`subq` have different, reduced semantics.
- Retain LDS allocation, partition stride, writes and all `bar(WG)` calls.
  Scalar LDS still stages real and imaginary components separately; a complex
  entry butterfly can only run after both components of its pair are available.
- Keep the existing `shufl_and_fft2` semantics for independent callers and
  preserve a selectable original path.
- Preserve FUSE_WEIGHT_BUTTERFLY, AUTO, PFA selection and release metadata.

## Validation status

Source inspection only. Compilation, GPU exactness and performance are
**NOT TESTED — intentionally deferred to hardware validation**. No measured
speedup is claimed. Implementation and hardware commands will be recorded
here as the individual stages are committed.
