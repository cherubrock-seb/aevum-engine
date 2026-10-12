IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED

# F1: defer the inverse GF61 radix-4 tail reduction into CRT

Baseline: 9db67c504bbd5877b4af0e44a538e1f0ca65e5f6.
Enable only with `-use AEVUM_CARRY_DEFER61_REDUCE_V1=1` (or the existing
`AEVUM_PRP_USE` profile); default is zero.

Scope: FFT3161 carryFused, WIDTH=1024, NW=4, WordSize=8, WMUL=1/2,
non-PFA AMD/NVIDIA OpenCL C 2.0+, native GF61 arithmetic. CUDA, limb32,
radix 8, other widths/types and unsupported profiles retain fft_WIDTH1.
Use the existing radix-4 1K profile; do not enable AEVUM_RADIX1K=8.
There is no AUTO selection change.

## Deleted work

Only the final inverse GF61 fft4 is changed: its eight scalar modM61
folds disappear. The existing constant 3*M61 biases remain on the
difference outputs. Weight application and CRT retain their existing
operations, including the final modM61 before balanced reconstruction.

The preceding four radix-4 stages, twiddles, shuffles, barriers and output
permutation match the original width transform. No additional LDS,
global traffic, launches or private array is introduced. Generic FFT
helpers, inverse GF31 and forward transforms are unchanged.

This removes a representation normalization at a producer/consumer
boundary, not a source scheduling or compiler-opacity trick. Actual
instructions, register pressure and timing remain unknown.

## Static range and congruence argument

Let M=2^61-1. For any unsigned 64-bit a, modM61(a) is at most M+7.
Immediately before the final radix-4 butterfly, every component therefore
lies in [0,M+7]: the preceding FFT and each cmul finish with modM61,
and shufl only transports those results.

The sum output is in [0,4M+28]. Each other output is a difference of
two sums of two input components, hence in [-2M-14,2M+14] before the
original 3M bias, and [M-14,5M+14] afterward. All deferred outputs
therefore lie in [0,B], B=5M+28. Unsigned intermediate subtraction wraps
exactly as in fft4by; adding 3M produces the stated nonnegative final
integer without loss. The original swap of outputs 1 and 2 is retained.

For inverse shift k in [0,60], shr(a,k) computes
floor(a/2^k) + ((a mod 2^k)*2^(61-k)).
It is congruent to a*2^(-k) modulo M. At k=0 it equals a; at k>=1
it is at most B/2+M < B. The actual OpenCL shifts are all in [0,61].

Let b=get_Z31(u31), so 0<=b<=2^31-2. The unchanged CRT forms
t=shr(a,k)+M-b, then t+shl(t,31). Thus 0<t<=6M+28.
The latter rotation is at most M+floor(t/2^30) < M+2^34.
Consequently t+shl(t,31) < 7M+28+2^34 < 2^64.
No unsigned overflow occurs in either CRT addition.

The omitted folds preserve congruence modulo M. The final CRT modM61
returns at most M+7; get_balanced_Z61 maps that representative to the
unique balanced residue, including zero represented by M and small
positive residues represented by M+1..M+7. Therefore the arithmetic
derivation predicts the same n61, roundoff statistic, i96 reconstruction,
LL adjustment, MUL3 and outgoing carry. This is a static argument, not
device validation or an exactness claim.

Radix-8 tails have larger quick ranges; this proof does not cover them.
Do not extend the gate by changing NW alone.

## Owner validation handoff

1. Confirm the active gate on actual 1K/NW4 carryFused for both vendors;
   compare knob zero and one with identical explicit plan and options.
2. Compare final-tail plus CRT outputs for all k=0..60, b at
   0/1/2^31-2, and butterfly components 0/1/M-1/M/M+1/M+7.
   Include alternating signs after the butterflies, both zero
   representatives and values around the balanced threshold.
3. Compare every residue word, outgoing carry and statistics in complete
   iterations at representative production exponents. Cover LL/MUL3,
   WMUL1/2, INPLACE0/1, both 101/202 variants where legal, and the cyclic
   last-group seam. Use an independent integer oracle for arithmetic
   boundaries; do not rely on RES64 alone.
4. Confirm disabled/unsupported profiles preserve original artifacts,
   especially NW8, width512, Type2, FFT323161 and CUDA.
5. Inspect generated tail reductions, registers/spills and LDS. Reject
   unchanged code generation. Only after correctness gates, perform
   alternating carryFused and full-iteration timings on RTX 3080/Radeon VII.

No compilation, execution, benchmark, SSH or workflow dispatch was
performed for this candidate.

## Campaign evidence

A2 immediate shuttle streaming, C1 opacity, A3 new LDS plane and A4 trivial
rescheduling are excluded. New owner dispositions also reject A1 packed
storage, D1 barrier elision, A5 order change, A6 LDS streaming, S1 first-word
conversion and PrMers P1 straight dispatch. This candidate does not repeat
any of them. Directional fusion #7 is also rejected and untouched.
Type2 integration #18 is already on this baseline and remains untouched.
PrMers P2 #200 has an owner KEEP disposition pending a clean production
transplant; that integration belongs to the owner's validation lane.

Next independent lead: eliminate a repeated legacy NTT representation
conversion at the inverse/carry boundary, without changing carry batching
or duplicating P2 mask reuse and K1 proof spectrum retention.
