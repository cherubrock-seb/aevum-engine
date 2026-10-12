# S1: shared encoding of the first finalized FFT3161 word

IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED

Baseline: `71e1b9819a7041c7f83889dd51e36ba2f1b3fa58`.
Opt-in: `AEVUM_CARRY_SHARED_RESIDUE_V1=1` through the existing use-option path.
Default zero; independent from every earlier campaign branch.

## Concrete deletion

The forward-weight loop converts each signed WordSize=8 integer separately:
make_Z31(i64) splits at bit 31, adjusts the upper part, adds and folds mod M31;
make_Z61(i64) conditionally adds M61 with a wide operation.
For the first word only, a shared 32-bit biased low limb supplies both residues.
No CRT work, carry scheduling, FFT ordering, allocation or synchronization moves.

Per first word, the candidate removes the generic GF31 split/fold/reduction and
the separate conditional 64-bit M61 addition. It introduces a 32-bit sign bias,
a low mask, a sign-derived high limb and limb assembly. NW=4/8 therefore offers
four/eight such replacements per work-item reaching the forward stage.
This is a source arithmetic ledger, not an instruction count or measured saving.

## Range contract

In FFT3161 with WordSize=8, carryFinal calls carryStepSignedSloppy(i64,i64*).
Both branches of that helper call carryStep(i64,i64*). For floor(EXP/NWORDS)
between 1 and 30, this obtains the first word by signed low-bit extraction with
b = floor(EXP/NWORDS) + biglit0, hence 1 <= b <= 31.

The resulting first word x obeys -2^(b-1) <= x < 2^(b-1).
In particular, -2^30 <= x < 2^30: the cast to i32 is exact.
This bound follows from the existing first-word normalization, not a presumed
bound on FFT residues or incoming carry. Existing arithmetic input contracts
remain unchanged.

No equivalent bound is claimed for the second word: carryFinal adds tmpCarry to
its previously sloppy value. Both original second-word converters remain.

## Encoding identity, including representatives

Let M31 = 2^31-1 and M61 = 2^61-1.
Define lo = (uint)x - (x < 0), using unsigned 32-bit subtraction.

For x >= 0, lo=x, the sign-derived high limb is zero, and both encoded values
are x. In particular zero stays zero, not a noncanonical modulus representative.

For x < 0, lo=2^32+x-1. Its low 31 bits are 2^31+x-1=M31+x.
Combining lo with high limb 0x1fffffff gives:
(2^29-1)*2^32 + 2^32+x-1 = M61+x.

These are the same representatives as the baseline converters on this interval.
For the baseline GF31 converter, the upper extracted part is -1 for negative x;
after adjustment and folding, its result is M31+x. For nonnegative x it is x.
The result is independent of the existing MODM31 implementation choice.
All subsequent shl calls, shifts 0/31/61 included, see identical inputs.

Boundary expectations before forward weights:

| x | GF31 | GF61 |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 1 | 1 |
| -1 | M31-1 | M61-1 |
| -2^30 | M31-2^30 | M61-2^30 |
| 2^30-1 | 2^30-1 | 2^30-1 |

The source uses arithmetic right shift of signed i32 to form the sign mask,
consistent with the existing OpenCL arithmetic. All subtraction used for the
biased limb is unsigned; negative signed left shifts are not introduced.

## Scope and fallback

Active only for FFT3161, WordSize=8, the range above, width512/1024, NW4/8,
WMUL1/2, non-PFA, non-CUDA, AMD/NVIDIA OpenCL C 2.0+.
The original two assignment lines remain verbatim in the fallback.
Other FFT types, Apple/OpenCL 1.2 and unsupported configurations are unaffected.
Shared arithmetic helpers and FFT323161 CRT/i128 are untouched.

## Owner arbitration

No compilation, preprocessing, kernel execution or benchmark was performed.

- Prove activation on both production families: 1:1K:8:512:202 and
  1:512:16:512:101/202 where legal, retaining identical plan/use settings.
- Compare baseline/candidate converter results on-device at the boundary values,
  adjacent values, powers of two and random signed values. Cover every legal
  GF31/GF61 weight shift, MODM31=0/1/2 and both big/little word sizes.
- Run dense full-word differentials, including LL and MUL3, WMUL1/2, INPLACE0/1,
  ROE/STATS variants, zero/negative first words and cyclic seam groups.
- Inspect generated code for removed wide M61 addition and GF31 fold, and record
  registers/spills/occupancy. The optimizer may already derive part of the
  baseline identity; reject a byte-identical result before timing.
- Only after exact validation, run counterbalanced kernel and full-PRP timing on
  RTX 3080 and Radeon VII at fixed production profiles. Do not infer a win from
  the source ledger.

## Campaign resume

P2 PrMers #200 is the independent predecessor (one-word packed carry masks).
A2 #9, C1 #10, A3 #11 and A4 #12 supply authoritative negative evidence:
immediate carry traffic, opacity tricks, extra LDS staging and pure rescheduling
are not repeated. A1 #8, D1 #13, A5 #14, A6 #15 and P1 #198 are already
implemented. PR #7, FFT323161 CRT/i128 and Type2 #16/#199 remain separate lanes.

Next: a materially different arithmetic deletion. Do not extend this conversion
to the second word without proving its range. The PR body carries the exact head
and authoritative PERF_CAMPAIGN resume block.
