# Inverse GF61 before GF31 in carryFused

IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED

Baseline Aevum: `71e1b9819a7041c7f83889dd51e36ba2f1b3fa58`.
Source implementation: `34cd41287c9a69d1400df2b13b20ce5ef2355d10`.
Sequence A5. Independent baseline branch, following A1/A2/C1/A3/A4/D1
and PrMers P1 (#198).

## Opt-in and exclusions

Use `AEVUM_CARRY_INVERSE61_FIRST_V1=1` through existing `-use` /
`AEVUM_PRP_USE`; default 0. The active gate is FFT3161, non-PFA,
AMD/NVIDIA OpenCL C 2.0+, width 512/1024, NW=4/8 and WMUL=1/2.
CUDA, Apple/OpenCL 1.2, other types/shapes and default zero retain the
original inverse order.

The gate also requires `AEVUM_CARRY_FRESH_LDS_V1=0`. D1 (PR #13)
requires GF31 to be the very first LDS user. Requesting that option
therefore suppresses this reordering, even if D1 itself would fall back
for the selected shape. Do not remove this exclusion when reconciling
the independent branches.

## Schedule change

The baseline loads/transforms GF31, then loads/transforms GF61, then
enters the CRT/carry loop. The candidate loads/transforms GF61 first,
then GF31. It changes no arithmetic within either transform and no
operation in or after the CRT/carry loop.

The retained result changes as follows:

| Interval | Baseline | Candidate |
| --- | --- | --- |
| GF61 inverse transform | completed GF31: 8*NW bytes | no completed other plane |
| GF31 inverse transform | no completed other plane | completed GF61: 16*NW bytes |
| CRT/carry entry | both complete planes | both complete planes |

This is deliberately different from A3's volatile LDS staging: no
explicit spill/reload and no extra local allocation. It asks whether
moving the larger retained plane to the cheaper transform interval
reduces the maximum resource requirement.

## Liveness condition

Let P31 and P61 denote each inverse transform's private peak including
its own data but excluding the other completed plane. Ignoring other
shared invariants, the inverse portion changes from
max(P31, P61+8*NW) to max(P61, P31+16*NW).

This can help only if the GF61 portion is sufficiently more expensive.
For example, P61-P31>8*NW makes the second candidate interval smaller
than the original GF61-plus-GF31-result interval in this payload model.
The opposite resource balance can lose. The CRT or forward FFT may
still dominate the whole kernel, producing no occupancy change.

These expressions are a source liveness model, not measured registers:
inlining, unrolling, invariant retention, scalarization and scheduling
can all change the actual allocation. No array is removed.

## Dependency and LDS argument

The GF31 and GF61 inputs, roots and private outputs are separate residue
planes. Neither inverse transform consumes the other's private result.
The unchanged CRT/carry loop is their first arithmetic join, and receives
the same two arrays in the same operand positions.

Both transforms reuse the existing LDS allocation. Each production
shuffle begins with the existing reuse barrier/fence, which protects
the preceding phase's reads before overwrite. Reversing which plane
goes first leaves that protocol intact. Every first/post-store/inter-phase
barrier is retained; the D1 fresh-LDS helper is not part of this branch.

dependentLaunchWait moves before the GF61 load. It is a no-op in the
selected OpenCL source. CUDA programmatic dependent launches are
excluded and retain the original GF31-before-wait schedule.

No stores to the input/output planes occur until both inverse reads and
the original carry operations complete. The group-zero read signal and
last-group INPLACE wait remain at their original sites. Carry publication,
seam rotation, LL/MUL3, weights, stats and forward plane order are unchanged.

## Cost and owner validation

Source counts are unchanged: two global plane reads, two inverse FFTs,
the same LDS allocation/accesses/barriers and the original arithmetic.
The global read and twiddle access order changes, so cache locality and
memory scheduling can improve or regress independently of registers.

Owner validation must compare knob=0 and knob=1 under identical legal
plans, with every other campaign option disabled. Start with
1:512:16:512:101 and 1:1K:8:512:202 subject to actual NW/WMUL gates.

- Dense full-word comparisons on RTX 3080 and Radeon VII for NW=4/8,
  WMUL=1/2, supported variants, low/central/high legal exponents,
  quick representations, LL/MUL3, ROE/STATS and INPLACE=0/1/2.
- Inspect generated inverse order, VGPR/register count, live ranges,
  spills, occupancy and trig/address retention. Verify all LDS barriers.
- Alternate carryFused and full-PRP timings; report resources and both
  timings even if a kernel-local gain fails to improve total time.
- Check default/excluded CUDA/Apple/PFA/FFT323161 paths separately.
  A5=1 with FRESH_LDS=1 must keep the original inverse order. Combinations
  with other independent candidates require their own review/validation.
- Reject mismatches, a worse resource peak or end-to-end regression.
  No exactness, occupancy or speedup result is claimed here.

## Resume

Published order now includes Aevum #8 A1, #9 A2, #10 C1, #11 A3,
#12 A4, #13 D1; PrMers #198 P1; then this A5 candidate.
All remain pending owner validation. PR #7 and the owner's FFT323161
CRT/i128 work remain separate.

Next source question: a WMUL=2 carry state stream would need explicit
cross-sub-workgroup staging. It must demonstrate a cost advantage over
A2's existing WMUL=1 stream without merely changing the batch count.
Do not implement another carry_batch1/4 or a generic grid prefix:
the original outgoing carry publication already precedes each wait.
