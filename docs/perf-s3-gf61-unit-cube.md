IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED

# S3: unit-norm cube for chained GF61 width twiddles

Baseline: 9db67c504bbd5877b4af0e44a538e1f0ca65e5f6.
Enable with `-use TABMUL_CHAIN61=1,AEVUM_GF61_UNIT_CUBE_V1=1`,
or the corresponding existing AEVUM_PRP_USE profile.
The new option defaults to zero. TABMUL_CHAIN61 also remains at its
existing default; this experiment does not select chained twiddles.

Scope: FFT3161 width512/1K, radix4/8, native GF61 on AMD/NVIDIA OpenCL
C 2.0+, non-PFA. Height transforms, other FFT types, CUDA, limb32 and
the alternate cmul implementation retain the original path.

## Mechanism and source cost

chainMul4 and chainMul8 already form w^2 with csq(w), then form w^3
using generic cmul(w^2,w). Replace only that cube with the existing
ccubeTrig(w^2,w) helper in math.cl.

The generic native cmul performs one mul64 and two weakMulAdd calls:
three 64x64-to-128 products and two 128-bit accumulations. ccubeTrig
uses two weakMul calls and 64-bit additions. Thus one wide product
and two wide accumulations disappear per chained twiddle cube.
Both paths end with two modM61 folds.

The later radix-8 recurrence, all data multiplications, w^2 generation,
table reads, shuffles, barriers, weighting, CRT and carry remain unchanged.
No helper implementation, extra array, traffic or kernel is added.
Actual instruction count, registers and timing are unverified.

Coverage is conditional: TABMUL_CHAIN61=0 cannot benefit. Compare OFF
and ON with TABMUL_CHAIN61=1 in both profiles. Separately compare any
winning chained profile with the owner's production profile; beating
a forced chained baseline alone is not sufficient for production use.

## Why the width twiddle has unit norm

TrigBufCache.cpp::genSmallTrigGF61 constructs powers of
GF61::root_one(size), with size=512 or 1024 in this gate. These are
roots z with z^size=1 in F_(M^2), M=2^61-1.

Since size divides M+1, z^M=z^(-1). Frobenius conjugation in this
extension sends a+i*b to a-i*b, so z*z^M=a^2+b^2=1 modulo M.
This applies to every column and stage twiddle loaded by tabMul.
No assertion about arbitrary data values is used.

Write w=a+i*b and s=w^2, so s.y=2ab. With t=2*s.y:

    w^3.real = a - t*b
    w^3.imag = t*a - b

because a^2+b^2=1. These are exactly the two products expressed by
ccubeTrig. This nontrivial field identity removes arithmetic that a
generic multiply cannot infer from the table value.

## Sloppy ranges

Host table components are canonical in [0,M-1]. csq(w) returns
components in [0,M+7]; therefore t is in [0,2M+14].
The existing helper uses 2M-b for -b, in [M+1,2M].

For its first weak product, t*(2M-b)<5M^2<2^125; after its weak fold
and addition of a, the result is below 7M and fits u64.
For the second, t*a<3M^2; the weak fold plus 2M-b stays below 6M.
The helper's final folds return the same [0,M+7] admissible range as
the original cmul. No new range reaches later butterflies.

The two paths may choose different noncanonical representatives of the
same field value. Intermediate bitwise equality is not required, but
canonical field equality and complete integer/residue equality are.
The derivation is static reasoning, not GPU validation.

## Owner gates

1. Confirm the active gate and TABMUL_CHAIN61=1 in actual FFT3161 width
   kernels. Cover radix4/8 and width512/1K where the plan is legal.
2. Generate every actual width-table twiddle using the existing host
   generator. Check unit norm and compare canonical w^3 from the
   original cmul and ccubeTrig against independent field arithmetic.
   Include w=1, -1, i and -i; inspect zero/M representatives.
3. Compare chained data outputs modulo M for hostile valid GF61
   inputs, then compare every final integer/residue word and statistics
   in full iterations. Cover LL/MUL3, INPLACE and WMUL variants.
4. Check option-off, chain-off and excluded type/height/backend paths
   retain the original artifacts.
5. Inspect wide product/accumulation instructions, registers, spills and
   occupancy. After correctness, alternate width/carryFused and full
   iteration timings on RTX 3080/Radeon VII with identical profiles.
   Reject no-codegen-effect or regressions.

No build, compile, execution, benchmark, SSH or Actions dispatch was
performed. Nothing changes AUTO selection or existing defaults.

## Continuity

S2 is independently published in PrMers #204. F1/F2 and K1 remain
pending. P2 has owner KEEP evidence pending clean production integration.
Closed A1/A2/C1/A3/A4/D1/A5/A6/S1/P1 and directional V2 are not repeated.
This uses a pre-existing field-identity helper to remove a concrete
wide product, not opacity, lifetime scheduling or generic limb32 math.

Next lead: FFT3161 CRT reconstruction fused with immediate word/carry
extraction, but only if an entire wide operation can be removed with
a complete range proof. Do not broaden pending twiddle candidates.
