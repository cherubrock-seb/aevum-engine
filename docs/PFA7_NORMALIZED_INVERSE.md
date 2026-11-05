# PFA7 inverse normalization inside NUSS8

**IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED**

The current inverse Nussbaumer transform computes eight scalar products per
component and then scales seven outputs by inverse7. This candidate folds
inverse7 into the eight constants and scales input[0] once: 15 products become
9. It adds no twiddle table, memory transfer, barrier or kernel launch.

## Activation and scope

Default OFF. Select the legal Type1 plan `pfa7:1:512:7:512:202` and hold
the Nussbaumer implementation fixed:

- Baseline: `-use AEVUM_PFA7_NUSS8=1,AEVUM_PFA7_FOLD_INV7=0`.
- Candidate: `-use AEVUM_PFA7_NUSS8=1,AEVUM_PFA7_FOLD_INV7=1`.

Omitted/0 retains the existing path. The existing NUSS8 host geometry checks
remain authoritative. This key uses the normal compilation configuration and
kernel cache; there is no new environment-variable alias. Manual plan
selection, tune-file precedence, capacity and production defaults are unchanged.

AMD uses the new GF31 and GF61 inverse helpers. NVIDIA uses the GF31 helper
but retains its existing GF61 prescaled-row path. The prescale marker has
priority over the new dispatch, so normalization occurs exactly once.
Forward transforms and PFA3/PFA9 are unchanged.

Explicit selection rejects other FFT types/radices, NUSS8=0, COSBASE, limb32
and Apple staging. Combination with the separate AEVUM_PFA_LAZY61 experiment
also rejects. Do not revive or combine rejected candidates for this screen.

Primary targets are Radeon VII/gfx906 OpenCL and RTX 3080 OpenCL/CUDA.
RX 7900 XTX/gfx1100 remains an additional target with hardware validation
pending; no architecture-specific default or performance claim is introduced.

## Algebra and ranges

Let p be M31 or M61. The original NUSS8 graph contains

```text
total = a1 + a2 + a3
center = x0 + C1*total
C1 = -1/6 mod p
```

Each of its eight inverse-direction constants K is replaced by
K*7^-1 mod p; the center uses x0*7^-1. Consequently all six nonzero-frequency
outputs are already normalized, since the remaining graph only adds and
subtracts these scaled terms.

The zero-frequency output follows from

```text
scaled_center = (x0 - total/6)/7
x0 - 6*scaled_center = (x0 + total)/7
```

This reconstructs the seventh output without another general product.
All constants are derived from Aevum's existing inverse roots and coefficient
ordering. The direct host DFT oracle uses those field roots independently
of this factorization.

GF31 retains [0,M31] using existing reduced additions, subtraction and
multiplication. Reconstructing six times center uses three reduced additions
and one reduced subtraction.

GF61 retains [0,M61+7] in the normal arithmetic. Only the zero-frequency
reconstruction temporarily uses unreduced values. For M=M61:

```text
0 <= 6*center <= 6M+42 < 2^64
M-42 <= x0 - 6*center + 7M <= 8M+7 = 2^64-1
```

Unsigned subtraction may wrap temporarily, but adding 7M produces the exact
nonnegative pre-fold integer within 64 bits. One modM61 fold restores the
original output range. All general products still use reduced operands and
constants less than the modulus.

The new helper bypasses the old seven-output scaling loop. A row already
marked with AEVUM_PFA7_MIDOUT_TWIDDLE_PRESCALE keeps the original unnormalized
DFT, preventing a second inverse7. CRT, carry and buffer ordering are unchanged.

## Evidence used to select this candidate

The baseline is main `118309fef172c26475f83fe51cdfa3015bb87d8a`, including
the accepted gfx906 H512 reverse16 implementation from #260.

Earlier PFA7 runtime twiddle fusion (#23) was exact but rejected for
performance. This candidate rescales fixed coefficients instead of adding a
normalization dependency on the common runtime twiddle before seven complex
multiplications. It also uses the later eight-product NUSS8 graph and the
zero-frequency identity above.

Cherub rejected #262 after current-main exactness passed: the gfx906 PFA7
screen improved only 0.733841%, and PFA3 regressed 0.466102%. That GF61
reduction-only candidate is not refreshed here. This candidate targets general
products in both fields. The previous result is evidence against assuming
that arithmetic counts predict full-operation speedups.

Larger coefficient values, longer x0 lifetime, extra GF31 additions and changed
scheduling may increase registers or scratch and erase the saving. NVIDIA
GF61 has no claimed gain because its current optimization remains selected.
There is no measured speedup or claim of PRPLLNTT parity.

## Pending validation

No compilation, preprocessing test, host test, GPU test or benchmark was run.

1. Run `make test-pfa7-inverse-normalized` with GCC or Clang.
   The C++ test includes the actual helper and checks both fields against
   an independent uint128 normalized inverse DFT, output ranges, normalized
   round trips, basis/sloppy/corner/random inputs and the nine-product count.
   Its primitive reductions model the source arithmetic; device instruction
   lowering remains unverified.
   The source gate preprocesses the actual option policy and checks
   inverse-only dispatch plus NVIDIA GF61 prescale priority.
2. Run the existing NUSS8 model and native PFA reference/source gates; build
   OpenCL and CUDA OFF/ON. Check flag=2/nonnumeric, radix=0/3/9, other type,
   NUSS8=0, COSBASE, limb32, Apple and LAZY61 combinations reject.
   Omitted/0 must reproduce the stock path; verify actual emitted definitions.
3. Require at least 15 GiB free before GPU execution. Screen M136279841 and
   M125000003 on the fixed PFA7 plan, with matching baseline/candidate settings,
   Radeon VII OpenCL and RTX 3080 CUDA/OpenCL. Compare complete canonical
   residues for square, factor-3, prepared multiply, carry-heavy/boundary
   inputs, available layouts and canonical/retained transitions.
4. If exact, exercise full PRP/Gerbicz, LL, P-1, ECM, proof verification and
   checkpoint/resume on affected operations, including independent prepared
   operands and multiple engine instances. Do not accept RES64 alone.
5. Only if promising, run at least three counterbalanced paired A/B
   measurements (AB, BA, AB). Fix exponent, plan, register count, backend,
   power/clocks and warmed cache state. Measure complete PRP wall time.
   Separately capture middle-out timings, registers, scratch, LDS and occupancy.
   Require at least 3% repeatable end-to-end improvement for performance KEEP.
6. Compare with best legal AUTO and PRPLLNTT only under equivalent hardware,
   workload and proof/check policies. The supplied gfx1100 quick-pass report
   has noisy square timings and inconsistent production estimates: use warmed
   repeated probes as a screen, not as proof of performance parity.

Keep Draft and default OFF until independently accepted. Do not modify the
embedded PrMers engine as a second implementation of this experiment.
