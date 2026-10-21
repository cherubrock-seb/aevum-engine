# PFA7 native carry: staged PRPLL parity investigation

**Experimental branch only — no AUTO change.**
Comparison target: PRPLL-NTT `0dec9d4`, 4070 Ti SUPER, M136279841,
`pfa7:1:512:7:512:202`. The Kaggle Tesla T4 is NOT the target GPU;
it can prove arithmetic and screen candidates, not establish 4070 Ti S parity.

## Why copying PRPLL's carryFused is not sufficient

PRPLL's `carryfused.cl` implements PFA-aware carry scheduling through
`pfaPair`, `pfaCombo` and rotated PFA carry lines. Aevum's Good-Thomas
implementation currently scatters inverse width FFT values into *canonical*
word pairs before `carryA` / `carryB`, and its width/gather row layout is
different from PRPLL's native pair-index layout. Reusing the PRPLL kernel
without adapting **both representations** could silently corrupt residues.

To reach PRPLL's mechanism a future stage must introduce a PFA-native
representation in the complete carry boundary, including weights,
PFA carry predecessor mapping, `fftP`/lead width input/output, mixed
GF31/GF61 CRT, ROE/proof canonicalization and CUDA graph resource lifetime.
This stage requires dense word-for-word differentials (squares, mul,
prepared mul, repeated residues), then production-class PRP checkpoints.

## Executable stage 1: speed up the existing canonical PFA7 carryB

`AEVUM_PFA7_CARRYB_FAST` accepts:

- `0`: unmodified canonical `carryB`
- `1`: algebraically identical predecessor lookup, no division or modulo
- `2`: same lookup plus experimental zero-carry early exit

Modes 1 and 2 only specialize NVIDIA FFT3161 PFA7 `1:512:7:512:202`.
Modes 0 and Type2 remain unchanged. Nothing is enabled by default.

Original predecessor address (per width-column `x`, carry-group `gy`):

```
HB = BIG_HEIGHT / CARRY_LEN
prev = (gy + HB*x - 1) mod (HB*WIDTH)
prevLine = prev mod HB
prevCol = prev / HB
```

Equivalent branch-only form:

```
prevLine = (gy == 0 ? HB-1 : gy-1)
prevCol  = (gy == 0 ? (x == 0 ? WIDTH-1 : x-1) : x)
```

The exhaustive host model verifies all **229,376** carry-group addresses
for the exact PFA7 geometry. Mode 2's early-exit arithmetic remains
**unvalidated until the GPU word-exact differential passes**.

## Required Kaggle gates

1. Compile CUDA + OpenCL separately and verify the exact source commit.
2. Per mode/process set `AEVUM_PFA7_LEAD_MODE=1` and the appropriate
   `AEVUM_PFA7_CARRYB_FAST=0/1/2`.
3. Compare a dense random input and 7 chained `square_mul` operations
   against independent Type2 (`AEVUM_EXPERIMENT_FFT3261=1`).
4. Benchmark only exact PASS candidates, short (~192 square operations),
   alternate control and candidate timings, retain logs and JSON evidence.
5. Do not merge based on a single T4 run. Reproduce on RTX 4070 Ti SUPER
   and compare to the PRPLL 3,025–3,080 iter/s baseline.

**Claim boundary:** this experiment improves canonical carryB indexing.
It is *not* PRPLL's native `carryFused`, and cannot plausibly explain or
erase the entire PRPLL vs PrMers gap by itself. Stage 2 targets the native
cross-workgroup PFA carry fusion after the layout model and exactness gates.
