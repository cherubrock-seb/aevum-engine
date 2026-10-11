# NVIDIA carry FFT invariant lifetime experiment

IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED

Baseline: `71e1b9819a7041c7f83889dd51e36ba2f1b3fa58`.
Source implementation: `a7167da7651fc80a96642dd700745ff659ed70bc`.
Independent C1 candidate; no dependency on A1, A2 or directional fusion PR #7.

## Motivation and mechanism

carryFused already documents excessive register retention when the compiler
reuses common expressions across the inverse and forward width transforms.
ZEROHACK_W perturbs LDS and trig pointers by an intended hidden zero to
discourage that reuse. This candidate uses an explicit, bit-preserving
NVIDIA instruction to make the inverse FFT lane index opaque instead.

Each inverse plane calls a helper that returns its u32 lane through a
volatile `mov.b32`. The OpenCL optimizer sees an opaque value rather than
the same SSA input used by the forward FFT. In fft_common, the lane
controls both shufl addresses and tabMul's `p = me & ~(f-1)` table indices.
Those first-pass intermediates can be discarded and recomputed on the
forward pass rather than retained across carry conversion and waiting.
The helper is called separately for GF31 and GF61.

The active path removes the hidden-zero quotient and pointer additions
from the source and adds two scalar identity assembly operations per
work-item. It does not change FFT arithmetic, ownership, LDS traffic,
barriers or any carry operation. This is an experiment about register
lifetime and recomputation, not about faster arithmetic or warp fusion.

## Scope and identity argument

Opt in with `AEVUM_CARRY_FFT_RECOMPUTE_V1=1` through existing `-use` or
`AEVUM_PRP_USE`. Default is 0. Source scope:

- FFT3161, power-of-two width 512 or 1024, WMUL=1/2;
- NVIDIA SM70+, wave32, PTX enabled, OpenCL C 2.0+;
- not CUDA, not PFA;
- ZEROHACK_W=0/1 and BIG_HEIGHT/WMUL < 131072.

The host launches H/WMUL+1 groups, numbered through H/WMUL inclusively.
Consequently ZEROHACK_W*gr/131072 is zero under this gate, even for the
duplicate final group. Removing its pointer offsets leaves the same
addresses. `mov.b32` copies every bit of the original lane, so the FFT's
indices and branch predicates have the same runtime values. There is no
pointer cast, shared-memory reinterpretation or arithmetic approximation.

The assembly is not a memory fence or an execution barrier and does not
replace any existing synchronization. All carry publication, loads, waits,
INPLACE coordination and output stores are unchanged. ZEROHACK_W itself
and all fallback code remain intact.

AMD/Radeon, Apple/OpenCL 1.2, CUDA, NO_ASM, older SMs, other shapes/types,
PFA and unsupported WMUL/hidden-zero configurations take the old path.
The experiment does not alter the math helpers or CRT/i128 work.

## Compiler risks

The PTX/SASS optimizer may recognize the identity and restore the original
common-expression retention. Conversely, obscuring the lane may lose
useful range/divergence information, duplicate too much address/twiddle
work or increase live ranges. Volatile source placement is not a claim
about final hardware scheduling. The mov may survive, coalesce or vanish.
No instruction-count, register-count or occupancy improvement is claimed.

Reject quickly if the emitted code does not reduce the problematic live
state or if the extra recomputation loses more than occupancy gains.

## Owner measurements

1. Build/JIT the exact candidate. Confirm the toolchain accepts the
   volatile identity helper and emits the intended width code.
2. Use a fixed profile with autotuning off. Compare knob=0/1 at both
   ZEROHACK_W=0 and ZEROHACK_W=1, changing one key per pair. The mandatory
   production comparison is against the original default ZEROHACK_W=1,
   not just against the deliberately disabled hidden-zero mechanism.
3. Dense all-word differential on `1:512:16:512:101` and
   `1:1K:8:512:202`, low/central/high legal exponents, NW=4/8,
   WMUL=1/2, LL/MUL3, INPLACE=0/1/2 and ROE/STATS where supported.
   Include cyclic seams, quick GF61 ranges and repeated iterations.
4. Inspect PTX and SASS: actual trig loads/recomputation, address arithmetic,
   mov placement, live registers, spills, occupancy and unchanged barrier
   structure. Mere source-array counts are insufficient.
5. Alternate equal-profile carryFused and full-PRP timing on RTX 3080.
   Check Radeon VII and CUDA fallback equality separately; the candidate
   is not active on those backends.
6. Keep a fixed-plan production result and full resource ledger for the
   exact head. A lower register count without a reproducible PRP benefit
   is insufficient to enable this by default.
