# GF31 staging across the inverse GF61 width transform

IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED

Baseline: `71e1b9819a7041c7f83889dd51e36ba2f1b3fa58`.
Source implementation: `8f3127ffb09720472f86186615662ce2e161bf99`.
Sequence A3. Independent of A1/A2/C1 and PR #7.

## Scope and selection

Opt in through existing `-use` / `AEVUM_PRP_USE` with
`AEVUM_CARRY_STAGE31_V1=1`; default 0. This selects power-of-two FFT3161
carryFused on AMD/NVIDIA OpenCL C 2.0+, width 512/1024 and WMUL=1/2,
only when `WMUL*(LDS_BYTES+WIDTH*8)<=32768`.
CUDA, Apple/OpenCL 1.2, PFA, other types/shapes and larger LDS totals
retain the original source path. No selector or host memory policy changes.

Target fixed plans: `1:512:16:512:101`, `1:512:16:512:202`,
`1:1K:8:512:101` and `1:1K:8:512:202`, subject to the actual radix,
padding and staging gate. Confirm the gate from the JIT defines; an
excluded profile silently uses the original path.

## Distinct hypothesis

The original inverse GF31 result u31[NW] remains live across the complete
inverse GF61 width transform. This experiment explicitly spills that one
plane to dedicated LDS and reads one pair per carry conversion. It targets
the inverse-FFT resource peak, before A1's packed-word interval or A2's
carry exchange. It changes neither GF61 arithmetic representation nor
the carry batching or readiness protocol.

The source removes the need to retain 8*NW bytes of GF31 values across
the GF61 inverse transform. It adds 8*WIDTH*WMUL bytes of declared LDS,
plus NW vector stores and NW vector loads per work-item. No barrier is
added. This is a deliberate register/LDS trade, not a traffic reduction.

## Ownership and lifetime

Let L=G_W*WMUL. Work-item me owns stage31[i*L+me], for i=0..NW-1.
All indices are in [0, WIDTH*WMUL-1]; distinct (i,me) pairs never collide.
Each work-item writes its own slots after fft_WIDTH1(GF31) and reads those
same slots after fft_WIDTH1(GF61). There is no cross-work-item exchange,
so no new inter-work-item fence or barrier is needed. The separate array
cannot overlap lds61/lds31, which remain the original shared FFT scratch.

The local array is volatile to prevent same-item store-to-load forwarding
from simply recreating the original long-lived private values. Each slot
is read into one GF31 scalar before SWAP_XY, whose macro otherwise expands
its operand twice. No volatile qualification is cast away.

Both 32-bit components are stored unchanged, including unreduced/quick
representations. The original weightAndCarryPairSloppy consumes the loaded
pair with identical operands and ordering. The original u31 array is
then reconstructed normally before the forward transform. LL/MUL3,
CARRY64 behavior, INPLACE coordination, statistics and all output indices
remain unchanged.

## Cost and rejection criteria

Actual register allocation is unknown. The compiler may reload earlier
than desired, retain other invariants, or spill more elsewhere. Additional
LDS can lower residency; volatile accesses can add bank conflicts,
waits or traffic that outweigh any register benefit. The 32 KiB source
allocation gate limits scope, but is not a claim about final JIT resources
or a guarantee that the host's local-memory policy accounts for this array.

Reject if the resource ledger shows no useful register/occupancy benefit,
or if total PRP time loses despite a local kernel improvement.

## Owner validation

- Build/JIT exact head and knob=0; record active gate, actual NW, WMUL,
  SHUFL_BYTES_W, LDSPAD_W, all compiler options and actual local memory.
- Dense full-word differential on RTX 3080 and Radeon VII, low/central/high
  legal exponents, quick GF31/GF61 boundaries, every slot/lane and cyclic
  seam, LL/MUL3, ROE/STATS, INPLACE=0/1/2 and NW=4/8 where supported.
- Inspect generated LDS stores/reloads, their placement around the GF61
  inverse transform, VGPR/register count, spills and occupancy. Check
  resources against device limits before long runs.
- Alternate identical-profile carryFused and full-PRP timings; compare
  WMUL=1 and WMUL=2 independently. A knob=1 fallback result is not active
  evidence. Check CUDA/Apple/PFA and the over-32-KiB exclusion separately.
- Preserve raw paired timings and exact SHA. No gain or exactness result
  is asserted by this source-only candidate.
