# Directional FFT fusion V2

**IMPLEMENTED — NOT COMPILED — NOT TESTED — NOT BENCHMARKED**

Baseline: `71e1b9819a7041c7f83889dd51e36ba2f1b3fa58` (main at branch creation, v0.3.101 preparation).
Branch: `exp/directional-fusion-v2-deep-opt`.
Draft PR: https://github.com/cherubrock-seb/aevum-engine/pull/7.
V1 reference: `b75b9b8bffa9827c5b3fe9c4cf56aff4623bf0c4` on `exp/openai-math-directional-fusion`.
No V1 implementation is merged or cherry-picked. This branch starts from current production source.

## Why change the dataflow

The owner's V1 results report an RTX 3080 regression and no robust Radeon VII gain.
V1 consumed the first layer of fft8_16a/b while reading the second transpose,
but retained the same LDS reads, writes and barriers. Its claimed opportunity
was register scheduling. The compiler could already schedule that way; added
unrolling/inlining could instead increase register pressure or issue cost.
These are hypotheses, not a profiler diagnosis of the supplied regression.

V2 moves the radix-2 operation to the **producer side** of that transpose.
The two contributing logical owners originally differed by 64, outside a
NVIDIA warp. A temporary ownership permutation makes them lane-xor-16 peers.
The first transpose establishes that ownership; the second restores the
original owner. The register butterfly therefore replaces a redundant LDS
operand load without introducing a third transpose.

## Implemented scope and controls

- `AEVUM_DIRECTIONAL_FUSION_V2` defaults to **0**; only 0 and 1 are accepted.
- Standalone: `-use AEVUM_DIRECTIONAL_FUSION_V2=1`.
- Engine PRP: `AEVUM_PRP_USE=AEVUM_DIRECTIONAL_FUSION_V2=1`, through the existing PRP profile interface.
- Set `AEVUM_RADIX1K=8` explicitly for a reproducible experiment. V2 does not alter radix policy.
- Enabled only for width, `WG=128`, `RADIX=8`, width variant 2, NVIDIA,
  `HAS_PTX>=700`, `WAVEFRONT=32`, `SHUFL_BYTES_W=8` or 16, `LDSPAD_W!=0`,
  `LDSSWIZ_W=0`, `NOWG2=0`, `NOLDS2=0`, and non-PFA plans.
- FP64, FP32, GF31 and GF61 have typed producer implementations.
- AMD (including Radeon VII), other vendors, older NVIDIA architectures,
  `NO_ASM`, height transforms, other shapes/variants and excluded layouts
  take the production path. Requesting 1 is not proof that the gate activated.
- No selector, AUTO/PFA policy, default tuning, release, workflow, ABI or PrMers change.

The default `1:1K:8:512:202` target exercises both GF31 and GF61 width code.
The new main baseline's `LDSPAD_H=0` policy for this shape is preserved.
For the 16-byte experiment use `WMUL=1`: the existing host policy disables
width padding at the 32 KiB threshold (16-byte staging with WMUL=2), which
would intentionally disable V2. No existing memory-limit rule is bypassed.

## Ownership and index derivation

Let p be the physical index within an aligned 128-thread FFT partition.
Define `P(p) = (p & ~80u) | ((p & 16u) << 2) | ((p & 64u) >> 2)`.
This swaps bits 4 and 6, preserves all other bits, and is an involution.
Thus `P(p xor 16) = P(p) xor 64`; the two owners are in one physical warp.

| Stage | Logical owner at physical thread p | Communication |
| --- | --- | --- |
| Initial FFT8 and first twiddle | p | Existing registers |
| First transpose | P(p) on read; stores still use p | Existing LDS store/read counts |
| Second FFT8 and second twiddle | P(p) | Twiddle index also uses P(p) |
| Radix-2 producer | P(p), paired with P(p) xor 64 | Bit-preserving warp exchange |
| Second transpose and final fft8_16a/b | p restored on read | One LDS result read per value |

Call the post-second-twiddle value at logical owner t and register j `U(t,j)`.
For output owner p and output slot i, set `r=p&7`, `j=(p>>3)&7`,
`d=p>>6`, `a=8*i+r` and `b=a+64`. The production second transpose forms
`U(a,j)+U(b,j)` for d=0 and `U(a,j)-U(b,j)` for d=1.

V2 computes those operations before the store. The lower logical owner stores
`addq(own,peer)`; the upper logical owner stores `subq(peer,own)`.
The exchange must occur before the directional selection. Both components
of every value use the same owner and preserve operand order.

For eight-byte elements the new store is `lds[136*j+t]`. Reading
`lds[136*((p/8)&7)+(p&7)+8*i+64*(p/64)]` selects owner `a+64*d`
and register j. For sixteen-byte GF61/FP64 elements the store is
`lds[8*j+8*(t&~7)+(t&7)]`; the read is
`lds[512*d+64*i+(p%64)]`, which selects exactly the same owner/register.
GF31/FP32 continue using eight-byte pair elements even with SHUFL_BYTES=16.

The padded first-transpose reads substitute P(p) for p in both components;
its writes are unchanged. Consequently each second FFT8 receives precisely
the production eight-value tuple for P(p). Only the second twiddle's index
changes, matching that tuple. No roots or twiddle table contents change.
The final a/b branch still uses physical p and remains uniform within a warp.

## Arithmetic and quick-range contracts

GF61 transports the original unsigned 64-bit pattern as two 32-bit words.
`addq` and `subq` are the same unreduced unsigned operations as production,
with the same operands and order. V2 does not replace an unsigned wrapped
difference with a signed C expression or normalize it early.

- At fft8_16a entry, sums retain the production `0..2*M61+epsilon` range.
- At fft8_16b entry, differences retain the unsigned representation of
  `-M61-epsilon..M61+epsilon`.
- The next original upper-half X2q/X2q_mul_t4 layer still has sums up to
  `4*M61+epsilon` and differences around `-2*M61..2*M61`.
- The original lower X2qt4 layer retains its `-2*M61..2*M61` range.
- Every `modM61q(...,0/3)`, `optional_add(...,2*M61)`, `mul_t8q(...,5)`,
  `mul_3t8q(...,3)`, nontrivial root and final swap in fft8.cl is unchanged.

GF31 uses its existing **reduced** scalar addq/subq overloads; it does not
inherit the GF61 quick arithmetic. FP32/FP64 use bitcasts for transport and
the same ordered +/-; their existing butterfly and FMA sequences are intact.
These are source equivalence arguments, not a claim of compiled exactness.

## Synchronization, storage and fallback

`shfl.sync.bfly.b32` uses a full mask, xor distance 16 and clamp 31.
Every current width call runs a whole 128-thread partition; each physical
warp executes all component exchanges before the +/- selection. No peer
crosses a warp or a numWG partition. Inline assembly is volatile so that
the exchange is an executed operation, not a removable pure expression.
Unsupported backends do not compile this assembly.

No barrier is removed or replaced by a warp shuffle. The entry barrier
protects the previous LDS consumer; the next makes producer stores visible.
Split GF61/FP64 staging retains the intervening barrier before imaginary
writes and their following producer barrier. Real reads overwrite only x;
the original y components remain available for their later peer exchange.
The next LDS user retains the pre-existing entry-barrier contract.

`LDS_BYTES` and `(get_local_id(0)/WG)*LDS_BYTES` partition offsets are unchanged.
The padded eight-byte stage reaches element 1079, within 1080 elements.
The sixteen-byte stage reaches 1023; its first-transpose padded allocation
contains 1031 elements. Eight-byte pair accesses in a sixteen-byte allocation
also fit. Every producer store has a unique (owner,register) address.

The original four `shufl_and_fft2` bodies are unchanged. The four eligible
tabulated FFT spans select the new producer explicitly; the old variant-0
caller and all disabled configurations retain the old second-stage helper.
The temporary f=1 read ownership is confined to the same compile-time gate.
Kernel local/global IDs, carry-ready flags, numWG, global I/O layouts and
carry shuffles remain unchanged.

## Static cost ledger

Counts are source operations per thread for one 1K transform of one numeric
representation. A complex equivalent is two components; scalar staging
doubles the element counts. These are not SASS instructions, transactions,
bank-conflict counts or measured timings.

| Resource | Main baseline | V1 | V2 active |
| --- | ---: | ---: | ---: |
| First transpose complex writes / reads | 8 / 8 | 8 / 8 | 8 / 8 |
| Second transpose complex writes / reads | 8 / 16 | 8 / 16 | 8 / 8 |
| Total transpose traffic, complex equivalents | 40 | 40 | 32 |
| GF61/FP64 traffic across both transposes | 640 B | 640 B | 512 B |
| GF31/FP32 traffic across both transposes | 320 B | 320 B | 256 B |
| Second-stage barriers, split 64-bit components | 4 | 4 | 4 |
| Second-stage barriers, vector staging | 2 | 2 | 2 |
| Added b32 peer shuffles, GF61/FP64 | 0 | 0 | 32 |
| Added b32 peer shuffles, GF31/FP32 | 0 | 0 | 16 |
| Padded LDS allocation, SHUFL_BYTES=8 / 16 | 8640 / 16496 B | same | same |
| Root multiplications / reductions | production | same graph | same graph |
| Additional private eight-value array | none | none | none |
| Registers, spills, occupancy, kernel/PRP time | not measured here | supplied qualitative results only | unknown |

For the paired GF31+GF61 target, the two transposes therefore move 960 ->
768 source-level bytes per thread: **20% less total transpose traffic**,
or **50% fewer reads in the second transpose**. This is not a 20% PRP gain.
There is no LDS-allocation or barrier saving. V1 has no source traffic saving.

Added costs are owner-bit manipulation, warp exchange instructions, and
direction-dependent +/- inside each producer warp. The compiler may issue
both arithmetic paths under predicates. First-transpose bank behavior and
twiddle access grouping change; address ALU cost may change too. Source
liveness needs only scalar peers plus result components, but unrolling and
register allocation can still increase registers/spills. All need profiling.

## Alternatives considered

- Whole-kernel owner permutation: rejected because it changes carry wave
  membership and makes the expensive final a/b branch divergent.
- Entirely warp-local final FFT16: deferred because restoring standard
  output ownership could require an extra cross-warp/LDS exchange.
- Folding more consumer arithmetic into loads: repeats V1 without removing
  the duplicate LDS operand. Not the V2 mechanism.
- Deleting entry or real/imag barriers: rejected without an ownership proof
  covering every previous reader and subsequent writer.
- AMD shuffle assumptions: rejected. Radeon keeps the proven production
  implementation; a separate measured AMD design would be a later task.

## Handoff and published checkpoints

See [directional-fusion-v2-validation.md](directional-fusion-v2-validation.md)
for owner-run commands, the hardware matrix, acceptance criteria and known gaps.

1. `ae7dc9bf14440dc8d94a3b366997a5264d5d4ccd` — docs: identify directional fusion v2 performance bottlenecks.
2. `656166a2916c57376e5461c69901f9b088abf800` — perf(opencl): add guarded warp ownership primitives for fusion v2.
3. `8834201c0bce2b2b0442fe9ade570ac7b4dbe3da` — perf(opencl): fuse radix-2 into warp producer before LDS transpose.
4. `4e79cf27a09e3249502513cac23344627199a695` — test(opencl): add same-plan directional v2 profile comparison.
5. `7754286ef15c76881dc3e1c47741d0999f660e4b` — refactor(opencl): keep legacy shuffle entry points intact.

The documentation completion commit follows this list; the PR head is the
authoritative final revision. All source stages have the status at the top.
After reading the request, implementation used GitHub read/tree/commit/ref/PR
operations only. No compiler, test, benchmark, SSH or Actions dispatch was run.

The connector's commit API exposes no author override. Git objects read back
report `cherubrock <iofabul@gmail.com>`, linked to GitHub account
`cherubrock-seb`; the requested `cherubrock-seb` author display name could
not be set through that API. No identity or repository config was rewritten.
