# First-use LDS barrier in FFT3161 carryFused

IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED

Baseline: `71e1b9819a7041c7f83889dd51e36ba2f1b3fa58`.
Source implementation: `8d497577f492228e9ef49d946b059f6018b09825`.
Sequence D1, following A1 (#8), A2 (#9), C1 (#10), A3 (#11), A4 (#12).
This is an independent synchronization-cost hypothesis, not a measured gain.

## Opt-in and scope

Use `AEVUM_CARRY_FRESH_LDS_V1=1` through existing `-use` /
`AEVUM_PRP_USE`; default 0. The active gate is FFT3161, width 512,
NW=8, G_W=64, SHUFL_BYTES_W=8, LDSPAD_W enabled, NOWG2=0,
FAST_BARRIER=0, WMUL=1/2, non-PFA AMD/NVIDIA OpenCL C 2.0+.
CUDA, Apple/OpenCL 1.2, other types/shapes/layouts and default zero
call the original fft_WIDTH1 unchanged.

This deliberately excludes width 1K and PR #7's directional transpose
ownership work. Shared fftbase.cl/fftwidth.cl are unchanged. There is
no FFT323161, CRT/i128, GF61 arithmetic or selector modification.

## Removed cost

The original shufl protocol starts with bar(WG) to keep a new writer
from overwriting data still being read by the preceding LDS user.
The first inverse GF31 transform in this kernel has no preceding LDS
user. Only global input loads and private arithmetic precede it.

The candidate skips exactly that first reuse-barrier call. The first
store-to-read barrier, the second transpose's entry and post-store
barriers, all GF61/forward transforms and all carry barriers remain.
LDS allocation, data traffic, twiddles, butterfly order and ownership
do not change.

On NVIDIA, G_W=64 exceeds the 32-lane wave width and the removed call
normally synchronizes multiple warps. On Radeon VII with wave64 and
FAST_BARRIER=0 it is a local memory fence rather than a workgroup
rendezvous. The benefit may be small or eliminated by the compiler;
this is not presented as a breakthrough without timing evidence.

## Ownership argument

Let m=lowMe in [0,63] and i in [0,7]. The first padded transpose writes
W(i,m)=66*i+m. Rows do not overlap. Its reads use
R(i,m)=8*i+floor(m/8)+66*(m mod 8), which equals
W(m mod 8,8*i+floor(m/8)). Every read therefore refers to a value
written in this same phase.

All work-items retain the original bar(G_W) after those writes. No
ordering with an earlier LDS read/write is required because there is
none. For WMUL=2, each sub-workgroup retains its original disjoint
LDS_BYTES/sizeof(F2) offset. The original F2 bit-transport casts are
also retained.

The next shuffle uses the production entry barrier to protect these
reads. The inverse GF61 phase aliases this same allocation and still
uses its production entry barrier. The special entry point must never
be reused for a later transform or a kernel with earlier LDS activity.

The transform loop has the original s=1,8 sequence, radix and table
calls, unroll hint and final radix. It substitutes only the first
padded shufl32 branch; NOWG2 and layout gates ensure that is the
production branch being replaced. No first-use claim is made for
other shapes.

## Owner validation

- Confirm the gate and actual NW/radix/padding/WMUL on identical plans.
  A knob=1 fallback is not active evidence.
- Dense all-word equality on RTX 3080 and Radeon VII, WMUL=1/2,
  INPLACE=0/1/2, LL/MUL3, legal exponent boundaries and quick GF31/GF61
  values. Include duplicate group zero, last group and every slot.
- Inspect barrier/fence count and placement, LDS mapping, register
  count, spills and occupancy. The expected source delta is one entry
  bar call; no post-store or later barrier may disappear by this edit.
- Alternate kernel and complete PRP timing. Reject unchanged generated
  code, any mismatch or a full-PRP regression.
- Check default and excluded CUDA, Apple, width1K, NW4, unpadded,
  split-LDS and FAST_BARRIER=1 paths separately.

## Resume

All campaign candidates remain awaiting owner GPU validation.
The next independent repository lane is PrMers' legacy carry helper:
replace digit-width pattern dispatch with a straight scalar carry
chain, retaining the same four-digit batch, unsigned operations and
default implementation. This is not carry_batch1/4 and changes no
Aevum carry protocol. Do not copy unvalidated candidates into defaults.
