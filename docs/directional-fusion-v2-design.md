# Directional FFT fusion V2

Status: DESIGN CHECKPOINT — NOT COMPILED — NOT TESTED — NOT BENCHMARKED.

Production baseline: `71e1b9819a7041c7f83889dd51e36ba2f1b3fa58` (current main, v0.3.101 preparation).
Branch: `exp/directional-fusion-v2-deep-opt`.
Previous experiment: `exp/openai-math-directional-fusion`, reference
`b75b9b8bffa9827c5b3fe9c4cf56aff4623bf0c4`, documented in that branch's
`README_DIRECTIONAL_FUSION.md`. No old source is merged or cherry-picked.

## Observations and hypothesis

The supplied V1 results report an RTX 3080 regression and no robust Radeon VII
gain. Source inspection confirms that V1 did not remove LDS reads, writes or
barriers. Its register scheduling was a hypothesis, not an established root
cause of the regression; no new profiler evidence is available here.

Current main still has the four production shufl_and_fft2 overloads, the
original fft8_16a/b tails, and split real/imaginary LDS staging for GF61 with
SHUFL_BYTES=8. The intended first target remains 1:1K:8:512:202. V2 must remove
communication rather than repeat V1's rearrangement of the consumer arithmetic.

Candidate directions, in priority order:

1. Change ownership/staging so a lane retains its own contribution and only
   exchanges the remote portion of a transpose; quantify LDS operations saved.
2. Use an explicitly supported NVIDIA warp shuffle for the within-warp part
   of a transpose, with a compile-time generic/AMD fallback. Do not infer a
   portable OpenCL subgroup size from a work-group size.
3. Evaluate a warp-local final radix-16 schedule only if its output ownership
   can be reconciled without adding another LDS round trip.
4. Reject barrier removal unless every producer/consumer and LDS-reuse edge
   is accounted for. A barrier saving is not required if traffic falls.

Selected candidate: change ownership only between the existing two transposes.
The first transpose reads logical owner P(p), where P swaps bits 4 and 6 of
the physical thread index. The second FFT8 and its twiddle use P(p).
Physical lane xor 16 then supplies the operand owned by P(p) xor 64.
Perform the +/- butterfly in registers **before** the second transpose stores.
That transpose stores one result and reads one result per value, restoring
the original physical/output owner p before fft8_16a/b.

This removes the second LDS operand load without adding an LDS round trip.
Global, carry, twiddle-table storage and external output layouts do not change.
In particular, the final a/b branch stays warp-uniform. A whole-kernel
permutation was rejected because it would complicate carry-ready flags and
make that final branch divergent. A wholly warp-local FFT16 was rejected at
this stage because restoring its output ownership would need more routing.

The first source checkpoint adds guarded ownership/transport helpers only;
the next source commit will wire them into both transposes atomically.
NVIDIA is the first experimental target; AMD remains a compile-time fallback.

## Correctness and scope invariants

- Preserve logical input/output indices, signs, roots, normalization and all
  GF31/GF61 quick representation contracts.
- Prefer bit-preserving transport changes; leave modM61q, optional_add,
  X2q_mul_t4 and delayed rotations untouched unless a separate range argument
  justifies each change.
- No implicit cross-warp synchronization. Full-mask warp instructions require
  uniform participation by complete physical warps and a guaranteed backend.
- Preserve LDS allocation and numWG partitioning unless an explicit derivation
  shows why a change is required and safe.
- New compile-time experiment defaults to disabled. No AUTO/PFA selector,
  release metadata, workflow, PrMers or public ABI changes are planned.

## Initial validation handoff

The owner must later compile both disabled/enabled configurations, compare
complete residue words (including hostile quick-range boundaries), and only
then run fixed-plan alternating A/B pairs on RTX 3080 and Radeon VII. Inspect
registers/VGPR, spills, LDS traffic, barriers, kernel time and full PRP time.
The Radeon comparison initially verifies the unchanged fallback. No manual
compilation, tests, benchmarks, shell commands, SSH or workflow dispatches
will be used for implementation. Automatically triggered CI is separate and
must not be presented as manually performed validation.

## Commit progress

- Initial checkpoint: current production source read; dedicated branch created.
- Source checkpoint 1: guarded owner permutation, bitwise NVIDIA peer transport,
  directional pair operations and padded read mapping. Helpers are not wired
  into the FFT yet.
- Source status: IMPLEMENTED — NOT COMPILED — NOT TESTED — NOT BENCHMARKED.
- Cost table and owner instructions follow with the integration checkpoint.
  No correctness or performance success is claimed.
