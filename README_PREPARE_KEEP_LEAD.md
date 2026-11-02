# Experimental retained-state preparation

**IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED**

The register API currently flushes pending arithmetic at every `prepare(dst,
src)`, even when the source already has a prepared transform. In PrMers P-1
stage 1, each set exponent bit can execute `square(state); prepare(temp, base);
mul(state, temp)`. The prepare flush materializes canonical words and breaks
the existing square/prepared-multiply lead bridge.

## Activation and fallback

This experiment is OFF by default. Enable all three existing/new switches:

```sh
export AEVUM_PREPARED_ALIAS_REUSE=1
export AEVUM_PREPARED_MUL_LEAD=1
export AEVUM_PREPARE_KEEP_LEAD=1
```

Supply them as environment variables to the application. The new flag is
captured per engine and accepts only 0 or 1. Enabling it requires alias reuse
and an active prepared-multiply lead bridge. The bridge already requires a
supported non-PFA short-carry plan with register lead caching. Apple rejects
the experiment. CUDA and OpenCL are candidates for RTX 3080, and OpenCL for
Radeon VII; all hardware validation is pending.

Unset `AEVUM_PREPARE_KEEP_LEAD` or set it to 0 for the original preparation
schedule. Keep the other two switches identical in A/B comparisons: otherwise
a result would combine three different mechanisms. The existing cache size,
manual FFT selection, tune-file precedence, device mapping and kernel-cache
keys are unchanged. No automatic tuning/promotion policy is added.

## Why cached preparation can retain the transform

The fast path requires a known prepared source and a pending result register
distinct from both source and destination. It then performs only canonical
register copying and alias/LRU bookkeeping. Neither operation writes the
shared transform scratch or any prepared buffer.

If both registers already belong to the same prepared slot, the canonical
copy is redundant and is omitted. The source still receives its ordinary LRU
touch. A different destination is invalidated and copied normally, then
attached to the cached source slot.

A source miss, or either register touching the pending result, follows the
original flush-before-copy/eviction/transform path. In particular, removing
the last alias of a slot used by a pending multiply does not overwrite that
slot's buffer. Any subsequent cache miss flushes the multiply before it can
acquire or rebuild a prepared buffer. Reads, sync, mutations and other
observable API boundaries retain their existing behavior.

The next compatible arithmetic call can now consume the retained transform
through the existing lead bridge. This removes a canonical carry/writeback
boundary and the next width transform on each eligible chain link, as well
as redundant device copies. It introduces no new arithmetic or GPU buffer.
Actual speedup depends on cache hits, the workload's call sequence and the
existing bridge configuration; no gain is claimed.

## Validation handoff

No builds or tests have been run. After the existing host suite and CUDA/
OpenCL builds, run the new test against each freshly built plugin:

```sh
python3 tests/engine_prepare_keep_lead_gpu_test.py /path/to/libaevum_engine.so --cache 2
```

Repeat with cache sizes 0, 1 and 8 and legal Type1/Type2 plans. The test creates
OFF/ON engines with identical prerequisites, compares complete canonical
words, checks small integer oracles and inspects actual Runtime flush
counters. The eight-bit cached square/prepare/mul chain must have nine
explicit pending flushes OFF and one ON when a prepared slot is available.
These counts establish that the intended path executed, not a speedup.

It also exercises distinct and identical aliases, a new destination while a
square is pending, pending-source/destination hazards, eviction after removal
of a live prepared operand's last alias, source mutation, dense/near-modulus/
zero words, square-minus-two, checkpoint-style read/upload and invalid flags.
Add production P-1 checks with Gerbicz enabled, source-base changes between
chunks, checkpoint/stop/resume, and injected errors. Verify exact residues
and valid diagnostic bounds; carry sampling positions may differ between
canonical and fused paths and must remain in bounds.

For performance, hold FFT, device, backend, cache size and both prerequisite
flags fixed. Compare a short repeated-bit screen and complete P-1 stage 1
(and an actual ECM call pattern if useful), rather than only individual
kernels. If promising, run at least three counterbalanced A/B pairs and
require at least 3% repeatable end-to-end improvement for KEEP. Include
complete PRP/LL controls, where little or no benefit may be expected.

Record full-operation timings, prepare misses, pending flushes, device copies
and transform/carry launches. Kernel source and register/scratch/occupancy
requirements should be unchanged. Keep at least 15 GiB free disk before
downstream GPU execution. Leave the experiment OFF pending acceptance.
