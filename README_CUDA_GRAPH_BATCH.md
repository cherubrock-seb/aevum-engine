# Experimental CUDA PRP graph batches

**IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED**

This opt-in repeats an unsampled retained-width production PRP square in one
serial CUDA graph. It removes repeated host square preparation and graph
submission. It does not change arithmetic, transform layouts, carry kernels or
the number of GPU kernels executed.

Enable both host options, for example:

```sh
AEVUM_CUDA_GRAPH_SQUARES=1 AEVUM_CUDA_GRAPH_BATCH=32 ./prpll <existing PRP arguments>
```

The same options can be supplied with `-use`. Batch sizes are exactly 1, 2, 4,
8, 16, 32, 64 or 128. Unset or 1 preserves the existing per-square path. A
conflicting environment/config selection or invalid value rejects at engine
construction. Batch sizes above 1 require graph squares to be enabled.

The existing graph restrictions apply: CUDA sm_75+, non-PFA short-carry
Type1/Type2, MULTI_Q=0, and no kernel profiling. RTX 3080 is the initial validation
target. OpenCL, including Radeon VII, has no batch path. Manual FFT selection,
tune-file precedence, device mapping and kernel-cache keys are unchanged.

## Execution invariants

Only the production `isPrimePRP` loop requests repetition. Register API calls,
LL, P-1, ECM, CERT and the tuner retain their current scheduling. In particular,
`timePRP` does not measure this experiment; use actual PRP operation timings.

A batch starts after the current ROE window is full and only while fused-carry
sampling (STATS bit 0) is disabled. Each batch ends strictly before the next
Gerbicz block boundary, proof residue, final residue and requested iteration
limit. Log and check intervals are multiples of the block size. Pending check
updates and stop requests use the ordinary loop. Sampled iterations, short
tails and every observable boundary also use that loop.

One fresh square snapshot supplies all copies. Explicit edges order every
kernel, including the final carry of one square before the next square's middle
input. The retained transform buffer is the output and next input each time.
The host counts every executed kernel for queue pacing.

Graphs are cached by topology and repeat count. Repeated graphs reuse parameters
only when every live argument byte matches the new snapshot. Changed parameters
are installed in every copy. A partial update failure invalidates the snapshot
and prevents submission; a subsequent attempt must refresh all nodes. The
single-square path still refreshes every node unconditionally. There are at
most 16 recorded kernels per square, 128 copies, and 8 cached topologies; one
fixed batch size avoids a family of remainder graphs.

## Validation handoff

No tests or measurements have been executed for this candidate.

1. Run `make test-cuda-graph-batch` and `make test-host`; build CUDA and OpenCL.
   The host recording facade checks argument snapshots, serial edges, repeated
   copies, changed parameters, aborts and partial-update recovery. The policy
   test compares batch admission with visiting each scalar iteration, including
   resume and near-overflow boundaries. Neither test proves GPU exactness.
2. Check invalid values, conflicting env/config values, batch without graph
   squares, profiling, MULTI_Q, unsupported FFT/carry plans, and unset/1 controls.
3. On RTX 3080 compare graphs OFF, graphs ON/batch=1 and batches 2/32/128. Compare
   complete checkpoint/final words, Gerbicz outcomes, ROE/carry diagnostics and
   proofs. Cover INPLACE modes, ROE exhaustion/reset, STATS bit 0, proof points
   within a potential batch, exact block boundaries, nonmultiple iteration
   limits, stop/resume, and repeated reload after a failed check. Include a
   small complete PRP with proof verification and a resumed run. Check OpenCL
   controls and the unaffected register API, LL, P-1 and ECM paths.
4. Use a legal small Type1/Type2 FFT and M136279841 with
   `1:512:8:512:202` as a larger control. Keep the same full profile for A/B.
   A 10,000-iteration production screen is long enough to cross the normal
   2,000-sample ROE window; a 300-iteration run may exercise no repeated graph.
   Confirm actual graph node/launch counts with external tracing. Separate
   graph construction from steady state, but report full-operation wall time.
5. If promising, run at least three counterbalanced paired measurements
   (AB, BA, AB), then full PRP including proof/checkpoint overhead. Require at
   least 3% repeatable end-to-end improvement over graphs ON/batch=1 and compare
   against graphs OFF. Record CPU submission cost, inter-kernel gaps, graph
   memory/setup cost and unchanged kernel register/scratch/occupancy data.

Preserve at least 15 GiB free disk for downstream GPU execution. Reset the batch
option to 1 or unset it to return to per-square submission; disable graph
squares for the ordinary launch path. Do not replay arithmetic after a driver
failure. Large graphs may regress startup cost, driver memory or latency on
long GPU-bound transforms. No speedup or production activation is claimed.
