# Streaming single-line carry state

IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED

Baseline: `71e1b9819a7041c7f83889dd51e36ba2f1b3fa58`.
Source implementation: `805cb8976361c7b697e0d0b0339f041297616afe`.
Independent A2 candidate. No dependency on A1 or directional fusion PR #7.

## Use and scope

Compare the same fixed power-of-two FFT3161 plan with
`WMUL=1,AEVUM_CARRY_STREAM_V1=0` and
`WMUL=1,AEVUM_CARRY_STREAM_V1=1` in the existing `-use` or
`AEVUM_PRP_USE` profile. Default is 0. Disable autotuning.
The source gate also requires AMD/NVIDIA OpenCL C 2.0+, not CUDA.
WMUL>1, PFA, Apple/OpenCL 1.2 and other FFT types retain the original path.

Do not compare candidate WMUL=1 against default WMUL=2 and attribute that
difference to this change. Measure that production-policy comparison
separately only after the same-profile comparison.

## Dataflow and synchronization argument

For WMUL=1 all outgoing carries go to carryShuttle; none is forwarded
through LDS. Each pair's outgoing carry is produced solely by the original
weightAndCarryPairSloppy. Store it immediately and discard the private
scalar before the next pair. Publication of ready remains at the original
location, after every pair and after the original global fence/barrier.

The consumer still waits, fences, clears ready and takes the final-group
barrier exactly as before. Move each shuttle load into the final weighting
loop, immediately before carryFinal. No load moves before the acquire-side
protocol. The store/load count and the global buffer layout are unchanged.
Only the scalar lifetime and instruction scheduling are different.

For ordinary gr, incoming[i] = shuttle[gr-1, me, i].
For the last group gr=H, the original code first loads lane (me-1) mod G_W.
Lane zero then rotates the loaded array by one slot. The streamed address is:

- lane = (me+G_W-1) mod G_W;
- slot = (i+NW-1) mod NW for me=0, otherwise i.

Thus no extra scratch slot or private rotation is needed. Group zero still
publishes its carries and follows the unchanged INPLACE read-completion
protocol before returning. The final group still waits before writing
overlapping input lines.

The i64 carry values, Word2 array, CRT construction, GF61 ranges, LL/MUL3,
weights, statistics and forward FFTs are unchanged. This is not the
previous carry_batch1/carry_batch4 design: workgroup geometry and the
number of handoffs are fixed, and no arithmetic batch size is introduced.

## Observable hypothesis and risks

The active source removes carry[NW+1] and holds one outgoing/incoming
scalar at a time. Earlier outgoing stores shorten the live range across
conversion of later pairs; incoming values no longer all exist before
the reconstruction loop. An optimizing JIT may still hoist loads or
retain stores, so no register reduction is asserted.

On AMD the original array loop can form vector global loads/stores.
Streaming may lose that combining or memory-level parallelism and regress.
On either GPU, added cyclic-address selection or longer load-use stalls
can cost more than the shorter live ranges save.

## Owner validation

1. Build/JIT exact candidate and knob=0 on RTX 3080 and Radeon VII; check
   excluded CUDA/Apple, WMUL=2, PFA and other-type fallback.
2. Dense all-word differential on `1:512:16:512:101` and
   `1:1K:8:512:202`, with low/central/high legal exponents. Exercise
   every NW value, first/last groups, lane zero, last lane and every slot.
3. Include INPLACE=0/1/2 where supported, LL/MUL3, ROE/STATS, repeated
   PRP iterations and restart/checkpoint comparisons. Do not accept a
   RES64-only check as exhaustive coverage.
4. Inspect register/VGPR count, scratch/spills, occupancy, global load/store
   widths, and actual load placement. Verify fences and ready publication
   still bracket all producer stores and consumer loads.
5. Alternate equal-profile kernel and complete PRP measurements. Record
   timing dispersion and both GPUs independently; reject if resources
   are unchanged and throughput offers no reproducible benefit.

## Carry-forwarding redesign assessment

The current algorithm publishes every group's outgoing carry before it
waits for incoming carry. It already uses a bounded two-word repair via
carryFinal: outgoing data is independent of that later repair under the
existing arithmetic contract. There is therefore no serial prefix over
all groups to remove. A scan would add work to an already one-hop
dependency and needs a different justification.

A synchronization-free predecessor-recompute prototype would duplicate
the inverse width transforms and CRT conversion, and would additionally
need an INPLACE-safe input strategy. No such prototype is included here:
without evidence that handoff stalls dominate those costs, it is a weak
next candidate. The useful next evidence is stall/wait attribution from
the same-profile owner run, followed by a design targeting that cost.
