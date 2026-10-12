# WMUL=2 carry streaming through existing LDS

IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED

Baseline Aevum: `71e1b9819a7041c7f83889dd51e36ba2f1b3fa58`.
Source implementation: `87c0f60678041672af0fe8fbafea9e81807991bd`.
Sequence A6, following Aevum #8–#14 and PrMers P1 #198.
This independent branch does not include the other candidates.

## Opt-in and distinct scope

Use `WMUL=2,AEVUM_CARRY_LDS_STREAM_V1=1` through existing `-use` /
`AEVUM_PRP_USE`. The experiment defaults to 0 and does not change WMUL.
Compare with the same WMUL=2 profile and knob=0.

The active gate requires FFT3161, non-PFA AMD/NVIDIA OpenCL C 2.0+,
width 512/1024, NW=4/8, SHUFL_BYTES_W=8/16 and
LDS_BYTES>=WIDTH*8. CUDA, Apple/OpenCL 1.2, WMUL!=2, split 4-byte
shuffle storage and other shapes/types retain the original path.

A2 (#9) streams WMUL=1, where no cross-line LDS handoff exists. A6
keeps the two-line workgroup and must explicitly stage first-line
carries for the second line. It is not another batch-count experiment.
Unlike A3, it allocates no additional LDS.

## Dataflow and ownership

Let L=G_W and S=LDS_BYTES/sizeof(i64). There are two sub-workgroups:
first-line items have me=m, second-line items have me=L+m, m in [0,L).
The existing shufl_carries_up implementation writes the first line's
carry i to the absolute LDS index S+i*L+m. The candidate uses that same
index, now immediately after weightAndCarryPairSloppy produces it.

The second line reads S+i*L+m directly at carryFinal. Each (i,m) has one
producer and one consumer. The gate S>=WIDTH=NW*L keeps every slot inside
the existing second LDS region. No index overlaps another carry slot.

Second-line outgoing carries go straight to the existing global shuttle
addresses while being produced. The first line loads predecessor carries
only at carryFinal. The private carry[NW+1] array and its last-group
rotation are absent from the active source.

## Synchronization proof obligations

The original easy WMUL=2 shuffle performs two full workgroup barriers:
one before overwriting FFT scratch and one after its carry stores.

The candidate relocates the first of these before the CRT/production
loop. That protects the inverse FFT's last LDS reads before any streamed
store. Its condition gr!=0 is workgroup-uniform. Group zero never stores
local carries, retains its original outgoing global publication and
read-completion signal, then returns at the original point.

The second full barrier remains after carry production/publication and
the group-zero return, before any staged load. It makes the first line's
writes visible to the second line, including across wavefronts.

No intervening code uses LDS: the statistics helper reduces directly
into global memory; ready publication/fences also use global memory.
Global readiness is still published only after every outgoing slot has
been produced. All original ready stores, waits, clears, global memory
fences, OLD_FENCE handling and last-group synchronization remain.

Later forward FFTs retain their entry reuse barriers. The second-line
readers own that second LDS region, so the existing per-sub-workgroup
reuse protocol protects them before it is overwritten. The first
forward transform's first region does not contain these staged carries.

This is a source dependency argument. Device exactness and generated
barrier placement remain untested.

## Cyclic seam

For the first line of a normal group, incoming slot i is read from
(gr-1)*WIDTH+CarryShuttleAccess(me,i), unchanged.

For the duplicate last group, sourceLane=(me+G_W-1)%G_W. If me!=0 the
slot is i. For me==0 the slot is (i+NW-1)%NW, expressing the original
array rotation directly. The second line uses its same-group staged
carry, including in the duplicate last group; it does not rotate.

## Source resource ledger

| Cost | Baseline WMUL=2 | Candidate WMUL=2 |
| --- | --- | --- |
| Private carry storage | i64 carry[NW+1] | scalar outgoing/incoming in separate phases |
| LDS allocation | existing FFT scratch | same allocation |
| First-line local stores / second-line loads | NW per participating item | same count and slots |
| Global outgoing stores / incoming loads | NW per participating item | same count and addresses |
| Local handoff barriers for gr!=0 | 2 | 2, first moved earlier |
| Local handoff for gr==0 | none | none |

The declared array occupies 8*(NW+1) bytes, but removal is not a measured
register saving. The compiler may already shorten its live ranges or
may reschedule scalar loads into an equivalent block.

Main risks: loss of AMD vector load/store combining, reduced global
memory-level parallelism, LDS instructions interleaved with CRT work,
and an earlier workgroup rendezvous. Address/predicate overhead can
outweigh any register or occupancy improvement.

## Owner validation

- Confirm actual active gate, exact SHA, NW, WMUL, padding, shuffle bytes,
  variant and compiler flags. Use identical WMUL=2 in both profiles.
- Dense full-word equality on RTX 3080 and Radeon VII across both widths,
  NW=4/8, all lanes/slots, both sub-workgroups, group zero, ordinary groups
  and the duplicate last group; include LL/MUL3, ROE/STATS, INPLACE=0/1/2,
  legal exponent boundaries and quick residue representations.
- Inspect the relocated full barrier, producer/consumer LDS addresses,
  final-FFT reuse barriers, ready ordering, registers/spills/occupancy and
  actual global/local load/store widths.
- Alternate carryFused and full-PRP timing; keep raw paired timings.
  Reject dense mismatches, synchronization faults or end-to-end losses.
- Verify knob=0 and excluded WMUL1, SHUFL4, CUDA, Apple, PFA and other
  FFT types. No candidate combination or GPU result is asserted.

## Resume

Published sequence: A1 #8, A2 #9, C1 #10, A3 #11, A4 #12, D1 #13,
PrMers P1 #198, A5 #14, then A6. Every candidate is opt-in and pending
owner validation. Do not repeat these implementations on resume.

PrMers' two-value width specialization is still only a source question:
a separate candidate needs a concrete cost case for constant shifts
versus added selects. A generic global carry prefix remains unjustified
because outgoing carries are already published before incoming waits.
Keep PR #7 and the owner's FFT323161 CRT/i128 work separate.
