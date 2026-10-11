# Directional fusion V2: owner validation handoff

**IMPLEMENTED — NOT COMPILED — NOT TESTED — NOT BENCHMARKED**

All commands below are proposed for the GPU owner. None were executed in the
implementation session. The design and static counts are in
[directional-fusion-v2-design.md](directional-fusion-v2-design.md).

## Build and identify the actual path

Use a separate checkout and empty run directories. Preserve main and existing
production work directories. Select device ordinals from the current host's
enumeration, record the driver/backend, and retain all build and run logs.

```bash
git fetch origin exp/directional-fusion-v2-deep-opt
git worktree add --detach ../aevum-directional-v2-review origin/exp/directional-fusion-v2-deep-opt
cd ../aevum-directional-v2-review
git rev-parse HEAD
make -j2 aevum engine-lib
mkdir -p build-tests
c++ -O2 -std=c++20 tests/native_pfa_dense_compare.cpp -ldl -o build-tests/native-pfa-dense-compare
# Separately, for NVIDIA's standalone CUDA backend:
make -j2 CUDA=1 aevum
```

`engine-lib` above is the OpenCL library. The CUDA standalone binary is a
separate backend comparison. Kernel compilation occurs at runtime; a host
build alone does not validate the inline PTX, any enabled kernel or its results.

Before recording enabled-path coverage, inspect the actual kernel defines
and generated code for all conditions in `AEVUM_DIRECTIONAL_V2_ACTIVE`.
Expect the xor-16 b32 warp exchanges in enabled width kernels and their
absence in the baseline and AMD fallback. Confirm radix 8, width variant 2
and width padding; printed option value 1 alone is insufficient.
Compiler diagnostics/dumps must come from the chosen backend's supported
tools. Do not assume Nsight Compute profiles an OpenCL program.

## Same-plan dense exactness

The new `--directional-v2` mode of `tests/native_pfa_dense_compare.cpp` uses
`aevum_engine_create_ex(..., AEVUM_WORKLOAD_PRP)` so PRP_USE is consumed.
It sets the same manual profile with V2=0 and V2=1, disables profile search,
requires the same explicit 1K-width/variant-2 plan and radix-8 override, and
checks the actual created plan against the resolved plan. It compares every
residue word after dense squaring (including multiply-by-3), generic multiply,
prepared multiply, and a consecutive pending square chain. It restores the
process environment and destroys each context before the next.

It does not detect the GPU's compiled gate and cannot prove that a fallback
run exercised V2. Its normal PFA comparison mode remains available unchanged.

```bash
v2_root="$PWD"
v2_device=0  # replace with the selected OpenCL ordinal
v2_dense_dir=$(mktemp -d "$v2_root/v2-dense.XXXXXX")
cd "$v2_dense_dir"
# Small fixed width cases: all four numeric representations, if retained by this frontend.
for v2_type in 0 1 4; do
  v2_plan="$v2_type:1K:2:256:202"
  AEVUM_RADIX1K=8 AEVUM_PRP_USE_TUNE=off AEVUM_PRP_USE= \
    "$v2_root/build-tests/native-pfa-dense-compare" \
    "$v2_root/build-engine/libaevum_engine.so" "$v2_device" 10000019 \
    "$v2_plan" "$v2_plan" 8 20261011 --directional-v2
done
# Primary paired-field target:
AEVUM_RADIX1K=8 AEVUM_PRP_USE_TUNE=off AEVUM_PRP_USE= \
  "$v2_root/build-tests/native-pfa-dense-compare" \
  "$v2_root/build-engine/libaevum_engine.so" "$v2_device" 164000047 \
  1:1K:8:512:202 1:1K:8:512:202 8 20261012 --directional-v2
cd "$v2_root"
```

Stop at any build failure, plan rewrite or residue mismatch. Never count a
rewritten type/plan as coverage of the requested case. Use more independent
seeds after the first passes. Repeat on RTX 3080 and Radeon VII, with the
Radeon result explicitly labeled **fallback-only**.

The dense comparator accepts other registered PRP flags identically for both
legs, for example `AEVUM_PRP_USE=WMUL=1` or `AEVUM_PRP_USE=NO_ASM=1`.
The engine's PRP profile parser rejects some legacy tuning keys, including
SHUFL_BYTES_W/LDSPAD_W. Exercise those through direct kernel defines or the
standalone `-use` path, not by assuming an engine profile accepted them.
That standalone path can print an inherited unrecognized-key warning while
still forwarding a legacy tuning key; confirm the resulting kernel defines.

## Required matrix and boundary work

| Case | Expected V2 selection | Evidence required |
| --- | --- | --- |
| RTX 3080, width 1K/radix8/variant2, pad1, 8-byte | active | Compile, full residue equality, generated code |
| RTX 3080, same, 16-byte, WMUL=1 | active | Same; LDS partition bounds |
| RTX 3080, 8-byte, WMUL=1 and 2 | active | Multi-partition correctness and full warp participation |
| Radeon VII, option 0 versus 1 | fallback | Same residues/code behavior; no NVIDIA assembly |
| NVIDIA NO_ASM=1 | fallback | Both profiles compile and agree |
| LDSPAD_W=0, LDSSWIZ_W=1, NOWG2=1 or NOLDS2=1 | fallback | Original eligible layout semantics |
| Width radix4, width variant1, height-only 1K, other sizes | fallback | Existing regression suite and full residues |
| PFA3/5/7/9 and other existing selectors | fallback | Existing supported plans remain unchanged |
| FP64 / FP32 / GF31 / GF61 | active only when all gates match | Retained actual type, exact residue words |

Keep `SHUFL_BYTES_W=16,WMUL=1` together; WMUL=2 causes the host to disable
padding at the memory threshold and therefore selects the fallback.
No new support for previously unsupported four-byte 1K/radix8 staging is claimed.

Dense canonical residues do not exhaust the internal GF61 quick ranges.
Before accepting the candidate, a GPU-level boundary fixture must compare
the stock and V2 spans for zero, all-equal, cancellation, alternating signs,
near-M31/near-M61 and valid widened quick inputs, with unique owner/slot tags
to expose sign and index errors. Check both components and both halves.
Derive admissible quick inputs from the preceding production FFT/twiddle
contract; arbitrary uint64 maxima are not necessarily legal FFT inputs.
This fixture is a remaining validation requirement, not a test delivered
or run by this branch. No field tolerance is acceptable for residue words.

Existing baseline regression commands, also not executed here:

```bash
make test-shufl-permutation test-fft-variant0-1k
make build-tests/aevum-fft8-skip1-test
./build-tests/aevum-fft8-skip1-test src/cl
```

These target their original/default profiles. They do not establish V2
coverage or replace the enabled dense and boundary comparisons.

## Fixed-plan timing after exactness

Use current-main baseline versus V2-disabled first to detect an unintended
default-path change. Then compare V2=0 versus V2=1 at the same V2 commit.
Treat old V1 timing as historical: its older base differs from current main.
Do not combine V1 and V2 flags or cherry-pick V1 into this experiment.

The following owner-run example alternates warm and measured profiles using
new directories every time, avoiding PRP resume state. Retain the logs and
compare iteration counts and residues before timing statistics.

```bash
set -euo pipefail
v2_bin="$PWD/build-release/aevum"
v2_device=0  # replace for this backend
v2_ab_root=$(mktemp -d "$PWD/v2-ab.XXXXXX")
run_v2_leg() {
  v2_leg_dir=$(mktemp -d "$v2_ab_root/leg-$1.XXXXXX")
  AEVUM_RADIX1K=8 "$v2_bin" -device "$v2_device" \
    -dir "$v2_leg_dir" -prp 164000047 -fft 1:1K:8:512:202 \
    -use "AEVUM_DIRECTIONAL_FUSION_V2=$1" \
    -proof 0 -iters 10000 -noclean 2>&1 | tee "$v2_leg_dir/output.log"
}
run_v2_leg 0
run_v2_leg 1
for v2_leg in 0 1 1 0 0 1; do run_v2_leg "$v2_leg"; done
```

Run RTX OpenCL first, then RTX CUDA with `build-cuda/aevum` and its own
device ordinal and fresh root. Run Radeon OpenCL separately as a fallback
regression comparison. Keep exponent, plan, backend, driver, carry policy,
WMUL, other flags, power state and measurement window fixed within a pair.
Discard warm-up/compilation; compare paired medians and spread of full PRP
microseconds/iteration, not process startup or one best sample.

Record carryFused, fftP and fftW kernel timings; registers, spills and local
memory; active warps/occupancy; shared load/store instructions/transactions
and bank conflicts; barrier instructions/stalls; shuffle instructions; global
traffic and full PRP time. No barrier-count or occupancy improvement is
predicted. The smaller source load count can lose to shuffle/ALU pressure.

Retain V2 only after exactness and a repeatable full-workload benefit beyond
noise on the target RTX. A neutral or losing result is a rejected performance
hypothesis, not a reason to enable the flag. AMD fallback should show no
material regression. Any default promotion requires a separate reviewed change.

## Results ledger

| Evidence | Current status |
| --- | --- |
| Source ownership/range/synchronization reasoning | documented, not machine validated |
| Host build and runtime kernel compilation | NOT COMPILED |
| Dense comparator and boundary GPU fixture | NOT TESTED |
| RTX OpenCL / RTX CUDA / Radeon fallback | NOT TESTED |
| Registers, spills, LDS transactions, barriers, timing | NOT BENCHMARKED |
| Claimed speedup or correctness PASS | none |

No manual build/test/benchmark or workflow dispatch was performed.
Automatically triggered CI, if any, is separate and has not been treated as
evidence of GPU correctness or speed. This remains an opt-in draft PR.
