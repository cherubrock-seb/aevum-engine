# Directional fusion experiment

Track: `OPENAI-MATH-DIRECTIONAL-FUSION`.
Baseline: `0769889c9dfffa7638d76e7c5f671de00d2334f6` (main).

The local architectural motivation is to consume permuted loads directly in
the adjacent butterfly, as suggested by the directional-operation viewpoint
of OpenAI Math Family 130. This is a fixed radix-16 entry experiment, not an
implementation of an asymptotic Fourier algorithm.

## Current-source inventory

`fftbase.cl` contains four typed `shufl_and_fft2` overloads. The 1K/radix-8
path uses `f=8`, `WG=128`, then selects `fft8_16a` for lanes 0..63 and
`fft8_16b` for lanes 64..127. FP64 has both variant-0 broadcast and tabulated
call sites; FP32, GF31 and GF61 have the tabulated call site. Width and height
instantiate this shared code. The preceding `f=1` shuffle is separate.

The existing FFT2 combines LDS locations `i*64+lane` and `512+i*64+lane`
in the unpadded layout, where `lane=lowMe%64`. The next independent pairs
are `(0,4)`, `(1,5)`, `(2,6)`, `(3,7)`. GF31 vector and GF61 scalar padded
paths use their own existing address mapping and must retain it.

GF61 `fft8_16a` starts with two `X2q` and two `X2q_mul_t4` operations;
`fft8_16b` starts with four `X2qt4` operations. These operations are now consumed while a pair is read, leaving the subsequent
normalization, nontrivial roots, radix-4 cores and final swaps unchanged.

## Invariants

- GF61 shuffle additions are unreduced `addq`, yielding 0..2*M61+epsilon;
  subtraction results represent -M61-epsilon..M61+epsilon in unsigned storage.
- The upper-half entry produces sums in 0..4*M61+epsilon and differences
  in -2*M61-epsilon..2*M61+epsilon. The lower-half rotated entry remains
  within -2*M61-epsilon..2*M61+epsilon. Preserve the exact original operation
  order within each pair, including the fused quarter-root operations.
- Keep every `modM61q` count, `optional_add`, delayed `mul_t8q` and
  `mul_3t8q` unchanged. A field identity alone does not justify changing the
  quick representation. GF31 `addq`/`subq` have different, reduced semantics.
- Retain LDS allocation, partition stride, writes and all `bar(WG)` calls.
  Scalar LDS still stages real and imaginary components separately; a complex
  entry butterfly can only run after both components of its pair are available.
- Keep the existing `shufl_and_fft2` semantics for independent callers and
  preserve a selectable original path.
- Preserve FUSE_WEIGHT_BUTTERFLY, AUTO, PFA selection and release metadata.

## Implemented dataflow

The five 1K call sites now call `shufl_and_fft16`. With the experiment enabled,
`shufl_and_fft2_impl(..., true)` shares the original LDS producers and barriers,
then calls a typed `shufl_fft2_entry16_read`. That reader forms `(q_i,q_(i+4))`,
consumes its first-layer butterfly and immediate rotations, and writes the pair
back to the original `u[]`. `fft8_16a_skip1` / `fft8_16b_skip1` finish the tail.
There is no additional array. Four independent pair iterations are unrolled so
that the per-index root choices can be folded during compilation.

| Type | Upper lanes: consumed first layer | Lower lanes: consumed first layer |
| --- | --- | --- |
| GF61 | `X2q` for i=0,1; `X2q_mul_t4` for i=2,3 | `X2qt4` for all four pairs |
| GF31 | Original `X2`, `X2_mul_t8`, `X2_mul_t4`, `X2_mul_3t8` | Original `X2t4` |
| FP64 / FP32 | Original add/sub and delayed eighth-root / quarter-root rotations | Original `X2t4` |

For FP64/FP32/GF31 the unchanged radix-4 finish is shared with ordinary fft8.
The FP32/GF31 inline overload-visibility workarounds remain. The separate FP64
`fft8_skip1` weight-fusion contract still applies its own delayed rotations;
it is not confused with the new radix-16 skip entry.

### Index and representation proof

Write `q_i = op(lds[row+i*stride], lds[row+i*stride+delta])`, with `op=addq`
in lanes 0..63 and `op=subq` in lanes 64..127. Unpadded reads use
`row=lowMe%64`, `stride=64`, `delta=512`. Padded reads use
`row=((lowMe/8)&7)*136+(lowMe&7)`, `stride=8`, `delta=64`. This is exactly
the old padded formula at its existing `WG=128` guard: `(i/2)*16+(i&1)*8=i*8`
and `(lowMe%64)/64=0`. It is a simplification of addresses, not a new layout.

The new reader visits output pairs `(0,4),(1,5),(2,6),(3,7)`. It evaluates each
`q_i` with the same operands and ordering, then executes the same first-layer
primitive as the old tail. No pair depends on another pair at this boundary.
All subsequent roots, output swaps and normalization are unchanged.

GF61 retains the unsigned encoding of quick signed values. The upper sums
can reach `4*M61+epsilon`; they are still reduced by `modM61q(...,0)`. The
upper differences and lower rotated outputs remain within `-2*M61-epsilon`
to `2*M61+epsilon`. The original `modM61q(...,3)`, component-specific
`optional_add(...,2*M61)`, `mul_t8q(...,5)` and `mul_3t8q(...,3)` calls are
unchanged. No reduction is removed or deferred further. In particular,
`X2q_mul_t4` is retained rather than replaced with a canonical `mul_t4` call.
`optsubqu` / `optsubqs` and the other GF61 FFT cores are untouched.

### Synchronization and cost

Each vector shuffle still has two `bar(WG)` calls and each split-component
shuffle still has four. The first protects prior LDS consumers from overwrite;
the following barrier makes the new writes visible. In split mode the middle
barrier protects real reads before imaginary writes, followed by their producer
barrier. Complex fusion starts only after all imaginary inputs are in LDS.
Overwriting `u[i]` then cannot destroy an unread imaginary input. The next LDS
user retains the existing entry-barrier contract. Work-group partition offsets
and `LDS_BYTES` are unchanged; no cross-wave assumption is introduced.

| Resource | Source-level change |
| --- | --- |
| LDS traffic | Unchanged: 8 complex writes and 16 complex reads per thread, or the corresponding split scalar accesses |
| LDS allocation / layout | Unchanged, including padding, swizzle policy and multiple logical groups |
| Barriers | Unchanged: 2 vector / 4 split-component |
| Arithmetic / reductions | Same operation graph; no claimed instruction-count reduction |
| Register lifetime | Consume two shuffled outputs immediately; avoid a separate complete eight-value intermediate stage |
| Address calculation | Padded mapping expressed as row + constant stride; the baseline compiler may already simplify it |

This is a scheduling and materialization experiment. The compiler may already
produce equivalent code, or unrolling/inlining may increase register pressure.
A gain is not guaranteed. Register allocation, spills, occupancy and full PRP
time decide whether to keep it.

## Controls and unchanged paths

- `AEVUM_DIRECTIONAL_FUSION` is a kernel compile-time define, default **0**;
  values other than 0 or 1 are rejected by `base.cl`.
- Standalone: existing `-use AEVUM_DIRECTIONAL_FUSION=1` syntax. Engine PRP:
  existing `AEVUM_PRP_USE=AEVUM_DIRECTIONAL_FUSION=1` profile. Only the shared
  `UseOptions.h` registry was extended; there is no new public CLI or engine ABI.
- Also set the existing `AEVUM_RADIX1K=8`; otherwise the 1K side remains radix-4
  and this experiment does not execute. Apple retains its forced radix-4 policy.
- `shufl_and_fft2` retains its original semantics for independent callers.
  Define 0 selects the original read loops and full tail entry.
- Only `WG=128`, `RADIX=8`, `f=8` changes. Width and height are covered; FP64
  variant 0 is covered as well as the tabulated paths. Existing supported
  8-byte and 16-byte shuffles remain supported. No new 4-byte mode is added.
- FUSED_LL, FUSE_WEIGHT_BUTTERFLY, selectors, PFA radix dispatch, bank layouts,
  PRP/proof logic, versions and PrMers are unchanged. No release or PR is opened.

### Other fusion boundaries reviewed

The `f=1` numeric-neutral `shufl64`/`shufl32` routines also transport GF values
through floating pointer casts. Applying arithmetic inside them would require
new typed scheduling across many existing padding/swizzle specializations.
That broader change was not adopted without hardware evidence; this is not a
claim that first-shuffle fusion is impossible. The `f=4` variant-2 paths also
have partial twiddle/cosine completion between the shuffle and the ordinary
butterfly. Their FMA contracts are left alone. The special 256/radix-8 family
is outside the current 256/radix-4 policy.

No successive whole shuffle stages were collapsed: a full FFT and a
lane-dependent `tabMul`/`chainMul` separate the two 1K communication stages.

PFA middle input does a binary-axis twiddle followed by the odd-radix transform.
PFA middle output does `ifft_MIDDLE`, `pfaMiddleTwiddle`, then output
transposition/global storage; the next binary FFT is in another kernel.
There is no adjacent in-register binary first layer to consume there. The
existing PFA7 prescaling and PFA9 inverse-scale options remain unchanged. No
secondary PFA implementation or PFA11/PFA14 support is added.

## Files and recoverable commits

Implementation files: `src/cl/fft8.cl`, `src/cl/fftbase.cl`, `src/cl/base.cl`,
`src/UseOptions.h`. Validation handoff: `tests/native_pfa_dense_compare.cpp`
and this document.

The dense comparator's optional `--directional-fusion` mode uses the existing
`create_ex` PRP workload, sets the compile profile to 0 then 1, compares the
same seeded dense residues word-for-word, and checks the actual created plans.
It refuses a missing radix-8 opt-in, a plan with no 1K side, PFA plans and Apple
staging. Its normal PFA mode is preserved. The generic `create` API does not
consume `AEVUM_PRP_USE`, so merely setting that environment variable for the
old comparator would not validate this experiment.

Published checkpoints, in order:

1. `3b8284c4ccf5c5329b96ad436dc2c4e0718491bf` — baseline and invariants.
2. `ffab0cdac6adb99f0c0993bbbcbebd19efb21481` — GF61 entry/tail separation.
3. `4ddb8cb4f0f743d635218f4b3f9073fb91dd12f0` — GF61 paired read fusion.
4. `d49cb8bb6a228984ab96359b9556ff06c435ecbf` — FP64/FP32/GF31 extension.
5. `70aa117eff54e173c4721799c638ee5023c2fa33` — shared paired readers.
6. `f8a5cd820b26e548ed640ede9822cce1317ed24b` — same-plan dense comparison mode.

The documentation completion commit follows these checkpoints. Recover the
exact final HEAD and ordered commit list with `git rev-parse HEAD` and
`git log --reverse --oneline 0769889c9dfffa7638d76e7c5f671de00d2334f6..HEAD`.

## Hardware validation commands (not executed)

These are Linux commands to run on the GPU host. Choose device indices from
that host's current enumeration; do not assume that either vendor is device 0.
Use a fresh worktree and isolated run directories to avoid resuming old PRP
savefiles or picking up a different config. No command below was run in this
implementation session.

### Checkout and builds

```bash
git fetch origin exp/openai-math-directional-fusion
git worktree add --detach ../aevum-directional-review origin/exp/openai-math-directional-fusion
cd ../aevum-directional-review
make -j2 aevum engine-lib
mkdir -p build-tests
c++ -O2 -std=c++20 tests/native_pfa_dense_compare.cpp -ldl -o build-tests/native-pfa-dense-compare
# Separate standalone CUDA backend, when validating NVIDIA/CUDA:
make -j2 CUDA=1 aevum
```

The engine-lib build above is the OpenCL library. Do not mistake `CUDA=1
engine-lib` for the standalone CUDA target. Kernel compilation happens when
the selected backend creates the kernels.

### Dense exactness, width and height, all four numeric representations

Set `fusion_device` to the desired current OpenCL ordinal. Repeat separately
on NVIDIA and AMD. The loop covers FP64 (type 0), GF31+GF61 (type 1), and
FP32+GF31+GF61 (type 4); the engine must retain the printed requested type.
The comparator internally sets 0 for the reference and 1 for the candidate.

```bash
fusion_device=0  # replace with the selected OpenCL device ordinal
for fft_type in 0 1 4; do
  for shape in 1K:2:256 256:2:1K; do
    fusion_plan="$fft_type:$shape:202"
    AEVUM_RADIX1K=8 AEVUM_PRP_USE_TUNE=off AEVUM_PRP_USE= \
      ./build-tests/native-pfa-dense-compare ./build-engine/libaevum_engine.so \
      "$fusion_device" 10000019 "$fusion_plan" "$fusion_plan" 8 20261009 --directional-fusion
  done
done
# Larger, fixed 8M GF31+GF61 plan:
AEVUM_RADIX1K=8 AEVUM_PRP_USE_TUNE=off AEVUM_PRP_USE= \
  ./build-tests/native-pfa-dense-compare ./build-engine/libaevum_engine.so \
  "$fusion_device" 164000047 1:1K:8:512:202 1:1K:8:512:202 8 20261010 --directional-fusion
```

Both profiles must appear in the verbose configuration output, and both
contexts must report the same actual plan. This exercises dense squarings,
generic multiplication, prepared multiplication and a pending square chain.
It is not a standalone proof of all quick-range boundary cases. Before
acceptance, also exercise near-modulus GF61 inputs, zero/cancellation patterns,
padded/unpadded LDS, 8/16-byte shuffle modes, and multiple logical groups. Keep
any failure as a rejection pending diagnosis; do not mask it with a tolerance
on the integer residue words.

Existing regression targets may also be run by the owner:

```bash
make test-shufl-permutation test-fft-variant0-1k
make build-tests/aevum-fft8-skip1-test
./build-tests/aevum-fft8-skip1-test src/cl
```

Those existing targets use their original device selection and default kernel
options; they check the preserved fallback, not the enabled experiment. A
separate enabled AMD variant-0 run is required before claiming that path is
hardware-validated.

### Fixed-plan PRP A/B on RTX and Radeon

Use the same exponent, plan, backend, carry policy and all other flags in both
legs. Run the OpenCL binary on each vendor first; the CUDA binary is a separate
NVIDIA comparison. Set `fusion_device` for the chosen backend's enumeration.
The function below creates a new run directory every time and stops on an
error. It does not compare or approve results automatically.

```bash
set -euo pipefail
fusion_bin="$PWD/build-release/aevum"
fusion_device=0  # replace for this backend
fusion_plan=1:1K:8:512:202
fusion_ab_root=$(mktemp -d "$PWD/directional-ab.XXXXXX")
run_fusion_leg() {
  fusion_leg_dir=$(mktemp -d "$fusion_ab_root/leg-$1.XXXXXX")
  AEVUM_RADIX1K=8 "$fusion_bin" -device "$fusion_device" \
    -dir "$fusion_leg_dir" -prp 164000047 -fft "$fusion_plan" \
    -use "AEVUM_DIRECTIONAL_FUSION=$1" -proof 0 -iters 10000 -noclean \
    2>&1 | tee "$fusion_leg_dir/output.log"
}
# Warm both paths, then collect three alternating pairs.
run_fusion_leg 0
run_fusion_leg 1
for fusion_leg in 0 1 1 0 0 1; do run_fusion_leg "$fusion_leg"; done
```

For Radeon, select its OpenCL ordinal and repeat this block with a new root.
For RTX CUDA, change `fusion_bin` to `$PWD/build-cuda/aevum`, select the CUDA
ordinal, create a new root and repeat. Verify equal residues and iteration
counts before comparing timing; exclude compilation/warm-up and compare
paired full-PRP microseconds/iteration, not process startup time.

Record kernel times, registers/VGPR, spills/local-memory traffic, LDS bytes,
barrier instructions, occupancy and the presence/absence of inlined paired
loads. Inspect both width and height kernels. Use supported compiler dumps
or profiling tools for the chosen backend; no Nsight Compute OpenCL route is
assumed. Keep a reproducible full-iteration improvement only after exactness.

## Validation status

Source-level review and a cheap text comparison verified that expanding the
new tail functions recovers the original arithmetic sequences and constants;
LDS writes, partition offsets and barrier counts match the original helpers;
the five call sites and preprocessor nesting were checked. The index/range
arguments above are static reasoning, not hardware validation.

Compilation, GPU exactness (including the new comparator), CUDA/NVIDIA,
Radeon and performance are **NOT TESTED — intentionally deferred to hardware
validation**. No measured speedup, compilation success or residue PASS is
claimed. The branch is experimental and defaults to the original path.
