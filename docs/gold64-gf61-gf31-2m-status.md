# Exact 2M Gold64 + GF61 + GF31 checkpoint

- Branch: `exp/gold64-gf61-gf31-exact-2m`, independent of PR #22.
- Base: `ce0b8b5ee3aa4d7c79d55858f5bb1fa783073610`.
- Last confirmed implementation commit (E): `e8d7791126280daaa1755754b2cb6cafb7357504`.
- Concurrent portable-scalar, Queue null-profile and API prefix corrections,
  plus external smoke harnesses, were
  preserved. This implementation session has not run that harness or reviewed
  external execution evidence.
- A implemented in `src/Gold3Plan.h` and `src/cl/gold3_arith.cl`: shared
  128-bit digit ABI, signed 192-bit arithmetic, exact Garner CRT constants,
  three residue conversions, balanced splitting, root/geometry checks.
- Integer capacity: ceiling bit grid, 64/65-bit digits, signed or unsigned
  magnitude below the per-digit radix; term weights 1 or 2 imply
  `|C| < 2^152 < Q/2`. This is conditional on all planes exporting the
  documented identical integer convolution. Factor 3 is post-CRT.
- B in source: `Gold3Gpu.h`, `Gold64Plan.h`, `gold64_transform.cl`,
  `gold64_arith.cl`, `gold3_bridge.cl`, and `Gpu.cpp::makeModularSetup`.
  Marin radix-64 edge + new radix-32 pass + packed 1024 tail, fused wide
  input/output edges. Existing optimized GF61/GF31 middle/tail/width kernels,
  exact modular weight shifts, tiled import, fused inverse/export.
- Source reference: PR #22 `f1c5662c7b0a49636e3ae37eab888536f58be716`.
  Marin license retained. No FP32 backend or old carry imported.
- Three GPU planes connected: yes, square and prepared multiply, one Queue.
  CRT arithmetic and reconstruction GPU dispatch: yes.
- C in source: `gold3_reconstruct.cl` / `Gold3Reconstruction.h`: exact CRT,
  post-CRT factor 3 and LL -2, range/overflow diagnostics, 8192 parallel
  256-digit segments, wide tails below 2^91, three-state correction maps,
  two-level prefix, deterministic cyclic fixed point and exit checks.
  Five bounded carry launches; no serial full-length sweep or convergence loop.
- D in source: `Gold3Engine.h`, `FFTConfig.h/.cpp`, `EngineApi.h/.cpp`,
  `Gpu.cpp` guards. Explicit `gold3-validation:512:4:512:202` / type 55;
  create_ex also requires `AEVUM_WORKLOAD_GOLD3_VALIDATION` (101).
  Compact upload/readback, raw 128-bit digit diagnostics, three residue
  diagnostics, register lifecycle, square, factor 1/3, prepared multiply,
  add/subtract/equality, dedicated square-minus-two and checked commit.
  Normal PRP/LL, stock Gpu, native CUDA and Apple reject this backend.
  Production capacity stays zero; AUTO and existing selector policies unchanged.
- Wide cyclic carry and Engine API: implemented, unvalidated.
- Full owner-led compilation/differential GPU validation can begin.
- E in source: GF31 reduced import uses 32-bit storage (8 MiB less scratch),
  final carry copies balanced remainders after the correction vanishes,
  and modular export canonically reduces even zero-shift redundant residues.
- Remaining validation blockers: compilation on RTX 3080 and gfx906, exact
  dense/sparse/hostile plane comparisons against an independent integer oracle,
  Gold64 radix-32 ordering, signs/digit-zero/normalization, cyclic long carry,
  64/65-bit compact packing, factors 1/3, LL -2, prepared-cache invalidation.
  Source capacity proof does not certify device code or production readiness.
- Next exact step: owner-led OpenCL compilation, then plane comparisons via
  `aevum_engine_gold3_set_digits` / `aevum_engine_gold3_get_residues`; no source
  implementation milestone is intentionally left as a stub.
- Production blocked. No builds, tests, benchmarks, SSH or CI polling run here.
