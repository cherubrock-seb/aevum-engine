# Exact 2M Gold64 + GF61 + GF31 checkpoint

- Branch: `exp/gold64-gf61-gf31-exact-2m`, independent of PR #22.
- Base: `ce0b8b5ee3aa4d7c79d55858f5bb1fa783073610`.
- Last confirmed implementation commit (A): `c3bafb5a2025096fbc16fdd0a97b856f5d7ee91e`.
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
  CRT arithmetic: yes; reconstruction GPU dispatch: not yet.
- Wide cyclic carry: split primitive only. Engine API: not yet.
- Next: `gold3_reconstruct.cl` CRT and segmented wide cyclic normalization,
  then the `Gold3Reconstruction.h` dispatch wrapper.
- Transform-level GPU validation can begin; full Engine API validation cannot.
- Production blocked. No builds, tests, benchmarks, SSH or CI polling run.
