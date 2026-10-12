# FP32 + Gold64 implementation checkpoint

- Last published SHA before this milestone: `79d43f151d90f19f337fe3d8fcb14fb14e59c4ad`.
  The containing milestone SHA is `git log -1 --format=%H -- docs/fp32-gold64-implementation-status.md`
  (a commit cannot embed its own SHA); the PR description records the confirmed remote SHA.
- A1 complete in source: canonical Gold64 arithmetic (host/OpenCL), packed
  Marin root tables, inverse half-length scale, digit widths and O(N) exact
  IBDWT weights. NVIDIA reduction has one inseparable PTX carry chain.
- A2 complete in source: selected Marin radix-64 and fused radix-1024
  square/prepared-multiply stages; signed input weighting and normalized modular
  output; `Gold64Gpu.h` dispatch using Aevum Queue/Buffer/KernelCompiler.
  Fixed N=4194304, five transform launches, 16 KiB LDS, in-place transforms,
  precombined inverse weights, lazily allocated reusable prepared RHS.
- B complete in source: `Fp32Gold64Gpu.h` schedules the existing Aevum FP32
  fftP/middle/tail/fftW and prepared-multiply kernels, retaining their FMA,
  radix-8, twiddle and 2^-24 scaling. A tiled export kernel removes weights,
  swaps inverse components and restores natural digit order for both planes.
  A common HybridConvolution API computes modular and approximate coefficients
  from the same signed input, exponent and prepared RHS. No CPU data roundtrip.
- State: callable two-plane coefficient convolution (OpenCL only).
  Original oracle retained; production engine and AUTO unchanged.
- Next: `src/cl/fp32_gold64_reconstruct.cl`: bounded signed lift and wide
  carry primitives; preserve a hard block on PRP until the error bound exists.
- Risks: arithmetic/codegen and root/weight agreement require validation;
  geometry legality is not a hybrid coefficient/carry capacity bound.
- Missing: certified quotient bound,
  wide carry, engine/type integration, all device validation.
- Explicit end-to-end Aevum path: **no**. Ready for GPU PRP validation: **no**.
  Gold64 arithmetic/tables and two-plane square/multiply are ready for owner-led
  compilation and GPU differential validation, not yet validated themselves.
- This session: source review only; no builds, tests, benchmarks or CI polling.
