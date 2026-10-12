# FP32 + Gold64 implementation checkpoint

- Last published SHA before this milestone: `17771185061dab96975f4c0e87ee742e762b95a9`.
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
- State: callable standalone Gold64 modular-convolution plane (OpenCL only).
  Original oracle retained; production engine and AUTO unchanged.
- Next: FP32 bridge at `src/cl/fftp.cl` / `src/cl/fftw.cl`: align existing
  Aevum FP32 signed digits/weights/scaling with the natural-order Gold64 output.
- Risks: arithmetic/codegen and root/weight agreement require validation;
  geometry legality is not a hybrid coefficient/carry capacity bound.
- Missing: FP32 layout bridge, certified quotient bound,
  wide carry, engine/type integration, all device validation.
- Explicit end-to-end Aevum path: **no**. Ready for GPU PRP validation: **no**.
  Gold64 arithmetic/tables and square/multiply plane are ready for owner-led
  compilation and GPU differential validation, not yet validated themselves.
- This session: source review only; no builds, tests, benchmarks or CI polling.
