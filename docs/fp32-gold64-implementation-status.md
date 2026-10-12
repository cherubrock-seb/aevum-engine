# FP32 + Gold64 implementation checkpoint

- Last published SHA before this milestone: `6abd2735ba79e7ed4bc6474bd32dd701f4475df2`.
  The containing milestone SHA is `git log -1 --format=%H -- docs/fp32-gold64-implementation-status.md`
  (a commit cannot embed its own SHA); the PR description records the confirmed remote SHA.
- A1 complete in source: canonical Gold64 arithmetic (host/OpenCL), packed
  Marin root tables, inverse half-length scale, digit widths and O(N) exact
  IBDWT weights. NVIDIA reduction has one inseparable PTX carry chain.
- State: geometry infrastructure; original oracle retained. No production
  configuration, buffer allocation or AUTO behavior changed.
- Next: `src/cl/gold64_transform.cl`, adapt Marin forward64_0/forward64,
  fused square/multiply1024 and reverse stages for N=4194304; then host dispatch.
- Risks: arithmetic/codegen and root/weight agreement require validation;
  geometry legality is not a hybrid coefficient/carry capacity bound.
- Missing: transform dispatch, FP32 layout bridge, certified quotient bound,
  wide carry, engine/type integration, all device validation.
- Explicit end-to-end Aevum path: **no**. Ready for GPU PRP validation: **no**.
  Arithmetic/table components may be validated separately by the owner.
- This session: source review only; no builds, tests, benchmarks or CI polling.
