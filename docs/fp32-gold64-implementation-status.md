# FP32 + Gold64 implementation checkpoint

- Last confirmed remote implementation SHA before this milestone: `04e6ffcef9f05d584b32c7411d8054627e84be25`.
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
- State: explicit supervised-validation Engine API backend (OpenCL only).
  Original oracle retained; production execution and AUTO unchanged.
- C1 complete in source: actual plane outputs feed signed 128-bit lift,
  factor 1/3, square-minus-two and an exact per-coefficient balanced carry split
  with signed-64 overflow detection. `Fp32Gold64Reconstruction.h` exposes only
  candidate coefficients: every status retains **UnprovenPrecision**.
  Final source review adds buffer-alias rejection before any reconstruction
  dispatch, including the differing 16-byte input / 8-byte carry output strides.
  The FP32 export preserves carry.cl's double-biased fractional counter
  (`fracBits(word)-1`), including digit zero's inverse-weight boundary.
- C2 complete in source: `gold64_carry_local/maps/scan/finish` and host
  `normalizeCandidates`, `squareNormalized`, `multiplyNormalized` perform full
  balanced cyclic carry, including negative carries and word-zero wrap.
  Each of 16384 segments owns 256 digits; two work-group scans compose exact
  three-state carry maps. Five launches, O(N) work for this fixed geometry,
  one N-word signed-64 scratch plus small segment metadata. No serial N-word
  sweep, convergence loop or assumption that one extra pass suffices.
  Signed-128 addition and signed-64 local-carry representability are checked;
  failed fixed-point/exit consistency reports CarryNonconvergence. Any fatal
  status invalidates the entire result. Input/output buffers must be distinct.
- Carry argument: local zero-input normalization leaves balanced digits D and
  outgoing T. For W bits in a segment, its correction for incoming x is
  floor((R+x)/2^W), where 0 <= R < 2^W. W >= 256 and signed-64 T imply that
  incoming T+delta, delta in {-1,0,1}, produces another correction in that set.
  The exact monotone segment maps compose associatively. A fixed point of the
  total map closes the cycle modulo 2^p-1; prefix maps select every block's
  incoming correction. The result can have redundant balanced encodings.
- C3 complete in source: two small metadata reductions expose global flags,
  first fatal 256-digit block, fatal-block count and unproven-block count.
  `squareChecked`/`multiplyChecked` reject their default RequireProven policy
  before arithmetic dispatch. Explicit SupervisedValidation returns normalized
  candidates only after global fatal-error rejection; diagnostics remain
  accessible on failure. No flag/tolerance can certify a near-integer quotient.
- Precision policy: no rigorous bound has been established for input rounding,
  generated weights/twiddles, all FMA/butterfly stages, inverse scaling/export
  and quotient rounding together. Target 136279841 / 4194304 remains unproven.
  Type2 limits are not reused. Factor 3 is applied exactly after the lift, so
  a wrong lift becomes a 3*p error, not a precision margin. Checked signed-wide
  carry handling does not validate the chosen multiple of the Gold64 prime.
- D complete in source: FFT32GOLD64=54, canonical explicit plan
  `gold64-validation:512:8:512:202`, independent NTT_GOLD64 flag and zero
  certified BPW/maxExp. Only format widths 1..33 are admitted; no AUTO/tune
  enumeration or validated selector window changes. Stock Gpu rejects type 54.
- E complete for supervised candidate execution: `Fp32Gold64Engine.h`
  RegisterEngine owns its ordered queue, register/scratch buffers, both planes,
  roots/weights, prepared RHS and complete carry pipeline. EngineApi Runtime
  selects it only with `create_ex(..., AEVUM_WORKLOAD_GOLD64_VALIDATION)` and
  an explicit Gold64 plan. Other workloads fail before device allocation.
  Register set/get compact words, copy, prepare, square, factor 3, multiply,
  add/subtract, equality and diagnostic square-minus-two are wired. One RHS
  cache is invalidated on register mutation and rebuilt on demand. Result
  registers swap only after global fatal-error rejection, preserving old data
  on arithmetic failure. There is no GF61 fallback.
- Diagnostics exported: `aevum_engine_gold64_diagnostics`, `_get_digits` and
  `_square_minus_two`. Every candidate operation keeps UnprovenPrecision.
  C API production entry points/ABI remain intact; no PrMers change required
  to call create_ex from a supervised C API harness. Existing PrMers workload
  values cannot enable this backend accidentally. CUDA rejects the new plan.
- Source refinement after engine integration: carry-map construction loads each
  local digit/width once for all three trajectories and stops reading a segment
  only when all three carries are zero. Balanced remaining digits then preserve
  zero exactly; hostile chains still traverse the complete segment. No measured
  speedup is claimed. Register import/linear dispatch clears stale diagnostics
  before starting, including compiler/enqueue failure paths.
- Next exact step: owner-led OpenCL compilation and differential validation
  through create_ex at q=136279841, including upload/readback, hostile cyclic
  carry, square/factor-3/minus-two, prepared RHS invalidation and exact residues.
  Establish a combined FP32 error certificate before considering PRP/LL enablement.
- Risks: arithmetic/codegen and root/weight agreement require validation;
  geometry legality is not a hybrid coefficient/carry capacity bound.
- Missing: certified quotient bound and all compilation/device validation.
  Production PRP/LL, standalone CLI execution, CUDA, production timing counters
  and performance tuning are intentionally unavailable for this experimental
  type. Compact upload/readback/equality reuse Aevum's host bit packing; the
  GPU arithmetic hot path never copies coefficients to the CPU.
- Full cyclic carry in source: **yes**. Real Aevum Engine API path: **yes**,
  explicit supervised validation only. Ready to BEGIN owner-led GPU compilation
  and differential validation: **yes**. Ready for production PRP/LL: **no**.
- This session: source review only; no builds, tests, benchmarks or CI polling.
