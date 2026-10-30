# RFC: Measured RTX 5090 / GB202 FFT profile selection

**Status:** Host policy scaffold implemented; production wiring and real GB202 validation pending.
**Canonical owner:** Aevum; PrMers must consume the validated engine behavior without duplicating FFT policy.
**Related:** PrMers issue [#36](https://github.com/cherubrock-seb/PrMers/issues/36), `README_RTX5090_GB202_TUNE.md`, Pass-4 shape and Pass-5 implementation tuning.

## Existing behavior (verified on Aevum main at `d9cedb424525008b6c9a404ba1ab54ff03acb6b5`)

`src/EngineApi.cpp` recognizes a GB202/RTX 5090 device name and exponent interval `146000000..150000000`. Without an explicit FFT or user tune directory, `AEVUM_GB202_TUNE != 0` sets the built-in FFT shape `1:512:8:512:202`. The resulting `gb202_profile` flag excludes the generic runtime FFT autotuner via `autotune_eligible`, so the historical profile is not independently compared with the current native AUTO shape. The log describes the built-in profile as `validated=1`; that is *not* evidence of a new current-version paired performance test.

The original external RTX 5090 OpenCL report measured approximately 236 us/iter (old default) versus 203 us/iter (fully tuned) around exponents 147–148M using:

```text
-fft 1:512:8:512:202
-use INPLACE=1,LOADS=10040,STORES=22,TABMUL_CHAIN32=1,MODM31=2,ZEROHACK_W=0
```

These are historical measurements on a different software baseline, not an accepted speedup for current Aevum. The FFT *shape* and the complete `-use` *implementation vector* are distinct candidates and must not be conflated.

## Proposed behavior

A pure host-testable policy scaffold now codifies GB202 scope, override precedence, fail-closed measurement handling, and the >=3% acceptance threshold. It is deliberately not wired into `EngineApi.cpp`; current production behavior is unchanged until real current-source GB202 A/B evidence exists.

1. **Scope precisely.** Evaluate only PRP on a positively identified NVIDIA GB202/RTX 5090, the supported 146M–150M range, and a supported matching backend. Do not modify LL, P-1, ECM, Radeon, or older NVIDIA defaults. Reject ambiguous device mapping. Preserve `AEVUM_GB202_TUNE=0`, explicit FFT, explicit implementation flags, user-supplied tune data, and other existing override precedence. Document `AEVUM_AUTOTUNE=off` semantics before changing them.
2. **Treat the historical shape as a candidate, not an unconditional winner.** Obtain a legal current native AUTO reference before applying the GB202 built-in shape. Compare the **actual instantiated FFT plans** under the same physical device, backend, exponent, register count, driver, runtime, and source. Check FFT capacity, known-unsafe plan restrictions, PRP factor-3 headroom, and allocation feasibility before measurement. Do not insert the built-in GB202 plan into a generic cross-device tuning table.
3. **Preserve numerical exactness.** Require the established canonical word/residue and ROE/safety gates, including a representative PRP operation sequence. Any mismatch, error, unsupported device, timeout, or incomplete comparison must fail closed to a known-safe plan; never treat a proxy hash or log label as arithmetic proof.
4. **Measure the work that matters.** Use bounded, counterbalanced A/B screening, then >=3 paired runs and a representative end-to-end PRP confirmation before declaring a production KEEP. Report both full PRP us/iter and kernel contributions; exclude JIT compilation from steady-state throughput while separately reporting startup/tuning overhead. Require >=3% repeatable **end-to-end** improvement, not only a kernel-local win. Preserve the existing stronger paired acceptance/non-regression gates if applicable.
5. **Select and cache conservatively.** Accept the winning exact plan/profile only for the validated configuration. Key decisions by Aevum source/version or plugin identity, physical device, driver, runtime, CUDA versus OpenCL, PRP workload, exponent/range or exact exponent where needed, register class, FFT shape, implementation flags, and relevant policy constraints. Preserve negative/deferred records without repeatedly recompiling or measuring on every startup. Never persist an inconclusive trial as an accepted KEEP.
6. **Respect the two selectors.** Aevum owns FFT-shape and `-use` selection. PrMers independently selects CUDA versus OpenCL using the **actual plan selected by each plugin**, exact residue comparison, and its >3% safe-backend rule. Avoid contaminating the backend probe cache across different FFT plans, implementations, or plugin revisions.
7. **Keep safe behavior while hardware evidence is absent.** A host-only implementation may add a candidate route behind an opt-in/guard and synthetic routing tests. It may **not** turn an unmeasured candidate into a new unconditional AUTO default or claim current RTX 5090 performance. Seek new current-version measurements from the existing issue #36 reporter or an equivalent real GB202 host before promotion.

## Minimum tests / validation ledger

| Gate | Required evidence |
| --- | --- |
| Scope and precedence | Host tests for exact GB202 recognition, 146M/150M boundaries and outside-range, explicit FFT/use, user tune directory, disable/off mode, and non-GB202 devices |
| Candidate/fallback behavior | Deterministic synthetic tests: native AUTO faster, GB202 faster, ties/noisy results, exactness mismatch, compile/device failure, timeout, incomplete search, and cache corruption |
| Cache correctness | Invalidate across source/plugin, device, driver, backend, exponent band, register class, FFT, flags, and changed safety policy; no stale positive promotion |
| Portability | Existing Linux, Windows, and macOS source/host CI; keep OS/GPU-specific paths guarded |
| RTX 3080 / Radeon VII | Verify no behavioral/selector regressions on hardware available to the campaign; never use them to claim GB202 throughput |
| RTX 5090 | Real current-source, same-backend, same-exponent paired A/B PRP, bit-exact residues, full-operation and kernel timings, >=3 pairs, >=3% sustained gain, and a safe negative/fallback case |
| PrMers integration | Refresh embedded Aevum from its exact accepted revision; verify engine/vendor identity, source and correctness gates, actual plan truth, and CUDA/OpenCL AUTO policy |

## Suggested autonomous follow-through

- **Current state: HOST_POLICY_SCAFFOLD_COMPLETE / NEEDS_GB202_HARDWARE**, not MERGE_CANDIDATE or validated KEEP.
- Inspect current Aevum and PrMers HEADs and existing PRs before any changes; do not create duplicate proposals.
- Implement the smallest focused reusable change on **this PR branch**, or explicitly supersede it with a traceable implementation PR if a different branch is technically required.
- Use connected GitHub identity `cherubrock-seb` for persistent writes. Run host/synthetic tests first, preserve negative results, and obtain real hardware confirmation before changing production defaults.
- Do not block unrelated validated integration/release work. Do not merge this RFC merely because Markdown CI is green.
