# Bounded GF61 PFA3/PFA7 reductions

**IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED**

This opt-in candidate defers modular reductions inside the odd-axis DFT.
It adds no global-memory traffic, twiddle table, barrier or kernel launch.
PFA7 retains the existing eight-product Nussbaumer factorization and constants;
PFA3 retains the one-product radix-three identity. GF31 arithmetic, inverse
normalization, transform layout and carry code remain unchanged.

## Activation and scope

The default is OFF. Use the existing per-run `-use` interface:

- PFA3: `-use AEVUM_PFA_LAZY61=1` with an explicitly legal PFA3 plan.
- PFA7: `-use AEVUM_PFA7_NUSS8=1,AEVUM_PFA_LAZY61=1` with
  `pfa7:1:512:7:512:202`. Hold NUSS8=1 in both A/B arms.

There is no new environment-variable alias. To disable the candidate, omit the
key or set it to 0. This preserves manual FFT selection, per-plan configuration
precedence, tune files and existing kernel-cache keys; the new define uses the
normal compilation configuration. Do not globally enable it while an AUTO
scan includes unrelated radices.

The compilation guards require a transform containing GF61, radix 3 or 7,
native GF61 arithmetic (no limb32), and non-Apple OpenCL. PFA7 additionally
requires the existing NUSS8 option; its host-side geometry restriction remains
in force. PFA9 is deliberately outside this candidate. The GF31 kernels in a
mixed transform retain their existing implementation.

Targets are Radeon VII/gfx906 OpenCL and RTX 3080 OpenCL/CUDA.
RX 7900 XTX/gfx1100 is a useful additional target, but real-hardware validation
there is pending. There is no architecture-specific default or tuning claim.

## Mechanism and bounds

Let `M = 2^61 - 1`. Every input lane is in `[0,M+7]`.
The surrounding GF61 butterflies/twiddle multiplications supply this sloppy
residue range. Adding an integer multiple of M preserves the field element.
Unsigned intermediate subtraction may wrap, but every expression passed to
the final fold, after its M offset, represents a mathematical integer in
`[0,2^64)`. Therefore no unintended multiple of 2^64 survives into the fold.
Each fold returns a value at most M+7.

The helper's `weakMul(a,w,count,2)` uses the existing bounded product path.
For PFA7 the largest input is `total <= 6M+42`, with count=7;
all constants are less than M. Thus count+2 <= 9, a+w < 2^64 and the weak
reduction cannot overflow. Other count bounds are 6 for shifted even
differences, 3 for sums of two reduced sine inputs, 4 for a shifted sine
difference and 5 for the shifted three-term sine input. The final product
fold is retained for all eight PFA7 products.

PFA7 keeps three sine-input folds and three even-output folds. Deferring those
too would require wider bounds at the final combinations; this version does
not assume that extra headroom.

| PFA7 intermediate before its fold | Lower bound | Upper bound |
| --- | --- | --- |
| x0 + total | 0 | 7M+49 |
| center - tc + 3M | M-14 | 5M+14 |
| center + mc | 0 | 4M+28 |
| center + tc - mc + 3M | M-14 | 7M+28 |
| e1 + q0 - tq + 3M | M-14 | 5M+14 |
| e1 - q0 + tq + 2M | M-7 | 5M+21 |
| e2 + q0 + sq | 0 | 4M+28 |
| e2 - q0 - sq + 4M | M-21 | 5M+7 |
| e3 + sq - q0 - tq + 4M | M-21 | 7M+21 |
| e3 - sq + q0 + tq + 3M | M-14 | 7M+28 |

Here a1/a2/a3 <= 2M+14, total <= 6M+42, center/tc/mc <= 2M+14,
e1/e2/e3/q0 <= M+7, and tq/sq <= 2M+14. All listed upper bounds
are strictly below 2^64.

For PFA3, `d = a1 + 2M - a2` lies in `[M-7,3M+7]`.
Its unreduced weak product wd is at most 4M+7. The three final fold inputs
are `a0+a1+a2`, `a0-a2+wd+2M`, and `a0-a1-wd+6M`;
their upper bounds are 3M+21, 7M+14 and 7M+7, respectively.
Their lower bounds are 0, M-7 and M-14. This applies to both existing
forward and inverse cube roots.

Counting scalar modM61 folds per component in the odd DFT (excluding external
inverse scaling), the existing PFA7 NUSS8 code has 36 add/sub folds plus eight
product folds: 44. This candidate has 13 add/output folds plus eight product
folds: 21. PFA3 goes from seven add/sub folds plus one product fold to three
output folds. The multiplication counts stay eight and one, respectively.

These are source operation counts, not emitted instruction counts or speed
measurements. A compiler cannot generally delete the original folds without
proving these modular and overflow bounds. Larger unreduced operands and
changed lifetimes can still worsen instruction scheduling, register use or
occupancy. Memory-bound transforms may show little end-to-end benefit.

This differs from the earlier GF61 cosine/sine-base experiments: it keeps the
existing eight-product schedule and removes intermediate normalization rather
than introducing a different product graph or a larger memory stream.

## Pending validation

No compilation or test was run for this implementation.

1. Run `make test-pfa-lazy61`. It includes the actual helper, checks both
   directions against an independent uint128 modular DFT, verifies output
   ranges and normalized round trips, and covers basis vectors, sloppy residues,
   all binary range corners and deterministic random inputs.
   The host primitive models the existing weak reduction; it does not validate
   PTX/GCN lowering. Also run the existing `tests/pfa7_nuss8_model_test.py`
   and native PFA source/reference gates.
2. Build OpenCL and CUDA variants. Check omitted/0 settings reproduce the stock
   path. Reject value 2, a nonnumeric token, radix 0/9, transforms without GF61,
   limb32, Apple mode, and PFA7 with NUSS8=0. Check both standalone GF61 and
   mixed-prime kernel compilation where the plan is legal.
3. With at least 15 GiB free, screen exact full residues on Radeon VII and
   RTX 3080 for legal PFA3 shapes and M136279841 with the fixed PFA7 plan.
   Compare flag 0/1 with all other options held fixed. Include nonzero high
   limbs, boundary exponents, carry-heavy inputs, square/mul/factor-3,
   repeated transforms and both inverse-normalization paths where available.
4. Run the existing native-PFA engine comparisons and production PRP checks,
   then LL, P-1, ECM, proof verification and checkpoint/resume gates using
   affected engine operations. Canonical complete residues are authoritative;
   sloppy intermediate GF61 encodings need not be byte-identical.
5. If exact, perform at least three counterbalanced paired A/B measurements
   (AB, BA, AB), warmed separately, on identical plans/exponents and power
   settings. Record full-operation PRP wall times with Gerbicz enabled.
   Collect middle-in/out kernel timings and register/scratch/occupancy evidence
   separately from throughput runs. Require at least 3% repeatable end-to-end
   improvement for performance KEEP; reject spill/occupancy regressions.
6. Compare PRPLLNTT only with matching workload, exponent, device, driver,
   proof/check policy and power constraints. The supplied gfx1100 quick-mode
   report has substantial dispersion and inconsistent production estimates.
   Use warmed repeated Gerbicz probes before ranking plans; those estimates
   remain a screen, not evidence of PRPLLNTT parity.

Do not enable the option in production defaults or combine it with active
scatter/tail-layout candidates until each mechanism has independent evidence.
