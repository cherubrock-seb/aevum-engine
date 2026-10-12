// Experimental coefficient lift, not an acceptance test for PRP.
// UNPROVEN_PRECISION is ALWAYS set until a certified FP32 error bound exists.
#include "gold64_arith.cl"
#if !defined(GOLD64_N) || GOLD64_N != 4194304
#error Gold64 reconstruction currently requires the 4M-word convolution
#endif

#define G64_UNPROVEN_PRECISION 1u
#define G64_NONFINITE 2u
#define G64_QUOTIENT_RANGE 4u
#define G64_FRACTIONAL_HINT 8u
#define G64_INVALID_RESIDUE 16u
#define G64_UNSUPPORTED_OPERATION 32u
#define G64_CARRY_OVERFLOW 64u
#define G64_FATAL (G64_NONFINITE | G64_QUOTIENT_RANGE | G64_INVALID_RESIDUE | G64_UNSUPPORTED_OPERATION)

// Signed two's-complement 128-bit values are stored as (low64, high64).
static inline ulong2 gold64_wide_add(ulong2 a, ulong2 b) {
  const ulong lo = a.x + b.x;
  return (ulong2)(lo, a.y + b.y + (ulong)(lo < a.x));
}
static inline ulong2 gold64_wide_signed(long x) {
  return (ulong2)(as_ulong(x), x < 0 ? ~0UL : 0UL);
}
static inline ulong2 gold64_lift_signed(ulong r, int quotient) {
  const long q = (long)quotient;
  // q*p = q*2^64 - q*(2^32-1). One 64-bit product, no 128-bit multiply.
  if (q >= 0) {
    const ulong c = (ulong)q * GOLD64_CORRECTION;
    return (ulong2)(r - c, (ulong)(q - (long)(r < c)));
  }
  const ulong c = (ulong)(-q) * GOLD64_CORRECTION;
  const ulong lo = r + c;
  return (ulong2)(lo, (ulong)(q + (long)(lo < r)));
}

// Exact single-digit balanced carry primitive: x = digit + 2^bits * carry.
// bits in [1,33]. It reports overflow instead of truncating a wide carry.
// This does NOT propagate carry between digits; that remains a separate gate.
static inline long gold64_split_wide(ulong2 x, uint bits,
                                     __private long *carry, __private uint *status) {
  const ulong sign = 1UL << (bits - 1);
  const ulong low = x.x & ((sign << 1) - 1);
  const long digit = (long)(low & (sign - 1)) - (long)(low & sign);
  ulong2 q = (ulong2)((x.x >> bits) | (x.y << (64 - bits)),
                       as_ulong(as_long(x.y) >> bits));
  if (digit < 0) q = gold64_wide_add(q, (ulong2)(1UL, 0UL));
  const long c = as_long(q.x);
  if (q.y != (c < 0 ? ~0UL : 0UL)) {
    *status |= G64_CARRY_OVERFLOW;
    *carry = 0;
  } else *carry = c;
  return digit;
}

__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold64_lift_candidates(__global const ulong * restrict residue,
                            __global const float * restrict estimate,
                            __global ulong2 * restrict candidate,
                            __global uint * restrict status,
                            uint factor, uint subtract_two) {
  const uint j = (uint)get_global_id(0);
  const ulong r = residue[j];
  const float a = estimate[j];
  uint flags = G64_UNPROVEN_PRECISION;
  // Integer exponent-bit checks survive Aevum's -cl-finite-math-only option.
  if ((as_uint(a) & 0x7f800000U) == 0x7f800000U) flags |= G64_NONFINITE;
  if (r >= GOLD64_P) flags |= G64_INVALID_RESIDUE;
  if ((factor != 1u && factor != 3u) || subtract_two > 1u || (subtract_two && factor != 1u))
    flags |= G64_UNSUPPORTED_OPERATION;

  float hint = fma(a, 0x1p-64f, -convert_float(r) * 0x1p-64f);
  // 1/p = 2^-64*(1 + 2^-32 + O(2^-96)). Keep the first correction;
  // this does not bound the FFT error or make near-integer hints a proof.
  hint = fma(hint, 0x1p-32f, hint);
  if ((as_uint(hint) & 0x7f800000U) == 0x7f800000U) flags |= G64_NONFINITE;
  const float rounded = rint(hint);
  // Below 2^23 FP32 still distinguishes half-integers. This is only a
  // representability gate, NOT a bound on the correct quotient or exponent.
  if (fabs(rounded) >= 0x1p23f) flags |= G64_QUOTIENT_RANGE;
  if (fabs(hint - rounded) > 0.125f) flags |= G64_FRACTIONAL_HINT;
  if (flags & G64_FATAL) {
    candidate[j] = (ulong2)(0UL, 0UL);
    status[j] = flags;
    return;
  }
  ulong2 x = gold64_lift_signed(r, convert_int(rounded));
  if (factor == 3u) x = gold64_wide_add(gold64_wide_add(x, x), x);
  if (subtract_two && j == 0u) x = gold64_wide_add(x, gold64_wide_signed(-2L));
  candidate[j] = x;
  status[j] = flags;
}

// Inspect wide carry boundaries without claiming a normalized Mersenne result.
// All outputs retain UNPROVEN_PRECISION; the host exposes no PRP/LL entry point.
__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold64_split_candidates(__global const ulong2 * restrict candidate,
                             __global long * restrict digit,
                             __global long * restrict outgoing,
                             __global uint * restrict status, uint exponent) {
  const uint j = (uint)get_global_id(0);
  const ulong qj = (ulong)exponent * j;
  const uint width = (uint)(((qj + exponent + GOLD64_N - 1) >> 22) -
                            ((qj + GOLD64_N - 1) >> 22));
  uint flags = status[j] | G64_UNPROVEN_PRECISION;
  if (!width || width > 33u || (flags & G64_FATAL)) {
    status[j] = flags | G64_UNSUPPORTED_OPERATION;
    digit[j] = 0; outgoing[j] = 0;
    return;
  }
  long carry;
  digit[j] = gold64_split_wide(candidate[j], width, &carry, &flags);
  outgoing[j] = carry;
  status[j] = flags;
}
