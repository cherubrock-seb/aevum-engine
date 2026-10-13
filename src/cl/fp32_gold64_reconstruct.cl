// Gold64 reconstruction kernels.
// Estimate-based FP32 diagnostic lifts set UNPROVEN_PRECISION themselves.
// Exact CRT/linear paths enter the shared carry normalizer with status 0.
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
#define G64_CARRY_NONCONVERGENCE 128u
#define G64_FATAL (G64_NONFINITE | G64_QUOTIENT_RANGE | G64_INVALID_RESIDUE | G64_UNSUPPORTED_OPERATION | G64_CARRY_OVERFLOW | G64_CARRY_NONCONVERGENCE)

// Signed two's-complement 128-bit values are stored as (low64, high64).
static inline ulong2 gold64_wide_add(ulong2 a, ulong2 b) {
  const ulong lo = a.x + b.x;
  return (ulong2)(lo, a.y + b.y + (ulong)(lo < a.x));
}
static inline ulong2 gold64_wide_signed(long x) {
  return (ulong2)(as_ulong(x), x < 0 ? ~0UL : 0UL);
}
static inline ulong2 gold64_wide_add_checked(ulong2 a, ulong2 b, __private uint *status) {
  const ulong2 r = gold64_wide_add(a, b);
  if (((~(a.y ^ b.y) & (a.y ^ r.y)) >> 63) != 0UL) *status |= G64_CARRY_OVERFLOW;
  return r;
}
static inline uint gold64_digit_width(uint j, uint exponent) {
  const ulong qj = (ulong)exponent * j;
  return (uint)(((qj + exponent + GOLD64_N - 1) >> 22) -
                 ((qj + GOLD64_N - 1) >> 22));
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


#define GOLD64_M31_P 0x7fffffffU
#define GOLD64_M31_INV3 1431655765UL

static inline ulong2 gold64_wide_sub(ulong2 a, ulong2 b) {
  const ulong lo = a.x - b.x;
  return (ulong2)(lo, a.y - b.y - (ulong)(a.x < b.x));
}
static inline int gold64_wide_unsigned_gt(ulong2 a, ulong2 b) {
  return a.y > b.y || (a.y == b.y && a.x > b.x);
}
static inline ulong2 gold64_wide_unsigned_shr1(ulong2 a) {
  return (ulong2)((a.x >> 1) | (a.y << 63), a.y >> 1);
}
static inline uint gold64_reduce_m31(ulong x) {
  ulong y = (x & GOLD64_M31_P) + (x >> 31);
  y = (y & GOLD64_M31_P) + (y >> 31);
  y = (y & GOLD64_M31_P) + (y >> 31);
  if (y >= GOLD64_M31_P) y -= GOLD64_M31_P;
  return (uint)y;
}

// Diagnostic exact lift from the two modular convolution planes.  Since
// Gold64_P mod M31 == 3, k=(r31-r64)*3^-1 selects the unique CRT representative.
// For 4M digits with |digit|<=2^32, one convolution coefficient is <=2^86;
// factor 3 remains <2^88, well below (Gold64_P*M31)/2 (~2^94).
__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold64_crt_m31_candidates(__global const ulong * restrict residue,
                               __global const uint * restrict residue31,
                               __global ulong2 * restrict candidate,
                               __global uint * restrict status,
                               uint factor, uint subtract_two) {
  const uint j = (uint)get_global_id(0);
  const ulong r = residue[j];
  const uint r31 = residue31[j];
  // Exactness proof under the enforced input contract:
  // N=2^22, |digit|<=2^32 => |coefficient|<=2^86.
  // GOLD64_P>2^63 and M31>2^30 => GOLD64_P*M31/2>2^92.
  // Therefore the signed CRT representative is unique with >6 bits margin.
  uint flags = 0u;

  if (r >= GOLD64_P || r31 >= GOLD64_M31_P) flags |= G64_INVALID_RESIDUE;
  if ((factor != 1u && factor != 3u) || subtract_two > 1u || (subtract_two && factor != 1u))
    flags |= G64_UNSUPPORTED_OPERATION;
  if (flags & G64_FATAL) {
    candidate[j] = (ulong2)(0UL, 0UL);
    status[j] = flags;
    return;
  }

  const uint r64m = gold64_reduce_m31(r);
  const uint delta = r31 >= r64m ? r31 - r64m : r31 + GOLD64_M31_P - r64m;
  const ulong k = ((ulong)delta * GOLD64_M31_INV3) % (ulong)GOLD64_M31_P;

  ulong2 x = gold64_lift_signed(r, (int)k);
  const ulong2 modulus = gold64_lift_signed(0UL, (int)GOLD64_M31_P);
  const ulong2 halfModulus = gold64_wide_unsigned_shr1(modulus);
  if (gold64_wide_unsigned_gt(x, halfModulus)) x = gold64_wide_sub(x, modulus);

  if (factor == 3u) x = gold64_wide_add(gold64_wide_add(x, x), x);
  if (subtract_two && j == 0u) x = gold64_wide_add(x, gold64_wide_signed(-2L));
  candidate[j] = x;
  status[j] = flags;
}

// Exact single-digit balanced carry primitive: x = digit + 2^bits * carry.
// bits in [1,33]. It reports overflow instead of truncating a wide carry.
// Inter-digit/cyclic propagation is implemented by the segmented kernels below.
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

// Diagnostic-only quotient comparison.  This does NOT alter reconstruction.
// It recomputes the quotient in FP64 from the SAME FP32 estimate and exact
// Gold64 residue, separating quotient-stage rounding from upstream FP32 error.
// Bit 0: current FP32 fractional hint; bit 1: FP64 fractional hint;
// bit 2: rounded quotient disagreement; bit 3: FP64 quotient out of FP32-safe range.
#define G64_QDIAG_FP32_FRACTIONAL 1u
#define G64_QDIAG_FP64_FRACTIONAL 2u
#define G64_QDIAG_QUOTIENT_MISMATCH 4u
#define G64_QDIAG_FP64_RANGE 8u

static inline double gold64_fp64_quotient_hint(ulong r, float a) {
  const double rscaled =
      fma(convert_double((uint)(r >> 32)), 0x1p-32,
          convert_double((uint)r) * 0x1p-64);
  double hint = fma(convert_double(a), 0x1p-64, -rscaled);
  return fma(hint, 0x1p-32, hint);
}

__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold64_quotient_diagnostics(__global const ulong * restrict residue,
                                 __global const float * restrict estimate,
                                 __global uint * restrict output) {
  const uint j = (uint)get_global_id(0);
  const ulong r = residue[j];
  const float a = estimate[j];

  float hint32 = fma(a, 0x1p-64f, -convert_float(r) * 0x1p-64f);
  hint32 = fma(hint32, 0x1p-32f, hint32);
  const float rounded32 = rint(hint32);

  // Converting the FP32 estimate to double is exact; this intentionally cannot
  // recover error already present in the FP32 transform output.
  const double hint64 = gold64_fp64_quotient_hint(r, a);
  const double rounded64 = rint(hint64);

  uint flags = 0u;
  if (fabs(hint32 - rounded32) > 0.125f) flags |= G64_QDIAG_FP32_FRACTIONAL;
  if (fabs(hint64 - rounded64) > 0.125) flags |= G64_QDIAG_FP64_FRACTIONAL;
  if (fabs(rounded64) >= 0x1p23) flags |= G64_QDIAG_FP64_RANGE;
  if (fabs(rounded32) < 0x1p23f && fabs(rounded64) < 0x1p23 &&
      convert_int(rounded32) != convert_int(rounded64))
    flags |= G64_QDIAG_QUOTIENT_MISMATCH;
  output[j] = flags;
}

// Diagnostic-only alternative lift.  It uses the FP64 quotient arithmetic above
// but the SAME FP32 transform estimate.  It never feeds the normal reconstruction
// or register commit path; callers may normalize its scratch candidate solely to
// determine whether quotient-stage rounding explains a known mismatch.
__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold64_lift_candidates_fp64_diagnostic(
    __global const ulong * restrict residue,
    __global const float * restrict estimate,
    __global ulong2 * restrict candidate,
    __global uint * restrict status,
    uint factor) {
  const uint j = (uint)get_global_id(0);
  const ulong r = residue[j];
  const float a = estimate[j];
  uint flags = G64_UNPROVEN_PRECISION;

  if ((as_uint(a) & 0x7f800000U) == 0x7f800000U) flags |= G64_NONFINITE;
  if (r >= GOLD64_P) flags |= G64_INVALID_RESIDUE;
  if (factor != 1u && factor != 3u) flags |= G64_UNSUPPORTED_OPERATION;

  const double hint = gold64_fp64_quotient_hint(r, a);
  const double rounded = rint(hint);
  if (!isfinite(hint)) flags |= G64_NONFINITE;
  if (fabs(rounded) >= 0x1p23) flags |= G64_QUOTIENT_RANGE;
  if (fabs(hint - rounded) > 0.125) flags |= G64_FRACTIONAL_HINT;

  if (flags & G64_FATAL) {
    candidate[j] = (ulong2)(0UL, 0UL);
    status[j] = flags;
    return;
  }

  ulong2 x = gold64_lift_signed(r, convert_int(rounded));
  if (factor == 3u) x = gold64_wide_add(gold64_wide_add(x, x), x);
  candidate[j] = x;
  status[j] = flags;
}

// Inspect wide carry boundaries while preserving the precision status
// supplied by the producing lift. Exact and diagnostic lifts share this code.
__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold64_split_candidates(__global const ulong2 * restrict candidate,
                             __global long * restrict digit,
                             __global long * restrict outgoing,
                             __global uint * restrict status, uint exponent) {
  const uint j = (uint)get_global_id(0);
  const ulong qj = (ulong)exponent * j;
  const uint width = (uint)(((qj + exponent + GOLD64_N - 1) >> 22) -
                            ((qj + GOLD64_N - 1) >> 22));
  uint flags = status[j];
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

// Cyclic carry lookahead. A work-item owns 64 consecutive digits, never N.
// After a zero-input local normalization, block k has digits D_k and outgoing
// signed-64 carry T_k. Its true outgoing carry is T_k + delta_k, where
// delta_k = normalize_out(D_k, T_(k-1) + delta_(k-1)).
// The host accepts this layout only when every 64-digit block spans at least
// 256 exponent bits. For any signed-64 T and delta in
// {-1,0,1}, the correction is in {-1,0,1}: normalization of balanced D plus
// that input is floor((R + input)/2^W), 0 <= R < 2^W, W >= 256.
// Thus each block is an exact monotone map of THREE states. Two levels of
// work-group prefix scan compose these maps. A fixed point of the total map
// closes the cycle, including negative wrap into digit zero. No iteration or
// probabilistic stopping criterion is used. Multiple fixed points are valid
// redundant balanced encodings of the same residue; prefer the zero state.
#ifndef G64_CARRY_BLOCK
#define G64_CARRY_BLOCK 256u
#endif
#define G64_CARRY_BLOCKS (GOLD64_N / G64_CARRY_BLOCK)
#define G64_MAP_IDENTITY 36u
static inline int gold64_map_value(uint map, int input) {
  return (int)((map >> (2u * (uint)(input + 1))) & 3u) - 1;
}
// Composition: later(earlier(x)).
static inline uint gold64_map_compose(uint later, uint earlier) {
  uint result = 0u;
  for (int x = -1; x <= 1; ++x)
    result |= (uint)(gold64_map_value(later, gold64_map_value(earlier, x)) + 1)
              << (2u * (uint)(x + 1));
  return result;
}

__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold64_carry_local(__global const ulong2 * restrict candidate,
                        __global long * restrict local_digit,
                        __global long * restrict block_carry,
                        __global uint * restrict block_status,
                        __global uint * restrict status, uint exponent) {
  const uint k = (uint)get_global_id(0);
  long carry = 0L;
  // Precision provenance belongs to the coefficient-producing path. Preserve
  // per-digit status instead of forcing every exact carry to be unproven.
  uint flags = 0u;
  for (uint i = 0; i < G64_CARRY_BLOCK; ++i) {
    const uint j = k * G64_CARRY_BLOCK + i;
    flags |= status[j];
    const uint bits = gold64_digit_width(j, exponent);
    if (!bits || bits > 33u) flags |= G64_UNSUPPORTED_OPERATION;
    long d = 0L;
    if (!(flags & G64_FATAL)) {
      const ulong2 x = gold64_wide_add_checked(candidate[j], gold64_wide_signed(carry), &flags);
      d = gold64_split_wide(x, bits, &carry, &flags);
    } else carry = 0L;
    local_digit[j] = d;
    status[j] = flags;
  }
  block_carry[k] = carry;
  block_status[k] = flags;
}

__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold64_carry_maps(__global const long * restrict local_digit,
                       __global const long * restrict block_carry,
                       __global uint * restrict block_status,
                       __global uint * restrict maps, uint exponent) {
  const uint k = (uint)get_global_id(0);
  const long base = block_carry[(k + G64_CARRY_BLOCKS - 1u) % G64_CARRY_BLOCKS];
  uint flags = block_status[k];
  ulong2 incoming[3];
  long carry[3] = {0L, 0L, 0L};
  for (uint s = 0; s < 3u; ++s)
    incoming[s] = gold64_wide_add(gold64_wide_signed(base), gold64_wide_signed((long)s - 1L));
  for (uint i = 0; i < G64_CARRY_BLOCK; ++i) {
    const uint j = k * G64_CARRY_BLOCK + i;
    const uint bits = gold64_digit_width(j, exponent);
    if (!bits || bits > 33u) { flags |= G64_UNSUPPORTED_OPERATION; break; }
    // Load each digit/width once for all three exact carry trajectories.
    const ulong2 d = gold64_wide_signed(local_digit[j]);
    for (uint s = 0; s < 3u; ++s) {
      gold64_split_wide(gold64_wide_add(incoming[s], d), bits, &carry[s], &flags);
      incoming[s] = gold64_wide_signed(carry[s]);
    }
    // Every remaining local digit is already balanced. Three zero carries
    // remain zero through the rest of the segment, irrespective of its data.
    // Hostile long chains still traverse all 256 digits; none are truncated.
    if (carry[0] == 0L && carry[1] == 0L && carry[2] == 0L) break;
  }
  uint map = 0u;
  for (uint s = 0; s < 3u; ++s) {
    if (carry[s] < -1L || carry[s] > 1L) {
      flags |= G64_CARRY_NONCONVERGENCE;
      carry[s] = 0L;
    }
    map |= (uint)(carry[s] + 1L) << (2u * s);
  }
  if (gold64_map_value(map, -1) > gold64_map_value(map, 0) ||
      gold64_map_value(map, 0) > gold64_map_value(map, 1)) flags |= G64_CARRY_NONCONVERGENCE;
  maps[k] = map;
  block_status[k] = flags;
}

// First level: 16384 block maps -> 64 group maps. Second: 64 -> one total.
// Padded lanes are identities and still participate in every barrier.
__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold64_carry_scan(__global const uint * restrict maps,
                       __global uint * restrict prefix,
                       __global uint * restrict group_map, uint count) {
  __local uint scan[256];
  const uint j = (uint)get_global_id(0), lane = (uint)get_local_id(0);
  scan[lane] = j < count ? maps[j] : G64_MAP_IDENTITY;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint offset = 1u; offset < 256u; offset <<= 1u) {
    uint value = scan[lane];
    if (lane >= offset) value = gold64_map_compose(value, scan[lane - offset]);
    barrier(CLK_LOCAL_MEM_FENCE);
    scan[lane] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (j < count) prefix[j] = scan[lane];
  if (lane == 255u) group_map[get_group_id(0)] = scan[lane];
}

__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold64_carry_finish(__global const long * restrict block_carry,
                         __global uint * restrict block_status,
                         __global const uint * restrict prefix,
                         __global const uint * restrict group_prefix,
                         __global const uint * restrict total_map,
                         __global long * restrict digit,
                         __global uint * restrict status, uint exponent) {
  const uint k = (uint)get_global_id(0), group = k / 256u;
  uint flags = block_status[k];
  const uint total = total_map[0];
  int fixed = 0;
  if (gold64_map_value(total, 0) != 0) {
    if (gold64_map_value(total, -1) == -1) fixed = -1;
    else if (gold64_map_value(total, 1) == 1) fixed = 1;
    else flags |= G64_CARRY_NONCONVERGENCE;
  }
  const uint before_group = group ? group_prefix[group - 1u] : G64_MAP_IDENTITY;
  const uint before = (k % 256u) ? gold64_map_compose(prefix[k - 1u], before_group) : before_group;
  const uint through = gold64_map_compose(prefix[k], before_group);
  const int delta = gold64_map_value(before, fixed);
  const long base = block_carry[(k + G64_CARRY_BLOCKS - 1u) % G64_CARRY_BLOCKS];
  ulong2 incoming = gold64_wide_add(gold64_wide_signed(base), gold64_wide_signed((long)delta));
  long carry = 0L;
  for (uint i = 0; i < G64_CARRY_BLOCK; ++i) {
    const uint j = k * G64_CARRY_BLOCK + i;
    // carry_local already wrote balanced digits into the final buffer. Once
    // the incoming correction reaches zero, the untouched suffix is final.
    if (!(flags & G64_FATAL) && incoming.x == 0UL && incoming.y == 0UL) {
      carry = 0L;
      break;
    }
    const uint bits = gold64_digit_width(j, exponent);
    long d = 0L;
    if (bits && bits <= 33u && !(flags & G64_FATAL)) {
      const ulong2 x = gold64_wide_add(incoming, gold64_wide_signed(digit[j]));
      d = gold64_split_wide(x, bits, &carry, &flags);
    } else { flags |= G64_UNSUPPORTED_OPERATION; carry = 0L; }
    digit[j] = d;
    incoming = gold64_wide_signed(carry);
  }
  if (carry != (long)gold64_map_value(through, fixed)) flags |= G64_CARRY_NONCONVERGENCE;
  // Propagate a block's late error to every digit it owns. Blocks are disjoint.
  for (uint i = 0; i < G64_CARRY_BLOCK; ++i) status[k * G64_CARRY_BLOCK + i] |= flags;
  block_status[k] = flags;
}

// (OR of flags, first fatal block or UINT_MAX, fatal blocks, unproven blocks).
// Only segment metadata is read: no second N-word status sweep or atomics.
static inline uint4 gold64_report_merge(uint4 a, uint4 b) {
  return (uint4)(a.x | b.x, min(a.y, b.y), a.z + b.z, a.w + b.w);
}
static inline uint4 gold64_report_reduce(__local uint4 *data) {
  const uint lane = (uint)get_local_id(0);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = 128u; stride; stride >>= 1u) {
    if (lane < stride) data[lane] = gold64_report_merge(data[lane], data[lane + stride]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  return data[0];
}
__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold64_status_blocks(__global const uint * restrict block_status,
                          __global uint4 * restrict partial) {
  __local uint4 data[256];
  const uint j = (uint)get_global_id(0), lane = (uint)get_local_id(0);
  const uint flags = block_status[j], fatal = (uint)((flags & G64_FATAL) != 0u);
  data[lane] = (uint4)(flags, fatal ? j : ~0u, fatal,
                       (uint)((flags & G64_UNPROVEN_PRECISION) != 0u));
  const uint4 report = gold64_report_reduce(data);
  if (!lane) partial[get_group_id(0)] = report;
}
__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold64_status_finish(__global const uint4 * restrict partial,
                          __global uint4 * restrict summary) {
  __local uint4 data[256];
  const uint lane = (uint)get_local_id(0);
  data[lane] = lane < G64_CARRY_BLOCKS / 256u ? partial[lane] : (uint4)(0u, ~0u, 0u, 0u);
  const uint4 report = gold64_report_reduce(data);
  if (!lane) summary[0] = report;
}

// Register import / exact linear operations reuse the same cyclic normalizer.
// Inputs may coincide (read-only); neither may alias either output buffer.
__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold64_register_coefficients(__global const long * lhs,
                                  __global const long * rhs,
                                  __global ulong2 * restrict candidate,
                                  __global uint * restrict status,
                                  uint operation, uint value) {
  const uint j = (uint)get_global_id(0);
  ulong2 x = gold64_wide_signed(lhs[j]);
  // Inputs are normalized signed digits inside the same <=2^32 contract.
  // Linear add/subtract is exact integer arithmetic followed by exact carry.
  uint flags = 0u;
  if (operation == 1u) x = gold64_wide_add_checked(x, gold64_wide_signed(rhs[j]), &flags);
  else if (operation == 2u) {
    const ulong2 y = gold64_wide_signed(rhs[j]);
    x = gold64_wide_add_checked(x, gold64_wide_add(~y, (ulong2)(1UL, 0UL)), &flags);
  } else if (operation == 3u && !j) x = gold64_wide_add_checked(x, gold64_wide_signed(-(long)value), &flags);
  else if (operation > 3u) flags |= G64_UNSUPPORTED_OPERATION;
  candidate[j] = x;
  status[j] = flags;
}
