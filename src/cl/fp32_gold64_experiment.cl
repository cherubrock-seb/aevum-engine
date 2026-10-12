// Experimental FP32 + Goldilocks coefficient-level oracle.
// Isolated from production: not a 4M-word IBDWT implementation.
// Goldilocks arithmetic derived from Marin (MIT); copyright Yves Gallot.
// https://github.com/cherubrock-seb/PrMers/blob/main/kernels/marin.cl

#define AEVUM_GOLD64_P ((ulong)0xffffffff00000001UL)
#define AEVUM_GOLD64_NEGP ((ulong)0xffffffffUL)

static inline ulong aevum_gold64_add(ulong a, ulong b) {
    return a + b + (a >= AEVUM_GOLD64_P - b ? AEVUM_GOLD64_NEGP : 0UL);
}
static inline ulong aevum_gold64_sub(ulong a, ulong b) {
    return a - b - (a < b ? AEVUM_GOLD64_NEGP : 0UL);
}
static inline ulong aevum_gold64_reduce(ulong lo, ulong hi) {
    // a,b < p => lo + 2^64*hi < p^2. 2^64 == 2^32-1 (mod p).
    uint top = (uint)(hi >> 32);
    uint bottom = (uint)hi;
    ulong s = (((ulong)bottom << 32) | (ulong)(0xffffffffU - top)) - (ulong)bottom;
    ulong r = s + lo;
    if (r >= s) {
        // No 64-bit carry: undo the initial +2^32-1 adjustment.
        const ulong borrow_fix = r < AEVUM_GOLD64_NEGP ? AEVUM_GOLD64_NEGP : 0UL;
        r -= AEVUM_GOLD64_NEGP;
        r -= borrow_fix;
    }
    return r;
}
static inline ulong aevum_gold64_mul(ulong a, ulong b) {
    return aevum_gold64_reduce(a * b, mul_hi(a, b));
}

// Return signed 128-bit X = residue + quotient * p as (lo, signed-hi).
// The FP32 plane must provide a quotient hint accurate to strictly < 0.5;
// no absolute error guarantee is inferred from this kernel's status flag.
static inline ulong2 aevum_gold64_lift(ulong residue, int quotient) {
    const long q = (long)quotient;
    if (q >= 0) {
        const ulong correction = ((ulong)q) * AEVUM_GOLD64_NEGP;
        return (ulong2)(residue - correction, (ulong)(q - (residue < correction ? 1L : 0L)));
    }
    const ulong magnitude = (ulong)(-q);
    const ulong correction = magnitude * AEVUM_GOLD64_NEGP;
    const ulong lo = residue + correction;
    return (ulong2)(lo, (ulong)(q + (lo < residue ? 1L : 0L)));
}

// A small, exact cyclic convolution oracle: O(n^2), NEVER a production FFT.
// The FP32 lane computes an approximate coefficient; the Gold lane computes
// the residue. Each output is reconstructed to signed 128 bits using FP32.
// Preconditions for the reference tests: 0 <= a[i],b[i] <= 2^32-1, n <= 64.
// Global size n; local size may be any valid divisor. Gold input values < p.
__kernel void aevum_fp32_gold64_cyclic_reference(__global const ulong *a,
                                                   __global const ulong *b,
                                                   __global ulong2 *out128,
                                                   __global ulong *out_mod,
                                                   __global float *out_quotient_hint,
                                                   __global uint *out_warning,
                                                   uint n) {
    const uint k = (uint)get_global_id(0);
    if (k >= n || n == 0U) return;
    ulong r = 0UL;
    float approximate = 0.0f;
    for (uint j = 0U; j < n; ++j) {
        const uint bj = (k >= j) ? (k - j) : (n + k - j);
        const ulong av = a[j];
        const ulong bv = b[bj];
        r = aevum_gold64_add(r, aevum_gold64_mul(av, bv));
        approximate = fma(convert_float(av), convert_float(bv), approximate);
    }
    // GOLD_P rounds to 2^64 in FP32. The correction is provided by exact r.
    const float quotient_hint = (approximate - convert_float(r)) * 0x1p-64f;
    const float rounded = rint(quotient_hint);
    // Guard obvious precision loss, NaN and overflow. This is diagnostic only;
    // even a '0' warning is NOT a proof of correct reconstructed coefficients.
    const uint unsafe = !isfinite(quotient_hint) ||
                        fabs(quotient_hint - rounded) > 0.125f ||
                        fabs(rounded) > 2147483520.0f;
    const int quotient = unsafe ? 0 : convert_int(rounded);
    out128[k] = aevum_gold64_lift(r, quotient);
    out_mod[k] = r;
    out_quotient_hint[k] = quotient_hint;
    out_warning[k] = unsafe;
}

// Independent arithmetic vector check, including near-modulus operands.
__kernel void aevum_gold64_arithmetic_reference(__global const ulong *a,
                                                 __global const ulong *b,
                                                 __global ulong4 *out,
                                                 uint n) {
    const uint i = (uint)get_global_id(0);
    if (i >= n) return;
    const ulong x = a[i], y = b[i];
    out[i] = (ulong4)(aevum_gold64_add(x, y),
                      aevum_gold64_sub(x, y),
                      aevum_gold64_mul(x, y),
                      aevum_gold64_mul(x, x));
}
