// Exact fixed-size arithmetic shared by the Gold3 host and OpenCL paths.
// Three BASE primes, not quadratic extension cardinalities. No floating point.
#ifndef AEVUM_GOLD3_ARITH
#define AEVUM_GOLD3_ARITH
#if defined(__OPENCL_VERSION__) || defined(__OPENCL_C_VERSION__)
typedef ulong g3_u64;
typedef uint g3_u32;
#else
#include <cstdint>
namespace aevum_gold3 {
using g3_u64 = std::uint64_t;
using g3_u32 = std::uint32_t;
#endif

#define G3_G ((g3_u64)0xffffffff00000001ULL)
#define G3_P61 ((g3_u64)0x1fffffffffffffffULL)
#define G3_P31 ((g3_u32)0x7fffffffU)
#define G3_WORDS 2097152U

// Little-endian two's-complement limbs. Structs deliberately avoid ulong3:
// OpenCL pads that vector to 32 bytes, whereas the coefficient ABI is 24 bytes.
typedef struct { g3_u64 lo, hi; } g3_digit;
typedef struct { g3_u64 x, y, z; } g3_i192;

static inline g3_i192 g3_make(g3_u64 x, g3_u64 y, g3_u64 z) {
  g3_i192 r = {x, y, z};
  return r;
}
static inline g3_i192 g3_zero(void) { return g3_make(0, 0, 0); }
static inline int g3_negative(g3_i192 a) { return (int)(a.z >> 63); }
static inline int g3_equal(g3_i192 a, g3_i192 b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
static inline int g3_less_unsigned(g3_i192 a, g3_i192 b) {
  return a.z != b.z ? a.z < b.z : (a.y != b.y ? a.y < b.y : a.x < b.x);
}
static inline g3_i192 g3_add(g3_i192 a, g3_i192 b) {
  g3_u64 x = a.x + b.x, y = a.y + b.y;
  g3_u64 cy = y < a.y, c = x < a.x, yy = y + c;
  return g3_make(x, yy, a.z + b.z + cy + (yy < y));
}
static inline g3_i192 g3_negate(g3_i192 a) {
  return g3_add(g3_make(~a.x, ~a.y, ~a.z), g3_make(1, 0, 0));
}
static inline g3_i192 g3_sub(g3_i192 a, g3_i192 b) { return g3_add(a, g3_negate(b)); }
static inline g3_i192 g3_abs(g3_i192 a) { return g3_negative(a) ? g3_negate(a) : a; }
static inline int g3_add_overflow(g3_i192 a, g3_i192 b, g3_i192 sum) {
  return g3_negative(a) == g3_negative(b) && g3_negative(a) != g3_negative(sum);
}
static inline g3_i192 g3_from_digit(g3_digit d) {
  return g3_make(d.lo, d.hi, (d.hi >> 63) ? ~(g3_u64)0 : 0);
}
static inline g3_i192 g3_product(g3_u64 a, g3_u64 b) {
#if defined(__OPENCL_VERSION__) || defined(__OPENCL_C_VERSION__)
  return g3_make(a * b, mul_hi(a, b), 0);
#elif defined(__SIZEOF_INT128__)
  __uint128_t p = (__uint128_t)a * b;
  return g3_make((g3_u64)p, (g3_u64)(p >> 64), 0);
#else
  const g3_u64 a0 = (g3_u32)a, a1 = a >> 32, b0 = (g3_u32)b, b1 = b >> 32;
  const g3_u64 t0 = a0 * b0, t1 = a1 * b0 + (t0 >> 32);
  const g3_u64 t2 = a0 * b1 + (g3_u32)t1;
  return g3_make((t2 << 32) | (g3_u32)t0, a1 * b1 + (t1 >> 32) + (t2 >> 32), 0);
#endif
}
static inline g3_i192 g3_product128(g3_u64 lo, g3_u64 hi, g3_u64 b) {
  g3_i192 p = g3_product(lo, b), t = g3_product(hi, b);
  return g3_add(p, g3_make(0, t.x, t.y));
}

static inline g3_u64 g3_mod61(g3_u64 a) {
  a = (a & G3_P61) + (a >> 61);
  return a >= G3_P61 ? a - G3_P61 : a;
}
static inline g3_u64 g3_add61(g3_u64 a, g3_u64 b) { return g3_mod61(a + b); }
static inline g3_u64 g3_sub61(g3_u64 a, g3_u64 b) {
  return a >= b ? a - b : G3_P61 - (b - a);
}
static inline g3_u64 g3_mul61(g3_u64 a, g3_u64 b) {
  const g3_i192 p = g3_product(a, b); // a,b < 2^61; p.y << 3 cannot overflow.
  return g3_mod61((p.x & G3_P61) + (p.x >> 61) + (p.y << 3));
}
static inline g3_u32 g3_mod31(g3_u64 a) {
  a = (a & G3_P31) + (a >> 31);
  a = (a & G3_P31) + (a >> 31);
  return (g3_u32)(a >= G3_P31 ? a - G3_P31 : a);
}
static inline g3_u32 g3_mod128_31(g3_i192 a) {
  return g3_mod31((g3_u64)g3_mod31(a.x) + ((g3_u64)g3_mod31(a.y) << 2));
}
static inline g3_u64 g3_gold_add(g3_u64 a, g3_u64 b) {
  return a + b + (a >= G3_G - b ? (g3_u64)0xffffffffU : 0);
}

// Accept signed or unsigned natural-order digits with |d| < 2^width. This
// includes 65-bit unsigned input and all balanced 64/65-bit output digits.
static inline int g3_valid_digit(g3_digit d, g3_u32 width) {
  if (width != 64 && width != 65) { return 0; }
  const g3_i192 a = g3_abs(g3_from_digit(d));
  return a.z == 0 && a.y < ((g3_u64)1 << (width - 64));
}
// Caller checks g3_valid_digit first; the absolute high limb is at most one.
static inline g3_u64 g3_digit_gold(g3_digit d) {
  const g3_i192 s = g3_from_digit(d), a = g3_abs(s);
  const g3_u64 lo = a.x >= G3_G ? a.x - G3_G : a.x;
  const g3_u64 r = g3_gold_add(lo, a.y * (g3_u64)0xffffffffU);
  return g3_negative(s) && r ? G3_G - r : r;
}
static inline g3_u64 g3_digit61(g3_digit d) {
  const g3_i192 s = g3_from_digit(d), a = g3_abs(s);
  const g3_u64 r = g3_mod61(g3_mod61(a.x) + (a.y << 3));
  return g3_negative(s) && r ? G3_P61 - r : r;
}
static inline g3_u32 g3_digit31(g3_digit d) {
  const g3_i192 s = g3_from_digit(d), a = g3_abs(s);
  const g3_u32 r = g3_mod31((g3_u64)g3_mod31(a.x) + (a.y << 2));
  return g3_negative(s) && r ? G3_P31 - r : r;
}

static inline g3_i192 g3_modulus(void) {
  return g3_make(0x5ffffffe80000001ULL, 0xcfffffffb0000001ULL, 0x000000000fffffffULL);
}
static inline g3_i192 g3_half_modulus(void) {
  return g3_make(0xafffffff40000000ULL, 0xe7ffffffd8000000ULL, 0x0000000007ffffffULL);
}
// Garner: X = a + G*t1 + (G*P61)*t2. Every modular multiply is exact,
// uses fixed constants and has no integer division or floating point.
// Inputs MUST be canonical; the GPU reconstruction checks that contract.
static inline g3_i192 g3_crt(g3_u64 a, g3_u64 b, g3_u32 c) {
  const g3_u64 t1 = g3_mul61(g3_sub61(b, g3_mod61(a)), 189521343281878507ULL);
  g3_i192 x = g3_add(g3_make(a, 0, 0), g3_product(G3_G, t1));
  const g3_u32 xm = g3_mod128_31(x);
  const g3_u32 delta = c >= xm ? c - xm : G3_P31 - (xm - c);
  const g3_u32 t2 = g3_mod31((g3_u64)delta * 1431655764U);
  x = g3_add(x, g3_product128(0x20000000ffffffffULL, 0x1fffffffdfffffffULL, t2));
  return g3_less_unsigned(g3_half_modulus(), x) ? g3_sub(x, g3_modulus()) : x;
}
// Strict conservative bound: 2*N*(2^65-1)^2 < 2^152 < Q/2.
static inline int g3_coefficient_in_range(g3_i192 x) {
  return g3_abs(x).z < ((g3_u64)1 << 24);
}
// Arithmetic shift by a supported digit width; never shift a limb by 64.
static inline g3_i192 g3_shift_digit(g3_i192 a, g3_u32 width) {
  const g3_u64 sign = g3_negative(a) ? ~(g3_u64)0 : 0;
  if (width == 64) { return g3_make(a.y, a.z, sign); }
  return g3_make((a.y >> 1) | (a.z << 63), (a.z >> 1) | (sign << 63), sign);
}
// Balanced remainder d in [-2^(width-1), 2^(width-1)); exact wide quotient.
static inline g3_i192 g3_split(g3_i192 a, g3_u32 width, g3_digit* d) {
  g3_i192 q = g3_shift_digit(a, width);
  const g3_u64 negative = width == 64 ? a.x >> 63 : a.y & 1;
  d->lo = a.x;
  d->hi = negative ? ~(g3_u64)0 : 0;
  return negative ? g3_add(q, g3_make(1, 0, 0)) : q;
}

#if !defined(__OPENCL_VERSION__) && !defined(__OPENCL_C_VERSION__)
static_assert(sizeof(g3_digit) == 16, "Gold3 digit ABI");
static_assert(sizeof(g3_i192) == 24, "Gold3 coefficient ABI");
} // namespace aevum_gold3
#endif
#endif
