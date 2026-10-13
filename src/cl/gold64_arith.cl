// Gold64 arithmetic adapted from Marin (MIT), Copyright 2025 Yves Gallot.
// See licenses/Marin-MIT.txt. Canonical operands/results: 0 <= x < p.
#ifndef AEVUM_GOLD64_ARITH_CL
#define AEVUM_GOLD64_ARITH_CL
#define GOLD64_P ((ulong)0xffffffff00000001UL)
#define GOLD64_CORRECTION ((ulong)0xffffffffUL)

static inline ulong gold64_reduce(ulong lo, ulong hi) {
  const uint hi_hi = (uint)(hi >> 32), hi_lo = (uint)hi;
  const ulong s = upsample(hi_lo, (uint)0xffffffffU - hi_hi) - hi_lo;
#if defined(__NV_CL_C_VERSION)
  // Keep the carry chain in ONE asm block: condition codes cannot be assumed
  // live across separate inline-asm statements. Portable path on other GPUs.
  ulong r;
  __asm__ volatile (
    "{ .reg .u32 nc, c; .reg .u64 n64, c64;\n\t"
    "add.cc.u64 %0, %1, %2;\n\t"
    "addc.u32 nc, 0xffffffff, 0;\n\t"
    "cvt.u64.u32 n64, nc;\n\t"
    "sub.cc.u64 %0, %0, n64;\n\t"
    "subc.u32 c, 0, 0;\n\t"
    "cvt.u64.u32 c64, c;\n\t"
    "sub.u64 %0, %0, c64; }"
    : "=l"(r) : "l"(s), "l"(lo));
  return r;
#else
  ulong r = s + lo;
  if (r >= s) {
    const ulong borrow = r < GOLD64_CORRECTION ? GOLD64_CORRECTION : 0UL;
    r -= GOLD64_CORRECTION;
    r -= borrow;
  }
  return r;
#endif
}
static inline ulong gold64_add(ulong a, ulong b) {
  return a + b + (a >= GOLD64_P - b ? GOLD64_CORRECTION : 0UL);
}
static inline ulong gold64_sub(ulong a, ulong b) {
  return a - b - (a < b ? GOLD64_CORRECTION : 0UL);
}
static inline ulong gold64_mul(ulong a, ulong b) {
  return gold64_reduce(a * b, mul_hi(a, b));
}
static inline ulong gold64_sqr(ulong a) { return gold64_mul(a, a); }
static inline ulong gold64_muli(ulong a) {
  // sqrt(-1) = 2^48 mod p: replace a wide multiply with shifts and reduction.
  return gold64_reduce(a << 48, a >> 16);
}
static inline ulong2 gold64_add2(ulong2 a, ulong2 b) {
  return (ulong2)(gold64_add(a.x, b.x), gold64_add(a.y, b.y));
}
static inline ulong2 gold64_sub2(ulong2 a, ulong2 b) {
  return (ulong2)(gold64_sub(a.x, b.x), gold64_sub(a.y, b.y));
}
static inline ulong2 gold64_mul2(ulong2 a, ulong2 b) {
  return (ulong2)(gold64_mul(a.x, b.x), gold64_mul(a.y, b.y));
}
static inline ulong2 gold64_sqr2(ulong2 a) { return gold64_mul2(a, a); }
static inline ulong2 gold64_muli2(ulong2 a) {
  return (ulong2)(gold64_muli(a.x), gold64_muli(a.y));
}
#endif

