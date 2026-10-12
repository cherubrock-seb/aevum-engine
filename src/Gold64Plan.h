// Experimental Goldilocks transform support; not a PRP capacity certificate.
// Arithmetic, roots and IBDWT layout adapted from Marin, Copyright 2025 Yves Gallot.
// See licenses/Marin-MIT.txt. Aevum integration follows the repository license.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace aevum_gold64 {
using U64 = std::uint64_t;
using U32 = std::uint32_t;
constexpr U64 modulus = 0xffffffff00000001ULL;
constexpr U64 correction = 0xffffffffULL;

// Canonical operands/results in [0, modulus). No integer division in mul().
inline U64 add(U64 a, U64 b) {
  return a + b + (a >= modulus - b ? correction : 0);
}
inline U64 sub(U64 a, U64 b) {
  return a - b - (a < b ? correction : 0);
}
inline U64 reduce(U64 lo, U64 hi) {
  const U64 s = ((hi << 32) | (correction - (hi >> 32))) - U32(hi);
  U64 r = s + lo;
  if (r >= s) {
    const U64 borrow = r < correction ? correction : 0;
    r -= correction;
    r -= borrow;
  }
  return r;
}
inline U64 mul(U64 a, U64 b) {
#if defined(__SIZEOF_INT128__)
  const __uint128_t t = __uint128_t(a) * b;
  return reduce(U64(t), U64(t >> 64));
#else
  // Portable host setup fallback (including MSVC): four 32x32 products.
  const U64 a0 = U32(a), a1 = a >> 32, b0 = U32(b), b1 = b >> 32;
  const U64 t0 = a0 * b0;
  const U64 t1 = a1 * b0 + (t0 >> 32);
  const U64 t2 = a0 * b1 + U32(t1);
  return reduce((t2 << 32) | U32(t0), a1 * b1 + (t1 >> 32) + (t2 >> 32));
#endif
}
inline U64 power(U64 x, U64 e) {
  U64 r = 1;
  for (; e; e >>= 1) {
    if (e & 1) r = mul(r, x);
    x = mul(x, x);
  }
  return r;
}
inline U64 inverse(U64 x) {
  if (!x || x >= modulus) throw std::invalid_argument("Gold64 inverse: non-field operand");
  return power(x, modulus - 2);
}
inline U64 root(U64 order) {
  if (!order || (modulus - 1) % order)
    throw std::invalid_argument("Gold64 root: order must divide p-1");
  return power(7, (modulus - 1) / order);
}
inline U32 bitReverse(U32 j, U32 n) {
  U32 r = 0;
  for (; n > 1; n >>= 1, j >>= 1) r = (r << 1) | (j & 1);
  return r;
}

struct WeightPair { U64 forward, inverse; };
static_assert(sizeof(WeightPair) == 2 * sizeof(U64), "OpenCL ulong2 weight ABI");

// This layer checks root/weight geometry ONLY. In particular, a 4M-word table
// for M136279841 does not establish the FP32 quotient error or carry bounds.
class Tables {
public:
  U32 words, exponent;
  U64 inverseHalfLength;
  std::vector<U64> roots;               // Marin r2, r2i, r4, r4i layout: 3*N
  std::vector<WeightPair> weights;      // physical index (j % 4)*N/4 + j/4
  std::vector<std::uint8_t> widths;     // natural digit order

  static void requireGeometry(U32 n, U32 q) {
    if (n < 8 || (n & (n - 1)) || ((modulus - 1) / 192) % n)
      throw std::invalid_argument("Gold64: length must be a power of two in [8, 2^26]");
    if (q < n || (U64(q) + n - 1) / n > 63)
      throw std::invalid_argument("Gold64: digit widths must be in [1,63]");
  }

  explicit Tables(U32 n, U32 q) : words(n), exponent(q), inverseHalfLength(0) {
    requireGeometry(n, q);             // before allocation or shifting
    inverseHalfLength = inverse(n / 2);
    roots.resize(std::size_t(3) * n, 0);
    weights.resize(n);
    widths.resize(n);
    makeRoots();
    makeWeights();
  }

  static U32 weightIndex(U32 j, U32 n) { return (j & 3) * (n / 4) + j / 4; }

private:
  void makeRoots() {
    const U32 n = words;
    for (U32 s = 1; s <= n / 4; s *= 2) {
      const U64 step = root(U64(2) * s), invStep = inverse(step);
      U64 w = 1, wi = 1;
      for (U32 j = 0; j < s; ++j) {
        const U32 k = s + bitReverse(j, s);
        roots[k] = w;
        roots[n / 2 + k] = wi;
        w = mul(w, step);
        wi = mul(wi, invStep);
      }
    }
    // Only r4 stages used by the packed transform. s=n/4 would read across
    // the r2/r2i boundary and is unused (the final pair step uses r2 directly).
    for (U32 s = 1; s <= n / 8; s *= 2) {
      for (U32 j = 0; j < s; ++j) {
        const U32 k = s + j;
        roots[n + 2 * k] = roots[2 * k];
        roots[n + 2 * k + 1] = mul(roots[k], roots[2 * k]);
        roots[2 * n + 2 * k] = roots[n / 2 + 2 * k];
        roots[2 * n + 2 * k + 1] = mul(roots[n / 2 + k], roots[n / 2 + 2 * k]);
      }
    }
  }

  void makeWeights() {
    const U32 n = words, rem = exponent % n;
    // Marin's n-th root of two. requireGeometry prevents truncated orders.
    const U64 rho = power(554, (modulus - 1) / 192 / n);
    const U64 stepInv = power(rho, rem), step = inverse(stepInv);
    const U64 wrapStep = add(step, step);
    const U64 invTwo = (modulus + 1) / 2;
    const U64 wrapInv = mul(stepInv, invTwo);
    U64 w = 1, wi = 1, previous = 0;
    U32 e = 0;
    for (U32 j = 0; j < n; ++j) {
      weights[weightIndex(j, n)] = {w, wi};
      const U64 next = (U64(exponent) * (U64(j) + 1) + n - 1) / n;
      widths[j] = std::uint8_t(next - previous);
      previous = next;
      // e_j = (-q*j) mod n. A wrap adds n, hence an exact factor rho^n=2.
      // Two modular multiplies per digit; no per-digit pow or inverse.
      const bool wrap = e < rem;
      e = wrap ? e + n - rem : e - rem;
      w = mul(w, wrap ? wrapStep : step);
      wi = mul(wi, wrap ? wrapInv : stepInv);
    }
  }
};
} // namespace aevum_gold64
