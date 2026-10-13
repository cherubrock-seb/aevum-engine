// Explicit supervised-validation geometry for the exact three-prime backend.
#pragma once
#include "cl/gold3_arith.cl"
#include <stdexcept>
#include <vector>

namespace aevum_gold3 {
using U32 = std::uint32_t;
using U64 = std::uint64_t;
using Digit = g3_digit;
using Wide = g3_i192;

struct Plan {
  static constexpr U32 words = G3_WORDS;
  static constexpr U32 width = 512, middle = 4, height = 512;
  static constexpr U32 targetExponent = 136279841;
  U32 exponent;

  explicit Plan(U32 q) : exponent(q) {
    // Enforce the proof's input/carry assumptions before allocating buffers.
    if (q < U64(64) * words || q > U64(65) * words) {
      throw std::invalid_argument("Gold3 2M requires exclusively 64/65-bit digits");
    }
    // Marin's nth root of two; field transform roots have at least these orders.
    static_assert(((G3_G - 1) / 192) % words == 0, "Gold64 weight order");
    static_assert((G3_P61 + 1) % words == 0, "GF61 extension root order");
    static_assert((U64(G3_P31) + 1) % words == 0, "GF31 extension root order");
    static_assert(width * middle * height * 2 == words, "Aevum real-word geometry");
  }
  U32 bitOffset(U32 j) const {
    return U32((U64(exponent) * j + words - 1) / words);
  }
  U32 digitWidth(U32 j) const { return bitOffset(j + 1) - bitOffset(j); }
  std::vector<std::uint8_t> widths() const {
    std::vector<std::uint8_t> result(words);
    for (U32 j = 0; j < words; ++j) { result[j] = std::uint8_t(digitWidth(j)); }
    return result;
  }
  void requireDigits(const std::vector<Digit>& digits) const {
    if (digits.size() != words) { throw std::invalid_argument("Gold3 digit count"); }
    for (U32 j = 0; j < words; ++j) {
      if (!g3_valid_digit(digits[j], digitWidth(j))) {
        throw std::invalid_argument("Gold3 input digit exceeds its 64/65-bit bound");
      }
    }
  }
};

// Capacity lemma for the integer convolution, conditional on plane alignment:
// b_j=ceil(q*j/N), e_j=b_j-q*j/N. For k=(i+j)%N,
// b_i+b_j-b_k-q*floor((i+j)/N) is either zero or one. Thus every
// weighted cyclic term is d_i*d_j or 2*d_i*d_j. With |d_j|<2^65,
// |C_k| <= 2*N*(2^65-1)^2 < 2^152 < Q/2, for square AND multiply,
// signed or unsigned input. This proves unique centered CRT reconstruction
// ONLY when every plane exports this same C_k. Extension cardinalities do
// not contribute any extra integer range. Factor 3 and LL -2 are applied
// after reconstruction; |3*C_k-2| < 2^154, within signed 192-bit arithmetic.
// Device alignment, kernel correctness and carry still require validation.
} // namespace aevum_gold3
