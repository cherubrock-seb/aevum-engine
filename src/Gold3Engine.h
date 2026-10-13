// Explicit supervised register engine. No ordinary PRP/LL/AUTO acceptance.
#pragma once
#include "Gold3Reconstruction.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

#ifndef CUDA_BACKEND
namespace aevum_gold3 {
class RegisterEngine {
  static constexpr U32 n = Plan::words;
  static constexpr size_t noPrepared = std::numeric_limits<size_t>::max();
  Plan plan;
  Queue queue; // Outlives every kernel and buffer.
  Reconstruction reconstruction;
  Buffer<Digit> result;
  std::vector<Buffer<Digit>> registers;
  size_t preparedIndex = noPrepared;

  void invalidate(size_t index) { if (preparedIndex == index) { preparedIndex = noPrepared; } }
  void commit(size_t dst) {
    // Only a 16-byte diagnostic read at the supervised boundary; coefficients
    // and carries stay on GPU. A failed operation leaves the register intact.
    reconstruction.diagnostics().requireUsable();
    std::swap(registers.at(dst), result);
    invalidate(dst);
  }
  void linear(size_t dst, size_t src, U32 operation, U32 value = 0) {
    reconstruction.linearOperation(registers.at(dst), registers.at(src), result, operation, value);
    commit(dst);
  }
  static U32 wordAt(const U32* words, size_t count, size_t i) {
    return i < count ? words[i] : 0;
  }
  static U64 read64(const U32* words, size_t count, U32 bit) {
    const size_t k = bit / 32;
    const U32 shift = bit % 32;
    U64 v = U64(wordAt(words, count, k)) >> shift;
    v |= U64(wordAt(words, count, k+1)) << (32-shift);
    if (shift) { v |= U64(wordAt(words, count, k+2)) << (64-shift); }
    return v;
  }
  static void putBits(std::vector<U32>& words, U32 bit, U32 width, U64 lo, U64 hi) {
    U32 consumed = 0;
    while (consumed < width) {
      const U32 pos = bit + consumed, shift = pos % 32;
      const U32 take = std::min(U32(32-shift), width-consumed);
      U64 value;
      if (consumed < 64) {
        value = lo >> consumed;
        if (consumed && consumed + take > 64) { value |= hi << (64-consumed); }
      } else { value = hi; }
      const U32 mask = take == 32 ? ~U32(0) : (U32(1) << take)-1;
      words[pos/32] |= (U32(value) & mask) << shift;
      consumed += take;
    }
  }
  std::vector<U32> compact(size_t src) const {
    const auto digits = registers.at(src).read();
    std::vector<U32> words((plan.exponent + 31u)/32u, 0);
    Wide carry = g3_zero();
    // This host pass exists only at explicit readback/equality boundaries.
    // Arithmetic operations use the GPU cyclic carry, never this bit packer.
    for (U32 j = 0; j < n; ++j) {
      const Wide x = g3_add(g3_from_digit(digits[j]), carry);
      const U32 width = plan.digitWidth(j);
      putBits(words, plan.bitOffset(j), width, x.x, width == 65 ? x.y & 1 : 0);
      carry = g3_shift_digit(x, width);
    }
    const Wide minusOne = g3_make(~U64(0), ~U64(0), ~U64(0));
    if (g3_equal(carry, minusOne)) {
      // 2^q == 1 mod (2^q-1): a negative terminal carry subtracts one.
      bool borrow = true;
      for (auto& word : words) {
        if (!borrow) { break; }
        borrow = word == 0;
        --word;
      }
      if (borrow) { throw std::runtime_error("Gold3 compact negative wrap invariant"); }
    } else if (!g3_equal(carry, g3_zero())) {
      throw std::runtime_error("Gold3 compact carry outside bounded digit contract");
    }
    const U32 topMask = plan.exponent % 32 ? (U32(1) << (plan.exponent % 32))-1 : ~U32(0);
    words.back() &= topMask;
    bool modulus = words.back() == topMask;
    for (size_t i = 0; modulus && i+1 < words.size(); ++i) { modulus = words[i] == ~U32(0); }
    if (modulus) { std::fill(words.begin(), words.end(), 0); }
    return words;
  }
public:
  RegisterEngine(const Args& args, const Context& context, U32 exponent, size_t count)
    : plan(exponent), queue(context, args.profile), reconstruction(args, queue, exponent),
      result(nullptr, &queue, n) {
    if (!count) { throw std::invalid_argument("Gold3 register count must be positive"); }
    registers.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      registers.emplace_back(nullptr, &queue, n);
      queue.fillBuf(registers.back().get(), U32(0), n*sizeof(Digit), nullptr);
    }
  }
  ~RegisterEngine() noexcept { try { queue.finish(); } catch (...) {} }
  void sync() { queue.finish(); }
  Report diagnostics() const { return reconstruction.diagnostics(); }
  void getResidues(U64* g, U64* p, U32* s, size_t count) const {
    reconstruction.getResidues(g, p, s, count);
  }
  void setWords(size_t dst, const U32* words, size_t count) {
    (void)registers.at(dst);
    if (!words || count != (plan.exponent+31u)/32u) { throw std::invalid_argument("Gold3 compact input size"); }
    std::vector<Digit> digits(n);
    for (U32 j = 0; j < n; ++j) {
      const U32 bit = plan.bitOffset(j), width = plan.digitWidth(j);
      digits[j] = {read64(words, count, bit), width == 65 ? U64((wordAt(words, count, (bit+64)/32) >> ((bit+64)%32)) & 1) : 0};
    }
    reconstruction.invalidateDiagnostics();
    result.write(digits);
    reconstruction.linearOperation(result, result, result, 0);
    commit(dst);
  }
  void setU32(size_t dst, U32 value) {
    (void)registers.at(dst);
    reconstruction.invalidateDiagnostics();
    queue.fillBuf(result.get(), U32(0), n*sizeof(Digit), nullptr);
    // A complete 16-byte digit, not Buffer<Digit>::zero()'s scalar conversion.
    result.write(std::vector<Digit>{{value, 0}});
    reconstruction.linearOperation(result, result, result, 0);
    commit(dst);
  }
  void getWords(size_t src, U32* words, size_t count) const {
    if (!words || count != (plan.exponent+31u)/32u) { throw std::invalid_argument("Gold3 compact output size"); }
    const auto value = compact(src);
    std::copy(value.begin(), value.end(), words);
  }
  // Raw digit diagnostic ABI: exactly two little-endian two's-complement
  // uint64 limbs per digit. Range checked; no truncation or normalization.
  void setDigits(size_t dst, const U64* limbs, size_t count) {
    auto& reg = registers.at(dst);
    if (!limbs || count != n) { throw std::invalid_argument("Gold3 raw input digit count"); }
    std::vector<Digit> digits(n);
    std::memcpy(digits.data(), limbs, n*sizeof(Digit));
    plan.requireDigits(digits);
    reconstruction.invalidateDiagnostics();
    // Preserve the old register until the transfer has completed successfully.
    result.write(digits);
    queue.finish();
    std::swap(reg, result);
    invalidate(dst);
  }
  void getDigits(size_t src, U64* limbs, size_t count) const {
    if (!limbs || count != n) { throw std::invalid_argument("Gold3 raw output digit count"); }
    const auto digits = registers.at(src).read();
    std::memcpy(limbs, digits.data(), n*sizeof(Digit));
  }
  void copy(size_t dst, size_t src) {
    auto& out = registers.at(dst);
    const auto& in = registers.at(src);
    if (dst != src) {
      invalidate(dst);
      out << in;
    }
  }
  void prepare(size_t dst, size_t src) {
    copy(dst, src);
    preparedIndex = noPrepared;
    reconstruction.prepare(registers.at(dst));
    preparedIndex = dst;
  }
  void square(size_t dst, U32 factor, bool minusTwo = false) {
    reconstruction.square(registers.at(dst), result, factor, minusTwo);
    commit(dst);
  }
  void multiply(size_t dst, size_t src, U32 factor) {
    (void)registers.at(dst); (void)registers.at(src);
    if (factor != 1 && factor != 3) { throw std::invalid_argument("Gold3 multiplication factor must be 1 or 3"); }
    if (preparedIndex != src) { prepare(src, src); }
    reconstruction.multiply(registers.at(dst), result, factor);
    commit(dst);
  }
  void add(size_t dst, size_t src) { linear(dst, src, 1); }
  void subtract(size_t dst, size_t src) { linear(dst, src, 2); }
  void subtractU32(size_t dst, U32 value) { linear(dst, dst, 3, value); }
  bool equal(size_t lhs, size_t rhs) const { return compact(lhs) == compact(rhs); }
};
} // namespace aevum_gold3
#endif
