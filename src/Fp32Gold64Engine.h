// Explicit supervised-validation register backend. Never a PRP/LL certificate.
#pragma once
#include "Fp32Gold64Reconstruction.h"
#include "state.h"
#include <algorithm>
#include <limits>
#include <utility>

#ifndef CUDA_BACKEND
namespace aevum_gold64 {
class RegisterEngine {
  static constexpr U32 n = GpuPlane::words;
  static constexpr size_t noPrepared = std::numeric_limits<size_t>::max();
  const U32 exponent;
  Queue queue; // Must outlive every kernel/buffer below.
  ReconstructionExperiment reconstruction;
  KernelCompiler compiler;
  Buffer<WideCoefficient> candidates;
  Buffer<U32> status;
  Buffer<std::int64_t> result;
  Kernel coefficients;
  std::vector<Buffer<std::int64_t>> registers;
  size_t preparedIndex = noPrepared;

  void invalidate(size_t index) { if (preparedIndex == index) preparedIndex = noPrepared; }
  static void requireFactor(U32 factor) {
    if (factor != 1 && factor != 3)
      throw std::invalid_argument("Gold64 validation supports only factors 1 and 3");
  }
  void commit(size_t dst) {
    // Keep the old register intact on any fatal flag. The synchronous read is
    // deliberate at this unvalidated API boundary; it transfers only 16 bytes.
    diagnostics().requireUsable(PrecisionPolicy::SupervisedValidation);
    std::swap(registers.at(dst), result);
    invalidate(dst);
  }
  void normalize(size_t dst) {
    reconstruction.normalizeCandidates(candidates, result, status);
    commit(dst);
  }
  void linear(size_t dst, size_t src, U32 operation, U32 value = 0) {
    reconstruction.invalidateDiagnostics();
    coefficients(registers.at(dst), registers.at(src), candidates, status,
                 cl_uint(operation), cl_uint(value));
    normalize(dst);
  }

public:
  RegisterEngine(const Args& args, const Context& context, U32 q, size_t count)
    : exponent(q), queue(context, args.profile), reconstruction(args, queue, q),
      compiler(args, &context, "-DGOLD64_N=4194304"),
      candidates(nullptr, &queue, n), status(nullptr, &queue, n), result(nullptr, &queue, n),
      coefficients("gold64_register_coefficients", &compiler, nullptr, &queue,
                   "fp32_gold64_reconstruct.cl", "gold64_register_coefficients", n) {
    if (!count) throw std::invalid_argument("Gold64 requires at least one register");
    registers.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      registers.emplace_back(nullptr, &queue, n);
      registers.back().zero();
    }
  }
  ~RegisterEngine() noexcept { try { queue.finish(); } catch (...) {} }
  void sync() { queue.finish(); }
  ReconstructionReport diagnostics() const { return reconstruction.diagnostics(); }
  void getEstimates(float* output, size_t count) { reconstruction.getEstimates(output, count); }
  QuotientDiagnostics quotientDiagnostics() { return reconstruction.quotientDiagnostics(); }
  Words fp64DiagnosticWords(U32 factor) {
    requireFactor(factor);
    reconstruction.fp64DiagnosticNormalized(candidates, result, status, factor);
    return compactBits(result.read(), exponent);
  }

  // Compact little-endian 32-bit words are the existing Engine API ABI.
  // Reuse its host bit packing only at explicit upload/readback boundaries.
  void setWords(size_t dst, const U32* words, size_t count) {
    (void)registers.at(dst);
    if (!words || count != nWords(exponent)) throw std::invalid_argument("invalid Gold64 word buffer");
    Words compact(words, words + count);
    if (exponent % 32) compact.back() &= (U32(1) << (exponent % 32)) - 1;
    reconstruction.invalidateDiagnostics();
    result.write(expandBits(compact, n, exponent));
    // expandBits can leave a one-unit wrap on digit zero; normalize it fully.
    coefficients(result, result, candidates, status, cl_uint(0), cl_uint(0));
    normalize(dst);
  }
  void setU32(size_t dst, U32 value) {
    auto words = makeWords(exponent, value);
    setWords(dst, words.data(), words.size());
  }
  Words words(size_t src) { return compactBits(registers.at(src).read(), exponent); }
  void getWords(size_t src, U32* output, size_t count) {
    if (!output || count != nWords(exponent)) throw std::invalid_argument("invalid Gold64 output word buffer");
    auto compact = words(src);
    if (compact.size() != count) throw std::logic_error("Gold64 compact word count mismatch");
    std::copy(compact.begin(), compact.end(), output);
  }
  // Optional natural-order diagnostic ABI for exact per-digit comparison.
  void getDigits(size_t src, std::int64_t* output, size_t count) {
    if (!output || count != n) throw std::invalid_argument("Gold64 requires 4194304 output digits");
    registers.at(src).read(output, count);
  }
  void copy(size_t dst, size_t src) {
    auto& out = registers.at(dst);
    const auto& in = registers.at(src);
    if (dst != src) { out << in; invalidate(dst); }
  }
  void prepare(size_t dst, size_t src) {
    copy(dst, src);
    preparedIndex = noPrepared;
    reconstruction.prepare(registers.at(dst));
    preparedIndex = dst;
  }
  void square(size_t reg, U32 factor, bool subtractTwo = false) {
    requireFactor(factor);
    // RequireProven is intentionally not bypassable through an environment flag.
    reconstruction.squareChecked(registers.at(reg), candidates, result, status,
                                  factor, subtractTwo, PrecisionPolicy::SupervisedValidation);
    // squareChecked already read/rejected the global report.
    std::swap(registers.at(reg), result);
    invalidate(reg);
  }
  void multiply(size_t dst, size_t src, U32 factor) {
    requireFactor(factor);
    (void)registers.at(dst); (void)registers.at(src);
    if (preparedIndex != src) prepare(src, src);
    reconstruction.multiplyChecked(registers.at(dst), candidates, result, status,
                                    factor, PrecisionPolicy::SupervisedValidation);
    std::swap(registers.at(dst), result);
    invalidate(dst);
  }
  void add(size_t dst, size_t src) { linear(dst, src, 1); }
  void subtract(size_t dst, size_t src) { linear(dst, src, 2); }
  void subtractU32(size_t dst, U32 value) { linear(dst, dst, 3, value); }
  bool equal(size_t lhs, size_t rhs) { return words(lhs) == words(rhs); }
};
} // namespace aevum_gold64
#endif
