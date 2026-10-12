// Diagnostic reconstruction on the actual two-plane transform output.
// A nonzero UnprovenPrecision flag is unconditional: no correctness inference
// may be made from a near-integer quotient or from this API completing.
#pragma once
#include "Fp32Gold64Gpu.h"
#include <initializer_list>

#ifndef CUDA_BACKEND
namespace aevum_gold64 {
struct WideCoefficient { U64 lo, hi; };
static_assert(sizeof(WideCoefficient) == 16, "OpenCL ulong2 coefficient ABI");
enum ReconstructionStatus : U32 {
  UnprovenPrecision = 1, Nonfinite = 2, QuotientRange = 4, FractionalHint = 8,
  InvalidResidue = 16, UnsupportedOperation = 32, CarryOverflow = 64
};

class ReconstructionExperiment {
  static constexpr U32 n = GpuPlane::words;
  const U32 exponent;
  HybridConvolution convolution;
  KernelCompiler compiler;
  Buffer<U64> residues;
  Buffer<float> estimates;
  Kernel lift, split;

  template<class T> static void requireSize(const Buffer<T>& x) {
    if (x.size != n) throw std::invalid_argument("Gold64 reconstruction buffer size mismatch");
  }
  static void requireDistinct(std::initializer_list<cl_mem> buffers) {
    for (auto a = buffers.begin(); a != buffers.end(); ++a)
      for (auto b = a + 1; b != buffers.end(); ++b)
        if (*a == *b)
          throw std::invalid_argument("Gold64 reconstruction buffers must not alias");
  }
  static void requireOperation(U32 factor, bool ll) {
    if ((factor != 1 && factor != 3) || (ll && factor != 1))
      throw std::invalid_argument("Gold64 reconstruction supports square/multiply, times 3, or square minus 2");
  }
  void emit(Buffer<WideCoefficient>& candidates, Buffer<U32>& status, U32 factor, bool ll) {
    lift(residues, estimates, candidates, status, cl_uint(factor), cl_uint(ll));
  }

public:
  ReconstructionExperiment(const Args& args, Queue& queue, U32 q)
    : exponent(q), convolution(args, queue, q),
      compiler(args, queue.context, "-DGOLD64_N=4194304"),
      residues(nullptr, &queue, n), estimates(nullptr, &queue, n),
      lift("gold64_lift_candidates", &compiler, nullptr, &queue,
           "fp32_gold64_reconstruct.cl", "gold64_lift_candidates", n),
      split("gold64_split_candidates", &compiler, nullptr, &queue,
            "fp32_gold64_reconstruct.cl", "gold64_split_candidates", n) {}

  void squareCandidates(const Buffer<std::int64_t>& input,
                        Buffer<WideCoefficient>& candidates, Buffer<U32>& status,
                        U32 factor = 1, bool subtractTwo = false) {
    requireSize(input); requireSize(candidates); requireSize(status);
    requireDistinct({input.get(), candidates.get(), status.get()});
    requireOperation(factor, subtractTwo);
    convolution.square(input, residues, estimates);
    emit(candidates, status, factor, subtractTwo);
  }
  void prepare(const Buffer<std::int64_t>& rhs) { convolution.prepare(rhs); }
  void multiplyCandidates(const Buffer<std::int64_t>& lhs,
                          Buffer<WideCoefficient>& candidates, Buffer<U32>& status, U32 factor = 1) {
    requireSize(lhs); requireSize(candidates); requireSize(status);
    requireDistinct({lhs.get(), candidates.get(), status.get()});
    requireOperation(factor, false);
    convolution.multiply(lhs, residues, estimates);
    emit(candidates, status, factor, false);
  }
  // Only a per-coefficient split, not inter-digit propagation or a PRP step.
  void splitCandidates(const Buffer<WideCoefficient>& candidates,
                       Buffer<std::int64_t>& digits, Buffer<std::int64_t>& outgoing,
                       Buffer<U32>& status) {
    requireSize(candidates); requireSize(digits); requireSize(outgoing); requireSize(status);
    requireDistinct({candidates.get(), digits.get(), outgoing.get(), status.get()});
    split(candidates, digits, outgoing, status, cl_uint(exponent));
  }
};
} // namespace aevum_gold64
#endif
