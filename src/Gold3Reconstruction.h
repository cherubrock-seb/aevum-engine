#pragma once
#include "Gold3Gpu.h"
#include <array>

#ifndef CUDA_BACKEND
namespace aevum_gold3 {
struct CarryMap { U32 states[4]; };
static_assert(sizeof(CarryMap) == 16, "OpenCL uint4 carry map ABI");
struct Report {
  U32 flags, firstSegment, errorCount, cyclicState;
  void requireUsable() const {
    if (flags) { throw std::runtime_error("Gold3 reconstruction/carry flags=" + std::to_string(flags) +
                                        " first segment=" + std::to_string(firstSegment)); }
  }
};

class Reconstruction {
  static constexpr U32 n = Plan::words, segments = n/256, groups = segments/256;
  Plan plan;
  Convolution convolution;
  KernelCompiler compiler;
  Buffer<std::uint8_t> widths;
  Buffer<U64> gold, m61;
  Buffer<U32> m31, status;
  Buffer<Wide> coefficients, tails;
  Buffer<Digit> baseDigits;
  Buffer<CarryMap> maps, prefixes, totals, groupPrefixes;
  Kernel crt, linear, segmentZero, segmentMaps, scanSegments, scanGroups, applyCarry;
  bool hasReport = false;

  void begin() {
    hasReport = false;
    status.write(std::vector<U32>{0, ~U32(0), 0, 1});
  }
  void normalize(Buffer<Digit>& out) {
    if (out.size != n) { throw std::invalid_argument("Gold3 carry output size"); }
    segmentZero(coefficients, baseDigits, tails, widths, status);
    segmentMaps(baseDigits, tails, widths, maps, status);
    scanSegments(maps, prefixes, totals);
    scanGroups(totals, groupPrefixes, status);
    applyCarry(baseDigits, tails, widths, prefixes, groupPrefixes, out, status);
    hasReport = true;
  }
  static void requireFactor(U32 factor, bool minusTwo = false) {
    if ((factor != 1 && factor != 3) || (minusTwo && factor != 1)) {
      throw std::invalid_argument("Gold3 supports factor 1/3; square-minus-two requires factor 1");
    }
  }
public:
  Reconstruction(const Args& args, Queue& q, U32 exponent)
    : plan(exponent), convolution(args, q, exponent), compiler(args, q.context, ""),
      widths(q.context, plan.widths()), gold(nullptr, &q, n), m61(nullptr, &q, n),
      m31(nullptr, &q, n), status(nullptr, &q, 4), coefficients(nullptr, &q, n),
      tails(nullptr, &q, segments), baseDigits(nullptr, &q, n),
      maps(nullptr, &q, segments), prefixes(nullptr, &q, segments),
      totals(nullptr, &q, groups), groupPrefixes(nullptr, &q, groups),
      crt("gold3_crt", &compiler, nullptr, &q, "gold3_reconstruct.cl", "gold3_crt", n),
      linear("gold3_linear", &compiler, nullptr, &q, "gold3_reconstruct.cl", "gold3_linear", n),
      segmentZero("gold3_segment_zero", &compiler, nullptr, &q, "gold3_reconstruct.cl", "gold3_segment_zero", segments),
      segmentMaps("gold3_segment_maps", &compiler, nullptr, &q, "gold3_reconstruct.cl", "gold3_segment_maps", segments),
      scanSegments("gold3_scan_segments", &compiler, nullptr, &q, "gold3_reconstruct.cl", "gold3_scan_segments", segments),
      scanGroups("gold3_scan_groups", &compiler, nullptr, &q, "gold3_reconstruct.cl", "gold3_scan_groups", 256),
      applyCarry("gold3_apply_carry", &compiler, nullptr, &q, "gold3_reconstruct.cl", "gold3_apply_carry", segments) {}
  void invalidateDiagnostics() { hasReport = false; }
  Report diagnostics() const {
    if (!hasReport) { throw std::logic_error("Gold3 has no completed normalization diagnostics"); }
    std::array<U32, 4> r{};
    status.read(r.data(), r.size());
    return {r[0], r[1], r[2], r[3]};
  }
  void prepare(const Buffer<Digit>& rhs) {
    hasReport = false;
    convolution.prepare(rhs);
  }
  void square(const Buffer<Digit>& in, Buffer<Digit>& out, U32 factor, bool minusTwo = false) {
    requireFactor(factor, minusTwo);
    begin();
    convolution.square(in, gold, m61, m31);
    crt(gold, m61, m31, coefficients, status, cl_uint(factor), cl_uint(minusTwo));
    normalize(out);
  }
  void multiply(const Buffer<Digit>& in, Buffer<Digit>& out, U32 factor) {
    requireFactor(factor);
    begin();
    convolution.multiply(in, gold, m61, m31);
    crt(gold, m61, m31, coefficients, status, cl_uint(factor), cl_uint(0));
    normalize(out);
  }
  void linearOperation(const Buffer<Digit>& lhs, const Buffer<Digit>& rhs,
                       Buffer<Digit>& out, U32 operation, U32 value = 0) {
    if (lhs.size != n || rhs.size != n || operation > 3) {
      throw std::invalid_argument("Gold3 linear operation/buffer size");
    }
    begin();
    linear(lhs, rhs, coefficients, status, cl_uint(operation), cl_uint(value));
    normalize(out);
  }
  // Optional owner-led plane/alignment diagnostics after square or multiply.
  void getResidues(U64* g, U64* p, U32* s, size_t count) const {
    if (!g || !p || !s || count != n) { throw std::invalid_argument("Gold3 residue buffer size"); }
    gold.read(g, count); m61.read(p, count); m31.read(s, count);
  }
};
} // namespace aevum_gold3
#endif
