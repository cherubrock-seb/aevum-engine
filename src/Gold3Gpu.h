// Exact 2M convolution planes. Explicit validation only; no production routing.
#pragma once
#include "Gold3Plan.h"
#include "Gold64Plan.h"
#include "Kernel.h"
#include "KernelCompiler.h"
#include "TrigBufCache.h"
#include <memory>
#include <type_traits>

#ifndef CUDA_BACKEND
namespace aevum_gold3 {
struct ModularSetup {
  std::string defines;
  TrigPtr width, middle, height;
};
// Reuses clDefines and the existing root cache in Gpu.cpp.
ModularSetup makeModularSetup(const Args& args, Queue& queue, U32 exponent, bool gf61);

class GoldPlane {
  static constexpr U32 n = Plan::words;
  Queue* queue;
  KernelCompiler compiler;
  Buffer<U64> roots;
  Buffer<aevum_gold64::WeightPair> weights;
  Buffer<U64> work;
  std::unique_ptr<Buffer<U64>> prepared;
  bool ready = false;
  Kernel forward0, forward, squareTail, prepareTail, multiplyTail, backward, backward0;

  static aevum_gold64::Tables tables(Queue& q, U32 exponent) {
    Plan plan(exponent);
    std::size_t group = 0;
    cl_ulong local = 0;
    auto id = q.context->deviceId();
    if (clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(group), &group, nullptr) != CL_SUCCESS ||
        clGetDeviceInfo(id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local), &local, nullptr) != CL_SUCCESS ||
        group < 256 || local < 16384) {
      throw std::runtime_error("Gold3 requires 256 work-items and 16 KiB local memory");
    }
    aevum_gold64::Tables t(n, exponent);
    for (auto& w : t.weights) { w.inverse = aevum_gold64::mul(w.inverse, t.inverseHalfLength); }
    return t;
  }
  GoldPlane(const Args& args, Queue& q, aevum_gold64::Tables t)
    : queue(&q), compiler(args, q.context, "-DGOLD64_N=2097152"),
      roots(q.context, std::move(t.roots)), weights(q.context, std::move(t.weights)),
      work(nullptr, &q, n),
      forward0("gold3_gold_in", &compiler, nullptr, &q, "gold64_transform.cl", "gold64_forward64_0_weighted", n/8),
      forward("gold3_gold_f32", &compiler, nullptr, &q, "gold64_transform.cl", "gold64_forward32", n/8),
      squareTail("gold3_gold_square", &compiler, nullptr, &q, "gold64_transform.cl", "gold64_sqr1024", n/8),
      prepareTail("gold3_gold_prepare", &compiler, nullptr, &q, "gold64_transform.cl", "gold64_forward_mul1024", n/8),
      multiplyTail("gold3_gold_mul", &compiler, nullptr, &q, "gold64_transform.cl", "gold64_mul1024", n/8),
      backward("gold3_gold_b32", &compiler, nullptr, &q, "gold64_transform.cl", "gold64_backward32", n/8),
      backward0("gold3_gold_out", &compiler, nullptr, &q, "gold64_transform.cl", "gold64_backward64_0_unweighted", n/8) {}
  void begin(const Buffer<Digit>& in, Buffer<U64>& dst) {
    if (in.size != n || dst.size != n) { throw std::invalid_argument("Gold3 Gold64 buffer size"); }
    forward0(in, dst, roots, weights, cl_ulong(0));
    forward(dst, roots, cl_ulong(0), cl_uint(512), cl_uint(9));
  }
  void end(Buffer<U64>& out) {
    if (out.size != n) { throw std::invalid_argument("Gold3 Gold64 output size"); }
    backward(work, roots, cl_ulong(0), cl_uint(512), cl_uint(9));
    backward0(work, roots, out, weights, cl_ulong(0));
  }
public:
  GoldPlane(const Args& args, Queue& q, U32 exponent) : GoldPlane(args, q, tables(q, exponent)) {}
  void square(const Buffer<Digit>& in, Buffer<U64>& out) {
    begin(in, work); squareTail(work, roots, cl_ulong(0)); end(out);
  }
  void prepare(const Buffer<Digit>& in) {
    ready = false;
    if (!prepared) { prepared = std::make_unique<Buffer<U64>>(nullptr, queue, n); }
    begin(in, *prepared); prepareTail(*prepared, roots, cl_ulong(0)); ready = true;
  }
  void multiply(const Buffer<Digit>& in, Buffer<U64>& out) {
    if (!ready) { throw std::logic_error("Gold3 Gold64 RHS is not prepared"); }
    begin(in, work); multiplyTail(work, roots, cl_ulong(0), *prepared, cl_ulong(0)); end(out);
  }
};

// Storage is measured in doubles, as in Gpu.cpp; no floating-point arithmetic.
// Both paths reuse Aevum's optimized middle/tail/width kernels and root tables.
template<bool GF61> class ModularPlane {
  static constexpr U32 n = Plan::words, half = n/2;
  using Residue = std::conditional_t<GF61, U64, U32>;
  Queue* queue;
  ModularSetup setup;
  KernelCompiler compiler;
  Buffer<std::int64_t> transposed;
  Buffer<double> a, b;
  std::unique_ptr<Buffer<double>> prepared;
  bool ready = false;
  Kernel import, premul, middleIn, tailSquare, middleOut, exportResidues, tailMul;
  void begin(const Buffer<Digit>& in) {
    if (in.size != n) { throw std::invalid_argument("Gold3 modular input size"); }
    import(transposed, in);
    premul(b, transposed, setup.width);
    middleIn(a, b, setup.middle);
  }
  void end(Buffer<Residue>& out) {
    if (out.size != n) { throw std::invalid_argument("Gold3 modular output size"); }
    middleOut(a, b, setup.middle);
    exportResidues(out, a, setup.width);
  }
public:
  ModularPlane(const Args& args, Queue& q, U32 exponent)
    : queue(&q), setup(makeModularSetup(args, q, exponent, GF61)),
      compiler(args, q.context, setup.defines), transposed(nullptr, &q, n),
      a(nullptr, &q, GF61 ? n : half), b(nullptr, &q, GF61 ? n : half),
      import("gold3_import", &compiler, nullptr, &q, "gold3_bridge.cl", "gold3_import", half),
      premul("gold3_fftP", &compiler, nullptr, &q, "gold3_bridge.cl", "gold3_fftP", half/8),
      middleIn("gold3_middleIn", &compiler, nullptr, &q, "fftmiddlein.cl", GF61 ? "fftMiddleInGF61" : "fftMiddleInGF31", half/4),
      tailSquare("gold3_tailSquare", &compiler, nullptr, &q, "tailsquare.cl", GF61 ? "tailSquareGF61" : "tailSquareGF31", half/8),
      middleOut("gold3_middleOut", &compiler, nullptr, &q, "fftmiddleout.cl", GF61 ? "fftMiddleOutGF61" : "fftMiddleOutGF31", half/4),
      exportResidues("gold3_fftW_export", &compiler, nullptr, &q, "gold3_bridge.cl", "gold3_fftW_export", half/8),
      tailMul("gold3_tailMul", &compiler, nullptr, &q, "tailmul.cl", GF61 ? "tailMulGF61" : "tailMulGF31", half/16) {}
  void square(const Buffer<Digit>& in, Buffer<Residue>& out) {
    begin(in); tailSquare(b, a, setup.height); end(out);
  }
  void prepare(const Buffer<Digit>& in) {
    ready = false;
    if (!prepared) { prepared = std::make_unique<Buffer<double>>(nullptr, queue, a.size); }
    begin(in);
    // tailMul transforms the RHS itself; preserve the middle-in state.
    *prepared << a;
    ready = true;
  }
  void multiply(const Buffer<Digit>& in, Buffer<Residue>& out) {
    if (!ready) { throw std::logic_error("Gold3 modular RHS is not prepared"); }
    begin(in); tailMul(b, a, *prepared, setup.height); end(out);
  }
};

class Convolution {
  GoldPlane gold;
  ModularPlane<true> gf61;
  ModularPlane<false> gf31;
  bool ready = false;
public:
  Convolution(const Args& args, Queue& q, U32 exponent)
    : gold(args, q, exponent), gf61(args, q, exponent), gf31(args, q, exponent) {}
  // Inputs have already passed the digit-range gate; all outputs are natural
  // order, unweighted, canonical residues of the same integer coefficient.
  void square(const Buffer<Digit>& in, Buffer<U64>& g, Buffer<U64>& p, Buffer<U32>& s) {
    gold.square(in, g); gf61.square(in, p); gf31.square(in, s);
  }
  void prepare(const Buffer<Digit>& in) {
    ready = false;
    gold.prepare(in); gf61.prepare(in); gf31.prepare(in);
    ready = true;
  }
  void multiply(const Buffer<Digit>& in, Buffer<U64>& g, Buffer<U64>& p, Buffer<U32>& s) {
    if (!ready) { throw std::logic_error("Gold3 three-plane RHS is not prepared"); }
    gold.multiply(in, g); gf61.multiply(in, p); gf31.multiply(in, s);
  }
};
} // namespace aevum_gold3
#endif
