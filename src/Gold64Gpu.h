// Experimental Gold64 plane using Aevum's OpenCL objects and Marin's schedule.
// This returns MODULAR coefficients only. It cannot execute PRP/LL or carry.
#pragma once

#include "Gold64Plan.h"
#include "Kernel.h"
#include "KernelCompiler.h"
#include <cstdlib>

#ifndef CUDA_BACKEND
namespace aevum_gold64 {
class GpuPlane {
public:
  static constexpr U32 words = 4194304;

private:
  Queue* queue;
  KernelCompiler compiler;
  Buffer<U64> roots;
  Buffer<WeightPair> weights;
  Buffer<U64> work;
  std::unique_ptr<Buffer<U64>> prepared;
  bool preparedReady = false;
  bool fusedEdge = false;
  Kernel weight, unweight, forward0, forward0Weighted, forward, squareTail, prepareTail,
         multiplyTail, backward, backward0, backward0Unweighted;

  static bool fusedEdgeEnabled() {
    const char* value = std::getenv("AEVUM_GOLD64_FUSED_EDGE");
    return value && *value && *value != '0';
  }

  static Tables makeTables(Queue& q, U32 exponent) {
    // Reject unsupported resources before allocating large tables/buffers.
    std::size_t group = 0;
    cl_ulong local = 0;
    const auto id = q.context->deviceId();
    if (clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(group), &group, nullptr) != CL_SUCCESS ||
        clGetDeviceInfo(id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local), &local, nullptr) != CL_SUCCESS ||
        group < 256 || local < 16384)
      throw std::runtime_error("Gold64 4M plane requires 256 work-items and 16 KiB local memory");
    Tables t(words, exponent);
    for (auto& w : t.weights) w.inverse = mul(w.inverse, t.inverseHalfLength);
    return t;
  }

  GpuPlane(const Args& args, Queue& q, Tables tables)
    : queue(&q), compiler(args, q.context, "-DGOLD64_N=4194304"),
      roots(q.context, std::move(tables.roots)),
      weights(q.context, std::move(tables.weights)), work(nullptr, &q, words),
      fusedEdge(fusedEdgeEnabled()),
      weight("gold64_weight", &compiler, nullptr, &q, "gold64_transform.cl", "gold64_weight", words/4),
      unweight("gold64_unweight", &compiler, nullptr, &q, "gold64_transform.cl", "gold64_unweight", words/4),
      forward0("gold64_forward64_0", &compiler, nullptr, &q, "gold64_transform.cl", "gold64_forward64_0", words/8),
      forward0Weighted("gold64_forward64_0_weighted", &compiler, nullptr, &q,
                       "gold64_transform.cl", "gold64_forward64_0_weighted", words/8),
      forward("gold64_forward64", &compiler, nullptr, &q, "gold64_transform.cl", "gold64_forward64", words/8),
      squareTail("gold64_sqr1024", &compiler, nullptr, &q, "gold64_transform.cl", "gold64_sqr1024", words/8),
      prepareTail("gold64_forward_mul1024", &compiler, nullptr, &q, "gold64_transform.cl", "gold64_forward_mul1024", words/8),
      multiplyTail("gold64_mul1024", &compiler, nullptr, &q, "gold64_transform.cl", "gold64_mul1024", words/8),
      backward("gold64_backward64", &compiler, nullptr, &q, "gold64_transform.cl", "gold64_backward64", words/8),
      backward0("gold64_backward64_0", &compiler, nullptr, &q, "gold64_transform.cl", "gold64_backward64_0", words/8),
      backward0Unweighted("gold64_backward64_0_unweighted", &compiler, nullptr, &q,
                          "gold64_transform.cl", "gold64_backward64_0_unweighted", words/8) {}

  template<class T> static void requireSize(const Buffer<T>& b) {
    if (b.size != words) throw std::invalid_argument("Gold64 plane requires exactly 4194304 coefficients");
  }
  void begin(Buffer<U64>& x, const Buffer<std::int64_t>& digits) {
    requireSize(digits);
    if (fusedEdge) {
      forward0Weighted(digits, x, roots, weights, cl_ulong(0));
    } else {
      weight(digits, x, weights);
      forward0(x, roots, cl_ulong(0));
    }
    forward(x, roots, cl_ulong(0), cl_uint(1024), cl_uint(9));
  }
  void end(Buffer<U64>& residues) {
    backward(work, roots, cl_ulong(0), cl_uint(1024), cl_uint(9));
    if (fusedEdge) {
      backward0Unweighted(work, roots, residues, weights, cl_ulong(0));
    } else {
      backward0(work, roots, cl_ulong(0));
      unweight(work, residues, weights);
    }
  }

public:
  // q selects the IBDWT, not an advertised capacity. Queue must outlive this
  // object; all buffers must belong to its context. Calls are single-threaded.
  GpuPlane(const Args& args, Queue& q, U32 exponent)
    : GpuPlane(args, q, makeTables(q, exponent)) {}

  void squareResidues(const Buffer<std::int64_t>& digits, Buffer<U64>& residues) {
    requireSize(digits);
    requireSize(residues);
    begin(work, digits);
    squareTail(work, roots, cl_ulong(0));
    end(residues);
  }

  // Retain a prepared RHS across multiplications; allocate it only on demand.
  void prepare(const Buffer<std::int64_t>& rhs) {
    requireSize(rhs);
    preparedReady = false;
    if (!prepared) prepared = std::make_unique<Buffer<U64>>(nullptr, queue, words);
    begin(*prepared, rhs);
    prepareTail(*prepared, roots, cl_ulong(0));
    preparedReady = true;
  }
  void multiplyResidues(const Buffer<std::int64_t>& lhs, Buffer<U64>& residues) {
    requireSize(lhs);
    requireSize(residues);
    if (!preparedReady) throw std::logic_error("Gold64 multiply requires a prepared RHS");
    begin(work, lhs);
    multiplyTail(work, roots, cl_ulong(0), *prepared, cl_ulong(0));
    end(residues);
  }
};
} // namespace aevum_gold64
#endif // OpenCL-only experimental adapter; native CUDA backend not wired.
