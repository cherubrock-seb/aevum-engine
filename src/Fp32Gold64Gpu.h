// Experimental two-plane convolution. No carry or PRP execution is exposed.
#pragma once
#include "Gold64Gpu.h"
#include "TrigBufCache.h"
#include <cstdlib>
#include <chrono>
#include <iostream>

#ifndef CUDA_BACKEND
namespace aevum_gold64 {
struct Fp32Setup {
  std::string defines;
  std::vector<double> weights;
  TrigPtr width, middle, height;
};
struct M31Setup {
  std::string defines;
  TrigPtr width, middle, height;
};
// Implemented beside Aevum's existing clDefines/genWeights helpers in Gpu.cpp.
Fp32Setup makeFp32Setup(const Args& args, Queue& queue, U32 exponent);
M31Setup makeM31Setup(const Args& args, Queue& queue, U32 exponent);

class Fp32Plane {
  static constexpr U32 n = GpuPlane::words;
  static constexpr U32 half = n / 2;
  Queue* queue;
  Fp32Setup setup;
  KernelCompiler compiler;
  Buffer<double> weights;
  Buffer<std::int64_t> transposed;
  Buffer<double> a, b; // FP32 data, measured in doubles as in Gpu.cpp
  std::unique_ptr<Buffer<double>> prepared;
  bool preparedReady = false;
  Kernel transpose, premul, middleIn, tailSquare, middleOut, widthOut,
         heightIn, tailMul, exportCoefficients;

  template<class T> static void requireSize(const Buffer<T>& x) {
    if (x.size != n) throw std::invalid_argument("FP32/Gold64 requires 4194304 natural-order digits");
  }
  void begin(const Buffer<std::int64_t>& input) {
    requireSize(input);
    transpose(transposed, input);
    premul(b, transposed, setup.width, weights);
    middleIn(a, b, setup.middle);
  }
  void end(Buffer<float>& coefficients) {
    middleOut(a, b, setup.middle);
    widthOut(b, a, setup.width);
    exportCoefficients(b, coefficients, weights);
  }

public:
  Fp32Plane(const Args& args, Queue& q, U32 exponent)
    : queue(&q), setup(makeFp32Setup(args, q, exponent)),
      compiler(args, q.context, setup.defines),
      weights(q.context, std::move(setup.weights)),
      transposed(nullptr, &q, n), a(nullptr, &q, half), b(nullptr, &q, half),
      transpose("gold64_fp32_import", &compiler, nullptr, &q,
                "fp32_gold64_bridge.cl", "gold64_fp32_import", half),
      premul("gold64_fp32_fftP", &compiler, nullptr, &q, "fftp.cl", "fftP", half/8),
      middleIn("gold64_fp32_middleIn", &compiler, nullptr, &q, "fftmiddlein.cl", "fftMiddleIn", half/8),
      tailSquare("gold64_fp32_tailSquare", &compiler, nullptr, &q, "tailsquare.cl", "tailSquare", half/8),
      middleOut("gold64_fp32_middleOut", &compiler, nullptr, &q, "fftmiddleout.cl", "fftMiddleOut", half/8),
      widthOut("gold64_fp32_fftW", &compiler, nullptr, &q, "fftw.cl", "fftW", half/8),
      heightIn("gold64_fp32_fftHin", &compiler, nullptr, &q, "ffthin.cl", "fftHin", half/8),
      tailMul("gold64_fp32_tailMul", &compiler, nullptr, &q, "tailmul.cl", "tailMul", half/16),
      exportCoefficients("gold64_fp32_unweight", &compiler, nullptr, &q,
                         "fp32_gold64_bridge.cl", "gold64_fp32_unweight", half) {}

  void squareCoefficients(const Buffer<std::int64_t>& input, Buffer<float>& coefficients) {
    requireSize(coefficients);
    begin(input);
    tailSquare(b, a, setup.height);
    end(coefficients);
  }
  void prepare(const Buffer<std::int64_t>& rhs) {
    requireSize(rhs);
    preparedReady = false;
    if (!prepared) prepared = std::make_unique<Buffer<double>>(nullptr, queue, half);
    begin(rhs);
    // tailMul consumes the preserved RHS in the exact fftMidIn state.
    // Advancing it through fftHin here makes the FP32 quotient hint describe
    // a different convolution from the Gold64 residue plane.
    *prepared << a;
    preparedReady = true;
  }
  void multiplyCoefficients(const Buffer<std::int64_t>& lhs, Buffer<float>& coefficients) {
    requireSize(coefficients);
    if (!preparedReady) throw std::logic_error("FP32/Gold64 multiply requires a prepared RHS");
    begin(lhs);
    tailMul(b, a, *prepared, setup.height);
    end(coefficients);
  }
};


class M31Plane {
  static constexpr U32 n = GpuPlane::words;
  static constexpr U32 half = n / 2;
  Queue* queue;
  M31Setup setup;
  KernelCompiler compiler;
  Buffer<std::int64_t> transposed;
  Buffer<double> a, b; // GF31 data occupies one double per complex pair.
  std::unique_ptr<Buffer<double>> prepared;
  bool preparedReady = false;
  bool fusedOut = false;
  Kernel transpose, premul, middleIn, tailSquare, middleOut, widthOut,
         widthOutExport, tailMul, exportResidues;

  static bool fusedOutEnabled(Queue& q) {
    if (const char* value = std::getenv("AEVUM_GOLD64_M31_FUSED_OUT"))
      return *value && *value != '0';

    cl_uint vendor = 0u;
    const auto id = q.context->deviceId();
    if (clGetDeviceInfo(id, CL_DEVICE_VENDOR_ID, sizeof(vendor), &vendor, nullptr) != CL_SUCCESS)
      throw std::runtime_error("M31 fused-output selection could not query device vendor");
    // RTX benefits from eliminating the final GF31 materialization/reload.
    // gfx906 is neutral/slightly slower, so leave AMD on the established path.
    return vendor == GpuPlane::nvidiaVendor;
  }

  template<class T> static void requireSize(const Buffer<T>& x) {
    if (x.size != n) throw std::invalid_argument("M31/Gold64 requires 4194304 natural-order digits");
  }
  void begin(const Buffer<std::int64_t>& input) {
    requireSize(input);
    transpose(transposed, input);
    premul(b, transposed, setup.width);
    middleIn(a, b, setup.middle);
  }
  void end(Buffer<U32>& residues) {
    middleOut(a, b, setup.middle);
    if (fusedOut) {
      widthOutExport(residues, a, setup.width);
    } else {
      widthOut(b, a, setup.width);
      exportResidues(b, residues);
    }
  }

public:
  M31Plane(const Args& args, Queue& q, U32 exponent)
    : queue(&q), setup(makeM31Setup(args, q, exponent)),
      compiler(args, q.context, setup.defines),
      transposed(nullptr, &q, n), a(nullptr, &q, half), b(nullptr, &q, half),
      fusedOut(fusedOutEnabled(q)),
      transpose("gold64_m31_import", &compiler, nullptr, &q,
                "fp32_gold64_bridge.cl", "gold64_m31_import", half),
      premul("gold64_m31_fftP", &compiler, nullptr, &q, "fftp.cl", "fftP", half/8),
      middleIn("gold64_m31_middleIn", &compiler, nullptr, &q,
               "fftmiddlein.cl", "fftMiddleInGF31", half/8),
      tailSquare("gold64_m31_tailSquare", &compiler, nullptr, &q,
                 "tailsquare.cl", "tailSquareGF31", half/8),
      middleOut("gold64_m31_middleOut", &compiler, nullptr, &q,
                "fftmiddleout.cl", "fftMiddleOutGF31", half/8),
      widthOut("gold64_m31_fftW", &compiler, nullptr, &q,
               "fftw.cl", "fftWGF31", half/8),
      widthOutExport("gold64_m31_fftW_unweight", &compiler, nullptr, &q,
                     "fp32_gold64_bridge.cl", "gold64_m31_fftW_unweight", half/8),
      tailMul("gold64_m31_tailMul", &compiler, nullptr, &q,
              "tailmul.cl", "tailMulGF31", half/16),
      exportResidues("gold64_m31_unweight", &compiler, nullptr, &q,
                     "fp32_gold64_bridge.cl", "gold64_m31_unweight", half/8) {}

  void squareResidues(const Buffer<std::int64_t>& input, Buffer<U32>& residues) {
    requireSize(residues);
    begin(input);
    tailSquare(b, a, setup.height);
    end(residues);
  }
  void prepare(const Buffer<std::int64_t>& rhs) {
    requireSize(rhs);
    preparedReady = false;
    if (!prepared) prepared = std::make_unique<Buffer<double>>(nullptr, queue, half);
    begin(rhs);
    *prepared << a;
    preparedReady = true;
  }
  void multiplyResidues(const Buffer<std::int64_t>& lhs, Buffer<U32>& residues) {
    requireSize(residues);
    if (!preparedReady) throw std::logic_error("M31/Gold64 multiply requires a prepared RHS");
    begin(lhs);
    tailMul(b, a, *prepared, setup.height);
    end(residues);
  }
};

// Both outputs describe the same uncarried IBDWT coefficient X_j:
// residue[j] = X_j mod p; estimate[j] approximates X_j (not X_j/p).
// Preparing an RHS and changing it updates both planes together.
class HybridConvolution {
  GpuPlane gold;
  std::unique_ptr<Fp32Plane> fp32;
  M31Plane m31;
  Queue* queue;
  bool preparedReady = false;
  bool fp32Ready = false;
  bool profileStages = false;
  double lastGoldUs = 0.0;
  double lastM31Us = 0.0;
  double lastFp32Us = 0.0;

  static bool envEnabled(const char* name) {
    const char* value = std::getenv(name);
    return value && *value && *value != '0';
  }
  static bool enableFp32Diagnostics() {
    return envEnabled("AEVUM_GOLD64_FP32_DIAGNOSTICS");
  }
  static U32 requireGeometry(U32 q) {
    Tables::requireGeometry(GpuPlane::words, q);
    // Numeric input format for this experiment, NOT a proven FP32 error bound.
    if (q > U64(33) * GpuPlane::words)
      throw std::invalid_argument("FP32/Gold64 experiment supports digit widths up to 33 bits");
    return q;
  }
  static void requireBuffers(const Buffer<std::int64_t>& input,
                             const Buffer<U64>& residue, const Buffer<U32>& residue31,
                             const Buffer<float>& estimate) {
    if (input.size != GpuPlane::words || residue.size != GpuPlane::words ||
        residue31.size != GpuPlane::words || estimate.size != GpuPlane::words)
      throw std::invalid_argument("M31/Gold64 coefficient buffer size mismatch");
    if (input.get() == residue.get() || input.get() == residue31.get() ||
        input.get() == estimate.get() || residue.get() == residue31.get() ||
        residue.get() == estimate.get() || residue31.get() == estimate.get())
      throw std::invalid_argument("M31/Gold64 inputs and plane outputs must not alias");
  }
public:
  HybridConvolution(const Args& args, Queue& q, U32 exponent)
    : gold(args, q, requireGeometry(exponent)),
      fp32(enableFp32Diagnostics()
               ? std::make_unique<Fp32Plane>(args, q, exponent)
               : nullptr),
      m31(args, q, exponent), queue(&q),
      profileStages(envEnabled("AEVUM_GOLD64_STAGE_PROFILE")) {}

  bool fp32DiagnosticsReady() const { return fp32Ready; }
  bool fp32DiagnosticsEnabled() const { return bool(fp32); }
  bool stageProfileEnabled() const { return profileStages; }
  void profileFinish() const { queue->finish(); }
  double goldStageUs() const { return lastGoldUs; }
  double m31StageUs() const { return lastM31Us; }
  double fp32StageUs() const { return lastFp32Us; }

  template<class F>
  double profileStage(F&& fn) {
    queue->finish();
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    queue->finish();
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - t0).count();
  }

  void square(const Buffer<std::int64_t>& input, Buffer<U64>& residue,
              Buffer<U32>& residue31, Buffer<float>& estimate) {
    requireBuffers(input, residue, residue31, estimate);
    fp32Ready = false;
    lastGoldUs = lastM31Us = lastFp32Us = 0.0;
    if (profileStages) {
      lastGoldUs = profileStage([&] { gold.squareResidues(input, residue); });
      lastM31Us = profileStage([&] { m31.squareResidues(input, residue31); });
      if (fp32) {
        lastFp32Us = profileStage([&] { fp32->squareCoefficients(input, estimate); });
        fp32Ready = true;
      }
    } else {
      gold.squareResidues(input, residue);
      m31.squareResidues(input, residue31);
      if (fp32) {
        fp32->squareCoefficients(input, estimate);
        fp32Ready = true;
      }
    }
  }
  void prepare(const Buffer<std::int64_t>& rhs) {
    preparedReady = false;
    fp32Ready = false;
    gold.prepare(rhs);
    m31.prepare(rhs);
    if (fp32) fp32->prepare(rhs);
    preparedReady = true;
  }
  void multiply(const Buffer<std::int64_t>& lhs, Buffer<U64>& residue,
                Buffer<U32>& residue31, Buffer<float>& estimate) {
    requireBuffers(lhs, residue, residue31, estimate);
    if (!preparedReady) throw std::logic_error("M31/Gold64 multiply requires all prepared exact planes");
    fp32Ready = false;
    gold.multiplyResidues(lhs, residue);
    m31.multiplyResidues(lhs, residue31);
    if (fp32) {
      fp32->multiplyCoefficients(lhs, estimate);
      fp32Ready = true;
    }
  }
};
} // namespace aevum_gold64
#endif
