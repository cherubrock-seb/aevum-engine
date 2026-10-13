// Exact Gold64*M31 reconstruction plus retained FP32 diagnostics.
// The legacy FP32 lift remains explicitly UnprovenPrecision; the active CRT
// lane is proven only under the fixed N=2^22, |digit|<=2^32 input contract.
#pragma once
#include "Fp32Gold64Gpu.h"
#include <initializer_list>
#include <chrono>
#include <iostream>

#ifndef CUDA_BACKEND
namespace aevum_gold64 {
struct WideCoefficient { U64 lo, hi; };
static_assert(sizeof(WideCoefficient) == 16, "OpenCL ulong2 coefficient ABI");
enum ReconstructionStatus : U32 {
  UnprovenPrecision = 1, Nonfinite = 2, QuotientRange = 4, FractionalHint = 8,
  InvalidResidue = 16, UnsupportedOperation = 32, CarryOverflow = 64,
  CarryNonconvergence = 128
};
// UnprovenPrecision is reserved for legacy estimate-based diagnostic paths.
// No caller-supplied tolerance or near-integer FP hint can clear it.
enum class PrecisionPolicy { RequireProven, SupervisedValidation };
struct ReconstructionReport {
  U32 flags, firstFatalBlock, fatalBlocks, unprovenBlocks;
  // A large fractional quotient hint is empirically correlated with a wrong
  // Gold64 lift on hostile dense M136279841 inputs. It remains diagnostic
  // evidence, not a precision proof, but supervised execution must fail-stop
  // instead of returning a known-unreliable candidate residue.
  static constexpr U32 fatalMask = Nonfinite | QuotientRange | FractionalHint |
      InvalidResidue | UnsupportedOperation | CarryOverflow |
      CarryNonconvergence;
  bool proven() const { return !(flags & (fatalMask | UnprovenPrecision)); }
  void requireUsable(PrecisionPolicy policy) const {
    if (flags & fatalMask)
      throw std::runtime_error("Gold64 reconstruction rejected: flags=" + std::to_string(flags) +
                               ", first fatal carry block=" + std::to_string(firstFatalBlock));
    if (policy == PrecisionPolicy::RequireProven && !proven())
      throw std::runtime_error("Gold64 UnprovenPrecision: requested path is not covered by the exact CRT contract");
  }
};
static_assert(sizeof(ReconstructionReport) == 16, "OpenCL uint4 diagnostic ABI");

struct QuotientDiagnostics {
  U32 coefficients = 0;
  U32 fp32Fractional = 0;
  U32 fp64Fractional = 0;
  U32 quotientDisagreements = 0;
  U32 fp64Range = 0;
  U32 firstDisagreement = ~U32(0);
  U32 firstFp32Fractional = ~U32(0);
  U32 firstFp64Fractional = ~U32(0);
};

class ReconstructionExperiment {
  static constexpr U32 n = GpuPlane::words;
  static constexpr U32 amdVendor = 0x1002u;
  const U32 exponent;
  const U32 carryBlock;
  const U32 blocks;
  const U32 groups;
  HybridConvolution convolution;
  KernelCompiler compiler;
  Buffer<U64> residues;
  Buffer<U32> residues31;
  Buffer<float> estimates;
  Buffer<std::int64_t> blockCarry;
  Buffer<U32> blockStatus, maps, prefix, groupMaps, groupPrefix, totalMap;
  Buffer<U32> quotientFlags;
  Buffer<ReconstructionReport> partialReports, summary;
  Kernel lift, liftFp64Diagnostic, liftM31Diagnostic, split, quotientProbe;
  Kernel carryLocal, carryMaps, scanBlocks, scanGroups, carryFinish;
  Kernel statusBlocks, statusFinish;
  bool reportReady = false;

  static U32 chooseCarryBlock(Queue& queue) {
    if (const char* overrideValue = std::getenv("AEVUM_GOLD64_CARRY_BLOCK")) {
      char* end = nullptr;
      const unsigned long parsed = std::strtoul(overrideValue, &end, 10);
      if (!*overrideValue || !end || *end || (parsed != 64ul && parsed != 128ul && parsed != 256ul))
        throw std::invalid_argument("AEVUM_GOLD64_CARRY_BLOCK must be 64, 128, or 256");
      return static_cast<U32>(parsed);
    }
    cl_uint vendor = 0u;
    const auto id = queue.context->deviceId();
    if (clGetDeviceInfo(id, CL_DEVICE_VENDOR_ID, sizeof(vendor), &vendor, nullptr) != CL_SUCCESS)
      throw std::runtime_error("Gold64 carry block selection could not query device vendor");
    // gfx906 benefits strongly from four times more carry-level parallelism;
    // NVIDIA benefits from the longer serial block and lower block overhead.
    return vendor == amdVendor ? 64u : 256u;
  }
  static std::string carryCompilerDefines(U32 carryBlock) {
    return std::string("-DGOLD64_N=4194304 -DG64_CARRY_BLOCK=") +
           std::to_string(carryBlock);
  }

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
  double lastLiftUs = 0.0;
  double lastCarryLocalUs = 0.0;
  double lastCarryMapsUs = 0.0;
  double lastScanBlocksUs = 0.0;
  double lastScanGroupsUs = 0.0;
  double lastCarryFinishUs = 0.0;
  double lastStatusUs = 0.0;

  template<class F>
  double profileStage(F&& fn) {
    convolution.profileFinish();
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    convolution.profileFinish();
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - t0).count();
  }

  void emit(Buffer<WideCoefficient>& candidates, Buffer<U32>& status, U32 factor, bool ll) {
    // Exact supervised-validation reconstruction now uses two exact modular
    // convolution planes. FP32 estimates remain retained for diagnostics only.
    lastLiftUs = 0.0;
    if (convolution.stageProfileEnabled()) {
      lastLiftUs = profileStage([&] {
        liftM31Diagnostic(residues, residues31, candidates, status,
                          cl_uint(factor), cl_uint(ll));
      });
    } else {
      liftM31Diagnostic(residues, residues31, candidates, status,
                        cl_uint(factor), cl_uint(ll));
    }
  }

public:
  ReconstructionExperiment(const Args& args, Queue& queue, U32 q)
    : exponent(q), carryBlock(chooseCarryBlock(queue)),
      blocks(n / carryBlock), groups(blocks / 256u),
      convolution(args, queue, q),
      compiler(args, queue.context, carryCompilerDefines(carryBlock)),
      residues(nullptr, &queue, n), residues31(nullptr, &queue, n),
      estimates(nullptr, &queue, n),
      blockCarry(nullptr, &queue, blocks),
      blockStatus(nullptr, &queue, blocks), maps(nullptr, &queue, blocks),
      prefix(nullptr, &queue, blocks), groupMaps(nullptr, &queue, groups),
      groupPrefix(nullptr, &queue, groups), totalMap(nullptr, &queue, 1),
      quotientFlags(nullptr, &queue, n),
      partialReports(nullptr, &queue, groups), summary(nullptr, &queue, 1),
      lift("gold64_lift_candidates", &compiler, nullptr, &queue,
           "fp32_gold64_reconstruct.cl", "gold64_lift_candidates", n),
      liftFp64Diagnostic("gold64_lift_candidates_fp64_diagnostic", &compiler, nullptr, &queue,
                         "fp32_gold64_reconstruct.cl", "gold64_lift_candidates_fp64_diagnostic", n),
      liftM31Diagnostic("gold64_crt_m31_candidates", &compiler, nullptr, &queue,
                        "fp32_gold64_reconstruct.cl", "gold64_crt_m31_candidates", n),
      split("gold64_split_candidates", &compiler, nullptr, &queue,
            "fp32_gold64_reconstruct.cl", "gold64_split_candidates", n),
      quotientProbe("gold64_quotient_diagnostics", &compiler, nullptr, &queue,
                    "fp32_gold64_reconstruct.cl", "gold64_quotient_diagnostics", n),
      carryLocal("gold64_carry_local", &compiler, nullptr, &queue,
                 "fp32_gold64_reconstruct.cl", "gold64_carry_local", blocks),
      carryMaps("gold64_carry_maps", &compiler, nullptr, &queue,
                "fp32_gold64_reconstruct.cl", "gold64_carry_maps", blocks),
      scanBlocks("gold64_carry_scan_blocks", &compiler, nullptr, &queue,
                 "fp32_gold64_reconstruct.cl", "gold64_carry_scan", blocks),
      scanGroups("gold64_carry_scan_groups", &compiler, nullptr, &queue,
                 "fp32_gold64_reconstruct.cl", "gold64_carry_scan", 256),
      carryFinish("gold64_carry_finish", &compiler, nullptr, &queue,
                  "fp32_gold64_reconstruct.cl", "gold64_carry_finish", blocks),
      statusBlocks("gold64_status_blocks", &compiler, nullptr, &queue,
                   "fp32_gold64_reconstruct.cl", "gold64_status_blocks", blocks),
      statusFinish("gold64_status_finish", &compiler, nullptr, &queue,
                   "fp32_gold64_reconstruct.cl", "gold64_status_finish", 256) {
    if (!carryBlock || n % carryBlock || blocks % 256u || !groups || groups > 256u)
      throw std::logic_error("Gold64 carry block selection is incompatible with the two-level scan");
    // The three-state carry map assumes the incoming signed-64 correction is
    // negligible versus a block radix. Keep at least 256 exponent bits per
    // selected block on every device.
    if (U64(q) * carryBlock < U64(256) * n)
      throw std::invalid_argument("Gold64 carry blocks require at least 256 exponent bits per block");
  }

  static void requireExecution(PrecisionPolicy policy) {
    if (policy != PrecisionPolicy::SupervisedValidation)
      throw std::runtime_error("Gold64 exact CRT is still restricted to explicit supervised validation");
  }
  void invalidateDiagnostics() { reportReady = false; }
  ReconstructionReport diagnostics() const {
    if (!reportReady) throw std::logic_error("Gold64 has no completed normalization diagnostics");
    return summary.read(1).front();
  }
  void requireFp32Diagnostics() const {
    if (!convolution.fp32DiagnosticsEnabled())
      throw std::logic_error("Gold64 FP32 diagnostics require AEVUM_GOLD64_FP32_DIAGNOSTICS=1 before handle creation");
    if (!convolution.fp32DiagnosticsReady())
      throw std::logic_error("Gold64 FP32 diagnostics require a completed square or multiply operation");
  }
  void getEstimates(float* output, size_t count) {
    if (!output || count != n)
      throw std::invalid_argument("Gold64 estimate diagnostics require 4194304 floats");
    requireFp32Diagnostics();
    estimates.read(output, count);
  }
  void getResidues(U64* output, size_t count) {
    if (!output || count != n)
      throw std::invalid_argument("Gold64 residue diagnostics require 4194304 uint64 values");
    residues.read(output, count);
  }
  void getResidues31(U32* output, size_t count) {
    if (!output || count != n)
      throw std::invalid_argument("Gold64 M31 residue diagnostics require 4194304 uint32 values");
    residues31.read(output, count);
  }
  QuotientDiagnostics quotientDiagnostics() {
    if (!reportReady)
      throw std::logic_error("Gold64 quotient diagnostics require a completed candidate operation");
    requireFp32Diagnostics();
    quotientProbe(residues, estimates, quotientFlags);
    const auto flags = quotientFlags.read(n);
    QuotientDiagnostics out;
    out.coefficients = n;
    for (U32 i = 0; i < n; ++i) {
      const U32 f = flags[i];
      if (f & 1u) {
        ++out.fp32Fractional;
        if (out.firstFp32Fractional == ~U32(0)) out.firstFp32Fractional = i;
      }
      if (f & 2u) {
        ++out.fp64Fractional;
        if (out.firstFp64Fractional == ~U32(0)) out.firstFp64Fractional = i;
      }
      if (f & 4u) {
        ++out.quotientDisagreements;
        if (out.firstDisagreement == ~U32(0)) out.firstDisagreement = i;
      }
      if (f & 8u) ++out.fp64Range;
    }
    return out;
  }

  void fp64DiagnosticNormalized(Buffer<WideCoefficient>& candidates,
                               Buffer<std::int64_t>& digits,
                               Buffer<U32>& status, U32 factor) {
    requireSize(candidates); requireSize(digits); requireSize(status);
    requireDistinct({candidates.get(), digits.get(), status.get()});
    requireOperation(factor, false);
    requireFp32Diagnostics();
    liftFp64Diagnostic(residues, estimates, candidates, status, cl_uint(factor));

    // Reuse the exact cyclic carry pipeline but deliberately do NOT publish a
    // ReconstructionReport.  The last real operation's fail-stop diagnostics
    // remain intact, and no register is committed by this diagnostic lane.
    carryLocal(candidates, digits, blockCarry, blockStatus, status, cl_uint(exponent));
    carryMaps(digits, blockCarry, blockStatus, maps, cl_uint(exponent));
    scanBlocks(maps, prefix, groupMaps, cl_uint(blocks));
    scanGroups(groupMaps, groupPrefix, totalMap, cl_uint(groups));
    carryFinish(blockCarry, blockStatus, prefix, groupPrefix, totalMap,
                 digits, status, cl_uint(exponent));

    // FractionalHint is expected to remain diagnostic here.  Reject only true
    // operational/range failures before exposing scratch output to the harness.
    constexpr U32 operationalFatal =
        Nonfinite | QuotientRange | InvalidResidue | UnsupportedOperation |
        CarryOverflow | CarryNonconvergence;
    const auto flags = status.read(n);
    for (U32 f : flags)
      if (f & operationalFatal)
        throw std::runtime_error("Gold64 FP64 diagnostic lift encountered operational failure");
  }


  void m31DiagnosticNormalized(Buffer<WideCoefficient>& candidates,
                               Buffer<std::int64_t>& digits,
                               Buffer<U32>& status, U32 factor,
                               bool subtractTwo = false) {
    requireSize(candidates); requireSize(digits); requireSize(status);
    requireDistinct({candidates.get(), digits.get(), status.get()});
    requireOperation(factor, subtractTwo);
    liftM31Diagnostic(residues, residues31, candidates, status,
                      cl_uint(factor), cl_uint(subtractTwo));

    carryLocal(candidates, digits, blockCarry, blockStatus, status, cl_uint(exponent));
    carryMaps(digits, blockCarry, blockStatus, maps, cl_uint(exponent));
    scanBlocks(maps, prefix, groupMaps, cl_uint(blocks));
    scanGroups(groupMaps, groupPrefix, totalMap, cl_uint(groups));
    carryFinish(blockCarry, blockStatus, prefix, groupPrefix, totalMap,
                 digits, status, cl_uint(exponent));

    constexpr U32 operationalFatal =
        Nonfinite | QuotientRange | InvalidResidue | UnsupportedOperation |
        CarryOverflow | CarryNonconvergence;
    const auto flags = status.read(n);
    for (U32 f : flags)
      if (f & operationalFatal)
        throw std::runtime_error("Gold64 M31 CRT diagnostic lift encountered operational failure");
  }

  void squareCandidates(const Buffer<std::int64_t>& input,
                        Buffer<WideCoefficient>& candidates, Buffer<U32>& status,
                        U32 factor = 1, bool subtractTwo = false) {
    requireSize(input); requireSize(candidates); requireSize(status);
    requireDistinct({input.get(), candidates.get(), status.get()});
    requireOperation(factor, subtractTwo);
    reportReady = false;
    convolution.square(input, residues, residues31, estimates);
    emit(candidates, status, factor, subtractTwo);
  }
  void prepare(const Buffer<std::int64_t>& rhs) { convolution.prepare(rhs); }
  void multiplyCandidates(const Buffer<std::int64_t>& lhs,
                          Buffer<WideCoefficient>& candidates, Buffer<U32>& status, U32 factor = 1) {
    requireSize(lhs); requireSize(candidates); requireSize(status);
    requireDistinct({lhs.get(), candidates.get(), status.get()});
    requireOperation(factor, false);
    reportReady = false;
    convolution.multiply(lhs, residues, residues31, estimates);
    emit(candidates, status, factor, false);
  }
  // Only a per-coefficient split, not inter-digit propagation or a PRP step.
  void splitCandidates(const Buffer<WideCoefficient>& candidates,
                       Buffer<std::int64_t>& digits, Buffer<std::int64_t>& outgoing,
                       Buffer<U32>& status) {
    requireSize(candidates); requireSize(digits); requireSize(outgoing); requireSize(status);
    requireDistinct({candidates.get(), digits.get(), outgoing.get(), status.get()});
    reportReady = false;
    split(candidates, digits, outgoing, status, cl_uint(exponent));
  }
  // Complete cyclic normalization of the supplied coefficients, modulo 2^q-1.
  // Five carry launches plus two small diagnostic reductions; no host carry
  // loop/readback and no N-word serial sweep.
  // Output is balanced, but remains a diagnostic candidate until precision is proven.
  // Any fatal status invalidates the WHOLE result, not merely the flagged digit.
  void normalizeCandidates(const Buffer<WideCoefficient>& candidates,
                           Buffer<std::int64_t>& digits, Buffer<U32>& status) {
    requireSize(candidates); requireSize(digits); requireSize(status);
    requireDistinct({candidates.get(), digits.get(), status.get()});
    reportReady = false;
    lastCarryLocalUs = lastCarryMapsUs = lastScanBlocksUs = lastScanGroupsUs = 0.0;
    lastCarryFinishUs = lastStatusUs = 0.0;

    if (convolution.stageProfileEnabled()) {
      lastCarryLocalUs = profileStage([&] {
        carryLocal(candidates, digits, blockCarry, blockStatus, status, cl_uint(exponent));
      });
      lastCarryMapsUs = profileStage([&] {
        carryMaps(digits, blockCarry, blockStatus, maps, cl_uint(exponent));
      });
      lastScanBlocksUs = profileStage([&] {
        scanBlocks(maps, prefix, groupMaps, cl_uint(blocks));
      });
      lastScanGroupsUs = profileStage([&] {
        scanGroups(groupMaps, groupPrefix, totalMap, cl_uint(groups));
      });
      lastCarryFinishUs = profileStage([&] {
        carryFinish(blockCarry, blockStatus, prefix, groupPrefix, totalMap,
                 digits, status, cl_uint(exponent));
      });
      lastStatusUs = profileStage([&] {
        statusBlocks(blockStatus, partialReports);
        statusFinish(partialReports, summary);
      });
    } else {
      carryLocal(candidates, digits, blockCarry, blockStatus, status, cl_uint(exponent));
      carryMaps(digits, blockCarry, blockStatus, maps, cl_uint(exponent));
      scanBlocks(maps, prefix, groupMaps, cl_uint(blocks));
      scanGroups(groupMaps, groupPrefix, totalMap, cl_uint(groups));
      carryFinish(blockCarry, blockStatus, prefix, groupPrefix, totalMap,
                 digits, status, cl_uint(exponent));
      statusBlocks(blockStatus, partialReports);
      statusFinish(partialReports, summary);
    }
    reportReady = true;

    if (convolution.stageProfileEnabled()) {
      std::cout
          << "GOLD64_STAGE_PROFILE"
          << " gold_us=" << convolution.goldStageUs()
          << " m31_us=" << convolution.m31StageUs()
          << " fp32_us=" << convolution.fp32StageUs()
          << " lift_us=" << lastLiftUs
          << " carry_local_us=" << lastCarryLocalUs
          << " carry_maps_us=" << lastCarryMapsUs
          << " scan_blocks_us=" << lastScanBlocksUs
          << " scan_groups_us=" << lastScanGroupsUs
          << " carry_finish_us=" << lastCarryFinishUs
          << " status_us=" << lastStatusUs
          << std::endl;
    }
  }
  void squareNormalized(const Buffer<std::int64_t>& input,
                        Buffer<WideCoefficient>& candidates, Buffer<std::int64_t>& digits,
                        Buffer<U32>& status, U32 factor = 1, bool subtractTwo = false) {
    requireSize(digits);
    requireDistinct({input.get(), candidates.get(), digits.get(), status.get()});
    squareCandidates(input, candidates, status, factor, subtractTwo);
    normalizeCandidates(candidates, digits, status);
  }
  void multiplyNormalized(const Buffer<std::int64_t>& lhs,
                          Buffer<WideCoefficient>& candidates, Buffer<std::int64_t>& digits,
                          Buffer<U32>& status, U32 factor = 1) {
    requireSize(digits);
    requireDistinct({lhs.get(), candidates.get(), digits.get(), status.get()});
    multiplyCandidates(lhs, candidates, status, factor);
    normalizeCandidates(candidates, digits, status);
  }
  // The exact CRT lane remains restricted to the explicit validation backend.
  // FP32 estimate diagnostics are opt-in and never participate in correctness.
  ReconstructionReport squareChecked(const Buffer<std::int64_t>& input,
                        Buffer<WideCoefficient>& candidates, Buffer<std::int64_t>& digits,
                        Buffer<U32>& status, U32 factor = 1, bool subtractTwo = false,
                        PrecisionPolicy policy = PrecisionPolicy::RequireProven) {
    requireExecution(policy);
    squareNormalized(input, candidates, digits, status, factor, subtractTwo);
    auto report = diagnostics();
    report.requireUsable(policy);
    return report;
  }
  ReconstructionReport multiplyChecked(const Buffer<std::int64_t>& lhs,
                          Buffer<WideCoefficient>& candidates, Buffer<std::int64_t>& digits,
                          Buffer<U32>& status, U32 factor = 1,
                          PrecisionPolicy policy = PrecisionPolicy::RequireProven) {
    requireExecution(policy);
    multiplyNormalized(lhs, candidates, digits, status, factor);
    auto report = diagnostics();
    report.requireUsable(policy);
    return report;
  }
};
} // namespace aevum_gold64
#endif
