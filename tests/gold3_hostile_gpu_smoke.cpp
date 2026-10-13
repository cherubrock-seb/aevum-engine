// Hostile cross-vendor correctness smoke for Gold3.
// Exercises signed multi-index weighting/order and a full-length cyclic carry.
#include "Args.h"
#include "Buffer.h"
#include "Context.h"
#include "Gold3Reconstruction.h"
#include "Queue.h"
#include "clwrap.h"

#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using boost::multiprecision::cpp_int;
using namespace aevum_gold3;

static void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

static Args makeArgs(unsigned ordinal) {
  Args args(true);
  args.device = static_cast<int>(ordinal);
  args.verbose = true;
  args.safeMath = true;
  args.clean = true;
  args.useCache = true;
  args.cacheDir = std::filesystem::absolute(".aevum-kernel-cache");
  args.masterDir = std::filesystem::absolute(".");
  args.setDefaults();
  return args;
}

static Digit small(std::int64_t value) {
  if (value >= 0) return {U64(value), 0};
  return {U64(value), ~U64(0)};
}

static cpp_int signedDigit(Digit d) {
  cpp_int x = (cpp_int(d.hi) << 64) + d.lo;
  if (d.hi >> 63) x -= cpp_int(1) << 128;
  return x;
}

static U64 mod64(const cpp_int& x, U64 modulus) {
  cpp_int r = x % cpp_int(modulus);
  if (r < 0) r += modulus;
  return r.convert_to<U64>();
}

static U32 mod32(const cpp_int& x, U32 modulus) {
  cpp_int r = x % cpp_int(modulus);
  if (r < 0) r += modulus;
  return r.convert_to<U32>();
}

struct Entry {
  U32 index;
  Digit digit;
};

static std::vector<Entry> uniqueEntries(std::vector<Entry> entries) {
  std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.index < b.index; });
  entries.erase(std::unique(entries.begin(), entries.end(),
                            [](const Entry& a, const Entry& b) { return a.index == b.index; }),
                entries.end());
  return entries;
}

static std::vector<Digit> materialize(const Plan& plan, const std::vector<Entry>& entries) {
  std::vector<Digit> digits(Plan::words);
  for (const auto& e : entries) {
    require(e.index < Plan::words, "hostile index outside Gold3 geometry");
    require(g3_valid_digit(e.digit, plan.digitWidth(e.index)), "hostile digit outside width bound");
    digits[e.index] = e.digit;
  }
  return digits;
}

static std::unordered_map<U32, cpp_int> sparseConvolution(
    const Plan& plan, const std::vector<Entry>& lhs, const std::vector<Entry>& rhs,
    U64& weightOneTerms, U64& weightTwoTerms) {
  std::unordered_map<U32, cpp_int> result;
  weightOneTerms = weightTwoTerms = 0;
  for (const auto& a : lhs) {
    for (const auto& b : rhs) {
      const U64 sum = U64(a.index) + b.index;
      const bool wrap = sum >= Plan::words;
      const U32 k = U32(sum & (Plan::words - 1));
      const std::int64_t delta =
          std::int64_t(plan.bitOffset(a.index)) + plan.bitOffset(b.index) -
          plan.bitOffset(k) - (wrap ? std::int64_t(plan.exponent) : 0);
      require(delta == 0 || delta == 1, "Gold3 weight lemma violated in host oracle");
      if (delta) ++weightTwoTerms; else ++weightOneTerms;
      cpp_int term = signedDigit(a.digit) * signedDigit(b.digit);
      if (delta) term <<= 1;
      result[k] += term;
    }
  }
  return result;
}

static void checkResidues(const std::unordered_map<U32, cpp_int>& expected,
                          const std::vector<U64>& gold,
                          const std::vector<U64>& m61,
                          const std::vector<U32>& m31,
                          const char* label) {
  require(gold.size() == Plan::words && m61.size() == Plan::words && m31.size() == Plan::words,
          "residue vector size mismatch");
  U64 badG = 0, badP = 0, badS = 0;
  U32 firstG = ~U32(0), firstP = ~U32(0), firstS = ~U32(0);
  for (U32 j = 0; j < Plan::words; ++j) {
    auto it = expected.find(j);
    const cpp_int value = it == expected.end() ? cpp_int(0) : it->second;
    const U64 eg = mod64(value, G3_G);
    const U64 ep = mod64(value, G3_P61);
    const U32 es = mod32(value, G3_P31);
    if (gold[j] != eg) { if (!badG) firstG = j; ++badG; }
    if (m61[j] != ep) { if (!badP) firstP = j; ++badP; }
    if (m31[j] != es) { if (!badS) firstS = j; ++badS; }
  }
  std::cout << label << "_GOLD_MISMATCHES=" << badG << " FIRST=" << firstG << "\n";
  std::cout << label << "_GF61_MISMATCHES=" << badP << " FIRST=" << firstP << "\n";
  std::cout << label << "_GF31_MISMATCHES=" << badS << " FIRST=" << firstS << "\n";
  require(badG == 0 && badP == 0 && badS == 0, "hostile three-plane residue mismatch");
}

int main(int argc, char** argv) {
  try {
    const unsigned ordinal = argc > 1 ? unsigned(std::stoul(argv[1])) : 0u;
    const auto devices = getAllDeviceIDs();
    require(ordinal < devices.size(), "OpenCL device ordinal out of range");
    std::cout << "DEVICE=" << ordinal << ":" << getDeviceName(devices[ordinal]) << "\n";

    Args args = makeArgs(ordinal);
    Context context(devices[ordinal]);
    Queue queue(context, false);
    Plan plan(Plan::targetExponent);

    U32 boundary = 1;
    while (boundary + 1 < Plan::words && plan.digitWidth(boundary) == plan.digitWidth(boundary - 1)) ++boundary;
    require(boundary + 1 < Plan::words, "no 64/65 width transition found");
    std::cout << "WIDTH_BOUNDARY=" << boundary
              << " LEFT=" << plan.digitWidth(boundary - 1)
              << " RIGHT=" << plan.digitWidth(boundary) << "\n";

    std::vector<Entry> lhs = uniqueEntries({
      {0, Digit{1, 1}},             // +2^64+1, true 65-bit input.
      {1, small(-3)},
      {31, small(5)},
      {32, small(-7)},
      {511, small(11)},
      {512, small(-13)},
      {1023, small(17)},
      {1024, small(-19)},
      {boundary - 1, small(23)},
      {boundary, small(-29)},
      {Plan::words - 2, small(31)},
      {Plan::words - 1, small(-37)}
    });
    std::vector<Entry> rhs = uniqueEntries({
      {0, small(41)},
      {2, small(-43)},
      {63, small(47)},
      {64, small(-53)},
      {513, small(59)},
      {1025, small(-61)},
      {boundary, small(67)},
      {Plan::words - 1, small(-71)}
    });

    auto lhsHost = materialize(plan, lhs);
    auto rhsHost = materialize(plan, rhs);
    Buffer<Digit> lhsGpu(nullptr, &queue, Plan::words);
    Buffer<Digit> rhsGpu(nullptr, &queue, Plan::words);
    Buffer<U64> gold(nullptr, &queue, Plan::words);
    Buffer<U64> m61(nullptr, &queue, Plan::words);
    Buffer<U32> m31(nullptr, &queue, Plan::words);
    lhsGpu.write(lhsHost);
    rhsGpu.write(rhsHost);

    Convolution conv(args, queue, Plan::targetExponent);

    U64 w1 = 0, w2 = 0;
    auto squareOracle = sparseConvolution(plan, lhs, lhs, w1, w2);
    require(w1 && w2, "hostile square did not exercise both coefficient weights");
    std::cout << "SQUARE_WEIGHT1_TERMS=" << w1 << "\n";
    std::cout << "SQUARE_WEIGHT2_TERMS=" << w2 << "\n";
    conv.square(lhsGpu, gold, m61, m31);
    queue.finish();
    checkResidues(squareOracle, gold.read(), m61.read(), m31.read(), "HOSTILE_SQUARE");

    auto mulOracle = sparseConvolution(plan, lhs, rhs, w1, w2);
    require(w1 && w2, "hostile multiply did not exercise both coefficient weights");
    std::cout << "MUL_WEIGHT1_TERMS=" << w1 << "\n";
    std::cout << "MUL_WEIGHT2_TERMS=" << w2 << "\n";
    conv.prepare(rhsGpu);
    conv.multiply(lhsGpu, gold, m61, m31);
    queue.finish();
    checkResidues(mulOracle, gold.read(), m61.read(), m31.read(), "HOSTILE_MUL");

    // Full-length cyclic carry: d_j = 2^(w_j-1)-1. The zero-seeded
    // segment pass leaves tail +1; therefore the cyclic correction delta is 0
    // (encoded state 1), while the physical carry entering each segment is +1.
    // The normalized digit is -1 at every j.
    std::vector<Digit> dense(Plan::words);
    for (U32 j = 0; j < Plan::words; ++j) {
      const U32 width = plan.digitWidth(j);
      dense[j] = width == 64
          ? Digit{0x7fffffffffffffffULL, 0}
          : Digit{0xffffffffffffffffULL, 0};
    }
    Buffer<Digit> denseGpu(nullptr, &queue, Plan::words);
    Buffer<Digit> denseOut(nullptr, &queue, Plan::words);
    denseGpu.write(dense);
    Reconstruction carry(args, queue, Plan::targetExponent);
    carry.linearOperation(denseGpu, denseGpu, denseOut, 1);
    queue.finish();
    const Report report = carry.diagnostics();
    std::cout << "LONG_CARRY_FLAGS=" << report.flags << "\n";
    std::cout << "LONG_CARRY_ERRORS=" << report.errorCount << "\n";
    std::cout << "LONG_CARRY_STATE=" << report.cyclicState << "\n";
    require(report.flags == 0 && report.errorCount == 0 && report.cyclicState == 1,
            "full-length cyclic carry correction-state mismatch");

    const auto normalized = denseOut.read();
    U64 badDense = 0;
    U32 firstDense = ~U32(0);
    for (U32 j = 0; j < Plan::words; ++j) {
      if (normalized[j].lo != ~U64(0) || normalized[j].hi != ~U64(0)) {
        if (!badDense) firstDense = j;
        ++badDense;
      }
    }
    std::cout << "LONG_CARRY_DIGIT_MISMATCHES=" << badDense << " FIRST=" << firstDense << "\n";
    require(badDense == 0, "full-length cyclic carry output mismatch");

    std::cout << "GOLD3_HOSTILE_GPU_SMOKE=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "GOLD3_HOSTILE_GPU_SMOKE=FAIL " << e.what() << "\n";
    return 1;
  }
}
