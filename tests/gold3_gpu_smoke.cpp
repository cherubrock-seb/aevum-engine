// Exact GPU smoke for the experimental Gold3 2M three-plane backend.
#include "Args.h"
#include "Buffer.h"
#include "Context.h"
#include "Gold3Reconstruction.h"
#include "Gold64Plan.h"
#include "Queue.h"
#include "clwrap.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace aevum_gold3;

static void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

template<class T>
static std::uint64_t sparseMismatch(const std::vector<T>& got, T expected0) {
  std::uint64_t bad = got.empty() || got[0] != expected0;
  for (std::size_t i = 1; i < got.size(); ++i) bad += got[i] != T{};
  return bad;
}

static std::uint64_t digitMismatch(const std::vector<Digit>& a, const std::vector<Digit>& b) {
  require(a.size() == b.size(), "digit vector size mismatch");
  std::uint64_t bad = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    bad += a[i].lo != b[i].lo || a[i].hi != b[i].hi;
  return bad;
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
    require(plan.digitWidth(0) == 65, "target digit zero is not 65 bits");

    const Digit wide{1, 1}; // 2^64 + 1: cannot survive an accidental int64 narrowing.
    require(g3_valid_digit(wide, 65), "65-bit probe digit rejected");
    std::vector<Digit> rhs(Plan::words);
    rhs[0] = wide;

    const U64 dg = g3_digit_gold(wide);
    const U64 dp = g3_digit61(wide);
    const U32 ds = g3_digit31(wide);
    const U64 sqG = aevum_gold64::mul(dg, dg);
    const U64 sqP = g3_mul61(dp, dp);
    const U32 sqS = g3_mod31(U64(ds) * ds);
    const U64 mulG = aevum_gold64::mul(U64(3), dg);
    const U64 mulP = g3_mul61(U64(3), dp);
    const U32 mulS = g3_mod31(U64(3) * ds);

    {
      Buffer<Digit> in(nullptr, &queue, Plan::words);
      Buffer<Digit> lhs(nullptr, &queue, Plan::words);
      Buffer<U64> gold(nullptr, &queue, Plan::words);
      Buffer<U64> m61(nullptr, &queue, Plan::words);
      Buffer<U32> m31(nullptr, &queue, Plan::words);
      in.write(rhs);

      Convolution conv(args, queue, Plan::targetExponent);
      conv.square(in, gold, m61, m31);
      queue.finish();
      auto g = gold.read();
      auto p = m61.read();
      auto s = m31.read();
      const auto bg = sparseMismatch(g, sqG);
      const auto bp = sparseMismatch(p, sqP);
      const auto bs = sparseMismatch(s, sqS);
      std::cout << "SQUARE_GOLD_MISMATCHES=" << bg << "\n";
      std::cout << "SQUARE_GF61_MISMATCHES=" << bp << "\n";
      std::cout << "SQUARE_GF31_MISMATCHES=" << bs << "\n";
      require(bg == 0 && bp == 0 && bs == 0, "three-plane square mismatch");

      std::vector<Digit> lhsHost(Plan::words);
      lhsHost[0] = Digit{3, 0};
      lhs.write(lhsHost);
      conv.prepare(in);
      conv.multiply(lhs, gold, m61, m31);
      queue.finish();
      g = gold.read();
      p = m61.read();
      s = m31.read();
      const auto mbg = sparseMismatch(g, mulG);
      const auto mbp = sparseMismatch(p, mulP);
      const auto mbs = sparseMismatch(s, mulS);
      std::cout << "MUL_GOLD_MISMATCHES=" << mbg << "\n";
      std::cout << "MUL_GF61_MISMATCHES=" << mbp << "\n";
      std::cout << "MUL_GF31_MISMATCHES=" << mbs << "\n";
      require(mbg == 0 && mbp == 0 && mbs == 0, "three-plane prepared multiply mismatch");
    }

    {
      Buffer<Digit> in(nullptr, &queue, Plan::words);
      Buffer<Digit> out(nullptr, &queue, Plan::words);
      in.write(rhs);
      Reconstruction rec(args, queue, Plan::targetExponent);
      rec.square(in, out, 1, false);
      queue.finish();
      const Report report = rec.diagnostics();
      std::cout << "RECON_FLAGS=" << report.flags << "\n";
      std::cout << "RECON_ERROR_COUNT=" << report.errorCount << "\n";
      require(report.flags == 0 && report.errorCount == 0, "reconstruction diagnostics failed");

      std::vector<Digit> expected(Plan::words);
      g3_i192 carry = g3_crt(sqG, sqP, sqS);
      for (U32 j = 0; j < Plan::words && !g3_equal(carry, g3_zero()); ++j)
        carry = g3_split(carry, plan.digitWidth(j), &expected[j]);
      require(g3_equal(carry, g3_zero()), "host carry oracle did not terminate");
      const auto got = out.read();
      const auto bad = digitMismatch(got, expected);
      std::cout << "RECON_DIGIT_MISMATCHES=" << bad << "\n";
      require(bad == 0, "CRT/carry output mismatch");
    }

    std::cout << "GOLD3_GPU_SMOKE=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "GOLD3_GPU_SMOKE=FAIL " << e.what() << "\n";
    return 1;
  }
}
