// Targeted Gold3 performance probe after exactness gates.
// Compares public Engine API wall time for production AUTO and explicit Gold3.
#include "EngineApi.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static void fail(const std::string& what) {
  const char* e = aevum_engine_last_error();
  throw std::runtime_error(what + ": " + (e ? e : "no error"));
}

static void call(int ok, const char* what) {
  if (!ok) fail(what);
}

static std::string planOf(aevum_engine_handle h) {
  char plan[128]{};
  if (!aevum_engine_plan_spec(h, plan, sizeof(plan))) return "(unavailable)";
  return plan;
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n & 1 ? v[n/2] : 0.5 * (v[n/2-1] + v[n/2]);
}

enum class Lane { Square, LlGeneric, LlGold3Dedicated };

static double oneBatch(aevum_engine_handle h, Lane lane, int iterations) {
  call(aevum_engine_set_u32(h, 0, 3), "set_u32");
  call(aevum_engine_sync(h), "sync reset");

  const auto t0 = Clock::now();
  for (int i = 0; i < iterations; ++i) {
    if (lane == Lane::Square) {
      call(aevum_engine_square_mul(h, 0, 1), "square");
    } else if (lane == Lane::LlGeneric) {
      call(aevum_engine_square_mul(h, 0, 1), "LL square");
      call(aevum_engine_sub_u32(h, 0, 2), "LL sub2");
    } else {
      call(aevum_engine_gold3_square_minus_two(h, 0), "Gold3 dedicated LL");
    }
  }
  call(aevum_engine_sync(h), "sync measured");
  const auto t1 = Clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;
}

static std::vector<double> runLane(aevum_engine_handle h, Lane lane, int batches, int iterations) {
  // One unreported warmup batch ensures kernels/caches are hot.
  (void)oneBatch(h, lane, 3);
  std::vector<double> samples;
  samples.reserve(batches);
  for (int b = 0; b < batches; ++b) samples.push_back(oneBatch(h, lane, iterations));
  return samples;
}

static void printLane(const char* name, const std::vector<double>& samples) {
  std::cout << name << "_SAMPLES_US=";
  for (size_t i = 0; i < samples.size(); ++i) {
    if (i) std::cout << ",";
    std::cout << std::fixed << std::setprecision(3) << samples[i];
  }
  std::cout << "\n";
  std::cout << name << "_MEDIAN_US=" << std::fixed << std::setprecision(3) << median(samples) << "\n";
}

int main(int argc, char** argv) {
  try {
    const uint32_t device = argc > 1 ? uint32_t(std::stoul(argv[1])) : 0u;
    const int batches = argc > 2 ? std::stoi(argv[2]) : 5;
    const int iterations = argc > 3 ? std::stoi(argv[3]) : 12;
    if (batches < 3 || iterations < 4) throw std::invalid_argument("need >=3 batches and >=4 iterations");

    constexpr uint32_t exponent = 136279841u;
    constexpr const char* gold3Spec = "gold3-validation:512:4:512:202";

    aevum_engine_handle prp =
        aevum_engine_create_ex(exponent, 1, device, 0, nullptr, ".", AEVUM_WORKLOAD_PRP);
    if (!prp) fail("create production PRP");
    aevum_engine_handle ll =
        aevum_engine_create_ex(exponent, 1, device, 0, nullptr, ".", AEVUM_WORKLOAD_LL);
    if (!ll) fail("create production LL");
    aevum_engine_handle gold3 =
        aevum_engine_create_ex(exponent, 1, device, 0, gold3Spec, ".", AEVUM_WORKLOAD_GOLD3_VALIDATION);
    if (!gold3) fail("create Gold3");

    std::cout << "DEVICE=" << device << "\n";
    std::cout << "EXPONENT=" << exponent << "\n";
    std::cout << "PROD_PRP_PLAN=" << planOf(prp) << "\n";
    std::cout << "PROD_LL_PLAN=" << planOf(ll) << "\n";
    std::cout << "GOLD3_PLAN=" << planOf(gold3) << "\n";
    std::cout << "BATCHES=" << batches << "\n";
    std::cout << "ITERATIONS_PER_BATCH=" << iterations << "\n";

    const auto prodSquare = runLane(prp, Lane::Square, batches, iterations);
    const auto gold3Square = runLane(gold3, Lane::Square, batches, iterations);
    const auto prodLl = runLane(ll, Lane::LlGeneric, batches, iterations);
    const auto gold3LlGeneric = runLane(gold3, Lane::LlGeneric, batches, iterations);
    const auto gold3LlDedicated = runLane(gold3, Lane::LlGold3Dedicated, batches, iterations);

    printLane("PROD_PRP_SQUARE", prodSquare);
    printLane("GOLD3_SQUARE", gold3Square);
    printLane("PROD_LL", prodLl);
    printLane("GOLD3_LL_GENERIC", gold3LlGeneric);
    printLane("GOLD3_LL_DEDICATED", gold3LlDedicated);

    const double sqRatio = median(gold3Square) / median(prodSquare);
    const double llGenericRatio = median(gold3LlGeneric) / median(prodLl);
    const double llDedicatedRatio = median(gold3LlDedicated) / median(prodLl);
    std::cout << "GOLD3_SQUARE_OVER_PROD=" << std::fixed << std::setprecision(4) << sqRatio << "\n";
    std::cout << "GOLD3_LL_GENERIC_OVER_PROD=" << llGenericRatio << "\n";
    std::cout << "GOLD3_LL_DEDICATED_OVER_PROD=" << llDedicatedRatio << "\n";

    aevum_engine_destroy(gold3);
    aevum_engine_destroy(ll);
    aevum_engine_destroy(prp);
    std::cout << "GOLD3_PERF_PROBE=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "GOLD3_PERF_PROBE=FAIL " << e.what() << "\n";
    return 1;
  }
}
