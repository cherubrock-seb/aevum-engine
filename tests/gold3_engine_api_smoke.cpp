// Public Engine API smoke for the explicit-only Gold3 2M validation backend.
#include "EngineApi.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void fail(const char* what) {
  const char* e = aevum_engine_last_error();
  throw std::runtime_error(std::string(what) + ": " + (e ? e : "no error"));
}

static void require_call(int ok, const char* what) {
  if (!ok) fail(what);
}

static void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(what);
}

static void require_small_value(aevum_engine_handle h, size_t reg, uint32_t expected) {
  const size_t count = aevum_engine_word_count(h);
  std::vector<uint32_t> words(count);
  require_call(aevum_engine_get_words(h, reg, words.data(), words.size()), "get_words");
  require(!words.empty() && words[0] == expected, "unexpected low word");
  const uint64_t nonzero = std::count_if(words.begin(), words.end(), [](uint32_t x) { return x != 0; });
  require(nonzero == (expected ? 1u : 0u), "unexpected high compact words");
}

int main(int argc, char** argv) {
  try {
    const uint32_t device = argc > 1 ? uint32_t(std::stoul(argv[1])) : 0u;
    constexpr uint32_t exponent = 136279841u;
    constexpr size_t n = 2097152u;
    constexpr const char* spec = "gold3-validation:512:4:512:202";

    aevum_engine_handle badWorkload =
        aevum_engine_create_ex(exponent, 2, device, 0, spec, ".", AEVUM_WORKLOAD_GENERIC);
    require(badWorkload == nullptr, "Gold3 plan accepted without validation workload");
    std::cout << "GATE_PLAN_WITHOUT_WORKLOAD=PASS\n";

    aevum_engine_handle badPlan =
        aevum_engine_create_ex(exponent, 2, device, 0, nullptr, ".", AEVUM_WORKLOAD_GOLD3_VALIDATION);
    require(badPlan == nullptr, "Gold3 workload accepted without explicit plan");
    std::cout << "GATE_WORKLOAD_WITHOUT_PLAN=PASS\n";

    aevum_engine_handle h =
        aevum_engine_create_ex(exponent, 2, device, 1, spec, ".", AEVUM_WORKLOAD_GOLD3_VALIDATION);
    if (!h) fail("gold3 create_ex");

    require(aevum_engine_transform_size(h) == n, "unexpected Gold3 transform size");
    char plan[128]{};
    require_call(aevum_engine_plan_spec(h, plan, sizeof(plan)), "plan_spec");
    std::cout << "PLAN=" << plan << "\n";

    std::vector<uint64_t> limbs(2 * n, 0);
    limbs[0] = 1;
    limbs[1] = 1;
    require_call(aevum_engine_gold3_set_digits(h, 0, limbs.data(), n), "gold3_set_digits");
    std::fill(limbs.begin(), limbs.end(), 0);
    require_call(aevum_engine_gold3_get_digits(h, 0, limbs.data(), n), "gold3_get_digits");
    require(limbs[0] == 1 && limbs[1] == 1, "65-bit raw digit did not roundtrip");
    require(std::count_if(limbs.begin() + 2, limbs.end(), [](uint64_t x) { return x != 0; }) == 0,
            "raw digit roundtrip polluted other digits");
    std::cout << "RAW_65BIT_ROUNDTRIP=PASS\n";

    require_call(aevum_engine_set_u32(h, 0, 2), "set_u32(2)");
    require_call(aevum_engine_square_mul(h, 0, 1), "square 2");
    require_small_value(h, 0, 4);
    std::cout << "API_SQUARE_2_TO_4=PASS\n";

    require_call(aevum_engine_gold3_square_minus_two(h, 0), "square_minus_two");
    require_small_value(h, 0, 14);
    std::cout << "API_SQUARE_MINUS_TWO_4_TO_14=PASS\n";

    require_call(aevum_engine_set_u32(h, 1, 3), "set_u32(3)");
    require_call(aevum_engine_prepare(h, 1, 1), "prepare 3");
    require_call(aevum_engine_mul(h, 0, 1, 1), "mul 14 by 3");
    require_small_value(h, 0, 42);
    std::cout << "API_PREPARED_MUL_14_BY_3=PASS\n";

    aevum_engine_gold3_report report{};
    require_call(aevum_engine_gold3_diagnostics(h, &report), "gold3_diagnostics");
    std::cout << "DIAGNOSTICS_FLAGS=" << report.flags << "\n";
    std::cout << "DIAGNOSTICS_ERRORS=" << report.error_count << "\n";
    require(report.flags == 0 && report.error_count == 0, "Gold3 diagnostics reported failure");

    require_call(aevum_engine_copy(h, 1, 0), "copy 42");
    int equal = 0;
    require_call(aevum_engine_equal(h, 0, 1, &equal), "equal copied 42");
    require(equal == 1, "copied Gold3 registers are not equal");
    std::cout << "API_COPY_EQUAL=PASS\n";

    require_call(aevum_engine_sync(h), "sync");
    aevum_engine_destroy(h);
    std::cout << "GOLD3_ENGINE_API_SMOKE=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "GOLD3_ENGINE_API_SMOKE=FAIL " << e.what() << "\n";
    return 1;
  }
}
