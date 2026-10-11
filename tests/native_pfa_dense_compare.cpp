// Native Good-Thomas PFA GPU differential test with dense random residues.
//
// A pfa3:1 / pfa9:1 plan is compared word-for-word against a power-of-two
// FFT3161 plan on dense, seeded random residues: several squarings, a generic
// multiply (tailMul) and a prepared multiply (tailMulLow).
//
// Dense input is essential.  The former PFA check squared set_u32(3), whose
// transform spectrum is constant; an omitted twiddle that both the forward and
// the inverse transform skip is invisible on such input, while every word of
// a dense residue comes out wrong.
//
// usage: native-pfa-dense-compare libaevum_engine.so device exponent
//                                 reference_spec candidate_spec [iterations] [seed]
//                                 [--directional-v2]
// V2 mode: IMPLEMENTED — NOT COMPILED — NOT TESTED — NOT BENCHMARKED.
// Compare the same explicit 1K-width plan with the PRP compile profile off/on.
// A passing fallback device is not evidence that the NVIDIA warp path ran.
#include <dlfcn.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include "../src/EngineApi.h"
#include <stdexcept>
#include <string>
#include <vector>

using Handle = void*;

struct Api {
  void* so{};
  const char* (*last_error)(){};
  int (*resolve)(uint32_t, const char*, char*, size_t){};
  Handle (*create)(uint32_t, size_t, uint32_t, int, const char*, const char*){};
  decltype(&aevum_engine_create_ex) create_ex{};
  decltype(&aevum_engine_plan_spec) plan_spec{};
  void (*destroy)(Handle){};
  size_t (*transform_size)(Handle){};
  size_t (*word_count)(Handle){};
  int (*set_words)(Handle, size_t, const uint32_t*, size_t){};
  int (*get_words)(Handle, size_t, uint32_t*, size_t){};
  int (*prepare)(Handle, size_t, size_t){};
  int (*square_mul)(Handle, size_t, uint32_t){};
  int (*mul)(Handle, size_t, size_t, uint32_t){};

  template <typename T> T sym(const char* name) {
    void* p = dlsym(so, name);
    if (!p) throw std::runtime_error(std::string("missing symbol ") + name);
    return reinterpret_cast<T>(p);
  }

  explicit Api(const char* path, bool directional = false) {
    so = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!so) throw std::runtime_error(dlerror());
    last_error = sym<decltype(last_error)>("aevum_engine_last_error");
    resolve = sym<decltype(resolve)>("aevum_engine_resolve_fft");
    create = sym<decltype(create)>("aevum_engine_create");
    if (directional) {
      create_ex = sym<decltype(create_ex)>("aevum_engine_create_ex");
      plan_spec = sym<decltype(plan_spec)>("aevum_engine_plan_spec");
    }
    destroy = sym<decltype(destroy)>("aevum_engine_destroy");
    transform_size = sym<decltype(transform_size)>("aevum_engine_transform_size");
    word_count = sym<decltype(word_count)>("aevum_engine_word_count");
    set_words = sym<decltype(set_words)>("aevum_engine_set_words");
    get_words = sym<decltype(get_words)>("aevum_engine_get_words");
    prepare = sym<decltype(prepare)>("aevum_engine_prepare");
    square_mul = sym<decltype(square_mul)>("aevum_engine_square_mul");
    mul = sym<decltype(mul)>("aevum_engine_mul");
  }

  ~Api() { if (so) dlclose(so); }
};

static void require(Api& api, int rc, const char* what) {
  if (!rc) throw std::runtime_error(std::string(what) + ": " +
                                    (api.last_error() ? api.last_error() : "unknown"));
}

// Only this opt-in test mode changes the process environment; restore it on
// success and on exceptions. The engine's generic create() ignores PRP_USE.
struct ScopedEnvironment {
  const char* key;
  bool present;
  std::string previous;

  explicit ScopedEnvironment(const char* name) : key(name), present(std::getenv(name) != nullptr),
      previous(present ? std::getenv(name) : "") {}

  void set(const std::string& value) {
    if (setenv(key, value.c_str(), 1) != 0) { throw std::runtime_error("setenv failed"); }
  }

  ~ScopedEnvironment() {
    if (present) { setenv(key, previous.c_str(), 1); }
    else { unsetenv(key); }
  }
};

static void require_directional_plan(const std::string& spec) {
  std::istringstream input(spec);
  std::vector<std::string> parts;
  std::string part;
  while (std::getline(input, part, ':')) { parts.push_back(part); }
  if (parts.size() != 5 || (parts[1] != "1K" && parts[1] != "1024") ||
      parts[4].size() != 3 || parts[4][0] != '2') {
    throw std::runtime_error("V2 comparison requires an explicit non-PFA 1K-width, width-variant-2 plan");
  }
}

struct Run {
  std::string resolved;
  size_t transform{};
  std::vector<std::vector<uint32_t>> outputs;
  std::vector<std::string> labels;
};

static uint64_t hash_words(const std::vector<uint32_t>& words) {
  uint64_t h = 1469598103934665603ULL;
  for (uint32_t value : words) { h ^= value; h *= 1099511628211ULL; }
  return h;
}

// Uniformly random E-bit residue (the engine masks the top word to E bits).
static std::vector<uint32_t> dense_words(std::mt19937& rng, uint32_t exponent, size_t count) {
  std::vector<uint32_t> words(count);
  for (uint32_t& w : words) w = rng();
  if (exponent % 32) words.back() &= (uint32_t(1) << (exponent % 32)) - 1;
  return words;
}

static Run execute(Api& api, uint32_t exponent, uint32_t device,
                   const char* spec, unsigned iterations, unsigned seed, bool directional = false) {
  char resolved[256]{};
  require(api, api.resolve(exponent, spec, resolved, sizeof(resolved)), "resolve");
  if (directional) { require_directional_plan(resolved); }
  Handle h = directional
    ? api.create_ex(exponent, 2, device, 1, spec, ".", AEVUM_WORKLOAD_PRP)
    : api.create(exponent, 2, device, 1, spec, ".");
  if (!h) throw std::runtime_error(std::string("create ") + spec + ": " +
                                   (api.last_error() ? api.last_error() : "failed"));

  struct Cleanup {
    Api& api;
    Handle handle;
    ~Cleanup() { api.destroy(handle); }
  } cleanup{api, h};

  Run result;
  result.resolved = resolved;
  if (directional) {
    char actual[256]{};
    require(api, api.plan_spec(h, actual, sizeof(actual)), "actual plan");
    require_directional_plan(actual);
    if (result.resolved != actual) {
      throw std::runtime_error("requested/resolved plan changed at creation");
    }
    result.resolved = actual;
  }
  result.transform = api.transform_size(h);
  const size_t count = api.word_count(h);
  auto capture = [&](size_t reg, const std::string& label) {
    std::vector<uint32_t> words(count);
    require(api, api.get_words(h, reg, words.data(), words.size()), "get_words");
    result.outputs.push_back(std::move(words));
    result.labels.push_back(label);
  };

  // The same seed produces the same residues for both plans.
  std::mt19937 rng(seed);
  std::vector<uint32_t> x = dense_words(rng, exponent, count);
  require(api, api.set_words(h, 0, x.data(), x.size()), "set dense square input");
  for (unsigned i = 0; i < iterations; ++i) {
    require(api, api.square_mul(h, 0, (i & 1u) ? 3u : 1u), "square_mul");
    capture(0, "square " + std::to_string(i + 1));
  }

  // Generic multiply by a dense operand (tailMul).
  std::vector<uint32_t> y = dense_words(rng, exponent, count);
  require(api, api.set_words(h, 1, y.data(), y.size()), "set dense mul source");
  require(api, api.mul(h, 0, 1, 1), "mul");
  capture(0, "generic mul");

  // Prepared multiply by another dense operand (tailMulLow).
  std::vector<uint32_t> z = dense_words(rng, exponent, count);
  require(api, api.set_words(h, 1, z.data(), z.size()), "set dense prepared source");
  require(api, api.prepare(h, 1, 1), "prepare");
  require(api, api.mul(h, 0, 1, 3), "mul prepared");
  capture(0, "prepared mul");

  // One more squaring after the multiplies.
  require(api, api.square_mul(h, 0, 1), "square_mul final");
  capture(0, "final square");

  // Consecutive squarings without an intervening read: the engine keeps one
  // operation pending and executes it with a retained width transform, so a
  // PFA plan with a lead cache runs fftW + carryA + fftPCarryB (lazy carryB
  // inside the Good-Thomas gather) instead of the canonical carry sequence.
  for (unsigned i = 0; i < 4; ++i)
    require(api, api.square_mul(h, 0, 1), "square_mul chain");
  capture(0, "square chain x4");

  return result;
}

int main(int argc, char** argv) {
  try {
    if (argc < 6) {
      std::cerr << "usage: " << argv[0]
                << " libaevum_engine.so device exponent reference_spec candidate_spec"
                   " [iterations] [seed] [--directional-v2]\n";
      return 2;
    }
    const bool directional = argc == 9 && std::string(argv[8]) == "--directional-v2";
    if (argc > 8 && !directional) { throw std::runtime_error("unknown extra argument"); }
    Api api(argv[1], directional);
    const uint32_t device = static_cast<uint32_t>(std::stoul(argv[2]));
    const uint32_t exponent = static_cast<uint32_t>(std::stoul(argv[3]));
    const char* reference = argv[4];
    const char* candidate = argv[5];
    const unsigned iterations = argc > 6 ? static_cast<unsigned>(std::stoul(argv[6])) : 3u;
    const unsigned seed = argc > 7 ? static_cast<unsigned>(std::stoul(argv[7])) : 20261002u;

    Run left, right;
    if (directional) {
#if defined(__APPLE__)
      throw std::runtime_error("V2 NVIDIA ownership path is not enabled on Apple");
#endif
      const char* radix = std::getenv("AEVUM_RADIX1K");
      if (!radix || std::string(radix) != "8") {
        throw std::runtime_error("V2 comparison requires explicit AEVUM_RADIX1K=8");
      }
      if (std::string(reference) != candidate || iterations == 0) {
        throw std::runtime_error("V2 comparison requires identical plans and nonzero iterations");
      }
      ScopedEnvironment profile("AEVUM_PRP_USE");
      ScopedEnvironment tuning("AEVUM_PRP_USE_TUNE");
      // Preserve the owner's other flags identically in both profiles.
      if (profile.previous.find("AEVUM_DIRECTIONAL_FUSION") != std::string::npos) {
        throw std::runtime_error("remove directional flags from AEVUM_PRP_USE; the comparator sets them");
      }
      const std::string common = profile.previous.empty() ? "" : profile.previous + ",";
      tuning.set("off");
      profile.set(common + "AEVUM_DIRECTIONAL_FUSION_V2=0");
      std::cout << "reference PRP profile: " << std::getenv("AEVUM_PRP_USE") << std::endl;
      left = execute(api, exponent, device, reference, iterations, seed, true);
      profile.set(common + "AEVUM_DIRECTIONAL_FUSION_V2=1");
      std::cout << "candidate PRP profile: " << std::getenv("AEVUM_PRP_USE") << std::endl;
      right = execute(api, exponent, device, candidate, iterations, seed, true);
      if (left.resolved != right.resolved || left.transform != right.transform) {
        throw std::runtime_error("V2 actual-plan or transform-size mismatch");
      }
    } else {
      left = execute(api, exponent, device, reference, iterations, seed);
      right = execute(api, exponent, device, candidate, iterations, seed);
    }

    if (left.outputs.size() != right.outputs.size())
      throw std::runtime_error("output-count mismatch");

    for (size_t output = 0; output < left.outputs.size(); ++output) {
      if (left.outputs[output] != right.outputs[output]) {
        size_t word = 0, differing = 0;
        while (word < left.outputs[output].size() &&
               left.outputs[output][word] == right.outputs[output][word]) ++word;
        for (size_t i = 0; i < left.outputs[output].size(); ++i)
          differing += left.outputs[output][i] != right.outputs[output][i];
        std::cerr << "MISMATCH output=" << output << " (" << left.labels[output] << ")"
                  << " first_word=" << word
                  << " differing_words=" << differing << "/" << left.outputs[output].size()
                  << " reference_hash=0x" << std::hex << hash_words(left.outputs[output])
                  << " candidate_hash=0x" << hash_words(right.outputs[output]) << std::dec << "\n";
        const size_t begin = word > 3 ? word - 3 : 0;
        const size_t end = std::min(word + 5, left.outputs[output].size());
        for (size_t i = begin; i < end; ++i)
          std::cerr << "  word[" << i << "] " << reference << "=" << left.outputs[output][i]
                    << " " << candidate << "=" << right.outputs[output][i]
                    << (i == word ? "  <-- first" : "") << "\n";
        return 1;
      }
      std::cout << "exact output " << output << " (" << left.labels[output] << ") OK hash=0x"
                << std::hex << hash_words(left.outputs[output]) << std::dec << "\n";
    }

    std::cout << "reference=" << left.resolved << " size=" << left.transform << "\n";
    std::cout << "candidate=" << right.resolved << " size=" << right.transform
              << " bits/word=" << double(exponent) / double(right.transform) << "\n";
    if (directional) {
      std::cout << "DIRECTIONAL V2 PROFILE A/B RESIDUES MATCH\n"
                   "Confirm the compiled activation gate; fallback is not enabled-path coverage.\n";
    } else {
      std::cout << "NATIVE AEVUM PFA DENSE DIFFERENTIAL TEST PASSED\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
