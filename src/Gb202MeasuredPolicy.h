#pragma once

#include <cstdint>
#include <string_view>

namespace aevum_gb202 {

inline constexpr std::string_view kPolicyVersion = "gb202-measured-v1";
inline constexpr std::string_view kHistoricalPlan = "1:512:8:512:202";
inline constexpr double kMinimumEndToEndGain = 1.03;

struct Scope {
  std::string_view device_name;
  uint32_t exponent = 0;
  bool prp = false;
  bool explicit_fft = false;
  bool disabled = false;
  bool user_tune_dir = false;
  bool forced = false;
};

inline bool deviceMatches(std::string_view name) {
  return name.find("RTX 5090") != std::string_view::npos ||
         name.find("GB202") != std::string_view::npos;
}

inline bool candidateEligible(const Scope& scope) {
  if (!scope.prp ||
      scope.explicit_fft ||
      scope.disabled ||
      scope.user_tune_dir ||
      scope.exponent < 146000000u ||
      scope.exponent > 150000000u)
    return false;

  return scope.forced || deviceMatches(scope.device_name);
}

struct Measurement {
  double native_us = 0.0;
  double candidate_us = 0.0;
  bool exact = false;
  bool complete = false;
  bool candidate_valid = false;
};

enum class Decision {
  Native,
  Candidate,
};

inline Decision chooseMeasured(const Measurement& measurement,
                               double minimum_gain = kMinimumEndToEndGain) {
  if (!measurement.exact ||
      !measurement.complete ||
      !measurement.candidate_valid ||
      !(measurement.native_us > 0.0) ||
      !(measurement.candidate_us > 0.0) ||
      !(minimum_gain > 1.0))
    return Decision::Native;

  return measurement.candidate_us * minimum_gain <= measurement.native_us
      ? Decision::Candidate
      : Decision::Native;
}

}  // namespace aevum_gb202
