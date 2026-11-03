#include "Gb202MeasuredPolicy.h"

#include <cstdlib>
#include <iostream>
#include <string>

using aevum_gb202::Decision;
using aevum_gb202::Measurement;
using aevum_gb202::Scope;
using aevum_gb202::candidateEligible;
using aevum_gb202::chooseMeasured;

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

Scope baseScope() {
  Scope scope;
  scope.device_name = "NVIDIA GeForce RTX 5090";
  scope.exponent = 147800003u;
  scope.prp = true;
  return scope;
}

}  // namespace

int main() {
  check(aevum_gb202::kPolicyVersion == "gb202-measured-v1",
        "policy identity is stable");
  check(aevum_gb202::kHistoricalPlan == "1:512:8:512:202",
        "historical candidate plan is explicit");

  {
    auto scope = baseScope();
    check(candidateEligible(scope), "RTX 5090 PRP in range is eligible");

    scope.device_name = "NVIDIA GB202";
    check(candidateEligible(scope), "GB202 name is eligible");

    scope.exponent = 146000000u;
    check(candidateEligible(scope), "lower exponent boundary is inclusive");

    scope.exponent = 150000000u;
    check(candidateEligible(scope), "upper exponent boundary is inclusive");

    scope.exponent = 145999999u;
    check(!candidateEligible(scope), "below range is rejected");

    scope.exponent = 150000001u;
    check(!candidateEligible(scope), "above range is rejected");
  }

  {
    auto scope = baseScope();

    scope.prp = false;
    check(!candidateEligible(scope), "non-PRP workload is rejected");

    scope = baseScope();
    scope.explicit_fft = true;
    check(!candidateEligible(scope), "explicit FFT keeps precedence");

    scope = baseScope();
    scope.disabled = true;
    check(!candidateEligible(scope), "explicit disable keeps precedence");

    scope = baseScope();
    scope.user_tune_dir = true;
    check(!candidateEligible(scope), "user tune directory keeps precedence");

    scope = baseScope();
    scope.device_name = "NVIDIA GeForce RTX 3080";
    check(!candidateEligible(scope), "non-GB202 device is rejected by default");

    scope.forced = true;
    check(candidateEligible(scope),
          "force remains available for explicit validation only");
  }

  {
    Measurement m;
    m.native_us = 104.0;
    m.candidate_us = 100.0;
    m.exact = true;
    m.complete = true;
    m.candidate_valid = true;
    check(chooseMeasured(m) == Decision::Candidate,
          ">=3% complete exact end-to-end gain accepts candidate");

    m.native_us = 102.9;
    check(chooseMeasured(m) == Decision::Native,
          "<3% gain fails closed to native");

    m.native_us = 100.0;
    m.candidate_us = 104.0;
    check(chooseMeasured(m) == Decision::Native,
          "slower candidate keeps native");

    m.native_us = 104.0;
    m.candidate_us = 100.0;
    m.exact = false;
    check(chooseMeasured(m) == Decision::Native,
          "residue mismatch keeps native");

    m.exact = true;
    m.complete = false;
    check(chooseMeasured(m) == Decision::Native,
          "incomplete comparison keeps native");

    m.complete = true;
    m.candidate_valid = false;
    check(chooseMeasured(m) == Decision::Native,
          "invalid candidate keeps native");

    m.candidate_valid = true;
    m.candidate_us = 0.0;
    check(chooseMeasured(m) == Decision::Native,
          "invalid timing keeps native");
  }

  if (failures) return EXIT_FAILURE;
  std::cout << "GB202 measured policy: PASS\n";
  return EXIT_SUCCESS;
}
