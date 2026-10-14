// INPLACE default: in-place on nVidia, as upstream GPUOwl/PRPLL; unchanged elsewhere; never over an explicit value or a PFA plan.
#include "InPlaceDefault.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

static int failures = 0;

static void check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

static std::string inplace(const std::map<std::string, std::string>& config) {
  auto it = config.find("INPLACE");
  return it == config.end() ? "<unset>" : it->second;
}

int main(int argc, char** argv) {
  {
    std::map<std::string, std::string> c;
    applyDefaultInPlace(c, true, false);
    check(inplace(c) == "1", "nVidia non-PFA defaults to INPLACE=1");
  }
  {
    std::map<std::string, std::string> c;
    applyDefaultInPlace(c, false, false);
    check(inplace(c) == "<unset>", "other vendors keep the INPLACE=0 compiled default (key left unset)");
  }
  {
    std::map<std::string, std::string> c{{"INPLACE", "0"}};
    applyDefaultInPlace(c, true, false);
    check(inplace(c) == "0", "explicit INPLACE=0 wins on nVidia");
  }
  {
    std::map<std::string, std::string> c{{"INPLACE", "2"}};
    applyDefaultInPlace(c, true, false);
    check(inplace(c) == "2", "explicit INPLACE=2 wins on nVidia");
  }
  {
    std::map<std::string, std::string> c;
    applyDefaultInPlace(c, true, true);
    check(inplace(c) == "<unset>", "PFA plans are not given INPLACE=1");
  }
  check(std::strcmp(defaultInPlaceKeyTag(true), ";inplace-default=1") == 0, "nVidia cache keys are tagged");
  check(std::strcmp(defaultInPlaceKeyTag(false), "") == 0, "other vendors' cache keys are unchanged");

  // clDefines must apply the default after the PFA INPLACE=0 default and before the loop that reads INPLACE into in_place,
  // so the buffer sizing and the kernel -D come from the same value.
  if (argc > 1) {
    std::ifstream in(argv[1]);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string src = ss.str();
    const auto pfa = src.find("config.try_emplace(\"INPLACE\", \"0\");");
    const auto apply = src.find("applyDefaultInPlace(config, isNvidiaGpu(id), fft.isPfa());");
    const auto read = src.find("if (k == \"INPLACE\") in_place = atoi(v.c_str());");
    check(pfa != std::string::npos && apply != std::string::npos && read != std::string::npos,
          "Gpu.cpp has the PFA default, the nVidia default and the in_place read");
    check(pfa < apply && apply < read, "nVidia INPLACE default sits between the PFA default and the in_place read");
  }

  if (failures) return EXIT_FAILURE;
  std::printf("inplace_default_test: OK\n");
  return EXIT_SUCCESS;
}
