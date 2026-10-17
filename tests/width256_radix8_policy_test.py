"""Header-policy gate for explicit WIDTH=256 radix-8 capture; no GPU claim."""
from pathlib import Path
import shutil
import subprocess
import tempfile

SOURCE = (Path(__file__).resolve().parents[1] / "src/FFTConfig.h").read_text()

DRIVER = r'''
#include "FFTConfig.h"
#include <cassert>
#include <iostream>

FFTShape::FFTShape(FFT_TYPES t, u32 w, u32 m, u32 h) :
  fft_type(t), width(w), middle(m), height(h) {}

int main() {
  unsetenv("AEVUM_RADIX8_WIDTH256");
  setenv("AEVUM_RADIX1K", "4", 1);

  FFTShape baseline(FFT3161, 256, 16, 512);
  assert(baseline.nW() == 4 && baseline.nH() == 8);

  setenv("AEVUM_RADIX8_WIDTH256", "0", 1);
  FFTShape zero(FFT3161, 256, 16, 512);
  assert(zero.nW() == 4);

  setenv("AEVUM_RADIX8_WIDTH256", "1", 1);
  FFTShape enabled(FFT3161, 256, 16, 512);
  assert(enabled.nW() == 8 && enabled.nH() == 8);
  assert(enabled.experimentalRadix8Width256());

  FFTShape copy = enabled;
  FFTShape assigned;
  assigned = enabled;

  setenv("AEVUM_RADIX8_WIDTH256", "0", 1);
  FFTShape later(FFT3161, 256, 16, 512);

  assert(later.nW() == 4);
  assert(enabled.nW() == 8);
  assert(copy.nW() == 8);
  assert(assigned.nW() == 8);

  for (const char* invalid : {"", "2", "-1", "true", "01", " 1", "1 ", "x"}) {
    setenv("AEVUM_RADIX8_WIDTH256", invalid, 1);
    bool rejected = false;
    try {
      FFTShape bad;
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    assert(rejected);
  }

  setenv("AEVUM_RADIX8_WIDTH256", "1", 1);
  FFTShape heightOnly(FFT3161, 512, 8, 256);
  FFTShape oneK(FFT3161, 1024, 8, 512);
  assert(heightOnly.nW() == 8 && heightOnly.nH() == 4);
  assert(oneK.nW() == 4);

  std::cout << "width256 radix8 capture and strict values verified; GPU validation pending\n";
}
'''

COMMON = r'''
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using std::string;
using std::vector;
using std::array;
using std::max_element;
using std::to_string;
'''

compiler = shutil.which("g++") or shutil.which("c++")
if not compiler:
    raise SystemExit("C++ compiler required")

with tempfile.TemporaryDirectory(prefix="width256-radix8-") as tmp:
    root = Path(tmp)
    (root / "FFTConfig.h").write_text(SOURCE)
    (root / "common.h").write_text(COMMON)
    (root / "probe.cpp").write_text(DRIVER)
    binary = root / "probe"
    subprocess.run(
        [compiler, "-std=c++17", str(root / "probe.cpp"), "-o", str(binary)],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
