#pragma once

#include <map>
#include <string>

// Default for the INPLACE -use key when no -use, tune.txt, per-FFT or engine
// profile value was given.  nVidia defaults to the in-place layout, as
// upstream GPUOwl/PRPLL does (gwoltman/gpuowl 1c96a99): its -tune only ever
// times INPLACE=1 on nVidia, so INPLACE=0 is a layout the tuner never picks
// there.  Other vendors keep INPLACE=0.  PFA plans are left alone: their fused
// fftW scatter requires INPLACE=0, which clDefines() supplies separately.
//
// The value goes into the config map, so the C++ buffer sizing (in_place) and
// the kernels' -DINPLACE are taken from the same entry and cannot disagree.
inline void applyDefaultInPlace(std::map<std::string, std::string>& config, bool nvidia, bool pfa) {
  if (nvidia && !pfa) config.try_emplace("INPLACE", "1");
}

// Tag for autotune cache keys: records measured against the old INPLACE=0
// nVidia baseline must not be replayed as if they had been measured with the
// new default.
inline const char* defaultInPlaceKeyTag(bool nvidia) {
  return nvidia ? ";inplace-default=1" : "";
}
