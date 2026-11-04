#!/usr/bin/env python3
from pathlib import Path

gpu = Path("src/Gpu.cpp").read_text()
api = Path("src/EngineApi.cpp").read_text()

assert 'std::getenv("AEVUM_PFA7_LEAD_MODE")' in gpu
assert 'std::strcmp(pfa7_mode, "1")' in gpu
assert 'std::strcmp(pfa7_mode, "2")' in gpu
assert 'isNvidiaGpu(shared.context->deviceId())' in gpu
assert 'fft.pfa_radix == 7 && fft.shape.width == 512' in gpu
assert 'fft.shape.middle == 7 && fft.shape.height == 512 && fft.variant == 202' in gpu
assert 'args.value("AEVUM_PFA7_SEPARATE_CARRYB", 1)' in gpu
assert 'std::strcmp(std::getenv("AEVUM_PFA7_LEAD_MODE"), "2") == 0' in gpu
assert 'fftPCarryB(buf1, out);' in gpu
assert 'std::getenv("AEVUM_PFA7_LEAD_MODE")' in api
assert 'paths.lead_cache = paths.lead_cache && mode' in api
assert 'std::getenv("AEVUM_PFA_LEAD_BRIDGE")' in api

print("ok: PFA7 retained-width lead cache is NVIDIA/exact-geometry/default-OFF with separate and fused bridge modes")
