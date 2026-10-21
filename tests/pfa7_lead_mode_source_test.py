#!/usr/bin/env python3
"""Host-only source gate; actual arithmetic exactness runs on Kaggle GPUs."""
from pathlib import Path

gpu = Path("src/Gpu.cpp").read_text()
api = Path("src/EngineApi.cpp").read_text()

assert 'std::getenv("AEVUM_PFA7_LEAD_MODE")' in gpu
assert 'std::strcmp(pfa7_mode, "1")' in gpu
assert 'std::strcmp(pfa7_mode, "2")' in gpu
assert 'fft.pfa_radix == 7 && fft.shape.width == 512' in gpu
assert 'fft.shape.middle == 7 && fft.shape.height == 512' in gpu
assert 'isNvidiaGpu(id)' in gpu
assert 'fft.pfa_radix == 9' in gpu
assert 'args.value("AEVUM_PFA7_SEPARATE_CARRYB", 1)' in gpu
assert 'fftPCarryB(buf1, out);' in gpu
assert 'std::getenv("AEVUM_PFA7_LEAD_MODE")' in api
assert 'paths.lead_cache = paths.lead_cache && mode' in api
assert 'std::getenv("AEVUM_PFA_LEAD_BRIDGE")' in api
print("PFA7_EXPERIMENT_SOURCE_GATE=PASS")
