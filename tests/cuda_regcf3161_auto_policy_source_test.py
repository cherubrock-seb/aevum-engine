#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
engine = (root / "src" / "EngineApi.cpp").read_text()
gpu = (root / "src" / "Gpu.cpp").read_text()
use = (root / "src" / "PrpUseTune.cpp").read_text()

for token in (
    'workload_ == aevum_autotune::Workload::Prp',
    'device_name == "NVIDIA GeForce RTX 3080"',
    'getNvidiaComputeCapability(selected_device) == 806u',
    'exponent_ >= 135000013u',
    'exponent_ <= 158000003u',
    'fft.spec() == "1:512:8:512:202"',
    'pfa_use.emplace_back("REGCF3161", "2")',
    'args_.uses("REGCF3161") || args_.uses("NOREG")',
    'AEVUM_CUDA_REG_PROFILE source=rtx3080-type1-auto',
):
    assert token in engine, token

assert 'if (enabled && std::strcmp(enabled, "0") == 0) return {};' in gpu
assert 'for (auto it = config.begin(); it != config.end();)' in gpu
assert 'aevumRegisterUseKey(it->first) || it->first == "NOREG"' in gpu
assert 'AEVUM_REGISTER_PROFILES must be 0 or 1' in gpu
assert 'std::getenv("AEVUM_REGISTER_PROFILES") : "1"' in use
assert 'if (!enabled || std::strcmp(enabled, "1") != 0) return {};' not in gpu

print("CUDA REGCF3161 automatic policy source test PASS")
