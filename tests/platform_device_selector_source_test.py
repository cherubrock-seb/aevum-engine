from pathlib import Path

root = Path(__file__).resolve().parents[1]
args = (root / "src/Args.cpp").read_text()
gpuid_cpp = (root / "src/gpuid.cpp").read_text()
gpuid_h = (root / "src/gpuid.h").read_text()

assert '-device <P>:<D>' in args
assert 'getPlatformDeviceFromPos(i).c_str()' in args
assert "s.find(':')" in args
assert 'parseStrictInt(s.substr(0, colon), platform)' in args
assert 'parseStrictInt(s.substr(colon + 1), platformDevice)' in args
assert 'getPosFromPlatformDevice(platform, platformDevice)' in args
assert '-device expects <N> or <platform>:<device>' in args

assert 'getGpusByPlatform()' in gpuid_cpp
assert 'CL_DEVICE_TYPE_GPU' in gpuid_cpp
assert 'int getPosFromPlatformDevice(int platform, int device)' in gpuid_cpp
assert 'string getPlatformDeviceFromPos(int pos)' in gpuid_cpp
assert 'std::find(all.begin(), all.end(), devices[device])' in gpuid_cpp

assert 'int getPosFromPlatformDevice(int platform, int device);' in gpuid_h
assert 'std::string getPlatformDeviceFromPos(int pos);' in gpuid_h

print("Aevum OpenCL platform device selector source test passed")
