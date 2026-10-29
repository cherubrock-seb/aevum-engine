// The stand-in cuda.h in tests/cuda_stub declares only what the shim used before it answered CL_DEVICE_TOPOLOGY_AMD.
// This declares cuDeviceGetPCIBusId as the real cuda.h does; included with -include when building
// tests/cuda_topology_pci_test.cpp.
#pragma once

#include <cuda.h>

CUresult cuDeviceGetPCIBusId(char* pci, int len, CUdevice dev);
