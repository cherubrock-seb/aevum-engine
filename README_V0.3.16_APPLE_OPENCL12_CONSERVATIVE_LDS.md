# Aevum Engine v0.3.16 — Apple OpenCL 1.2 conservative-LDS port

This release keeps the upstream PRPLL/Aevum OpenCL 2.0 path unchanged on
Linux, Windows and CUDA. On macOS, it selects a compatibility layer for
Apple's OpenCL C 1.2 frontend:

- monolithic in-memory source expansion and `clBuildProgram`;
- `get_local_size()` compatibility for the OpenCL 2.0-only
  `get_enqueued_local_size()` use in `halfBar()`;
- corrected GF31 private/local address-space cast;
- OpenCL 1.2 ready-flag atomics for `carryFused`, while preserving the
  original LDS stairway, global fences and OpenCL 2.0 atomic expansion;
- conservative Apple middle work-group defaults with both middle LDS
  transposes retained;
- Apple-only compact staging of the `fftP` GF61 width path;
- one Apple-only retry with `-cl-opt-disable` if Metal rejects the optimized
  pipeline.

The source suite parses every OpenCL root instantiated by `Gpu.cpp` as OpenCL
C 1.2 in representative GF31/GF61 configurations. Real Apple Silicon runtime
validation remains required.
