# Aevum Engine v0.3.33 — Apple MIDDLE=8 and stock-reverse GF61 tailMul

- Fixes Apple OpenCL C 1.2 compilation of GF31 `fft8Core` for large
  `MIDDLE=8` plans by inlining the exact `fft4Core` body only under
  `AEVUM_APPLE_OPENCL12`.
- Replaces the Apple staged GF61 direct-coordinate multiply with the literal
  stock sequence: reverse partner operands, execute pair multiplication at
  matching coordinates, then undo the partner reverse.
- Adds full production-root syntax coverage for the `WIDTH=1024`, `MIDDLE=8`,
  `CARRY64` plan.
- Leaves non-Apple arithmetic and queue behavior unchanged.
