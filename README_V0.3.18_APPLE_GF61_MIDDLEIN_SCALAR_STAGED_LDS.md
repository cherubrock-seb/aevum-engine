# Aevum Engine v0.3.18 — Apple GF61 middle-in scalar staged LDS

The Apple OpenCL 1.2 path now scalarizes the front of `fftMiddleInGF61` into
load, factor-generation and element-wise apply kernels. Each work-item owns one
GF61 value, avoiding the private `GF61 u[MIDDLE]` array in the pipeline that
Apple Metal rejected. The existing scratch stores factors temporarily; the
stock middle FFT, post multiply and LDS transpose remain unchanged. Other
platforms keep the original kernel and compiler path.
