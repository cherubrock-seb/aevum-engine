# Aevum Engine v0.3.17 — Apple GF61 middle-in staged LDS

The Apple OpenCL 1.2 path now decomposes `fftMiddleInGF61` into four in-order
kernels. Arithmetic stages use global ping-pong through existing buffers; the
last stage retains the upstream local-memory transpose. Other platforms keep
the original kernel and compiler path.
