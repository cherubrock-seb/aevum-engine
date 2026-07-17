# Aevum v0.3.47 — Apple GF61 middle-in restrict alias fix

The v0.3.46 arithmetic probe fails on its first operation, the pure square
`5^2`, before prepared multiplication is reached. The failure therefore belongs
to the Apple square pipeline, not to `tailMulLowGF61`.

The Apple GF61 middle-in decomposition intentionally performs two operations
in place. In v0.3.46 those kernels exposed separate `P(T2)` output and `CP(T2)`
input arguments. Both macros are `restrict`-qualified, while the host passed the
same `cl_mem` for input and output. That violates the kernel no-alias contract
and permits the Apple OpenCL-to-Metal compiler to reorder or eliminate reads.

v0.3.47 gives each in-place operation a single data pointer:

```text
fftMiddleInGF61ApplyScalarApple(io, factor)
fftMiddleInGF61MulApple(io, trig)
```

The arithmetic, transform layout, weights, trig tables and launch geometry are
unchanged. The change is compiled and dispatched only under `__APPLE__` /
`AEVUM_APPLE_OPENCL12`. Non-Apple `fftMiddleInGF61`, Linux, Windows and CUDA
remain unchanged.
