# Aevum Engine v0.3.21 — Apple GF61 special-tail radix-stage FFT

The Apple OpenCL/Metal backend accepts the scalar special-line load but rejects
the complete GF61 height FFT.  v0.3.21 decomposes `fft_common` into fixed stage
kernels (`s=1`, `RADIX`, `RADIX^2`, `RADIX^3`) and a final radix kernel.
Inactive stages are compile-time no-ops.  Each active stage retains the stock
GF61 radix, twiddle multiplication and LDS shuffle.
