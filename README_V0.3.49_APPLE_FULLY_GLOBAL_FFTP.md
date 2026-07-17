# Aevum v0.3.49 — Apple fully-global fftP

Apple FFT3161 `fftP` no longer dispatches the LDS-based `fftP31Apple` kernel.
Both GF31 and GF61 are prepared with scalar weighting and global
radix/twiddle/shuffle stages.  Each destination is written exactly once and the
final plane is returned to the stock transform output buffer.

This code is compiled and dispatched only by the `__APPLE__` FFT3161 branch.
Non-Apple builds continue to call the upstream monolithic `kfftP` kernel.
