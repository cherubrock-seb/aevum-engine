# Aevum Engine v0.3.43

- Restores the GitHub prepared-multiplicand cache on macOS.
- Keeps strict FFT3161 / GF31+GF61 arithmetic and the original FFT plans.
- macOS only: prepares invariant operands through `fftHin` and consumes them
  through upstream `tailMulLowGF31` and `tailMulLowGF61`.
- macOS only: avoids `restrict` self-alias in small-factor doubling with a
  two-buffer ping-pong.
- Linux, Windows, CUDA and non-Apple OpenCL execution branches are unchanged.
