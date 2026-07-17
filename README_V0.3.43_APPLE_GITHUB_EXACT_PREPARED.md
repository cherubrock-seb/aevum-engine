# Aevum Engine v0.3.43 — Apple compatibility, unchanged FFT3161 arithmetic

Aevum remains strict `FFT3161` with both exact CRT components
`GF(M31^2)` and `GF(M61^2)`.  No Apple-specific FFT plan, modulus, weight,
carry formula, transform size or exponentiation algorithm is introduced.

On Apple only, prepared invariant multiplicands are transformed through the
existing upstream `fftHin` step and multiplied with the existing upstream
`MUL_LOW` tail kernels.  Non-Apple uses the original `regPrepare` and
`regMulPrepared` branches unchanged.

The Apple small-factor helper uses two scratch registers to avoid passing the
same OpenCL buffer as both `restrict` destination and source during doubling.
It performs the same binary add-and-double algorithm.
