# Aevum Engine v0.3.19 — Apple GF61 tailSquare two-kernel LDS

v0.3.19 is a macOS-only compatibility refinement on top of v0.3.18.
Linux, Windows and CUDA retain their existing compiler path, kernel sources,
configuration and dispatch.

The M2 smoke test created the complete staged `fftMiddleInGF61` chain, including
the final LDS transpose, and then failed while Metal created the default
`tailSquareGF61` pipeline (`TAIL_KERNELS=2`).  This proves that neither GF61 nor
LDS is generally unsupported.

For Apple FFT3161 only, Aevum now forces the existing upstream
`TAIL_KERNELS=3` mode:

- double-wide processing is retained;
- `tailSquareZeroGF61` handles lines 0 and H/2;
- `tailSquareGF61` handles the remaining paired lines;
- both kernels retain their original local-memory FFT/reverse algorithms;
- no extra transform-sized buffer is allocated.

The change only selects an already-supported execution policy.  It does not
rewrite the GF61 arithmetic, the LDS routines, or the non-Apple path.
