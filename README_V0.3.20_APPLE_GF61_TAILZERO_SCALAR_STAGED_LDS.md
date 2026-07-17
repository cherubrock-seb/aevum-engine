# Aevum Engine v0.3.20 — Apple GF61 special-tail scalar staged LDS

v0.3.20 refines only the Apple FFT3161 path. The M2 smoke showed that all
staged middle-in kernels were accepted and that Metal then rejected the
monolithic `tailSquareZeroGF61` pipeline.

The two exceptional lines, 0 and H/2, now use a small scratch buffer and six
kernels: load, FFT1, reverse, scalar pair-square, FFT2 and write. The height FFT
and reverse stages preserve their local-memory implementation. The normal
`tailSquareGF61` remains the upstream double-wide LDS kernel for all other
lines.

The non-Apple compiler, kernels, ABI and dispatch are unchanged.
