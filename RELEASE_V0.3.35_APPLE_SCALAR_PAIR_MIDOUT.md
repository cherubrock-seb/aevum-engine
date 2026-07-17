# Aevum Engine v0.3.35

## Purpose

Advance the Apple OpenCL 1.2 port past two Metal pipeline failures observed in
v99.39:

- `INVALID_KERNEL (-48) tailMulGF61PairLocalApple`;
- `INVALID_KERNEL (-48) fftMiddleOutGF61CoreApple`.

## Implementation

### GF61 generic multiply

The central stock operation is decomposed into two scalar kernels.  The special
self-paired lines and normal partner lines are handled separately.  No local
memory and no private GF61 arrays are used in either pipeline.

### GF61 middle-out

For `MIDDLE >= 8`, the monolithic middle-out operation is split into scalar
load/multiply/multiply2/write stages plus one isolated `fft_MIDDLE` stage.

## Compatibility

All changes are guarded by Apple compilation or Apple FFT3161 dispatch.
Non-Apple and Marin paths are unchanged.

## Validation

- OpenCL C 1.2 production matrix, including WIDTH=1024/MIDDLE=8/CARRY64;
- exhaustive scalar pair coordinate coverage;
- middle-out scratch/output bijection;
- Aevum host/source audits;
- engine API load and plan selection;
- `Gpu.cpp` syntax in Apple and non-Apple branches.

No GitHub tag is created before real M2 validation.
