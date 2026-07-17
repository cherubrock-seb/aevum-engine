# Aevum Engine v0.3.47

## Fix

- Removes invalid `restrict` aliasing from the two in-place Apple GF61
  middle-in stages.
- Updates the fixed trig argument index after the kernel signature change.
- Adds a source regression test that rejects separate restrict-qualified
  input/output arguments for these in-place stages.

## Scope

Apple FFT3161 only. No arithmetic or dispatch change on Linux, Windows or CUDA.

## Expected validation

The standalone arithmetic probe must first report:

```text
PASS  square: 5^2
```

Only after that result is established is prepared multiplication testing
meaningful.
