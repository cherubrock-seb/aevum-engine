# Aevum v0.3.48 — Apple stage trace

The v0.3.47 probe still fails at the first pure square (`5^2`) on Apple M2, so
prepared multiplication is not involved. This diagnostic release does not
claim an arithmetic fix. It adds diagnostics only; the normal
Linux/Windows/CUDA paths are unchanged.

1. `AEVUM_APPLE_STAGE_FINISH=1` calls `clFinish` after every decomposed GF61
   Apple stage, testing command visibility/order independently of formulas.
2. `aevum_engine_debug_square_trace` returns separate GF31/GF61 hashes after
   fftP, middle-in, tail-square, middle-out and fftW, followed by the final raw
   word-buffer hash and low word.

Build the engine, then run:

```bash
tests/run_aevum_apple_stage_trace_v4.sh "$PWD" 0 0
tests/run_aevum_apple_stage_trace_v4.sh "$PWD" 0 1
```

The logs are `aevum-apple-stage-trace-0.log` and
`aevum-apple-stage-trace-1.log`. Run the normal-mode probe on a known-good Linux
Aevum build to identify the first stage whose hash diverges.
