# Aevum v0.3.52 — Apple canonical `set_u32`

The v0.3.51 plane-isolation run still failed on `roundtrip 0` even after the
register transpose was replaced.  Inspection showed that the probe did not use
`set_words()` or `transposeIn`: `aevum_engine_set_u32()` called
`Buffer::set()`, which performs a device fill and writes the first internal
word directly.

That is not a canonical integer import for Aevum's balanced-digit register
layout.  On Apple OpenCL the large-buffer fill also produced an invalid zero
roundtrip.

On Apple only, `Gpu::regSetU32()` now executes:

```cpp
writeIn(dst, makeWords(E, value));
```

This reuses the validated compact-word expansion and register-upload path.  It
also fixes the internal zero initialization used by small-factor
multiplication.  Non-Apple builds retain the previous fast `Buffer::set()`
path unchanged.
