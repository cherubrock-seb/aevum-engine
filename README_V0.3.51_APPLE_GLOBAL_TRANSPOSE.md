# Aevum v0.3.51 — Apple direct global register transpose

The v0.3.50 plane-isolation probe showed that `set_words()` followed immediately
by `get_words()` failed even for an all-zero register. This happens before
`fftP`, middle-in, tail arithmetic, carry, or CRT reconstruction.

On Apple OpenCL 1.2, Aevum v0.3.51 therefore replaces only the register
sequential/transposed conversion kernels:

- `transposeInAppleGlobal`
- `transposeOutAppleGlobal`

Each work-item copies one `Word2` using the exact matrix-transpose mapping. The
Apple kernels use no local memory and no barriers. Non-Apple builds continue to
use the upstream 64x64 LDS tiled kernels unchanged.

The first validation criterion is a successful plane-isolation roundtrip for
0, 1, 2, 3, and 5. Arithmetic tests should only be interpreted after the
roundtrip passes.
