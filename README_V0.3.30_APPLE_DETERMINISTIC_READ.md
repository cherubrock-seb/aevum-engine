# Aevum Engine v0.3.30 — Apple deterministic checked reads

The v0.3.29 M2 run confirmed that LL-UNSAFE completes its smoke path, but the Apple `sum64` verification still produced a transient mismatch in both forced and automatic LL-SAFE runs. The prior `queue.finish()` therefore did not fix the underlying issue: the highly contended checksum assembled from global 32-bit atomics is not dependable on Apple's legacy OpenCL implementation.

On Apple only, `readChecked` now performs two complete synchronous `transposeOut + read` operations and compares every returned word. Matching reads are accepted; a mismatch is retried up to three times and reported with the first differing word. Non-Apple platforms retain the original GPU `sum64` verification path.

This change affects infrequent host reads only and does not modify the arithmetic kernels or their hot path.
