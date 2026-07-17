# Aevum Engine v0.3.32 — Apple-only queue-marker flush

v0.3.31 fixed an Apple OpenCL command-queue stall by flushing immediately after enqueueing the periodic marker. v0.3.32 places that `clFlush` behind `#if defined(__APPLE__)`, preserving the original NVIDIA, Linux and Windows queue behavior.

No arithmetic kernels changed.
