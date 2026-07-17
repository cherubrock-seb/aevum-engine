# Aevum Engine v0.3.27 — Apple GF61 normal-line main-tail global staging

The Apple OpenCL 1.2 path stages every non-special GF61 tail-square line through the existing input/output transform planes. It preserves the stock line pairing, height-FFT shuffle mapping, cross-line reverse mapping, pair-square coverage, and trig indexing without allocating another transform-sized buffer. The original double-wide LDS kernel remains the non-Apple implementation.
