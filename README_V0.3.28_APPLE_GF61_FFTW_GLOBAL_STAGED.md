# Aevum Engine v0.3.28 — Apple GF61 fftW global staging

The Apple M2 OpenCL 1.2 translator accepts `fftMiddleOutGF61` but rejects the subsequent monolithic `fftWGF61` pipeline. This release stages only the Apple FFT3161 GF61 `fftW` path:

1. scalar load from the exact PAD=0 `readCarryFusedLine` layout;
2. in-place lane-owned radix;
3. twiddle plus global shuffle into the alternate existing buffer;
4. final radix into the requested output buffer.

The output buffer and the consumed middle-out input buffer are reused as ping-pong banks. No transform-sized allocation is added. The original LDS kernel is unchanged and selected on non-Apple builds.
