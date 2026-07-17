# Aevum Engine v0.3.34 — Apple GF61 middle-out / local-pair tailMul

- Stages large-plan `fftMiddleOutGF61` through the existing raw GF61 scratch plane.
- Writes the LDS transpose destination directly using the exact `middleShuffle` thread permutation.
- Uses one local-memory workgroup per tailMul line pair and the upstream `reverse`, `reverseLine`, and `pairMul` functions.
- Apple-only queue flush remains compile-time guarded.
