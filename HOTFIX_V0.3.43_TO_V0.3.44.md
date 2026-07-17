# Aevum hotfix v0.3.43 to v0.3.44

Apple-only: replace creation of the rejected monolithic `fftHinGF61` Metal pipeline with an exact global-memory decomposition of the same GF61 height transform. Non-Apple code and FFT selection are unchanged.
