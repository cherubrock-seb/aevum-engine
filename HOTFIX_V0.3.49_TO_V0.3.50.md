# Hotfix Aevum v0.3.49 -> v0.3.50

Overlay the hotfix onto an Aevum v0.3.49 source tree, run `make clean`, then
rebuild `engine-lib`.

The new single-plane plans are diagnostic-only.  They require an Apple build
and `AEVUM_APPLE_DIAGNOSTIC_PLANES=1`; otherwise FFT3161 remains mandatory.
