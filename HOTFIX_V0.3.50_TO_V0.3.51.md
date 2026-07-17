# Hotfix Aevum v0.3.50 to v0.3.51

This hotfix changes the Apple-only register upload/readback transpose from the
stock 64x64 LDS kernel to a direct global one-work-item-per-`Word2` mapping.

No Linux, Windows, CUDA, or non-Apple OpenCL dispatch is changed.
