# Experimental NVIDIA PFA7 retained-width carry (Kaggle)

Branch: `exp/pfa7-nvidia-retained-carry-kaggle`. The default AUTO/production path does **not** change.

This is a deliberately small experiment, not a port of PRPLL's full PFA-aware `carryFused` arithmetic. It tests a prerequisite: whether keeping `LEAD_WIDTH` between PFA7 squarings and fusing the existing `carryB + fftP` gather improves PRP squaring throughput.

At the exact FFT3161 NVIDIA geometry `pfa7:1:512:7:512:202`, set:

- `AEVUM_PFA7_LEAD_MODE=0` or unset: unchanged canonical PFA7 baseline.
- `AEVUM_PFA7_LEAD_MODE=1`: retained-width PFA7 path, separate `carryB` and `fftP` (control).
- `AEVUM_PFA7_LEAD_MODE=2`: retained-width PFA7 path, existing `fftPCarryB` fusion (candidate).

The experimental support is gated in `Gpu::regSupportsLeadCache()` and the PRP `productionRegPaths()`. It is NVIDIA-only, PFA7-only, FFT3161-only and requires the exact dimensions. Long-carry and Apple paths remain excluded.

**Correctness first:** `tools/pfa7_kaggle_probe.py` creates a Type2 reference `2:512:8:512:202`, runs dense seeded input through a sequence of squares and a factor-3 square, compares **all 32-bit output words**, and times consecutive squares only on a match. Run each mode as a *separate process*.

The paired comparison is short and synthetic, not a full PRP. It must not be treated as sufficient to enable production defaults; further independent dense/GMP checks, long PRP exactness and GPU-specific A/B are required.

Kaggle quick runner: see attached `PrMers_Kaggle_PFA7_Carry_Experimental.ipynb` in the corresponding ChatGPT conversation. It builds CUDA engine-lib only, runs modes 0 and 2 on device 0, and emits `pfa7-carry-kaggle-evidence.zip`.

**Do not merge based on successful compilation alone.** In particular, the existing fused `fftPCarryB` was historically slower than separate `carryB + fftP` on RTX 3080; Tesla T4 may behave differently.
