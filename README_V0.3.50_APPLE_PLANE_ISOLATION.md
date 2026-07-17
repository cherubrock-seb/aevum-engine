# Aevum v0.3.50 — Apple plane isolation diagnostic

This release adds no replacement arithmetic kernel.  It allows explicit FFT31
or FFT61 engine plans only on Apple and only while
`AEVUM_APPLE_DIAGNOSTIC_PLANES=1` is set.  Normal engine creation remains
restricted to FFT3161.

The supplied probe runs set/read and square tests for 0, 1, 2, 3 and 5 with
FFT3161, FFT31-only and FFT61-only plans.  This separates a plane-local defect
from a hybrid buffer-offset, CRT or carry interaction.

Non-Apple builds do not compile the diagnostic exception.  Linux, Windows and
CUDA retain the exact FFT3161-only validation.

Probe SHA-256:

```
5a13998ce3a0f83506c0ee78ea0e70bb02e63572a9c5809b74e8baaedfc26980  run_aevum_apple_plane_isolation_v5.sh
```
