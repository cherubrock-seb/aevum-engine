# Hotfix Aevum v0.3.46 -> v0.3.47

Replace the following files:

- `src/cl/fftmiddlein.cl`
- `src/Gpu.cpp`
- `src/version.inc`
- `Makefile`
- `tests/apple_gf61_middlein_staging_test.py`
- `tests/apple_gf61_middlein_restrict_alias_test.py`
- `tests/source_audit.sh`

The functional change is Apple-only and removes two illegal input/output aliases
between `restrict`-qualified kernel arguments.
