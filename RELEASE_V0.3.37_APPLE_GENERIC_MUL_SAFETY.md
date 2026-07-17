# Aevum Engine v0.3.37

- Preserve the native small-factor `square_mul` implementation.
- Disable generic register multiplication by default on Apple.
- Add an explicit diagnostic override:
  `AEVUM_APPLE_UNSAFE_GENERIC_MUL=1`.
- Preserve all non-Apple paths unchanged.
