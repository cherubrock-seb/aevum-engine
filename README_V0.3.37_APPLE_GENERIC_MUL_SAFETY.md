# Aevum v0.3.37 — Apple generic multiplication safety

The Aevum API already implements `square_mul(reg, factor)` by combining one
square with register additions.  This path does not use generic transform
multiplication and is suitable for P-1 base-3 exponentiation.

On Apple, `aevum_engine_mul` now fails closed by default while the staged GF61
generic multiplication remains under arithmetic validation.  Set
`AEVUM_APPLE_UNSAFE_GENERIC_MUL=1` only for bounded diagnostic comparisons.
Non-Apple behavior is unchanged.
