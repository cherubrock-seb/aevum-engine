# Aevum Engine v0.3.36

Correct the Apple staged GF61 generic-multiply special-line trig derivation.
Both special lines now start from the stock line-zero base trig; only `H/2`
receives one `TAILTGF61` factor.  Non-Apple paths are unchanged.
