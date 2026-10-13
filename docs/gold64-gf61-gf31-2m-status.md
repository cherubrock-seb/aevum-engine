# Exact 2M Gold64 + GF61 + GF31 checkpoint

- Branch: `exp/gold64-gf61-gf31-exact-2m`, independent of PR #22.
- Base / last confirmed remote commit before milestone A: `ce0b8b5ee3aa4d7c79d55858f5bb1fa783073610`.
- A implemented in `src/Gold3Plan.h` and `src/cl/gold3_arith.cl`: shared
  128-bit digit ABI, signed 192-bit arithmetic, exact Garner CRT constants,
  three residue conversions, balanced splitting, root/geometry checks.
- Integer capacity: ceiling bit grid, 64/65-bit digits, signed or unsigned
  magnitude below the per-digit radix; term weights 1 or 2 imply
  `|C| < 2^152 < Q/2`. This is conditional on all planes exporting the
  documented identical integer convolution. Factor 3 is post-CRT.
- Three GPU planes connected: no. CRT arithmetic: yes; GPU dispatch: no.
- Wide cyclic carry: split primitive only. Engine API: no.
- Next: adapt PR #22's Marin schedule to 2M and wide input, then couple
  existing GF61/GF31 transforms with explicit natural-order bridges.
- Full GPU validation cannot begin yet; foundation validation can begin.
- Production blocked. No builds, tests, benchmarks, SSH or CI polling run.
