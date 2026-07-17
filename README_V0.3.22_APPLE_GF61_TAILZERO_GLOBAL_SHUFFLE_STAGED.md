# Aevum Engine v0.3.22 — Apple GF61 special-tail global-shuffle staging

Apple Metal rejects even the first combined GF61 radix/twiddle/LDS-shuffle
stage used by the two exceptional tail lines.  v0.3.22 separates that stage
into a private radix kernel, a scalar twiddle kernel and a global permutation
kernel.

The permutation uses two tiny scratch banks and exactly reproduces the logical
result of `shufl`.  A final radix kernel writes back to bank zero.  The normal
double-wide tail, reverse, middle and carry LDS paths are unchanged.
