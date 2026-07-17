# Aevum Engine v0.3.23 — Apple GF61 special-tail global reverse staging

The M2 smoke for v0.3.22 proved that the private radix, scalar twiddle,
global shuffle and final radix kernels all create successfully.  Apple Metal
then rejected the remaining LDS-based `tailSquareZeroGF61ReverseApple` kernel.

For the two exceptional GF61 tail lines only, v0.3.23 replaces that reverse by
a race-free global permutation between the existing two tiny scratch banks.
The first reverse maps bank 0 to bank 1, the scalar pair-square operates in
bank 1, and the second reverse maps bank 1 back to bank 0.

The mapping is exactly equivalent to the stock `reverse` LDS write/read
semantics, including the one-position bump used by line zero.  No additional
scratch is allocated.  The main double-wide GF61 tail kernel and all
non-Apple LDS paths remain unchanged.
