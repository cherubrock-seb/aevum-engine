#!/usr/bin/env python3
"""Model the incompatible PFA7 layouts used by Aevum and PRPLL-NTT.

PRPLL's PFA carry is indexed by PAIRS (p mod 7), whereas Aevum's
Good-Thomas gather assigns rows per WORD (n mod 7). Before any native
carryFused port, the two representations must be mapped explicitly.
This is a model, not a GPU kernel or an arithmetic exactness proof.
"""
R = 7
WIDTH = SMALL_HEIGHT = 512
LP = WIDTH * SMALL_HEIGHT       # pairs per radix-7 subtransform
LW = 2 * LP                     # words per radix-7 subtransform
INV_LW = pow(LW, -1, R)
INV_LP = pow(LP, -1, R)
assert LW * R == 3670016 and LP * R == 1835008

def aevum_word(row, binary_word):
    n = binary_word + LW * (((row - binary_word) * INV_LW) % R)
    assert n % R == row and n % LW == binary_word
    return n

def prpll_pair(x, line):
    q = x * SMALL_HEIGHT + line % SMALL_HEIGHT
    return q + LP * (((line % R - q) * INV_LP) % R)

def native_coords_from_word(n):
    pair = n // 2
    row = pair % R
    q = pair % LP
    x, y = divmod(q, SMALL_HEIGHT)
    line = y + SMALL_HEIGHT * ((row - y) % R)
    assert prpll_pair(x, line) == pair
    return x, line

tested = 0
samples = list(range(1024)) + list(range(LW - 1024, LW)) + list(range(0, LW, 131))
for row in range(R):
    for binary_word in samples:
        n = aevum_word(row, binary_word)
        x, line = native_coords_from_word(n)
        assert 0 <= x < WIDTH and 0 <= line < R * SMALL_HEIGHT
        # PRPLL packs two consecutive words into one pair; the two words
        # occupy different Aevum radix rows, r and (r+1)%7.
        pair = n // 2
        assert (2 * pair) % R != (2 * pair + 1) % R
        tested += 1
print(f"PFA7_PRPLL_NATIVE_LAYOUT_MODEL=PASS {tested} Aevum-to-PRPLL scalar mappings")
