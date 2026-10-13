"""Arithmetic model for the opt-in PFA7 convolution decomposition; not a GPU test."""
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/cl/fft-middle.cl").read_text()
FIELDS = (
    (31, 1513477735, (141971251, 405239506, 526531066),
     (752284155, 1347360268, 1073059520)),
    (61, 1669582390241348315,
     (1773674115463894065, 901087505085424499, 784002893271222362),
     (673484497713682504, 831884051671952528, 381307857222696375)),
)

def dft(values, root, p):
    return [sum(x * pow(root, j*k, p) for j, x in enumerate(values)) % p
            for k in range(len(values))]

def conv(values, spectrum, root, p):
    transformed = dft(values, root, p)
    return dft([a*b % p for a, b in zip(transformed, spectrum)], pow(root, -1, p), p)

def candidate(u, kc, ks, r, p):
    a = [(u[j] + u[7-j]) % p for j in (1, 2, 3)]
    b = [(u[j] - u[7-j]) % p for j in (1, 2, 3)]
    t = [(x + u[0]) % p for x in conv([a[0], a[2], a[1]], kc, r, p)]
    q1, nq3, q2 = conv([b[0], b[1], -b[2] % p], ks, r, p)
    q = (q1, q2, -nq3 % p)
    out = [sum(u) % p] + [0]*6
    for j in range(3):
        out[j+1] = (t[j] + q[j]) % p
        out[6-j] = (t[j] - q[j]) % p
    return out

count = 0
for bits, r, cosines, sines in FIELDS:
    p = (1 << bits) - 1
    root = (cosines[0] + sines[0]) % p
    assert pow(root, 7, p) == 1 and root != 1
    assert pow(r, 3, p) == 1 and r != 1
    inv2 = pow(2, -1, p)
    for j in range(1, 4):
        z = pow(root, j, p)
        iz = pow(z, -1, p)
        assert cosines[j-1] == (z + iz)*inv2 % p
        assert sines[j-1] == (z - iz)*inv2 % p
    kc = [v*pow(3, -1, p) % p for v in dft(cosines, r, p)]
    ks = [v*pow(3, -1, p) % p for v in dft((sines[0], -sines[2] % p, sines[1]), r, p)]
    # Check the generated constants actually embedded in both kernel branches.
    for value in kc + ks + [-v % p for v in ks]:
        suffix = "U" if bits == 31 else "UL"
        assert f"(Z{bits}){value}{suffix}" in SOURCE
    vectors = [[int(i == j) for i in range(7)] for j in range(7)]
    vectors += [[v]*7 for v in (0, 1, p-1, p, p+1, p+7)]
    vectors += [[0, p-1, p, p+1, p+7, 1, 2]]
    rng = random.Random(bits)
    vectors += [[rng.randrange(p+8) for _ in range(7)] for _ in range(128)]
    for inverse in (False, True):
        selected_root = pow(root, -1, p) if inverse else root
        selected_spectrum = [-v % p for v in ks] if inverse else ks
        for u in vectors:
            assert candidate(u, kc, selected_spectrum, r, p) == dft(u, selected_root, p)
            count += 1
print(f"Arithmetic model: {count} field/direction/vector comparisons; GPU validation pending.")
