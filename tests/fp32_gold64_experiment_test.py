#!/usr/bin/env python3
"""CPU oracle and optional GPU validation for the isolated FP32/Gold64 experiment.

Usage:
  python3 tests/fp32_gold64_experiment_test.py
  python3 tests/fp32_gold64_experiment_test.py --gpu   # pyopencl + GPU/OpenCL ICD
"""
import argparse
import pathlib
import random
import shutil
import struct
import subprocess

P = (1 << 64) - (1 << 32) + 1
MASK64 = (1 << 64) - 1
K = (1 << 32) - 1
SRC = pathlib.Path(__file__).resolve().parents[1] / 'src/cl/fp32_gold64_experiment.cl'


def gold_add(a, b):
    return (a + b + (K if a >= P - b else 0)) & MASK64


def gold_sub(a, b):
    return (a - b - (K if a < b else 0)) & MASK64


def gold_reduce(a, b):
    lo = (a * b) & MASK64
    hi = (a * b) >> 64
    top = hi >> 32
    bottom = hi & K
    s = (((bottom << 32) | (K - top)) - bottom) & MASK64
    r = (s + lo) & MASK64
    if r >= s:
        borrow_fix = K if r < K else 0
        return (r - K - borrow_fix) & MASK64
    return r


def lift(residue, quotient):
    assert -(1 << 31) <= quotient < (1 << 31)
    if quotient >= 0:
        c = quotient * K
        lo = (residue - c) & MASK64
        hi = (quotient - (residue < c)) & MASK64
    else:
        c = (-quotient) * K
        lo = (residue + c) & MASK64
        hi = (quotient + ((residue + c) > MASK64)) & MASK64
    x = (hi << 64) | lo
    if x & (1 << 127):
        x -= 1 << 128
    return lo, hi, x


def cyclic(a, b):
    n = len(a)
    return [sum(a[j] * b[(k - j) % n] for j in range(n)) for k in range(n)]


def f32(x):
    return struct.unpack('f', struct.pack('f', float(x)))[0]


def hint_sim(a, b, residue, k):
    """Non-fused FP32 emulation for coverage, not a guarantee about device FMA."""
    n = len(a)
    accum = 0.0
    for j in range(n):
        accum = f32(accum + f32(f32(a[j]) * f32(b[(k - j) % n])))
    return f32(f32(accum - f32(residue)) * f32(2**-64))


def cpu_tests():
    rng = random.Random(0xA5E601)
    boundary = [0, 1, 2, (1 << 32) - 1, (1 << 32), (1 << 61) - 1,
                P // 2, P - 2, P - 1]
    for a in boundary:
        for b in boundary:
            assert gold_add(a, b) == (a + b) % P, (a, b, 'add')
            assert gold_sub(a, b) == (a - b) % P, (a, b, 'sub')
            assert gold_reduce(a, b) == (a * b) % P, (a, b, 'mul')
    for _ in range(30000):
        a = rng.randrange(P)
        b = rng.randrange(P)
        assert gold_add(a, b) == (a + b) % P
        assert gold_sub(a, b) == (a - b) % P
        assert gold_reduce(a, b) == (a * b) % P
    for _ in range(20000):
        q = rng.randrange(-(1 << 31), 1 << 31)
        r = rng.randrange(P)
        assert lift(r, q)[2] == r + q * P
    for n in (4, 8, 16, 32, 64):
        for _ in range(15):
            a = [rng.randrange(1 << 32) for __ in range(n)]
            b = [rng.randrange(1 << 32) for __ in range(n)]
            expected = cyclic(a, b)
            for k, x in enumerate(expected):
                residue = sum((a[j] * b[(k - j) % n]) % P for j in range(n)) % P
                assert residue == x % P
                guess = hint_sim(a, b, residue, k)
                assert abs(guess - round(guess)) < 0.125, (n, k, guess)
                assert lift(residue, round(guess))[2] == x, (n, k, guess, x)
    clang = shutil.which('clang')
    if clang:
        proc = subprocess.run([clang, '-x', 'cl', '-cl-std=CL1.2', '-fsyntax-only', str(SRC)],
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        print('OpenCL C 1.2 syntax: PASS')
    else:
        print('OpenCL C syntax: SKIPPED (clang unavailable)')
    print('Gold64 arithmetic + signed CRT lift + FP32/Gold64 small-convolution CPU oracle: PASS')


def gpu_tests():
    try:
        import numpy as np
        import pyopencl as cl
    except ImportError as exc:
        raise SystemExit(f'GPU tests need numpy and pyopencl: {exc}')
    gpu_devices = [d for platform in cl.get_platforms() for d in platform.get_devices()
                   if d.type & cl.device_type.GPU]
    if not gpu_devices:
        raise SystemExit('No OpenCL GPU device available')
    device = gpu_devices[0]
    ctx = cl.Context(devices=[device])
    queue = cl.CommandQueue(ctx)
    program = cl.Program(ctx, SRC.read_text()).build(options=['-cl-std=CL1.2'])
    rng = random.Random(0xA5E601)
    flags = cl.mem_flags
    for n in (4, 8, 16, 32, 64):
        a = np.asarray([rng.randrange(1 << 32) for _ in range(n)], dtype=np.uint64)
        b = np.asarray([rng.randrange(1 << 32) for _ in range(n)], dtype=np.uint64)
        expected = cyclic(list(map(int, a)), list(map(int, b)))
        ga = cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=a)
        gb = cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=b)
        out128 = np.empty((n, 2), dtype=np.uint64)
        outmod = np.empty(n, dtype=np.uint64)
        outhint = np.empty(n, dtype=np.float32)
        outwarning = np.empty(n, dtype=np.uint32)
        xo = cl.Buffer(ctx, flags.WRITE_ONLY, out128.nbytes)
        mo = cl.Buffer(ctx, flags.WRITE_ONLY, outmod.nbytes)
        ho = cl.Buffer(ctx, flags.WRITE_ONLY, outhint.nbytes)
        wo = cl.Buffer(ctx, flags.WRITE_ONLY, outwarning.nbytes)
        program.aevum_fp32_gold64_cyclic_reference(
            queue, (n,), None, ga, gb, xo, mo, ho, wo, np.uint32(n)).wait()
        for h, buf in [(out128, xo), (outmod, mo), (outhint, ho), (outwarning, wo)]:
            cl.enqueue_copy(queue, h, buf).wait()
        for i, x in enumerate(expected):
            signed = (int(out128[i, 1]) << 64) | int(out128[i, 0])
            if signed >= 1 << 127:
                signed -= 1 << 128
            assert int(outwarning[i]) == 0, (device.name, n, i, float(outhint[i]))
            assert int(outmod[i]) == x % P, (device.name, n, i, 'mod')
            assert signed == x, (device.name, n, i, 'lift')
    # Independently probe multiplication near p, where reduction corner cases occur.
    edges = [0, 1, P - 1, P - 2, (1 << 32), (1 << 61) - 1]
    aa = np.asarray([x for x in edges for _ in edges], dtype=np.uint64)
    bb = np.asarray([y for _ in edges for y in edges], dtype=np.uint64)
    ga = cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=aa)
    gb = cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=bb)
    dst = np.empty((len(aa), 4), dtype=np.uint64)
    do = cl.Buffer(ctx, flags.WRITE_ONLY, dst.nbytes)
    program.aevum_gold64_arithmetic_reference(
        queue, (len(aa),), None, ga, gb, do, np.uint32(len(aa))).wait()
    cl.enqueue_copy(queue, dst, do).wait()
    for i, (a, b) in enumerate(zip(aa, bb)):
        x, y = int(a), int(b)
        assert list(map(int, dst[i])) == [(x + y) % P, (x - y) % P, (x * y) % P, (x * x) % P]
    print(f'OpenCL GPU {device.name}: exact reference + Gold64 arithmetic PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='also run OpenCL GPU differential tests')
    args = parser.parse_args()
    cpu_tests()
    if args.gpu:
        gpu_tests()
