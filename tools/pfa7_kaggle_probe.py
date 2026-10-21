#!/usr/bin/env python3
"""Short, isolated CUDA PFA7 PRP-square differential and throughput probe.

Launch this script in a fresh process for each AEVUM_PFA7_LEAD_MODE=0/1/2.
GPU output is compared word-for-word with an independent FFT3161/FFT3261 Type2
reference before timing. This is a diagnostic only, not a full PRP proof.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
import random
import struct
import sys
import time

PFA7 = "pfa7:1:512:7:512:202"
TYPE2 = "2:512:8:512:202"

class Engine:
    def __init__(self, library):
        self.dll = C.CDLL(str(library))
        def bind(name, result, *args):
            f = getattr(self.dll, "aevum_engine_" + name)
            f.restype = result
            f.argtypes = list(args)
            return f
        P = C.c_void_p
        U = C.c_uint32
        S = C.c_size_t
        self.error = bind("last_error", C.c_char_p)
        self.create = bind("create", P, U, S, U, C.c_int, C.c_char_p, C.c_char_p)
        self.destroy = bind("destroy", None, P)
        self.word_count = bind("word_count", S, P)
        self.plan_spec = bind("plan_spec", C.c_int, P, C.c_char_p, S)
        self.set_words = bind("set_words", C.c_int, P, S, C.POINTER(U), S)
        self.get_words = bind("get_words", C.c_int, P, S, C.POINTER(U), S)
        self.set_u32 = bind("set_u32", C.c_int, P, S, U)
        self.square_mul = bind("square_mul", C.c_int, P, S, U)
        self.sync = bind("sync", C.c_int, P)

    def check(self, status, action):
        if not status:
            error = self.error()
            raise RuntimeError(f"{action}: {error.decode(errors='replace') if error else 'unknown error'}")

    def new(self, exp, device, plan, kernels):
        h = self.create(exp, 2, device, 1, plan.encode(), kernels.encode())
        if not h:
            error = self.error()
            raise RuntimeError(f"create {plan}: {error.decode(errors='replace') if error else 'unknown error'}")
        active = C.create_string_buffer(256)
        self.check(self.plan_spec(h, active, 256), "plan_spec")
        if active.value.decode() != plan:
            self.destroy(h)
            raise RuntimeError(f"Active plan {active.value!r} differs from {plan}")
        return h

    def exact(self, h, exp, seed):
        n = self.word_count(h)
        if not n:
            raise RuntimeError("empty word count")
        rng = random.Random(seed)
        words = (C.c_uint32 * n)()
        for i in range(n):
            words[i] = rng.getrandbits(32)
        if exp % 32:
            words[n - 1] &= (1 << (exp % 32)) - 1
        self.check(self.set_words(h, 0, words, n), "set_words dense")
        # Five squares share the register pipeline; factor=3 forces its flush.
        for index, factor in enumerate([1, 1, 1, 1, 1, 3, 1]):
            self.check(self.square_mul(h, 0, factor), f"square[{index}]")
        result = (C.c_uint32 * n)()
        self.check(self.get_words(h, 0, result, n), "get_words")
        return bytes(result)

    def perf(self, h, niter):
        self.check(self.set_u32(h, 0, 3), "set_u32 warmup")
        for _ in range(12):
            self.check(self.square_mul(h, 0, 1), "warmup square")
        self.check(self.sync(h), "warmup sync")
        self.check(self.set_u32(h, 0, 3), "reset perf seed")
        t0 = time.perf_counter()
        for _ in range(niter):
            self.check(self.square_mul(h, 0, 1), "perf square")
        self.check(self.sync(h), "perf sync")
        return (time.perf_counter() - t0) / niter

def first_diff(a, b):
    for i in range(0, min(len(a), len(b)), 4):
        if a[i:i+4] != b[i:i+4]:
            return {"word": i // 4, "reference": struct.unpack_from("<I", a, i)[0],
                    "candidate": struct.unpack_from("<I", b, i)[0]}
    return {"length_reference": len(a), "length_candidate": len(b)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", required=True)
    parser.add_argument("--kernels", default=".")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--exponent", type=int, default=136279841)
    parser.add_argument("--iterations", type=int, default=192)
    parser.add_argument("--mode", choices=["0", "1", "2"], required=True)
    args = parser.parse_args()
    if os.getenv("AEVUM_PFA7_LEAD_MODE", "0") != args.mode:
        raise RuntimeError("Pass AEVUM_PFA7_LEAD_MODE equal to --mode")
    api = Engine(args.lib)
    h = api.new(args.exponent, args.device, TYPE2, args.kernels)
    type2_us = None
    try:
        reference = api.exact(h, args.exponent, 20261021)
        # One Type2 control per backend. This is a different plan; it is
        # context, NOT an A/B measure of the same transform.
        if args.mode == "0" and os.getenv("AEVUM_PFA7_CARRYB_FAST", "0") == "0":
            type2_us = api.perf(h, args.iterations) * 1e6
            print("TYPE2_REFERENCE=" + json.dumps(
                {"us_per_iter": round(type2_us, 3),
                 "iter_s": round(1e6 / type2_us, 3)}), flush=True)
    finally:
        api.destroy(h)
    h = api.new(args.exponent, args.device, PFA7, args.kernels)
    try:
        candidate = api.exact(h, args.exponent, 20261021)
        valid = reference == candidate
        print("PFA7_WORD_EXACT=" + ("PASS" if valid else "FAIL"), flush=True)
        if not valid:
            print("FIRST_DIFFERENCE=" + json.dumps(first_diff(reference, candidate)), flush=True)
            return 3
        us = api.perf(h, args.iterations) * 1e6
        summary = {"mode": args.mode, "carry_fast": os.getenv("AEVUM_PFA7_CARRYB_FAST", "0"),
                   "exact": valid, "p": args.exponent,
                   "plan": PFA7, "reference": TYPE2,
                   "iterations": args.iterations,
                   "us_per_iter": round(us, 3), "iter_s": round(1e6 / us, 3),
                   "sha256": hashlib.sha256(candidate).hexdigest()}
        print("PFA7_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)
    finally:
        api.destroy(h)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("PFA7_PROBE_FAILED=" + repr(exc), file=sys.stderr, flush=True)
        sys.exit(2)
