#!/usr/bin/env python3
"""Exercise cached preparation through an existing Aevum plugin on a real GPU.
No build or benchmark is performed. Compare complete words and actual Runtime
flush counters with the experiment OFF and ON in coexisting engine instances.
"""
import argparse
import ctypes as C
import os
import random
import tempfile


class Timing(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in (
        "square_calls", "pending_flush_calls", "pending_flush_ns",
        "queue_sync_calls", "queue_sync_ns", "copy_calls", "copy_ns",
        "equal_calls", "equal_ns", "readback_calls", "readback_ns")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--exponent", type=int, default=21000029)
    parser.add_argument("--fft", default="1:256:8:256:202")
    parser.add_argument("--cache", type=int, choices=(0, 1, 2, 8), default=2)
    args = parser.parse_args()
    api = C.CDLL(os.path.abspath(args.library))
    handle, size, u32 = C.c_void_p, C.c_size_t, C.c_uint32
    words = C.POINTER(u32)
    declarations = {
        "create": (handle, [u32, size, u32, C.c_int, C.c_char_p, C.c_char_p]),
        "destroy": (None, [handle]),
        "last_error": (C.c_char_p, []),
        "word_count": (size, [handle]),
        "set_u32": (C.c_int, [handle, size, u32]),
        "set_words": (C.c_int, [handle, size, words, size]),
        "get_words": (C.c_int, [handle, size, words, size]),
        "copy": (C.c_int, [handle, size, size]),
        "prepare": (C.c_int, [handle, size, size]),
        "square_mul": (C.c_int, [handle, size, u32]),
        "mul": (C.c_int, [handle, size, size, u32]),
        "add": (C.c_int, [handle, size, size]),
        "sub_reg": (C.c_int, [handle, size, size]),
        "sub_u32": (C.c_int, [handle, size, u32]),
        "sync": (C.c_int, [handle]),
        "timing_reset": (C.c_int, [handle]),
        "timing_get": (C.c_int, [handle, C.POINTER(Timing)]),
    }
    for name, (result, params) in declarations.items():
        fn = getattr(api, "aevum_engine_" + name)
        fn.restype, fn.argtypes = result, params

    def call(name, *values):
        result = getattr(api, "aevum_engine_" + name)(*values)
        if not result:
            raise RuntimeError(name + ": " + api.aevum_engine_last_error().decode())
        return result

    controls = {
        "AEVUM_AUTOTUNE": "off",
        "AEVUM_TIMING_DECOMP": "1",
        "AEVUM_REG_LEAD_CACHE": "1",
        "AEVUM_FUSED_LL": "1",
        "AEVUM_PREPARED_ALIAS_REUSE": "1",
        "AEVUM_PREPARED_MUL_LEAD": "1",
        "AEVUM_PREPARED_CACHE": str(args.cache),
        "AEVUM_PREPARE_KEEP_LEAD": "0",
    }
    previous = {key: os.environ.get(key) for key in controls}
    handles = []
    try:
        os.environ.update(controls)
        with tempfile.TemporaryDirectory(prefix="aevum-prepare-chain-") as directory:
            def create():
                return api.aevum_engine_create(
                    args.exponent, 8, args.device, 1, args.fft.encode(), directory.encode())

            for enabled in ("0", "1"):
                os.environ["AEVUM_PREPARE_KEEP_LEAD"] = enabled
                h = create()
                if not h:
                    raise RuntimeError(api.aevum_engine_last_error().decode())
                handles.append(h)
            count = call("word_count", handles[0])
            if call("word_count", handles[1]) != count:
                raise RuntimeError("word count mismatch")
            array = u32 * count

            def both(name, *values):
                for h in handles:
                    call(name, h, *values)

            def compare(*registers):
                result = None
                for reg in registers:
                    outputs = []
                    for h in handles:
                        out = array()
                        call("get_words", h, reg, out, count)
                        outputs.append(bytes(out))
                    if outputs[0] != outputs[1]:
                        raise RuntimeError("full-word mismatch in register " + str(reg))
                    result = outputs[0]
                return result

            def flush_counts():
                counts = []
                for h in handles:
                    stats = Timing()
                    call("timing_get", h, C.byref(stats))
                    counts.append(stats.pending_flush_calls)
                return counts

            def check_integer(expected, reg=0):
                out = compare(reg)
                value = array()
                for i in range(count):
                    value[i] = expected & 0xffffffff
                    expected >>= 32
                if expected or out != bytes(value):
                    raise RuntimeError("integer oracle mismatch in register " + str(reg))

            # Actual P-1 bit path: square, repeated prepare of the fixed base, mul.
            both("set_u32", 0, 3)
            both("set_u32", 1, 5)
            both("prepare", 2, 1)
            both("sync")
            both("timing_reset")
            expected = 3
            modulus = (1 << args.exponent) - 1
            rounds = 8
            for _ in range(rounds):
                both("square_mul", 0, 1)
                both("prepare", 2, 1)
                both("mul", 0, 2, 1)
                expected = expected * expected * 5 % modulus
            both("sync")
            if args.cache and flush_counts() != [rounds + 1, 1]:
                raise RuntimeError("cached bit chain did not eliminate prepare flushes: " + str(flush_counts()))
            check_integer(expected)

            # A cached source can supply a new canonical destination while an
            # unrelated square is pending; prepare(dst,dst) can also retain it.
            both("set_u32", 0, 3)
            both("set_u32", 3, 11)
            both("timing_reset")
            both("square_mul", 0, 1)
            both("square_mul", 0, 1)
            both("prepare", 3, 1)
            if args.cache and flush_counts() != [1, 0]:
                raise RuntimeError("disjoint cached copy unexpectedly flushed the candidate")
            both("prepare", 3, 3)
            both("mul", 0, 3, 1)
            both("sync")
            check_integer(405)
            compare(1, 2, 3)

            # Source/destination touching the pending result must materialize.
            both("square_mul", 0, 1)
            both("prepare", 4, 0)
            both("mul", 3, 4, 1)
            compare(0, 3, 4)
            both("square_mul", 0, 1)
            both("timing_reset")
            both("prepare", 0, 1)
            if flush_counts() != [1, 1]:
                raise RuntimeError("prepare overwriting the pending result did not flush")
            check_integer(5)

            # Replace the last canonical alias of an operand used by a pending
            # multiply. Its transform must remain live until the miss below
            # flushes that multiply, before any LRU slot can be overwritten.
            both("set_u32", 1, 5)
            both("prepare", 2, 1)
            both("set_u32", 1, 7)
            both("set_u32", 3, 11)
            both("prepare", 4, 3)
            both("set_u32", 5, 13)
            both("set_u32", 0, 3)
            both("mul", 0, 2, 1)
            both("prepare", 2, 3)
            both("prepare", 6, 5)
            both("square_mul", 0, 1)
            both("mul", 0, 2, 1)
            check_integer(2475)
            compare(1, 2, 3, 4, 5, 6)

            rng = random.Random(0x50524550415245)
            dense = array.from_buffer_copy(rng.randbytes(count * 4))
            high = array(*([0xffffffff] * count))
            if args.exponent % 32:
                mask = (1 << (args.exponent % 32)) - 1
                dense[-1] &= mask
                high[-1] &= mask
            high[0] -= 1
            for seed in (array(), high, dense):
                both("set_words", 0, seed, count)
                both("set_u32", 1, 5)
                both("prepare", 2, 1)
                for i in range(6):
                    both("square_mul", 0, 1)
                    both("prepare", 2, 1)
                    both("mul", 0, 2, 1)
                    if i == 2:
                        # A write invalidates the source alias even though the
                        # other alias still keeps the old transform cached.
                        both("set_words", 1, seed, count)
                both("square_mul", 0, 1)
                both("sub_u32", 0, 2)
                both("prepare", 2, 1)
                both("square_mul", 0, 1)
                compare(0, 1, 2)
                both("add", 1, 2)
                both("sub_reg", 2, 1)
                both("prepare", 3, 2)
                both("square_mul", 0, 1)
                both("prepare", 2, 1)
                both("mul", 0, 2, 1)
                both("copy", 7, 1)
                both("prepare", 2, 7)
                both("mul", 0, 2, 1)
                compare(0, 1, 2, 3, 7)
                snapshot = array.from_buffer_copy(compare(0))
                both("set_words", 0, snapshot, count)
                both("square_mul", 0, 1)
                both("prepare", 2, 1)
                both("mul", 0, 2, 1)
                compare(0)

            for key, value, message in (
                ("AEVUM_PREPARE_KEEP_LEAD", "invalid", "must be 0 or 1"),
                ("AEVUM_PREPARED_ALIAS_REUSE", "0", "requires prepared alias reuse"),
                ("AEVUM_PREPARED_MUL_LEAD", "0", "requires prepared alias reuse"),
            ):
                os.environ.update(controls)
                os.environ["AEVUM_PREPARE_KEEP_LEAD"] = "1"
                os.environ[key] = value
                h = create()
                if h:
                    api.aevum_engine_destroy(h)
                    raise RuntimeError("unsupported selection was accepted: " + key)
                if message not in api.aevum_engine_last_error().decode():
                    raise RuntimeError("unexpected rejection for " + key)
            for h in handles:
                if api.aevum_engine_prepare(h, 8, 1):
                    raise RuntimeError("out-of-range destination accepted")
                if api.aevum_engine_prepare(h, 2, 8):
                    raise RuntimeError("out-of-range source accepted")
            print("cached preparation full-word, flush-count and invalidation checks completed")
    finally:
        for h in handles:
            api.aevum_engine_destroy(h)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    main()
