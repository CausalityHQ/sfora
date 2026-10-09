#!/usr/bin/env python3
"""Stdlib source falsifier. Does NOT compile, load, or execute CUDA/native code."""

import ast
import hashlib
import math
from pathlib import Path
import random
import re
import sys
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "rust/sfora-cuda-sha256/sha256_occurrences.cu"
MASK = (1 << 32) - 1


def expression(text):
    """Accept only the finite integer-expression subset used by this source."""
    text = re.sub(r"(?<=\d)[uU](?:[lL]{2})?\b", "", text).replace("/", "//")
    tree = ast.parse(text.strip(), mode="eval")
    allowed = (
        ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp,
        ast.Call, ast.Subscript, ast.Compare, ast.Add, ast.Sub, ast.Mult,
        ast.FloorDiv, ast.Mod, ast.LShift, ast.RShift, ast.BitAnd, ast.BitOr,
        ast.BitXor, ast.Invert, ast.USub, ast.Lt, ast.GtE, ast.Eq,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"unsupported source expression: {text}")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise ValueError("only named integer helpers are allowed")
    code = compile(tree, "<extracted CUDA expression>", "eval")
    return lambda env: eval(code, {"__builtins__": {}}, env)


class SourceOperations:
    """Extract operations; Python drives blocks/rounds, not CUDA scheduling."""

    def __init__(self, source):
        self.source = source
        self.constants = {}
        for name, count in (("K", 64), ("IV", 8)):
            body = re.search(rf"{name}\[{count}\] = \{{(.*?)\}};", source, re.S)
            if body is None:
                raise ValueError(f"missing {name} constants")
            self.constants[name] = [int(x.strip().rstrip("uU"), 16)
                                    for x in body[1].split(",") if x.strip()]
            if len(self.constants[name]) != count:
                raise ValueError(f"wrong {name} constant count")
        self.helpers = {"uint32_t": lambda x: x & MASK}
        for name in ("rotr", "small0", "small1", "big0", "big1", "choose", "majority",
                     "padded_blocks", "load_word"):
            match = re.search(rf"\b{name}\(([^{{}}]*)\) \{{\s*return ([^;]*);\s*\}}", source)
            if match is None:
                raise ValueError(f"unsupported helper body: {name}")
            args = [arg.strip().split()[-1].lstrip("*") for arg in match[1].split(",")]
            evaluate = expression(match[2])
            def helper(*values, args=args, evaluate=evaluate, name=name):
                result = evaluate(dict(self.helpers, **dict(zip(args, values, strict=True))))
                return result if name == "padded_blocks" else result & MASK
            self.helpers[name] = helper
        body = re.search(r"\bpadded_byte\([^{}]*\) \{([^{}]*)\}", source)[1]
        self.padding = [(expression(condition), expression(value)) for condition, value in
                        re.findall(r"if \((.*?)\) return (.*?);", body)]
        self.padding_default = expression(re.findall(r"return ([^;]*);", body)[-1])
        self.helpers["padded_byte"] = self.padded_byte
        self.initial = expression(re.search(r"state\[i\] = ([^;]*);", source)[1])
        self.block_count = expression(re.search(r"uint64_t blocks = ([^;]*);", source)[1])
        self.load = expression(re.search(r"w\[i\] = (load_word[^;]*);", source)[1])
        self.schedule = expression(re.search(r"w\[i\] = (small1[^;]*);", source)[1])
        self.registers = [(name, expression(value)) for name, value in
                          re.findall(r"([a-h]) = (state\[[0-7]\])", source)]
        self.t1 = expression(re.search(r"uint32_t t1 = ([^;]*);", source)[1])
        self.t2 = expression(re.search(r"uint32_t t2 = ([^;]*);", source)[1])
        updates = re.search(r"// round state\n(.*?)\n", source)[1]
        self.updates = [(name, expression(value)) for name, value in
                        re.findall(r"([a-h]) = ([^;]*);", updates)]
        self.working = [expression(value) for value in
                        re.search(r"uint32_t working\[8\] = \{([^{}]*)\};", source)[1].split(",")]
        self.feed = expression(re.search(r"state\[i\] \+= ([^;]*);", source)[1])
        self.output = expression(re.search(r"output\[i\] = ([^;]*);", source)[1])

    def padded_byte(self, data, n, pos, total):
        env = dict(self.helpers, data=data, n=n, pos=pos, total=total)
        for condition, value in self.padding:
            if condition(env):
                return value(env) & 255
        return self.padding_default(env) & 255

    def digest(self, data):
        state = [self.initial(dict(IV=self.constants["IV"], i=i)) for i in range(8)]
        n = len(data)
        blocks = self.block_count(dict(self.helpers, n=n))
        for block in range(blocks):
            w = [self.load(dict(self.helpers, data=data, n=n, block=block, i=i, blocks=blocks))
                 for i in range(16)] + [0] * 48
            env = dict(self.helpers, w=w, K=self.constants["K"])
            for i in range(16, 64):
                w[i] = self.schedule(dict(env, i=i)) & MASK
            for name, value in self.registers:
                env[name] = value(dict(state=state))
            for i in range(64):
                env["i"] = i
                env["t1"] = self.t1(env) & MASK
                env["t2"] = self.t2(env) & MASK
                for name, operation in self.updates:
                    env[name] = operation(env) & MASK
            working = [value(env) for value in self.working]
            state = [(old + self.feed(dict(working=working, i=i))) & MASK
                     for i, old in enumerate(state)]
        return bytes(self.output(dict(state=state, i=i)) & 255 for i in range(32))


def root_floor(value, degree):
    lo, hi = 0, 1 << ((value.bit_length() + degree - 1) // degree)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid ** degree <= value:
            lo = mid
        else:
            hi = mid
    return lo


class ShaSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text() if SOURCE.is_file() else ""
        cls.ops = SourceOperations(cls.source) if cls.source else None

    def setUp(self):
        if not self.source and self._testMethodName != "test_source_exists":
            self.skipTest("prototype source is absent")

    def test_source_exists(self):
        self.assertTrue(self.source, f"required prototype source missing: {SOURCE}")

    def test_independently_derived_constants(self):
        primes = [n for n in range(2, 312)
                  if all(n % d for d in range(2, math.isqrt(n) + 1))]
        self.assertEqual(self.ops.constants["IV"], [math.isqrt(p << 64) & MASK for p in primes[:8]])
        self.assertEqual(self.ops.constants["K"], [root_floor(p << 96, 3) & MASK for p in primes[:64]])

    def test_vectors_padding_and_fp32_literal_bytes(self):
        rng = random.Random(20261009)
        vectors = [b"", b"abc", b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                   bytes.fromhex("0000000000000080"), bytes.fromhex("0100c07f0000807f000080ff")]
        vectors += [rng.randbytes(n) for n in range(257)]
        vectors += [rng.randbytes(n * 4) for n in (1, *range(13, 18), *range(29, 34), 1024)]
        for raw in vectors:
            with self.subTest(length=len(raw)):
                self.assertEqual(self.ops.digest(raw), hashlib.sha256(raw).digest())
        self.assertNotEqual(self.ops.digest(bytes(4)), self.ops.digest(bytes.fromhex("00000080")))

    def test_occurrence_order_alias_mutation_restore_and_offset(self):
        storage = bytearray(range(132))
        leaf = memoryview(storage)[4:124]
        other = bytearray(bytes.fromhex("00000080"))
        leaves = [leaf, other, leaf, memoryview(storage)[8:12], b""]
        def check(items):
            got = [self.ops.digest(item) for item in items]
            self.assertEqual(got, [hashlib.sha256(item).digest() for item in items])
            return got
        before = check(leaves)
        self.assertEqual(before[0], before[2])
        self.assertEqual(check(list(reversed(leaves))), list(reversed(before)))
        storage[7] ^= 1
        changed = check(leaves)
        self.assertNotEqual(changed[0], before[0])
        self.assertEqual(changed[0], changed[2])
        storage[7] ^= 1
        self.assertEqual(check(leaves), before)

    def test_each_wrong_constant_is_detected_by_digest_oracle(self):
        for name in ("K", "IV"):
            for i in range(len(self.ops.constants[name])):
                mutant = SourceOperations(self.source)
                mutant.constants[name][i] ^= 1
                with self.subTest(constant=name, index=i):
                    self.assertNotEqual(mutant.digest(b"abc"), hashlib.sha256(b"abc").digest())

    def test_schedule_padding_round_and_endian_mutants(self):
        mutants = [("w[i - 7]", "w[i - 6]", bytes(range(68))),
                   ("n % 64 >= 56", "n % 64 >= 60", bytes(range(56))),
                   ("return 0x80", "return 0x81", b""),
                   ("(total - 1 - pos) * 8", "(pos - (total - 8)) * 8", b"abc"),
                   ("e = d + t1", "e = d + t2", b"abc"),
                   ("a = state[0]", "a = state[1]", b"abc"),
                   ("{a, b, c, d, e, f, g, h}", "{b, a, c, d, e, f, g, h}", bytes(range(128))),
                   ("x >> n", "x << n", b"abc"),
                   ("pos, total)) << 24", "pos, total)) << 16", b"abc"),
                   ("24 - 8 * (i % 4)", "8 * (i % 4)", b"abc")]
        for old, new, raw in mutants:
            self.assertIn(old, self.source)
            mutant = SourceOperations(self.source.replace(old, new))
            with self.subTest(mutation=old):
                self.assertNotEqual(mutant.digest(raw), hashlib.sha256(raw).digest())

    def test_control_skeleton_is_explicitly_bounded(self):
        # This checks source structure only; it does not execute the kernel or ABI.
        for snippet in ("for (uint64_t block = 0; block < blocks; ++block)",
                        "for (uint32_t i = 0; i < 16; ++i)",
                        "for (uint32_t i = 16; i < 64; ++i)",
                        "for (uint32_t i = 0; i < 64; ++i)",
                        "state[i] += working[i]", "if (index >= occurrence_count) return",
                        "device_ptrs[index], device_byte_lengths[index], device_output + index * 32",
                        "<<<blocks, 128, 0, current_stream>>>", "return cudaGetLastError()"):
            self.assertIn(snippet, self.source)
        self.assertEqual(self.source.count('extern "C"'), 1)
        self.assertEqual(len(self.ops.updates), 8)
        self.assertEqual(len(self.ops.registers), 8)
        self.assertEqual(len(self.ops.working), 8)
        self.assertEqual(len(self.ops.padding), 3)


if __name__ == "__main__":
    if len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        SOURCE = Path(sys.argv.pop())
    print("SOURCE EMULATION ONLY: CUDA compilation/execution/streams/lifetimes/performance UNRUN.")
    unittest.main()
