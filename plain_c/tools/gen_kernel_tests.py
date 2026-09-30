"""
Generate tests/data/kernels.bin: small random cases for every C kernel, with the
expected outputs computed by the functions in reference.py.

    python3 plain_c/tools/gen_kernel_tests.py [out.bin]

Record format (little-endian), repeated until EOF:
    char   op[16]         NUL-padded op name
    int32  dims[8]        op-specific sizes, unused entries 0
    int32  n_arrays
    n_arrays x { int32 count; float32 data[count] }
Arrays are the op's inputs in the order the C test passes them, then the
expected output last. The file is committed so `make test` needs no Python.
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as ref  # noqa: E402

rng = np.random.default_rng(1234)


def randn(*shape):
    return rng.standard_normal(shape).astype(np.float32)


cases = []


def add(op, dims, *arrays):
    cases.append((op, dims, [np.ascontiguousarray(a, dtype=np.float32) for a in arrays]))


# sizes: include cin != cout, H != W, odd sizes where the op allows, and 1-sized dims
for cin, cout, H, W in [(1, 1, 1, 1), (3, 5, 4, 7), (8, 2, 6, 3), (4, 4, 9, 11)]:
    x, w, b = randn(cin, H, W), randn(cout, cin, 1, 1), randn(cout)
    add("conv1x1", [cin, cout, H, W], x, w, b, ref.conv1x1(x, w, b))

# the fast conv3x3 tiles 4 output channels x 32 columns: include widths that are
# multiples of 32 (fast path), channel counts that aren't multiples of 4 (tail
# tile), a single row, and one tile wide (both column edges in one tile)
for cin, cout, H, W in [(1, 1, 1, 1), (1, 1, 3, 3), (3, 5, 4, 7), (8, 2, 6, 3), (4, 4, 9, 11),
                        (3, 4, 1, 32), (5, 6, 3, 64), (2, 9, 5, 96), (7, 8, 2, 128)]:
    x, w, b = randn(cin, H, W), randn(cout, cin, 3, 3), randn(cout)
    add("conv3x3", [cin, cout, H, W], x, w, b, ref.conv3x3(x, w, b))

for cin, cout, H, W in [(1, 1, 2, 2), (3, 5, 4, 8), (6, 9, 8, 2), (4, 4, 10, 6), (3, 5, 4, 64)]:
    x, w, b = randn(cin, H, W), randn(cout, cin, 2, 2), randn(cout)
    add("conv2x2_s2", [cin, cout, H, W], x, w, b, ref.conv2x2_s2(x, w, b))

for cin, cout, H, W in [(1, 1, 1, 1), (5, 3, 2, 4), (9, 6, 4, 1), (4, 4, 5, 3), (5, 3, 2, 32)]:
    x, w, b = randn(cin, H, W), randn(cin, cout, 2, 2), randn(cout)  # ConvTranspose: [cin][cout]
    add("convT2x2_s2", [cin, cout, H, W], x, w, b, ref.convT2x2_s2(x, w, b))

for C, T, fin, fout in [(1, 1, 1, 1), (3, 4, 16, 2), (2, 5, 2, 16), (4, 3, 13, 7), (3, 3, 64, 8), (2, 3, 8, 64)]:
    x, w = randn(C, T, fin), randn(fin, fout)
    add("matmul_lastdim", [C * T, fin, fout], x, w, ref.matmul_lastdim(x, w))

for C, H, W in [(1, 1, 1), (3, 4, 5), (6, 2, 9)]:
    x = randn(C, H, W)
    scale, bias, mean = randn(C), randn(C), randn(C)
    var = (rng.random(C) * 2 + 1e-3).astype(np.float32)  # variance must be positive
    add("batchnorm", [C, H, W], x, scale, bias, mean, var, ref.batchnorm(x, scale, bias, mean, var))

for n in [1, 7, 100]:
    x = randn(n)
    add("relu", [n], x, ref.relu(x))

for n in [1, 7, 100]:
    a, b = randn(n), randn(n)
    add("add", [n], a, b, a + b)
    add("mul", [n], a, b, a * b)

for C, H, W in [(1, 1, 1), (2, 3, 5), (3, 8, 4)]:
    x = randn(C, H, W)
    add("transpose_last2", [C, H, W], x, ref.transpose_last2(x))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    dst = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "tests", "data", "kernels.bin")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as f:
        for op, dims, arrays in cases:
            f.write(op.encode().ljust(16, b"\0"))
            f.write(struct.pack("<8i", *(dims + [0] * (8 - len(dims)))))
            f.write(struct.pack("<i", len(arrays)))
            for a in arrays:
                f.write(struct.pack("<i", a.size))
                f.write(a.astype("<f4").tobytes())
    print(f"wrote {dst}: {len(cases)} cases, {os.path.getsize(dst)} bytes")


if __name__ == "__main__":
    main()
