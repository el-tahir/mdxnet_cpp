"""
Export UVR_MDXNET_KARA_2.onnx to the plain_c weight file (kara.bin).

    python3 plain_c/tools/export.py [model.onnx] [out.bin]

File layout (all little-endian):

    header, 64 bytes
        u32 magic      'MDXN' (0x4E58444D)
        u32 version    1
        u32 dim_c      4      input/output channels (L.re, L.im, R.re, R.im)
        u32 dim_f      2048   frequency bins seen by the model
        u32 dim_t      256    time frames per chunk
        u32 n_scales   5      encoder/decoder levels
        u32 growth     48     channels added per level
        u32 n_tfc      3      conv3x3 layers per TFC
        u32 bn_factor  8      TDF bottleneck F -> F/bn_factor
        f32 bn_eps     1e-5
        zero padding up to 64 bytes
    then every tensor as raw float32, back to back, in the order of
    tensor_list() below, which is also the order the forward pass uses them.

Also writes <out>.manifest: one line per tensor with its name, shape, and a few
values, which tests/test_load.c compares against what the C loader mapped.
"""
import struct
import sys

import numpy as np
import onnx
from onnx import numpy_helper

MAGIC = 0x4E58444D  # bytes 'M','D','X','N' read as a little-endian u32
VERSION = 1
HEADER_SIZE = 64

CONFIG = dict(dim_c=4, dim_f=2048, dim_t=256, n_scales=5, growth=48, n_tfc=3, bn_factor=8)
BN_EPS = 1e-5


def tensor_list(dim_c, dim_f, dim_t, n_scales, growth, n_tfc, bn_factor):
    """(name, shape) of every tensor in file order. mdx.c walks the same order."""
    out = []

    def conv(name, cout, cin, k):
        out.append((f"{name}.w", (cout, cin, k, k)))
        out.append((f"{name}.b", (cout,)))

    def bn(name, c):
        for p in ("scale", "bias", "mean", "var"):
            out.append((f"{name}.{p}", (c,)))

    def tfc_tdf(name, c, f):
        for i in range(n_tfc):
            conv(f"{name}.tfc{i}", c, c, 3)
        out.append((f"{name}.tdf1.w", (f, f // bn_factor)))
        bn(f"{name}.tdf1.bn", c)
        out.append((f"{name}.tdf2.w", (f // bn_factor, f)))
        bn(f"{name}.tdf2.bn", c)

    conv("first", growth, dim_c, 1)
    c, f = growth, dim_f
    for i in range(n_scales):
        tfc_tdf(f"enc{i}", c, f)
        conv(f"enc{i}.down", c + growth, c, 2)
        c, f = c + growth, f // 2
    tfc_tdf("mid", c, f)
    for i in range(n_scales):
        # ConvTranspose weight is [Cin, Cout, kh, kw]
        out.append((f"dec{i}.up.w", (c, c - growth, 2, 2)))
        out.append((f"dec{i}.up.b", (c - growth,)))
        bn(f"dec{i}.up.bn", c - growth)
        c, f = c - growth, f * 2
        tfc_tdf(f"dec{i}", c, f)
    conv("final", dim_c, growth, 1)
    return out


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "models/UVR_MDXNET_KARA_2.onnx"
    dst = sys.argv[2] if len(sys.argv) > 2 else "plain_c/models/kara.bin"

    graph = onnx.load(src).graph
    inits = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    # initializers in the order the graph's nodes consume them (= forward order)
    graph_order = [i for n in graph.node for i in n.input if i in inits]
    assert len(graph_order) == len(set(graph_order)) == len(inits), "initializer reused or unused"

    # the one BN epsilon we store must be the one every BN node uses
    for n in graph.node:
        if n.op_type == "BatchNormalization":
            eps = next(a.f for a in n.attribute if a.name == "epsilon")
            assert abs(eps - BN_EPS) < 1e-9, eps

    expected = tensor_list(**CONFIG)
    assert len(expected) == len(graph_order), (len(expected), len(graph_order))

    header = struct.pack("<9If", MAGIC, VERSION, *CONFIG.values(), BN_EPS)
    header += b"\0" * (HEADER_SIZE - len(header))

    total = 0
    with open(dst, "wb") as f, open(dst + ".manifest", "w") as man:
        f.write(header)
        for (name, shape), onnx_name in zip(expected, graph_order):
            a = inits[onnx_name]
            assert a.dtype == np.float32, (name, a.dtype)
            assert tuple(a.shape) == shape, f"{name}: onnx '{onnx_name}' has {a.shape}, expected {shape}"
            a = np.ascontiguousarray(a)
            f.write(a.tobytes())
            flat = a.ravel().astype(np.float64)
            man.write(f"{name} {a.size} {flat[0]:.9g} {flat[a.size // 2]:.9g} {flat[-1]:.9g} {flat.sum():.17g}\n")
            total += a.size

    print(f"wrote {dst}: {len(expected)} tensors, {total} floats, {HEADER_SIZE + 4 * total} bytes")


if __name__ == "__main__":
    main()
