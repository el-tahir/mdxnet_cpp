"""
Reference forward pass of UVR_MDXNET_KARA_2.onnx in plain numpy.

This is the spec for plain_c/mdx.c: every op here maps to one C kernel, and the
weights are consumed in exactly the order the ONNX graph uses them (which is also
the order export.py writes them to kara.bin).

    python3 plain_c/tools/reference.py [path/to/model.onnx]

runs the reference on a fixed random input, runs ONNX Runtime on the same input,
and checks they agree (bottleneck and final output).

Needs: numpy, onnx, onnxruntime (onnxruntime only for the check).
"""
import sys
import time

import numpy as np
import onnx
from onnx import numpy_helper

EPS = 1e-5  # BatchNormalization epsilon, same for all 27 BN nodes in the graph

# ----------------------------------------------------------------------------
# ops. All tensors are [C, H, W] (batch 1), float32.


def relu(x):
    return np.maximum(x, 0)


def batchnorm(x, scale, bias, mean, var):
    """y[c] = (x[c] - mean[c]) / sqrt(var[c] + eps) * scale[c] + bias[c]"""
    k = scale / np.sqrt(var + EPS)
    return x * k[:, None, None] + (bias - mean * k)[:, None, None]


def conv1x1(x, w, b):
    """w [Cout, Cin, 1, 1]: y[o,h,w] = b[o] + sum_i w[o,i] * x[i,h,w]"""
    return np.einsum("oi,ihw->ohw", w[:, :, 0, 0], x, optimize=True) + b[:, None, None]


def conv3x3(x, w, b):
    """w [Cout, Cin, 3, 3], stride 1, zero pad 1, cross-correlation (no flip):
    y[o,h,w] = b[o] + sum_i sum_ky sum_kx w[o,i,ky,kx] * x[i, h+ky-1, w+kx-1]"""
    _, H, W = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    y = np.zeros((w.shape[0], H, W), np.float32) + b[:, None, None]
    for ky in range(3):
        for kx in range(3):
            y += np.einsum("oi,ihw->ohw", w[:, :, ky, kx], xp[:, ky:ky + H, kx:kx + W], optimize=True)
    return y


def conv2x2_s2(x, w, b):
    """w [Cout, Cin, 2, 2], stride 2, no pad (non-overlapping windows):
    y[o,h,w] = b[o] + sum_i sum_ky sum_kx w[o,i,ky,kx] * x[i, 2h+ky, 2w+kx]"""
    C, H, W = x.shape
    xr = x.reshape(C, H // 2, 2, W // 2, 2)  # [i, h, ky, w, kx]
    return np.einsum("oiyx,ihywx->ohw", w, xr, optimize=True) + b[:, None, None]


def convT2x2_s2(x, w, b):
    """w [Cin, Cout, 2, 2]  (note: Cin first, unlike Conv), stride 2:
    y[o, 2h+ky, 2w+kx] = b[o] + sum_i w[i,o,ky,kx] * x[i,h,w]"""
    _, H, W = x.shape
    y = np.einsum("ioyx,ihw->ohywx", w, x, optimize=True).reshape(w.shape[1], 2 * H, 2 * W)
    return y + b[:, None, None]


def matmul_lastdim(x, w):
    """w [F_in, F_out] (not nn.Linear's [out, in]): y[c,t,j] = sum_f x[c,t,f] * w[f,j]"""
    return x @ w


def transpose_last2(x):
    """y[c,j,i] = x[c,i,j]"""
    return np.ascontiguousarray(x.transpose(0, 2, 1))


# ----------------------------------------------------------------------------
# weights: every initializer, in the order the graph's nodes consume them


class Weights:
    def __init__(self, onnx_path):
        g = onnx.load(onnx_path).graph
        inits = {t.name: numpy_helper.to_array(t).astype(np.float32) for t in g.initializer}
        self.tensors = [inits[i] for n in g.node for i in n.input if i in inits]
        assert len(self.tensors) == len(inits) == 220
        self.pos = 0

    def take(self, n=1):
        out = self.tensors[self.pos:self.pos + n]
        self.pos += n
        return out

    def done(self):
        return self.pos == len(self.tensors)


# ----------------------------------------------------------------------------
# model


def tfc_tdf(x, wt):
    """TFC: 3x (conv3x3 + ReLU), BN already folded into the convs by the exporter.
    TDF: Linear along F down to F/8 and back, each followed by BN + ReLU.
    Residual add of the two."""
    for _ in range(3):
        x = relu(conv3x3(x, *wt.take(2)))
    h = relu(batchnorm(matmul_lastdim(x, *wt.take(1)), *wt.take(4)))
    h = relu(batchnorm(matmul_lastdim(h, *wt.take(1)), *wt.take(4)))
    return x + h


def forward(inp, wt, n_scales=5):
    """inp [4, F=2048, T=256] -> [4, 2048, 256]"""
    x = relu(conv1x1(inp, *wt.take(2)))       # first_conv: 4 -> 48, [48, F, T]
    x = transpose_last2(x)                     # [C, T, F]: F is now the contiguous axis

    skips = []
    for _ in range(n_scales):                  # encoder
        x = tfc_tdf(x, wt)
        skips.append(x)
        x = relu(conv2x2_s2(x, *wt.take(2)))   # C -> C+48, T/2, F/2 (BN folded)

    x = tfc_tdf(x, wt)                         # bottleneck [288, 8, 64]
    bottleneck = x

    for i in range(n_scales):                  # decoder
        x = relu(batchnorm(convT2x2_s2(x, *wt.take(2)), *wt.take(4)))  # C -> C-48, 2T, 2F
        x = x * skips[-1 - i]                  # multiplicative skip connection
        x = tfc_tdf(x, wt)

    x = transpose_last2(x)                     # back to [C, F, T]
    out = conv1x1(x, *wt.take(2))              # final_conv: 48 -> 4, no activation
    assert wt.done(), "not every weight was consumed"
    return out, bottleneck


# ----------------------------------------------------------------------------
# check against ONNX Runtime


def main():
    import onnxruntime as ort

    path = sys.argv[1] if len(sys.argv) > 1 else "models/UVR_MDXNET_KARA_2.onnx"
    rng = np.random.default_rng(0)
    inp = (rng.standard_normal((1, 4, 2048, 256)) * 0.5).astype(np.float32)

    # expose the bottleneck output (ONNX tensor "571") as an extra graph output
    model = onnx.load(path)
    model.graph.output.append(onnx.helper.make_tensor_value_info("571", onnx.TensorProto.FLOAT, None))
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])

    t = time.time()
    ort_out, ort_bott = sess.run(["output", "571"], {"input": inp})
    print(f"onnxruntime: {time.time() - t:.2f}s")

    t = time.time()
    ref_out, ref_bott = forward(inp[0], Weights(path))
    print(f"reference:   {time.time() - t:.2f}s")

    ok = True
    for name, a, b in [("bottleneck", ort_bott[0], ref_bott), ("output", ort_out[0], ref_out)]:
        rel = np.abs(a - b).max() / np.abs(a).max()
        passed = rel < 1e-5
        ok &= passed
        print(f"{name:10s} max|diff|/max|ort| = {rel:.2e}  {'OK' if passed else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
