"""
Run ONNX Runtime on a fixed input and save intermediate activations ("taps")
for tests/test_forward.c.

    python3 plain_c/tools/dump_acts.py [model.onnx] [T] [out_dir]

The graph's time dim is fixed at 256; the model is fully convolutional in T
(only F is baked into the TDF weights), so we relax it and use a smaller T to
keep tests fast. T must be divisible by 2^n_scales = 32.

Writes to out_dir (default plain_c/tests/data/acts_T<T>/, gitignored):
    input.bin          float32 [4][2048][T]
    <tap>.bin          float32, one per tap below
    taps.txt           "<tap> <onnx tensor> <dim0> <dim1> <dim2>" per line
"""
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort

# C-side tap name -> ONNX tensor name (see PLAN.md section 7)
TAPS = [
    ("first", "447"),   # relu(conv1x1(input)), [48][F][T]  (before the transpose)
    ("enc0", "466"),    # encoder block outputs = skips, [C][T][F]
    ("enc1", "487"),
    ("enc2", "508"),
    ("enc3", "529"),
    ("enc4", "550"),
    ("mid", "571"),     # bottleneck block output
    ("dec0", "593"),    # decoder block outputs
    ("dec1", "615"),
    ("dec2", "637"),
    ("dec3", "659"),
    ("dec4", "681"),    # [48][T][F] (before the final transpose)
    ("output", "output"),  # [4][F][T]
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "models", "UVR_MDXNET_KARA_2.onnx")
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    dst = sys.argv[3] if len(sys.argv) > 3 else os.path.join(here, "..", "tests", "data", f"acts_T{T}")
    assert T > 0 and T % 32 == 0, "T must be a positive multiple of 32"

    model = onnx.load(src)
    for v in list(model.graph.input) + list(model.graph.output):
        v.type.tensor_type.shape.dim[3].dim_param = "T"  # relax the fixed 256
    model.graph.ClearField("value_info")  # stale inferred shapes would pin T again
    existing = {o.name for o in model.graph.output}
    for _, name in TAPS:
        if name not in existing:
            model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    inp = (rng.standard_normal((1, 4, 2048, T)) * 0.5).astype(np.float32)
    outs = sess.run([name for _, name in TAPS], {"input": inp})

    os.makedirs(dst, exist_ok=True)
    inp[0].astype("<f4").tofile(os.path.join(dst, "input.bin"))
    with open(os.path.join(dst, "taps.txt"), "w") as f:
        for (tap, name), a in zip(TAPS, outs):
            a = a[0]  # drop batch
            assert a.ndim == 3
            a.astype("<f4").tofile(os.path.join(dst, f"{tap}.bin"))
            f.write(f"{tap} {name} {a.shape[0]} {a.shape[1]} {a.shape[2]}\n")
    print(f"wrote {len(TAPS)} taps for T={T} to {dst}")


if __name__ == "__main__":
    main()
