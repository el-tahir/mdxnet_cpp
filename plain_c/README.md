# plain_c: MDX-Net Kara separator in dependency-free C

A second implementation of the vocal remover in this repo, in plain C99. It
needs only libc and libm: no ONNX Runtime, no C++, no FFT library. The MDX-Net
forward pass, the FFT, the STFT and the WAV handling are all loops in this
directory, so you can read every computation from input samples to output
samples. It is in the spirit of karpathy/llama2.c.

The C++/ONNX Runtime version at the repo root is kept as the reference. The two
produce the same output (116 dB SNR between them on the parity clip; see
"Verification").

## Build and run

```bash
cd plain_c
make                                  # builds ./separator (and the tests)

# one time: convert the ONNX model to the raw weight file (needs python3 + numpy + onnx)
pip install numpy onnx
python3 tools/export.py ../models/UVR_MDXNET_KARA_2.onnx models/kara.bin

./separator song.mp3 instrumental.wav            # default model: models/kara.bin
./separator song.wav instrumental.wav path/to/kara.bin
```

`../models/UVR_MDXNET_KARA_2.onnx` is downloaded by the top-level C++ build, or
fetch it directly from
`https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR_MDXNET_KARA_2.onnx`.

If `ffmpeg` is on the PATH it is run once at the start to decode any input to a
44.1 kHz 16-bit stereo WAV. Without it the input must already be a 44.1 kHz WAV
(16-bit PCM or 32-bit float, mono or stereo). Output is 32-bit float stereo WAV.

**Speed:** the kernels are currently naive loops, written straight from the
formulas. That's about 5.5 min per 256-frame chunk (about 6 s of audio) on one
core, against about 4 s for ONNX Runtime. Optimizing them is the next milestone
(see `../PROGRESS.md`).

## What happens to a song

| Step | Where |
|---|---|
| ffmpeg → 44.1 kHz stereo WAV (once, optional) | `main.c` `preprocess_input` |
| read WAV, split L/R | `wav.c`, `main.c` |
| reflect-pad 2048, periodic Hann, 4096-point FFT every 1024 samples | `stft.c`, `fft.c` |
| 256 frames → tensor `[4][2048][256]` (L.re, L.im, R.re, R.im; bins 0–2 zeroed) | `stft.c` `pack_chunk` |
| **MDX-Net forward pass** | `mdx.c` `mdx_forward` |
| tensor → frames (mirror the conjugate upper half, Nyquist = 0) | `stft.c` `unpack_chunk` |
| inverse FFT, window, overlap-add, crop, ÷1.5 | `stft.c`, `main.c` |
| noise gate (−40 dB), write float32 WAV | `main.c`, `wav.c` |

## The network

`tools/reference.py` is the whole forward pass in about 80 lines of numpy code. It is
the spec, and `mdx.c` follows it op for op.

```
x = relu(conv1x1(input, 4→48));  x = transpose(x)            # [C][T][F], F contiguous
5×  x = TFC_TDF(x); skip = x; x = relu(conv2x2_stride2(x, C→C+48))
    x = TFC_TDF(x)                                           # bottleneck [288][8][64]
5×  x = relu(BN(convT2x2_stride2(x, C→C-48))) * skip; x = TFC_TDF(x)
x = transpose(x);  output = conv1x1(x, 48→4)

TFC_TDF(x) = 3× relu(conv3x3(x)), then  x + relu(BN(relu(BN(x @ W1)) @ W2))   # W1: F→F/8, W2: F/8→F
```

Every kernel (`mdx_conv3x3`, `mdx_convT2x2_s2`, `mdx_matmul_lastdim`, …) is
declared in `mdx.h` with its exact index formula. `../PLAN.md` §1–2 has the
full trace of the ONNX graph.

## Files

| File | |
|---|---|
| `main.c` | CLI and the separation pipeline |
| `mdx.c`, `mdx.h` | weight loading, kernels, forward pass |
| `fft.c`, `fft.h` | radix-2 complex FFT |
| `stft.c`, `stft.h` | window, padding, STFT/ISTFT frames, frames ↔ model tensor |
| `wav.c`, `wav.h` | WAV read/write |
| `tools/reference.py` | numpy forward pass (the spec); checks itself against ONNX Runtime |
| `tools/export.py` | ONNX → `models/kara.bin` (64-byte header + raw float32 weights) |
| `tools/dump_acts.py` | saves ONNX Runtime's per-block activations for `test_forward` |
| `tools/gen_kernel_tests.py` | regenerates `tests/data/kernels.bin` |
| `tools/compare_wav.c` | SNR / header / gate comparison of two output WAVs |
| `tools/make_test_clip.py` | the synthetic clip used for the parity check |
| `tests/` | see below |

## Verification

```bash
make test     # first run also creates models/kara.bin and tests/data/acts_T32 (needs python3 + onnxruntime)
```

| Test | Checks |
|---|---|
| `test_fft` | FFT against a double-precision DFT, n = 1…4096 |
| `test_stft` | window, padding, STFT→ISTFT→overlap-add gain at every sample, pack/unpack layout |
| `test_wav` | WAV read/write/reject cases |
| `test_load` | all 220 weight tensors land in the right struct field |
| `test_kernels` | each kernel against numpy on small random cases |
| `test_forward` | the full forward pass against ONNX Runtime, block by block (13 checkpoints, T=32) |

End to end, `./compare_wav` of this separator against the C++ `build/separator`
on the same input gives byte-identical headers, identical noise-gate decisions
and 116 dB SNR. `../PROGRESS.md` has the exact procedure and results for every
milestone.
