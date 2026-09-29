# Plan: dependency-free C port of the separator, including the `UVR_MDXNET_KARA_2.onnx` forward pass

Goal: write a second, self-contained implementation of the whole separator in
plain C99 under `plain_c/`, in the spirit of
karpathy/llama2.c: WAV I/O, FFT, STFT/ISTFT, the MDX-Net forward pass and the
CLI. No ONNX Runtime, no C++, no vendored libraries — every computation from
input samples to output samples is a loop in our own source. Runtime needs only
libc + libm. Python is used only offline (weight export + verification), never
at runtime.

The existing C++/ORT implementation stays in the repo untouched and is kept as
the reference: every C result is checked against it (or against ORT directly).

---

## 1. What we are implementing (verified)

The graph was traced node-by-node and re-implemented in numpy; the numpy version
matches ORT on a full `[1,4,2048,256]` input to ~3e-6 relative error and consumes
all 220 initializers.

- PyTorch 1.9 export, opset 13, no metadata.
- I/O: `input [B,4,2048,256]` → `output [B,4,2048,256]`
  (channels = L.re, L.im, R.re, R.im; 2048 freq bins; 256 time frames). We run B=1.
- 178 nodes, 8 op types: Conv(40) ConvTranspose(5) MatMul(22) BatchNorm(27)
  Relu(66) Add(11) Mul(5) Transpose(2).
- 13,191,108 float32 params (52.8 MB).

```
x = relu(conv1x1(input, 4→48))                   # [48,2048,256]
x = transpose_last2(x)                           # [48,256,2048]  = [C,T,F], F contiguous

for i in 0..4:                                   # encoder, C = 48·(i+1)
    x = TFC_TDF(x);  skip[i] = x
    x = relu(conv2x2_s2(x, C → C+48))            # T,F halved
x = TFC_TDF(x)                                   # bottleneck [288,8,64]

for i in 4..0:                                   # decoder
    x = relu(BN(convT2x2_s2(x, C → C-48)))       # T,F doubled
    x = x * skip[i]                              # multiplicative skip
    x = TFC_TDF(x)

x = transpose_last2(x)                           # [48,2048,256]
output = conv1x1(x, 48→4)                        # no activation

TFC_TDF(x):  (C channels, T×F)
    repeat 3:  x = relu(conv3x3_p1(x))           # BN already folded into conv by exporter
    h = relu(BN(x @ W1))                         # W1 [F, F/8], no bias
    h = relu(BN(h @ W2))                         # W2 [F/8, F], no bias
    return x + h
```

| Level | C | T × F | Tensor size |
|---|---|---|---|
| 0 | 48 | 256 × 2048 | 96 MiB |
| 1 | 96 | 128 × 1024 | 48 MiB |
| 2 | 144 | 64 × 512 | 18 MiB |
| 3 | 192 | 32 × 256 | 6 MiB |
| 4 | 240 | 16 × 128 | 1.9 MiB |
| 5 (bottleneck) | 288 | 8 × 64 | 0.6 MiB |

Compute per 256-frame chunk: ~472 GFLOP — conv3x3 81%, TDF matmuls 14%,
down/up convs 5%, 1×1 convs ~0%.

---

## 2. Exact op definitions (what each C kernel computes)

Tensors are dense row-major `[C][H][W]`, batch 1. `x` is input, `y` output.

| Op | Definition | ONNX weight layout |
|---|---|---|
| `conv1x1` | `y[o,h,w] = b[o] + Σ_i W[o,i]·x[i,h,w]` | `[Cout,Cin,1,1]` |
| `conv3x3_p1` | `y[o,h,w] = b[o] + Σ_i Σ_ky Σ_kx W[o,i,ky,kx]·x[i,h+ky-1,w+kx-1]`, zero outside bounds. Cross-correlation, **no kernel flip**. | `[Cout,Cin,3,3]` |
| `conv2x2_s2` | `y[o,h,w] = b[o] + Σ_i Σ_ky Σ_kx W[o,i,ky,kx]·x[i,2h+ky,2w+kx]` (non-overlapping windows) | `[Cout,Cin,2,2]` |
| `convT2x2_s2` | `y[o,2h+ky,2w+kx] = b[o] + Σ_i W[i,o,ky,kx]·x[i,h,w]` (each output pixel gets exactly one term per input channel) | **`[Cin,Cout,2,2]`** (note: transposed vs Conv) |
| `matmul_lastdim` | `y[c,t,j] = Σ_f x[c,t,f]·W[f,j]` | `[F_in,F_out]` (**not** nn.Linear's `[out,in]`) |
| `batchnorm` | `y[c,…] = (x[c,…] − mean[c]) / sqrt(var[c] + eps) · scale[c] + bias[c]`, eps = 1e-5 | 4 × `[C]`: scale, bias, mean, var |
| `relu` | `y = max(x, 0)` | — |
| `add`, `mul` | elementwise, same shape (no broadcasting in this model) | — |
| `transpose_last2` | `y[c,j,i] = x[c,i,j]` | — |

BatchNorm is kept explicit (not folded into the preceding MatMul/ConvT) so the C
code mirrors the ONNX graph 1:1. Folding it is an optional later optimization.

---

## 3. Repository layout (target)

All new code lives in `plain_c/`. Nothing outside it changes (apart from
`.gitignore` entries for its build outputs).

```
mdxnet_cpp/
├── src/, include/, tests/, third_party/, CMakeLists.txt, Makefile   C++/ORT reference — kept as-is
├── models/UVR_MDXNET_KARA_2.onnx                                     shared, gitignored
└── plain_c/
    ├── main.c              CLI + separation pipeline (mirrors src/main.cpp)
    ├── wav.c   / wav.h     WAV read/write (mirrors include/WAVHeader.h)
    ├── fft.c   / fft.h     our own radix-2 complex FFT (instead of kiss_fft)
    ├── stft.c  / stft.h    Hann window, reflect pad, STFT/ISTFT, pack/unpack (mirrors DSPCore.cpp, utils.cpp)
    ├── mdx.c   / mdx.h     model: weight loading, buffers, kernels, forward pass (instead of ModelHandler.cpp + ORT)
    ├── tests/test_fft.c    FFT vs naive O(N²) DFT
    ├── tests/test_stft.c   STFT→ISTFT round trip reconstructs the input
    ├── tests/test_mdx.c    kernels + forward pass vs ORT dumps
    ├── tools/reference.py  numpy forward pass (the verified spec, ~60 lines)
    ├── tools/export.py     ../models/*.onnx → models/kara.bin (header + raw float32 weights)
    ├── tools/dump_acts.py  runs ORT, writes fixed input + tapped activations for tests
    ├── models/             kara.bin (gitignored)
    └── Makefile            `cc -O3 -o separator *.c -lm`, plus `test` target
```

`plain_c/` shares no code with the C++ tree (it does not use kiss_fft or any
header from `include/`), so `cd plain_c && make` needs only a C compiler.
OpenMP pragmas in the perf phase are optional and compile away without
`-fopenmp`.

---

## 4. Weight file format (`kara.bin`)

llama2.c-style: fixed header, then every tensor as raw little-endian float32,
back to back, in forward-pass order. Loaded with `fread` (or `mmap`) and
pointer-bumped into a `Weights` struct — no names, no parsing.

```
Header (64 bytes)
  u32 magic      = 'MDXN'
  u32 version    = 1
  u32 dim_c      = 4       // input channels
  u32 dim_f      = 2048
  u32 dim_t      = 256
  u32 n_scales   = 5
  u32 growth     = 48      // channels added per scale
  u32 n_tfc      = 3       // conv3x3 layers per TFC
  u32 bn_factor  = 8       // TDF bottleneck F → F/8
  f32 bn_eps     = 1e-5
  (reserved, zero padded to 64 bytes)

Tensors, in this order:
  first_conv.w [48,4,1,1], first_conv.b [48]
  for each of 11 blocks (enc0..enc4, bottleneck, dec0..dec4):
      if decoder: up.w [Cin,Cout,2,2], up.b, up_bn.{scale,bias,mean,var}
      tfc[0..2].{w [C,C,3,3], b [C]}
      tdf1.w [F,F/8], tdf1_bn.{scale,bias,mean,var}
      tdf2.w [F/8,F], tdf2_bn.{scale,bias,mean,var}
      if encoder: down.w [C+48,C,2,2], down.b
  final_conv.w [4,48,1,1], final_conv.b [4]
```

`export.py` walks the ONNX nodes in graph order (already confirmed to be forward
order), asserts every tensor's shape against what the header predicts, writes
it, and asserts all 220 initializers were written exactly once. The C loader
performs the same shape arithmetic and checks the file size matches exactly.

---

## 5. The rest of the app in C

The C version reproduces the C++ pipeline step for step, so the C++ binary
serves as the reference for end-to-end parity, now and after future changes.

| Step | Current C++ | C replacement | Exact behaviour to preserve |
|---|---|---|---|
| Decode | `preprocess_input` → `system("ffmpeg …")` | `system()` in `main.c` | called **exactly once, before anything else**: converts any input to s16le 44.1 kHz stereo WAV in a temp file; falls back to the original file if ffmpeg fails; temp file removed at exit. Nothing after this step invokes an external program — everything from WAV samples to output WAV is our C code. |
| Read | `read_wav` | `wav_read` | 44.1 kHz only; PCM s16 (`/32768`) or float32; mono duplicated to stereo; deinterleave to L/R |
| Pad | `DSPCore::pad_audio` | `stft_pad` | reflect pad `n_fft/2 = 2048` samples each side: `p[i] = x[2047−i]`, tail `x[N−1−i]` |
| Window | `create_hann_window` | `stft_init` | periodic Hann: `w[n] = 0.5·(1 − cos(2πn/4096))` |
| STFT | `DSPCore::stft` | `stft_frame` | frame every `hop = 1024`; `X = FFT(w·x)`, full 4096 complex bins |
| Pack | `stft_to_tensor` | `pack_chunk` | 256 frames per chunk → `[4,2048,256]` = (L.re, L.im, R.re, R.im) × bins 0..2047 × frames; **bins 0–2 forced to 0**; short last chunk zero-filled |
| Model | `ModelHandler::run_inference` | `mdx_forward` | sections 1–2 |
| Unpack | `tensor_to_stft` | `unpack_chunk` | bins 0..2047 from tensor; `X[0].im = 0`; Nyquist `X[2048] = 0`; `X[4096−k] = conj(X[k])` for k = 1..2047 |
| ISTFT | `DSPCore::istft` | `istft_frame` | `y = w · Re(IFFT(X)) / 4096` |
| Overlap-add | `run_seperation` | `main.c` | add each frame at `frame·1024`, crop the 2048-sample pad, divide by 1.5 (Σw² for 75% overlap) |
| Noise gate | `apply_noise_gate` | `noise_gate` | threshold −40 dB, RMS over ±2048 interleaved samples; zero both channels of a frame below threshold |
| Write | `write_wav` | `wav_write` | 44-byte header, float32 stereo |

**FFT:** kiss_fft is replaced by our own iterative radix-2 Cooley–Tukey FFT
(n = 4096 is a power of two): bit-reversal permutation, then log₂N butterfly
stages with precomputed twiddles `e^{−2πik/N}`; inverse uses `+` sign and no
scaling (matching kiss_fft, the `/4096` stays in ISTFT). ~60 lines. Verified
against a naive O(N²) DFT in double precision.

**Quirks carried over as-is for parity** (fix after parity is proven, each as its
own change): the WAV reader assumes a bare 44-byte header (works because ffmpeg
is run with `-fflags +bitexact -map_metadata -1`); the noise gate is
O(N·4096); the whole song's STFT is held in memory.

---

## 6. Memory plan

All buffers allocated once in `mdx_load`; `mdx_forward` does no allocation.

| Buffer | Size | Purpose |
|---|---|---|
| `skip[0..4]` | 96+48+18+6+1.9 MiB | encoder block outputs; block output is written straight into its skip slot |
| `a`, `b` | 96 MiB each | ping-pong for conv chains (conv is not in-place) |
| `h` | 12 MiB | TDF hidden `[C,T,F/8]` |

≈ 375 MiB total. Each TFC_TDF: conv a→b→a→b, `h = tdf1(b)`, `a = tdf2(h)`,
`a += b`. Level sizes come from the header, so nothing is hardcoded.

---

## 7. Verification strategy

1. `dump_acts.py` feeds a fixed seeded input to ORT, with these node outputs
   added as extra graph outputs, and saves each as raw float32:

   | Tap (ONNX tensor) | Meaning |
   |---|---|
   | `447` | after first_conv + ReLU |
   | `466`, `487`, `508`, `529`, `550` | encoder block outputs (= skips) |
   | `571` | bottleneck output |
   | `593`, `615`, `637`, `659`, `681` | decoder block outputs |
   | `output` | final |

   Any other node can be tapped on demand while debugging a mismatch.
2. The C test runs the same input and compares at each tap:
   `max|c − ort| / max|ort| < 1e-4` (fp32 summation order differs, so bitwise
   equality isn't expected; the numpy ref already lands at ~3e-6).
3. Per-kernel unit tests on small random tensors against numpy, so a bug is
   pinned to one kernel before it pollutes a whole block.
4. DSP: `test_fft` (vs naive DFT, max error < 1e-4 relative) and `test_stft`
   (pad → STFT → ISTFT → OLA → crop → /1.5 reproduces the input, > 90 dB SNR).
5. End-to-end: run the C++ `separator` and `plain_c/separator` on the same
   songs; the outputs must match with SNR > 60 dB. A `plain_c/tools/compare_wav.py`
   (or a small C tool) reports the SNR. Since the C++ build stays in the repo,
   this check can be rerun any time.

Speed-up for tests: the model is fully convolutional in T (only F is baked into
the TDF weights), so `dump_acts.py` can relax the input dim to allow e.g.
T=32 — 8× less compute per test run. The C code handles any T divisible by 32.

---

## 8. Milestones

Each milestone ends with a passing check; nothing proceeds on a red check.

| # | Deliverable | Done when |
|---|---|---|
| M0 | `tools/reference.py` committed | matches ORT output < 1e-5 rel |
| M1 | `tools/export.py` → `kara.bin`; `mdx_load` in C | loader reads header, maps all tensors, file size exact, spot-check tensors equal ONNX values |
| M2 | All 9 kernels, naive loops, + unit tests | each kernel matches numpy < 1e-5 rel |
| M3 | first_conv + transpose + enc0 TFC_TDF | taps `447`, `466` match |
| M4 | full encoder + bottleneck | taps through `571` match |
| M5 | decoder + final conv | `output` matches < 1e-4 rel |
| M6 | `fft.c`, `stft.c` + tests | `test_fft`, `test_stft` pass |
| M7 | `wav.c`, `main.c`: full pipeline in C | `plain_c/separator` output SNR > 60 dB vs the C++ `separator` on the same songs |
| M8 | `plain_c/Makefile` with `test` target; README section | `cd plain_c && make && make test` works on a clean checkout with only a C compiler (plus `kara.bin` from `export.py`) |
| M9 | Performance (see below), naive kernels kept behind a flag | taps still match; per-chunk time within ~2× of ORT |

---

## 9. Performance (M9)

Budget: a 3-min song ≈ 7.7k frames ≈ 31 chunks ≈ 14.6 TFLOP.
ORT on this 4-core box: 4.4 s/chunk (~2.3 min/song).

| Stage | Expected throughput | Song time |
|---|---|---|
| Naive loops, `-O2` | ~1 GFLOP/s | hours |
| Loop order so the inner loop runs over contiguous W (auto-vectorized), `-O3 -march=native` | 10–20 GFLOP/s | 12–25 min |
| + OpenMP over output channels | ×cores | 3–6 min |
| conv3x3 as im2col + cache-blocked SGEMM microkernel (still plain C) | 50–100+ GFLOP/s | ~ORT |

conv3x3 is 81% of FLOPs, so it's the only kernel that matters for speed; the
TDF matmuls reuse the same SGEMM. Everything stays dependency-free (no BLAS).

---

## 10. Open decisions

1. **Language**: decided — everything in `plain_c/` is C99; the C++/ORT
   implementation stays in the repo as the reference.
2. **ffmpeg**: decided — kept, invoked once via `system()` to decode the input
   before the pipeline starts; never called afterwards.
3. **BatchNorm**: keep explicit (current plan, 1:1 with ONNX) vs fold at export.
4. **Transposes**: keep explicit (mirrors ONNX) vs fold into first/final conv
   indexing (saves 2 × 96 MiB copies, negligible time).
5. **Other MDX-Net models**: header already carries the hyperparameters, so
   supporting e.g. dim_f=3072 models is mostly an `export.py` concern — out of
   scope until Kara is bit-for-bit solid.
