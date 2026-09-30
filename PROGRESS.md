# Progress: plain C port (`plain_c/`)

Read this first, then `PLAN.md` (the full design: op formulas, file format,
memory plan, verification strategy, milestones). This file tracks where the
work stands and what a new agent needs to pick it up. **Update it at the end of
every milestone** (status table, "Next up", and anything learned the hard way).

Branch: `claude/chat-session-5b6r4k`

---

## Status

| # | Milestone | Status | Commit |
|---|---|---|---|
| — | Trace ONNX graph, write `PLAN.md` | done | `3d75fbd` |
| M0 | `plain_c/tools/reference.py` numpy forward pass matches ORT | **done** | `4b2681f` |
| M1 | `tools/export.py` → `kara.bin`; `mdx_load` in C; `tests/test_load.c` | **done** | `8452b6d` |
| M2 | 9 kernels (naive loops) + unit tests vs numpy | **done** | `f6f7190` |
| M3 | first_conv + transpose + enc0 TFC_TDF; taps `447`, `466` match | **done** | `7cd6f52` |
| M4 | full encoder + bottleneck; taps through `571` match | **done** (with M3) | `7cd6f52` |
| M5 | decoder + final conv; `output` matches < 1e-4 rel | **done** (with M3) | `7cd6f52` |
| M6 | `fft.c`, `stft.c` + tests | **done** | `b2d71de` |
| M7 | `wav.c`, `main.c`: full C pipeline, SNR > 60 dB vs C++ `separator` | **done** | `5274de0`, `cfba5ef` |
| M8 | `plain_c/Makefile` `test` target complete; README section | **done** | `2ff2fd2` |
| M9 | performance | next | |

### Verified results so far
- M0: reference vs ORT on seeded input `[1,4,2048,256]` (`np.random.default_rng(0)`, ×0.5):
  bottleneck rel err 9.4e-7, output 3.4e-6. Reference takes ~20 s, ORT ~2.6–4.4 s on this 4-core box.
- M1: `make test` → 220/220 tensors match the manifest by name; truncated / +1 float /
  header-only / empty files all rejected. Clean under `-fsanitize=address,undefined`.
  `kara.bin` = 52,764,496 bytes (64 header + 13,191,108 floats).
- M2: `make test` → 36/36 kernel cases pass, worst rel err 2.8e-7 (tolerance 1e-5).
  Mutation check: flipping the 3×3 kernel, or indexing ConvTranspose weights as
  `[o][i]` instead of `[i][o]`, makes the relevant cases fail (so the tests have teeth).
  Clean under ASan/UBSan.
- M3–M5 (landed together: the whole `mdx_forward` is ~60 lines, and the test
  checks every block's output, so the milestones stayed separately verified):
  `test_forward` at T=32, all 13 taps match ORT. Worst rel err per tap ≤ 2.2e-6
  inside the net, 4.8e-6 at `output` (tolerance 1e-4). Naive loops: 34 s at T=32
  (`-O2`, 1 thread).
  Full size T=256 (`python3 tools/dump_acts.py ../models/UVR_MDXNET_KARA_2.onnx 256`, 450 MB dump):
  all 13 taps match, worst 2.3e-6 inside the net, 4.8e-6 at `output`; 287 s naive.
  T=32 run clean under ASan/UBSan (no errors, no leaks).
- M6: `test_fft`: forward, inverse and inverse∘forward vs a double-precision
  naive DFT for n = 1..4096, worst rel err 4.0e-7; non-powers of two rejected.
  `test_stft`: Hann periodic with Σw² = 1.5 at hop n/4; reflect pad matches
  `DSPCore::pad_audio`; full pad → STFT → ISTFT → overlap-add → crop → ÷1.5 on
  3 s of noise equals `x·S(p)/1.5` at every sample (138 dB SNR) and `x` in the
  interior; pack/unpack layout and dropped bins exact. One-off check against the
  real C++ `DSPCore` + kiss_fft (harness compiled in scratch, not committed):
  STFT frame rel err 4.1e-8, ISTFT 1.5e-7. Both tests clean under ASan/UBSan.
- M7: end-to-end on a 10 s synthetic clip (`tools/make_test_clip.py`, encoded to
  mp3 so both binaries go through ffmpeg): `plain_c/separator` vs C++
  `build/separator` → headers byte-identical, same length (441,000 samples/ch),
  **SNR 116.2 dB**, max |diff| 4.2e-7 (peak 0.71), noise-gate decisions identical
  (0 samples zeroed in only one file), temp file cleaned up. C++: 10 s total;
  C (naive): 681 s (~335 s per 256-frame chunk). `test_wav`: 7/7.
- M8: `plain_c/README.md` + a short section in the top-level `README.md`.
  Fresh `git clone` + the ONNX file → `cd plain_c && make && make test`: builds
  with zero warnings (gcc, `-std=c99 -Wall -Wextra -pedantic`), exports
  `kara.bin` and dumps `acts_T32` itself, all 6 tests pass. Also builds warning-free
  with clang (`make CC=clang`).

---

## Ground rules (decided with the user — don't relitigate)

- **All new code lives in `plain_c/`.** Nothing outside it changes except
  `.gitignore`, `PLAN.md`, `PROGRESS.md`.
- **The C++/ORT implementation (`src/`, `include/`, `tests/`, `CMakeLists.txt`,
  top-level `Makefile`) is the reference. Never delete or modify it.**
- **Everything in `plain_c/` is C99**, libc + libm only. No C++, no kiss_fft, no
  BLAS, no headers from `include/`. OpenMP pragmas allowed later (M9) only if the
  code still builds and runs without `-fopenmp`.
- **ffmpeg** stays, called exactly once via `system()` at program start to decode
  the input to a temp WAV; nothing after that runs an external program.
- Python (numpy, onnx, onnxruntime) is for offline tools/tests only, never at runtime.
- Goal is readability: every computation of the forward pass visible as a loop.
  BatchNorm and the two transposes stay explicit (1:1 with the ONNX graph).
- Don't open PRs unless the user asks. Commit + push to the branch above.

---

## Environment setup (fresh container)

```bash
cd mdxnet_cpp
pip install onnx onnxruntime numpy
mkdir -p models
curl -L -o models/UVR_MDXNET_KARA_2.onnx \
  https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR_MDXNET_KARA_2.onnx
md5sum models/UVR_MDXNET_KARA_2.onnx   # 3a00bfec5b627a8ca6f121ea0e2a76a2

cd plain_c
make test          # exports models/kara.bin on first run, builds + runs C tests
python3 tools/reference.py ../models/UVR_MDXNET_KARA_2.onnx   # numpy vs ORT (~30 s)
```

`models/` (both top-level and `plain_c/models/`) is gitignored; so are
`plain_c/test_*`, `separator`, `compare_wav` binaries.

For the end-to-end parity check (M7) you also need ffmpeg and the C++ build:

```bash
# this container had no ffmpeg and apt couldn't install it; a static build via pip works:
pip install imageio-ffmpeg
ln -sf "$(python3 -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" /usr/local/bin/ffmpeg

make                                   # at repo root: C++ reference → build/separator (CMake downloads ORT 1.16.3)
python3 plain_c/tools/make_test_clip.py /tmp/clip.wav 10 && ffmpeg -y -i /tmp/clip.wav /tmp/clip.mp3
./build/separator /tmp/clip.mp3 /tmp/out_cpp.wav        # run from repo root (model path is relative)
cd plain_c && make && ./separator /tmp/clip.mp3 /tmp/out_c.wav   # ~11 min with the naive kernels
./compare_wav /tmp/out_cpp.wav /tmp/out_c.wav           # expect > 60 dB (got 116.2)
```

---

## Code map (`plain_c/`)

| File | What it is |
|---|---|
| `tools/reference.py` | numpy forward pass — **the spec**. One function per C kernel, index formula in each docstring. `Weights.take()` consumes initializers in graph order. |
| `tools/export.py` | ONNX → `models/kara.bin` + `kara.bin.manifest`. `tensor_list()` defines names/shapes/order of every tensor from the header config; asserts every ONNX initializer matches its slot. |
| `mdx.h` | `MdxConfig` (header), `MdxConv`/`MdxBN`/`MdxBlock`/`MdxWeights` (pointers into one float buffer), `mdx_load`/`mdx_free`. |
| `mdx.c` | Loader: `map_weights()` walks the same order as `tensor_list()`; with `data == NULL` it only counts, used to require an exact file size before reading. |
| `tests/test_load.c` | Walks `MdxWeights` by field name, compares count / first / mid / last / sum to the manifest. |
| `mdx.c` kernels | `mdx_conv1x1`, `mdx_conv3x3`, `mdx_conv2x2_s2`, `mdx_convT2x2_s2`, `mdx_matmul_lastdim`, `mdx_batchnorm` (in place), `mdx_relu`/`mdx_add`/`mdx_mul` (in place), `mdx_transpose_last2`. Declared in `mdx.h` with their formulas. Naive: loops follow the formula one output element at a time. Sizes passed as `int` (largest tensor 25.2M elements fits), offsets computed in `long`. |
| `tools/gen_kernel_tests.py` | Random small cases per kernel (incl. 1-sized, odd, cin≠cout, H≠W) → `tests/data/kernels.bin` (committed, 30 KB) using `reference.py` for expected outputs. Record format documented in its docstring. |
| `tests/test_kernels.c` | Runs each C kernel on each case; pass if max\|diff\|/max\|ref\| < 1e-5. |
| `mdx.c` forward | `MdxState` (`mdx_state_init(s, cfg, T)`): buffers `a`, `b` (level-0 size), `h` (TDF hidden), `skip[i]`. `tfc_tdf()` ping-pongs between two buffers and **returns the one holding its output**. Encoder blocks run in place on `skip[i]` (downsample writes straight into `skip[i+1]`); bottleneck/decoder alternate `a`/`b`. Optional tap callback `s->tap(ctx, name, t, c, h, w)` after every block (names: `first`, `enc0..4`, `mid`, `dec0..4`, `output`). At T=256 the buffers total ~375 MB. |
| `fft.c` / `fft.h` | `Complex {re, im}`, `FFTPlan` (`fft_init(p, n)`, power of 2 only): iterative radix-2 DIT, bit-reversal table + cos/sin twiddle tables (computed in double, stored float). `fft_forward` (e^{-i}), `fft_inverse` (e^{+i}, **no 1/n**, like kiss_fft). |
| `stft.c` / `stft.h` | `stft_hann` (periodic, double precision like the C++), `stft_reflect_pad` (edge sample repeated, like `DSPCore::pad_audio`), `stft_frame`, `istft_frame` (÷n and window applied here), `pack_chunk` / `unpack_chunk` (frames ↔ `[4][F][T]` model tensor; bins 0–2 zeroed on pack; unpack zeroes bins F..n/2 and mirrors conjugates). Framing / overlap-add / ÷1.5 live in the caller (see `tests/test_stft.c` §3 for the exact loop `main.c` needs). |
| `tests/test_fft.c`, `tests/test_stft.c` | See M6 results above. No data files, no Python. |
| `wav.c` / `wav.h` | Port of `include/WAVHeader.h`: 44-byte header parsed field by field (little-endian, no `#pragma pack`), 44.1 kHz PCM16 (`/32768`) or float32, mono → stereo. `wav_write` copies the input header and patches format fields (float, 2 ch, 32 bit, sizes) → byte-identical header to the C++ output. Rejects what the C++ would misread (24-bit, 48 kHz…). |
| `main.c` | `separator in out [models/kara.bin]`. ffmpeg once via `system()` (paths single-quoted) → temp WAV, fallback to the original file. Then the C++ pipeline step for step, **but chunk by chunk**: STFT 256 frames → `pack_chunk` → `mdx_forward` → `unpack_chunk` → ISTFT + overlap-add, so memory stays at ~375 MB model state + the signal (the C++ holds the whole song's STFT; output identical since frames are independent and added in the same order). Then crop, ÷1.5, `noise_gate` (in place, same float math: `powf`, `sqrtf`, later windows see zeroed samples), write. Errors out on inputs < 2048 samples (the C++ reads out of bounds there). |
| `tools/compare_wav.c` | `compare_wav ref.wav test.wav [min_snr]`: header match, SNR, max diff, bit-identical count, gate mismatches. Built by `make`. |
| `tools/make_test_clip.py` | Deterministic synthetic 10 s clip (chords, centred "voice", drums, silent gap). |
| `tests/test_wav.c` | Hand-written WAV files: pcm16 mono/stereo, write header patching, float round trip, rejects. |
| `tools/dump_acts.py` | Runs ORT with relaxed T (default 32) and the 13 taps as extra outputs; writes `tests/data/acts_T<T>/{input,<tap>}.bin` + `taps.txt` (57 MB at T=32, gitignored). |
| `tests/test_forward.c` | Runs `mdx_forward` on the dumped input; the tap callback compares each block's output to ORT as it's produced (< 1e-4 rel), printing elapsed time. T is read from the dump. |
| `Makefile` | `make` builds tests; `make test` creates `kara.bin` and `acts_T32/` if missing (needs Python for those two), then runs all three tests. Flags: `-O2 -std=c99 -Wall -Wextra -pedantic`. |

---

## Key facts about the model (details in PLAN.md §1–2)

- Tensor names from the ONNX graph used as test taps: `447` (after first conv),
  `466 487 508 529 550` (encoder blocks = skips), `571` (bottleneck),
  `593 615 637 659 681` (decoder blocks), `output`.
- Layout inside the net is `[C][T][F]` (F contiguous) between the two transposes.
- Skip connections are **multiplicative** (`x * skip`), not concat/add.
- Conv BN was folded by the PyTorch exporter; BN is explicit only after TDF
  MatMuls and after ConvTranspose.
- ConvTranspose weight is `[Cin][Cout][2][2]`; MatMul weight is `[F_in][F_out]`.
- ONNX input dim T is fixed at 256 in the graph, but the model is fully
  convolutional in T: relaxing `dim[3]` of input/output (and clearing
  `value_info`) lets ORT run any T that's a multiple of 32. `dump_acts.py` does this.

## Gotchas hit so far
- Don't size things by walking a NULL pointer (UB) — `map_weights` uses offsets.
- Manifest sums must be printed with `%.17g`; 9 digits is too coarse for the
  1e-12 relative tolerance in `test_load.c`.
- C++ Hann is `0.5f * (1.0f - std::cos(2.0f * M_PI * n / n_fft))`: `M_PI` is a
  double, so it's evaluated in **double** and rounded once. Match that, not a
  float `cosf` version.
- `/usr/bin/time` isn't installed here; time runs with `date +%s`.
- The C++ `build/separator` resolves `models/UVR_MDXNET_KARA_2.onnx` relative to
  the cwd — run it from the repo root. `plain_c/separator` defaults to
  `models/kara.bin` relative to `plain_c/`.
- The C++ framing attenuates the first/last ~1024 output samples (gain down to
  0.63, fewer overlapping windows than the interior, fixed ÷1.5). That is
  reference behaviour — reproduce it for parity, don't "fix" it yet (PLAN.md §5 quirks).

---

## Next up: M9 (performance)

Baseline: naive `mdx_forward` ≈ 335 s per 256-frame chunk (gcc `-O2`, 1 thread,
this 4-core box); ORT ≈ 4 s. FLOPs per chunk ≈ 472 G: conv3x3 81%, TDF matmuls
14%, down/up convs 5% (PLAN.md §9). So the naive code runs at ~1.4 GFLOP/s.

Approach, one measurable step at a time, each keeping `make test` green and
re-checking M7 parity (> 60 dB vs the C++ output) at the end:
1. Add a benchmark (`tests/bench_forward.c` or a flag on test_forward) that times
   each kernel type over one T=256 forward pass, so each step shows where time goes.
2. Keep the naive kernels as the readable reference (`mdx_conv3x3_ref` etc.);
   test_kernels checks fast vs reference on random cases, including edge sizes.
3. conv3x3: reorder so the innermost loop runs over contiguous W with the 3×3
   taps and input channels outside (accumulate whole output rows), then
   `-O3 -march=native` so gcc vectorises it. Then register/cache blocking over
   output channels, or im2col + a blocked SGEMM shared with the TDF matmuls.
4. matmul_lastdim: loop order r, f, j (contiguous W rows) → vectorises.
5. OpenMP `#pragma omp parallel for` over output channels / rows; must still
   build and pass without `-fopenmp`.
Report GFLOP/s per step in this file.
