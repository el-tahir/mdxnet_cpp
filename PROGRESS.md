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
| M2 | 9 kernels (naive loops) + unit tests vs numpy | next | |
| M3 | first_conv + transpose + enc0 TFC_TDF; taps `447`, `466` match | todo | |
| M4 | full encoder + bottleneck; taps through `571` match | todo | |
| M5 | decoder + final conv; `output` matches < 1e-4 rel | todo | |
| M6 | `fft.c`, `stft.c` + tests | todo | |
| M7 | `wav.c`, `main.c`: full C pipeline, SNR > 60 dB vs C++ `separator` | todo | |
| M8 | `plain_c/Makefile` `test` target complete; README section | todo | |
| M9 | performance | todo | |

### Verified results so far
- M0: reference vs ORT on seeded input `[1,4,2048,256]` (`np.random.default_rng(0)`, ×0.5):
  bottleneck rel err 9.4e-7, output 3.4e-6. Reference takes ~20 s, ORT ~2.6–4.4 s on this 4-core box.
- M1: `make test` → 220/220 tensors match the manifest by name; truncated / +1 float /
  header-only / empty files all rejected. Clean under `-fsanitize=address,undefined`.
  `kara.bin` = 52,764,496 bytes (64 header + 13,191,108 floats).

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
`plain_c/test_*` binaries.

---

## Code map (`plain_c/`)

| File | What it is |
|---|---|
| `tools/reference.py` | numpy forward pass — **the spec**. One function per C kernel, index formula in each docstring. `Weights.take()` consumes initializers in graph order. |
| `tools/export.py` | ONNX → `models/kara.bin` + `kara.bin.manifest`. `tensor_list()` defines names/shapes/order of every tensor from the header config; asserts every ONNX initializer matches its slot. |
| `mdx.h` | `MdxConfig` (header), `MdxConv`/`MdxBN`/`MdxBlock`/`MdxWeights` (pointers into one float buffer), `mdx_load`/`mdx_free`. |
| `mdx.c` | Loader: `map_weights()` walks the same order as `tensor_list()`; with `data == NULL` it only counts, used to require an exact file size before reading. |
| `tests/test_load.c` | Walks `MdxWeights` by field name, compares count / first / mid / last / sum to the manifest. |
| `Makefile` | `make` builds tests; `make test` builds `kara.bin` if missing and runs tests. Flags: `-O2 -std=c99 -Wall -Wextra -pedantic`. |

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
- ONNX input dim T is fixed at 256 in the graph; to test with smaller T in ORT
  you must relax the input dim (planned for `dump_acts.py`).

## Gotchas hit so far
- Don't size things by walking a NULL pointer (UB) — `map_weights` uses offsets.
- Manifest sums must be printed with `%.17g`; 9 digits is too coarse for the
  1e-12 relative tolerance in `test_load.c`.

---

## Next up: M2 (kernels)

Add to `mdx.c` (static or exposed via a test-only header) naive-loop kernels for
the 9 ops in PLAN.md §2: `conv1x1`, `conv3x3_p1`, `conv2x2_s2`, `convT2x2_s2`,
`matmul_lastdim`, `batchnorm`, `relu`, `add`/`mul`, `transpose_last2`.
Tensors are `[C][H][W]` row-major float arrays, batch 1.

Test plan: a Python tool generates small random inputs/weights per op (e.g.
C=3..8, H,W = 4..10, even sizes for the stride-2 ops), runs the matching
`reference.py` function, writes inputs + expected outputs as raw float32;
`tests/test_kernels.c` runs each C kernel and requires max rel err < 1e-5.
Wire into `make test`.
