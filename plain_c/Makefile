CC ?= cc
# OPT: speed flags. -fopenmp uses all cores (the code builds and runs without it);
# -march=native lets the compiler use this CPU's vector units. For a plain,
# portable, single-threaded build: make OPT=-O2
# -mprefer-vector-width=512 (x86 with AVX-512; added only if the compiler accepts
# it) makes gcc use full 512-bit vectors: conv3x3 ~110 -> ~175 GFLOP/s here.
WIDE := $(shell $(CC) -mprefer-vector-width=512 -E -x c /dev/null >/dev/null 2>&1 && echo -mprefer-vector-width=512)
OPT ?= -O3 -march=native $(WIDE) -fopenmp
CFLAGS ?= -std=c99 -Wall -Wextra -pedantic -Wno-unknown-pragmas
CFLAGS += $(OPT)
LDLIBS = -lm

ONNX = ../models/UVR_MDXNET_KARA_2.onnx
MODEL = models/kara.bin

TESTS = test_fft test_stft test_wav test_load test_kernels test_forward
ACTS = tests/data/acts_T32

SRC = main.c mdx.c kernels.c stft.c fft.c wav.c
HDR = mdx.h stft.h fft.h wav.h
MDX = mdx.c kernels.c mdx.h

all: separator compare_wav bench_forward $(TESTS)

separator: $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

# time one forward pass per kernel type: ./bench_forward [models/kara.bin] [T] [--ref]
bench_forward: tests/bench_forward.c $(MDX)
	$(CC) $(CFLAGS) -o $@ tests/bench_forward.c mdx.c kernels.c $(LDLIBS)

# compare two separator outputs: ./compare_wav reference.wav test.wav [min_snr_db]
compare_wav: tools/compare_wav.c wav.c wav.h
	$(CC) $(CFLAGS) -o $@ tools/compare_wav.c wav.c $(LDLIBS)

test_fft: tests/test_fft.c fft.c fft.h
	$(CC) $(CFLAGS) -o $@ tests/test_fft.c fft.c $(LDLIBS)

test_stft: tests/test_stft.c stft.c stft.h fft.c fft.h
	$(CC) $(CFLAGS) -o $@ tests/test_stft.c stft.c fft.c $(LDLIBS)

test_wav: tests/test_wav.c wav.c wav.h
	$(CC) $(CFLAGS) -o $@ tests/test_wav.c wav.c $(LDLIBS)

test_load: tests/test_load.c $(MDX)
	$(CC) $(CFLAGS) -o $@ tests/test_load.c mdx.c kernels.c $(LDLIBS)

# tests/data/kernels.bin is committed; regenerate with: python3 tools/gen_kernel_tests.py
test_kernels: tests/test_kernels.c $(MDX)
	$(CC) $(CFLAGS) -o $@ tests/test_kernels.c mdx.c kernels.c $(LDLIBS)

test_forward: tests/test_forward.c $(MDX)
	$(CC) $(CFLAGS) -o $@ tests/test_forward.c mdx.c kernels.c $(LDLIBS)

# ORT activations for test_forward (57 MB, gitignored); needs python3 + onnxruntime
$(ACTS)/taps.txt: $(ONNX) tools/dump_acts.py
	python3 tools/dump_acts.py $(ONNX) 32 $(ACTS)

# needs python3 with numpy + onnx; only run once
$(MODEL): $(ONNX) tools/export.py
	@mkdir -p models
	python3 tools/export.py $(ONNX) $(MODEL)

test: $(TESTS) $(MODEL) $(ACTS)/taps.txt
	./test_fft
	./test_stft
	./test_wav
	./test_load $(MODEL)
	./test_kernels tests/data/kernels.bin
	./test_forward $(MODEL) $(ACTS)

clean:
	rm -f separator compare_wav bench_forward $(TESTS)

.PHONY: all test clean
