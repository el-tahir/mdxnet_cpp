CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -pedantic
LDLIBS = -lm

ONNX = ../models/UVR_MDXNET_KARA_2.onnx
MODEL = models/kara.bin

TESTS = test_load test_kernels test_forward
ACTS = tests/data/acts_T32

all: $(TESTS)

test_load: tests/test_load.c mdx.c mdx.h
	$(CC) $(CFLAGS) -o $@ tests/test_load.c mdx.c $(LDLIBS)

# tests/data/kernels.bin is committed; regenerate with: python3 tools/gen_kernel_tests.py
test_kernels: tests/test_kernels.c mdx.c mdx.h
	$(CC) $(CFLAGS) -o $@ tests/test_kernels.c mdx.c $(LDLIBS)

test_forward: tests/test_forward.c mdx.c mdx.h
	$(CC) $(CFLAGS) -o $@ tests/test_forward.c mdx.c $(LDLIBS)

# ORT activations for test_forward (57 MB, gitignored); needs python3 + onnxruntime
$(ACTS)/taps.txt: $(ONNX) tools/dump_acts.py
	python3 tools/dump_acts.py $(ONNX) 32 $(ACTS)

# needs python3 with numpy + onnx; only run once
$(MODEL): $(ONNX) tools/export.py
	@mkdir -p models
	python3 tools/export.py $(ONNX) $(MODEL)

test: $(TESTS) $(MODEL) $(ACTS)/taps.txt
	./test_load $(MODEL)
	./test_kernels tests/data/kernels.bin
	./test_forward $(MODEL) $(ACTS)

clean:
	rm -f $(TESTS)

.PHONY: all test clean
