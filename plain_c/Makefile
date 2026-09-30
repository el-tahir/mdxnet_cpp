CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -pedantic
LDLIBS = -lm

ONNX = ../models/UVR_MDXNET_KARA_2.onnx
MODEL = models/kara.bin

TESTS = test_load

all: $(TESTS)

test_load: tests/test_load.c mdx.c mdx.h
	$(CC) $(CFLAGS) -o $@ tests/test_load.c mdx.c $(LDLIBS)

# needs python3 with numpy + onnx; only run once
$(MODEL): $(ONNX) tools/export.py
	@mkdir -p models
	python3 tools/export.py $(ONNX) $(MODEL)

test: $(TESTS) $(MODEL)
	./test_load $(MODEL)

clean:
	rm -f $(TESTS)

.PHONY: all test clean
