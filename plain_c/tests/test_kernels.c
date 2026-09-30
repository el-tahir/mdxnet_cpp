/* Every kernel against the numpy reference on small random cases, both the
 * naive *_ref kernels (mdx.c) and the fast ones (kernels.c).
 *
 * Cases come from tests/data/kernels.bin (tools/gen_kernel_tests.py): inputs and
 * expected output per case. Pass if max|c - ref| / max|ref| < 1e-5.
 *
 *     ./test_kernels [tests/data/kernels.bin]
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../mdx.h"

#define MAX_ARRAYS 8
#define TOL 1e-5

typedef struct {
    char op[17];
    int32_t d[8];
    int n;
    float *a[MAX_ARRAYS];
    int32_t count[MAX_ARRAYS];
} Case;

static int read_case(FILE *f, Case *c) {
    memset(c, 0, sizeof(*c));
    if (fread(c->op, 1, 16, f) != 16) return 0; /* clean EOF */
    int32_t n;
    if (fread(c->d, sizeof(int32_t), 8, f) != 8 || fread(&n, sizeof(n), 1, f) != 1 || n < 1 || n > MAX_ARRAYS) {
        fprintf(stderr, "corrupt case header\n");
        exit(1);
    }
    c->n = n;
    for (int i = 0; i < n; i++) {
        if (fread(&c->count[i], sizeof(int32_t), 1, f) != 1 || c->count[i] < 0) exit(1);
        c->a[i] = malloc((size_t)c->count[i] * sizeof(float) + 1);
        if (fread(c->a[i], sizeof(float), c->count[i], f) != (size_t)c->count[i]) {
            fprintf(stderr, "corrupt case data\n");
            exit(1);
        }
    }
    return 1;
}

static void free_case(Case *c) {
    for (int i = 0; i < c->n; i++) free(c->a[i]);
}

/* compares got against the case's last array (the expected output) */
static int compare(const Case *c, const float *got, long n, double *rel_out) {
    const float *want = c->a[c->n - 1];
    if (n != c->count[c->n - 1]) {
        printf("  output size %ld, expected %d\n", n, c->count[c->n - 1]);
        return 0;
    }
    double maxdiff = 0, maxref = 0;
    for (long i = 0; i < n; i++) {
        double d = fabs((double)got[i] - want[i]);
        if (d > maxdiff) maxdiff = d;
        if (fabs(want[i]) > maxref) maxref = fabs(want[i]);
    }
    *rel_out = maxref > 0 ? maxdiff / maxref : maxdiff;
    return *rel_out < TOL;
}

/* runs the case's op into y; fast selects kernels.c over the *_ref kernels.
 * Returns 0 for an unknown op. */
static int run(const Case *c, float *y, int fast) {
    const int32_t *d = c->d;
    float *const *a = c->a;
    long out_n = c->count[c->n - 1];
    if (!strcmp(c->op, "conv1x1")) {
        (fast ? mdx_conv1x1 : mdx_conv1x1_ref)(y, a[0], a[1], a[2], d[0], d[1], d[2], d[3]);
    } else if (!strcmp(c->op, "conv3x3")) {
        (fast ? mdx_conv3x3 : mdx_conv3x3_ref)(y, a[0], a[1], a[2], d[0], d[1], d[2], d[3]);
    } else if (!strcmp(c->op, "conv2x2_s2")) {
        (fast ? mdx_conv2x2_s2 : mdx_conv2x2_s2_ref)(y, a[0], a[1], a[2], d[0], d[1], d[2], d[3]);
    } else if (!strcmp(c->op, "convT2x2_s2")) {
        (fast ? mdx_convT2x2_s2 : mdx_convT2x2_s2_ref)(y, a[0], a[1], a[2], d[0], d[1], d[2], d[3]);
    } else if (!strcmp(c->op, "matmul_lastdim")) {
        (fast ? mdx_matmul_lastdim : mdx_matmul_lastdim_ref)(y, a[0], a[1], d[0], d[1], d[2]);
    } else if (!strcmp(c->op, "batchnorm")) { /* single version: in place */
        MdxBN bn = {a[1], a[2], a[3], a[4]};
        memcpy(y, a[0], (size_t)out_n * sizeof(float));
        mdx_batchnorm(y, &bn, 1e-5f, d[0], d[1] * d[2]);
    } else if (!strcmp(c->op, "relu")) {
        memcpy(y, a[0], (size_t)out_n * sizeof(float));
        mdx_relu(y, d[0]);
    } else if (!strcmp(c->op, "add")) {
        memcpy(y, a[0], (size_t)out_n * sizeof(float));
        mdx_add(y, a[1], d[0]);
    } else if (!strcmp(c->op, "mul")) {
        memcpy(y, a[0], (size_t)out_n * sizeof(float));
        mdx_mul(y, a[1], d[0]);
    } else if (!strcmp(c->op, "transpose_last2")) {
        mdx_transpose_last2(y, a[0], d[0], d[1], d[2]);
    } else {
        return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "tests/data/kernels.bin";
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }

    int n_cases = 0, n_failed = 0;
    Case c;
    while (read_case(f, &c)) {
        long out_n = c.count[c.n - 1];
        float *y = malloc((size_t)out_n * sizeof(float) + 1);
        for (int fast = 0; fast < 2; fast++) {
            int known = run(&c, y, fast);
            n_cases++;
            double rel = 0;
            int ok = known && compare(&c, y, out_n, &rel);
            if (!ok) n_failed++;
            printf("%-4s %-4s %-16s dims [%d %d %d %d]  rel err %.2e%s\n", ok ? "ok" : "FAIL", fast ? "fast" : "ref",
                   c.op, c.d[0], c.d[1], c.d[2], c.d[3], rel, known ? "" : "  (unknown op)");
        }
        free(y);
        free_case(&c);
    }
    fclose(f);

    printf("%d cases, %d failed\n%s\n", n_cases, n_failed, n_failed || !n_cases ? "FAILED" : "OK");
    return n_failed || !n_cases ? 1 : 0;
}
