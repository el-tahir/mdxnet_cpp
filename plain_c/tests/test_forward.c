/* M3-M5 check: mdx_forward against ONNX Runtime, block by block.
 *
 * tools/dump_acts.py saves ORT's input and the output of every block ("taps")
 * for a fixed input. This test runs mdx_forward on the same input and, through
 * the tap callback, compares each block's output as soon as it is computed.
 * Pass if max|c - ort| / max|ort| < 1e-4 for every tap.
 *
 *     ./test_forward models/kara.bin tests/data/acts_T32
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../mdx.h"

#define TOL 1e-4
#define MAX_TAPS 32

typedef struct {
    const char *dir;
    char names[MAX_TAPS][16];
    int dims[MAX_TAPS][3];
    int n_taps, n_seen, n_failed;
    clock_t start;
} Ctx;

static float *read_floats(const char *path, long n) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }
    float *p = malloc((size_t)n * sizeof(float));
    long got = p ? (long)fread(p, sizeof(float), (size_t)n, f) : 0;
    int extra = fgetc(f) != EOF;
    fclose(f);
    if (got != n || extra) {
        fprintf(stderr, "%s: expected exactly %ld floats\n", path, n);
        free(p);
        return NULL;
    }
    return p;
}

static void on_tap(void *vctx, const char *name, const float *t, int c, int h, int w) {
    Ctx *ctx = vctx;
    double secs = (double)(clock() - ctx->start) / CLOCKS_PER_SEC;
    int k = 0;
    while (k < ctx->n_taps && strcmp(ctx->names[k], name)) k++;
    if (k == ctx->n_taps) {
        printf("FAIL %-6s no reference dump for this tap\n", name);
        ctx->n_failed++;
        return;
    }
    ctx->n_seen++;
    if (ctx->dims[k][0] != c || ctx->dims[k][1] != h || ctx->dims[k][2] != w) {
        printf("FAIL %-6s shape [%d %d %d], ORT [%d %d %d]\n", name, c, h, w, ctx->dims[k][0], ctx->dims[k][1],
               ctx->dims[k][2]);
        ctx->n_failed++;
        return;
    }
    char path[1024];
    long n = (long)c * h * w;
    snprintf(path, sizeof(path), "%s/%s.bin", ctx->dir, name);
    float *want = read_floats(path, n);
    if (!want) {
        ctx->n_failed++;
        return;
    }
    double maxdiff = 0, maxref = 0;
    long worst = 0;
    for (long i = 0; i < n; i++) {
        double d = fabs((double)t[i] - want[i]);
        if (d > maxdiff) maxdiff = d, worst = i;
        if (fabs(want[i]) > maxref) maxref = fabs(want[i]);
    }
    double rel = maxref > 0 ? maxdiff / maxref : maxdiff;
    int ok = rel < TOL;
    if (!ok) ctx->n_failed++;
    printf("%-4s %-6s [%3d %4d %4d]  rel err %.2e  (max|ort| %.3g)  t=%.1fs", ok ? "ok" : "FAIL", name, c, h, w, rel,
           maxref, secs);
    if (!ok) printf("  worst at %ld: c %.6g vs ort %.6g", worst, t[worst], want[worst]);
    printf("\n");
    fflush(stdout);
    free(want);
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "models/kara.bin";
    Ctx ctx = {0};
    ctx.dir = argc > 2 ? argv[2] : "tests/data/acts_T32";

    char path[1024];
    snprintf(path, sizeof(path), "%s/taps.txt", ctx.dir);
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot open %s (run tools/dump_acts.py)\n", path);
        return 1;
    }
    char onnx_name[64];
    while (ctx.n_taps < MAX_TAPS && fscanf(f, "%15s %63s %d %d %d", ctx.names[ctx.n_taps], onnx_name,
                                           &ctx.dims[ctx.n_taps][0], &ctx.dims[ctx.n_taps][1],
                                           &ctx.dims[ctx.n_taps][2]) == 5)
        ctx.n_taps++;
    fclose(f);

    MdxModel m;
    if (mdx_load(&m, model_path) != 0) return 1;
    const MdxConfig *cfg = &m.config;

    /* T from the "output" tap: [dim_c][F][T] */
    int T = 0;
    for (int k = 0; k < ctx.n_taps; k++)
        if (!strcmp(ctx.names[k], "output")) T = ctx.dims[k][2];

    MdxState s;
    if (mdx_state_init(&s, cfg, T) != 0) return 1;
    s.tap = on_tap;
    s.tap_ctx = &ctx;

    long n_io = (long)cfg->dim_c * cfg->dim_f * T;
    snprintf(path, sizeof(path), "%s/input.bin", ctx.dir);
    float *in = read_floats(path, n_io);
    float *out = malloc((size_t)n_io * sizeof(float));
    if (!in || !out) return 1;

    printf("T=%d, %d taps from %s\n", T, ctx.n_taps, ctx.dir);
    ctx.start = clock();
    mdx_forward(&m, &s, in, out);

    if (ctx.n_seen != ctx.n_taps) {
        printf("FAIL %d of %d reference taps were never produced\n", ctx.n_taps - ctx.n_seen, ctx.n_taps);
        ctx.n_failed++;
    }
    printf("%d taps checked, %d failed\n%s\n", ctx.n_seen, ctx.n_failed, ctx.n_failed ? "FAILED" : "OK");

    free(in);
    free(out);
    mdx_state_free(&s);
    mdx_free(&m);
    return ctx.n_failed ? 1 : 0;
}
