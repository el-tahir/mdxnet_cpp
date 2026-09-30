/* M1 check: the C loader maps every tensor to the right place.
 *
 * tools/export.py writes <bin>.manifest with, per tensor in file order:
 *     name  count  first  middle  last  sum
 * computed by numpy straight from the ONNX initializers. This test walks the
 * MdxWeights struct *by field* (not by file offset), recomputes the same
 * numbers from the pointers the loader produced, and compares them to the
 * manifest line of the same name. It also checks that corrupted files are
 * rejected.
 *
 *     ./test_load models/kara.bin
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../mdx.h"

static FILE *manifest;
static int n_checked, n_failed;

static void check(const char *name, const float *t, unsigned long long n) {
    char mname[128];
    unsigned long long mcount;
    double first, mid, last, sum;
    if (fscanf(manifest, "%127s %llu %lf %lf %lf %lf", mname, &mcount, &first, &mid, &last, &sum) != 6) {
        printf("FAIL %s: manifest ended early\n", name);
        n_failed++;
        return;
    }
    n_checked++;
    if (strcmp(name, mname) != 0) {
        printf("FAIL order: C expects %s, manifest has %s\n", name, mname);
        n_failed++;
        return;
    }
    if (t == NULL || n != mcount) {
        printf("FAIL %s: count %llu, manifest %llu\n", name, n, mcount);
        n_failed++;
        return;
    }
    double s = 0, abs_s = 0;
    for (unsigned long long i = 0; i < n; i++) {
        s += t[i];
        abs_s += fabs(t[i]);
    }
    /* first/mid/last are printed with 9 significant digits, which round-trips float32 exactly */
    int ok = t[0] == (float)first && t[n / 2] == (float)mid && t[n - 1] == (float)last &&
             fabs(s - sum) <= 1e-12 * abs_s;
    if (!ok) {
        printf("FAIL %s: C (%.9g %.9g %.9g sum %.17g) vs manifest (%.9g %.9g %.9g sum %.17g)\n", name, t[0],
               t[n / 2], t[n - 1], s, first, mid, last, sum);
        n_failed++;
    }
}

static void check_conv(const char *name, const MdxConv *c, unsigned long long cout, unsigned long long cin,
                       unsigned long long k) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s.w", name);
    check(buf, c->w, cout * cin * k * k);
    snprintf(buf, sizeof(buf), "%s.b", name);
    check(buf, c->b, cout);
}

static void check_bn(const char *name, const MdxBN *bn, unsigned long long ch) {
    char buf[128];
    const float *p[4] = {bn->scale, bn->bias, bn->mean, bn->var};
    const char *s[4] = {"scale", "bias", "mean", "var"};
    for (int i = 0; i < 4; i++) {
        snprintf(buf, sizeof(buf), "%s.%s", name, s[i]);
        check(buf, p[i], ch);
    }
}

static void check_block(const char *name, const MdxBlock *b, const MdxConfig *cfg, unsigned long long ch,
                        unsigned long long f) {
    char buf[128];
    for (uint32_t i = 0; i < cfg->n_tfc; i++) {
        snprintf(buf, sizeof(buf), "%s.tfc%u", name, i);
        check_conv(buf, &b->tfc[i], ch, ch, 3);
    }
    snprintf(buf, sizeof(buf), "%s.tdf1.w", name);
    check(buf, b->tdf1, f * (f / cfg->bn_factor));
    snprintf(buf, sizeof(buf), "%s.tdf1.bn", name);
    check_bn(buf, &b->tdf1_bn, ch);
    snprintf(buf, sizeof(buf), "%s.tdf2.w", name);
    check(buf, b->tdf2, (f / cfg->bn_factor) * f);
    snprintf(buf, sizeof(buf), "%s.tdf2.bn", name);
    check_bn(buf, &b->tdf2_bn, ch);
}

/* Write the first `keep` bytes of src to dst, then append `extra` zero bytes. */
static int write_variant(const char *src, const char *dst, long keep, long extra) {
    FILE *in = fopen(src, "rb"), *out = fopen(dst, "wb");
    if (!in || !out) return -1;
    char buf[1 << 16];
    long left = keep;
    while (left > 0) {
        size_t n = fread(buf, 1, left < (long)sizeof(buf) ? (size_t)left : sizeof(buf), in);
        if (n == 0) break;
        fwrite(buf, 1, n, out);
        left -= (long)n;
    }
    for (long i = 0; i < extra; i++) fputc(0, out);
    fclose(in);
    fclose(out);
    return 0;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "models/kara.bin";
    char mpath[1024];
    snprintf(mpath, sizeof(mpath), "%s.manifest", path);

    MdxModel m;
    if (mdx_load(&m, path) != 0) return 1;
    const MdxConfig *cfg = &m.config;
    printf("config: dim_c=%u dim_f=%u dim_t=%u n_scales=%u growth=%u n_tfc=%u bn_factor=%u bn_eps=%g\n",
           cfg->dim_c, cfg->dim_f, cfg->dim_t, cfg->n_scales, cfg->growth, cfg->n_tfc, cfg->bn_factor,
           cfg->bn_eps);
    printf("weights: %llu floats (%.1f MB)\n", (unsigned long long)m.n_floats, m.n_floats * 4 / 1e6);

    manifest = fopen(mpath, "r");
    if (!manifest) {
        fprintf(stderr, "cannot open %s (run tools/export.py)\n", mpath);
        return 1;
    }

    const MdxWeights *w = &m.weights;
    char buf[64];
    unsigned long long ch = cfg->growth, f = cfg->dim_f;
    check_conv("first", &w->first, cfg->growth, cfg->dim_c, 1);
    for (uint32_t i = 0; i < cfg->n_scales; i++) {
        snprintf(buf, sizeof(buf), "enc%u", i);
        check_block(buf, &w->enc[i], cfg, ch, f);
        snprintf(buf, sizeof(buf), "enc%u.down", i);
        check_conv(buf, &w->down[i], ch + cfg->growth, ch, 2);
        ch += cfg->growth;
        f /= 2;
    }
    check_block("mid", &w->mid, cfg, ch, f);
    for (uint32_t i = 0; i < cfg->n_scales; i++) {
        snprintf(buf, sizeof(buf), "dec%u.up", i);
        check_conv(buf, &w->up[i], ch - cfg->growth, ch, 2);
        snprintf(buf, sizeof(buf), "dec%u.up.bn", i);
        check_bn(buf, &w->up_bn[i], ch - cfg->growth);
        ch -= cfg->growth;
        f *= 2;
        snprintf(buf, sizeof(buf), "dec%u", i);
        check_block(buf, &w->dec[i], cfg, ch, f);
    }
    check_conv("final", &w->final, cfg->dim_c, cfg->growth, 1);

    char extra[128];
    if (fscanf(manifest, "%127s", extra) == 1) {
        printf("FAIL manifest has more tensors than the C struct (next: %s)\n", extra);
        n_failed++;
    }
    fclose(manifest);
    printf("tensors: %d checked against manifest, %d failed\n", n_checked, n_failed);
    long file_bytes = MDX_HEADER_SIZE + (long)m.n_floats * 4;
    mdx_free(&m);

    /* corrupted files must be rejected */
    const char *tmp = "models/_test_load_tmp.bin";
    struct { const char *what; long keep, extra; } bad[] = {
        {"truncated by one float", file_bytes - 4, 0},
        {"one extra float", file_bytes, 4},
        {"header only", MDX_HEADER_SIZE, 0},
        {"empty", 0, 0},
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        if (write_variant(path, tmp, bad[i].keep, bad[i].extra) != 0) {
            printf("FAIL cannot write %s\n", tmp);
            n_failed++;
            continue;
        }
        MdxModel bm;
        fprintf(stderr, "  (expected error follows) ");
        int rc = mdx_load(&bm, tmp);
        if (rc == 0) {
            printf("FAIL %s file was accepted\n", bad[i].what);
            mdx_free(&bm);
            n_failed++;
        } else {
            printf("rejected: %s\n", bad[i].what);
        }
    }
    remove(tmp);

    printf(n_failed ? "FAILED\n" : "OK\n");
    return n_failed ? 1 : 0;
}
