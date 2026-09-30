#include "mdx.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- */
/* weight loading
 *
 * The file is a 64-byte header followed by raw float32 tensors. There are no
 * names or shapes in the file: the header's hyperparameters fully determine the
 * size and order of every tensor, and map_weights() walks that order, handing
 * out consecutive slices. The same order is written by tools/export.py
 * (tensor_list) and consumed by the forward pass.
 */

typedef struct {
    const float *base; /* NULL when only counting */
    uint64_t off;      /* next unread float */
    uint64_t n;        /* floats available */
    int overrun;
} Cursor;

static const float *take(Cursor *c, uint64_t n) {
    if (c->overrun || c->n - c->off < n) {
        c->overrun = 1;
        return NULL;
    }
    const float *t = c->base ? c->base + c->off : NULL;
    c->off += n;
    return t;
}

static void take_conv(Cursor *c, MdxConv *conv, uint64_t cout, uint64_t cin, uint64_t k) {
    conv->w = take(c, cout * cin * k * k);
    conv->b = take(c, cout);
}

static void take_bn(Cursor *c, MdxBN *bn, uint64_t ch) {
    bn->scale = take(c, ch);
    bn->bias = take(c, ch);
    bn->mean = take(c, ch);
    bn->var = take(c, ch);
}

static void take_block(Cursor *c, MdxBlock *b, const MdxConfig *cfg, uint64_t ch, uint64_t f) {
    for (uint32_t i = 0; i < cfg->n_tfc; i++) take_conv(c, &b->tfc[i], ch, ch, 3);
    b->tdf1 = take(c, f * (f / cfg->bn_factor));
    take_bn(c, &b->tdf1_bn, ch);
    b->tdf2 = take(c, (f / cfg->bn_factor) * f);
    take_bn(c, &b->tdf2_bn, ch);
}

/* Walk the tensors in file order. Returns the number of floats consumed.
 * With data == NULL it only counts (all pointers in w are left NULL). */
static uint64_t map_weights(MdxWeights *w, const MdxConfig *cfg, const float *data, uint64_t n_floats, int *overrun) {
    Cursor c = {data, 0, n_floats, 0};
    uint64_t ch = cfg->growth, f = cfg->dim_f;

    take_conv(&c, &w->first, cfg->growth, cfg->dim_c, 1);
    for (uint32_t i = 0; i < cfg->n_scales; i++) {
        take_block(&c, &w->enc[i], cfg, ch, f);
        take_conv(&c, &w->down[i], ch + cfg->growth, ch, 2);
        ch += cfg->growth;
        f /= 2;
    }
    take_block(&c, &w->mid, cfg, ch, f);
    for (uint32_t i = 0; i < cfg->n_scales; i++) {
        /* ConvTranspose weight is [Cin][Cout][2][2]; same element count as [Cout][Cin] */
        take_conv(&c, &w->up[i], ch - cfg->growth, ch, 2);
        take_bn(&c, &w->up_bn[i], ch - cfg->growth);
        ch -= cfg->growth;
        f *= 2;
        take_block(&c, &w->dec[i], cfg, ch, f);
    }
    take_conv(&c, &w->final, cfg->dim_c, cfg->growth, 1);

    *overrun = c.overrun;
    return c.off;
}

static int check_config(const MdxConfig *cfg) {
    uint32_t div = 1u << cfg->n_scales; /* T and F are halved n_scales times */
    if (cfg->n_scales == 0 || cfg->n_scales > MDX_MAX_SCALES) return 0;
    if (cfg->n_tfc == 0 || cfg->n_tfc > MDX_MAX_TFC) return 0;
    if (cfg->dim_c == 0 || cfg->growth == 0 || cfg->bn_factor == 0) return 0;
    if (cfg->dim_f == 0 || cfg->dim_f % div || cfg->dim_t == 0 || cfg->dim_t % div) return 0;
    if ((cfg->dim_f >> cfg->n_scales) % cfg->bn_factor) return 0; /* F/bn_factor exact at every level */
    return 1;
}

int mdx_load(MdxModel *m, const char *path) {
    memset(m, 0, sizeof(*m));

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "mdx_load: cannot open %s\n", path);
        return -1;
    }

    unsigned char hdr[MDX_HEADER_SIZE];
    uint32_t u[9];
    if (fread(hdr, 1, sizeof(hdr), fp) != sizeof(hdr)) {
        fprintf(stderr, "mdx_load: %s: file shorter than header\n", path);
        fclose(fp);
        return -1;
    }
    memcpy(u, hdr, sizeof(u)); /* file is little-endian; so is every target we build for */
    if (u[0] != MDX_MAGIC || u[1] != MDX_VERSION) {
        fprintf(stderr, "mdx_load: %s: bad magic/version (%08x v%u)\n", path, u[0], u[1]);
        fclose(fp);
        return -1;
    }
    MdxConfig *cfg = &m->config;
    cfg->dim_c = u[2];
    cfg->dim_f = u[3];
    cfg->dim_t = u[4];
    cfg->n_scales = u[5];
    cfg->growth = u[6];
    cfg->n_tfc = u[7];
    cfg->bn_factor = u[8];
    memcpy(&cfg->bn_eps, hdr + sizeof(u), sizeof(float));
    if (!check_config(cfg)) {
        fprintf(stderr, "mdx_load: %s: invalid hyperparameters in header\n", path);
        fclose(fp);
        return -1;
    }

    /* size the weight block from the header alone, then require the file to match exactly */
    int overrun;
    MdxWeights counting;
    uint64_t want = map_weights(&counting, cfg, NULL, UINT64_MAX, &overrun);

    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return -1;
    }
    long file_size = ftell(fp);
    uint64_t want_bytes = MDX_HEADER_SIZE + want * sizeof(float);
    if (file_size < 0 || (uint64_t)file_size != want_bytes) {
        fprintf(stderr, "mdx_load: %s: size %ld bytes, header implies %llu\n", path, file_size,
                (unsigned long long)want_bytes);
        fclose(fp);
        return -1;
    }
    fseek(fp, MDX_HEADER_SIZE, SEEK_SET);

    m->data = malloc(want * sizeof(float));
    if (!m->data) {
        fprintf(stderr, "mdx_load: out of memory (%llu floats)\n", (unsigned long long)want);
        fclose(fp);
        return -1;
    }
    if (fread(m->data, sizeof(float), want, fp) != want) {
        fprintf(stderr, "mdx_load: %s: short read\n", path);
        fclose(fp);
        mdx_free(m);
        return -1;
    }
    fclose(fp);

    m->n_floats = want;
    uint64_t used = map_weights(&m->weights, cfg, m->data, want, &overrun);
    if (overrun || used != want) { /* cannot happen after the size check; belt and braces */
        fprintf(stderr, "mdx_load: internal error mapping weights\n");
        mdx_free(m);
        return -1;
    }
    return 0;
}

void mdx_free(MdxModel *m) {
    free(m->data);
    memset(m, 0, sizeof(*m));
}

/* ------------------------------------------------------------------------- */
/* kernels
 *
 * Deliberately naive: the loops follow the formulas in mdx.h one to one, output
 * element by output element. Speed comes later (PLAN.md M9) without changing
 * these reference versions.
 */

void mdx_conv1x1(float *y, const float *x, const float *W, const float *b, int cin, int cout, int H, int Wd) {
    long hw = (long)H * Wd;
    for (int o = 0; o < cout; o++) {
        for (long p = 0; p < hw; p++) {
            float acc = b[o];
            for (int i = 0; i < cin; i++) acc += W[(long)o * cin + i] * x[i * hw + p];
            y[o * hw + p] = acc;
        }
    }
}

void mdx_conv3x3(float *y, const float *x, const float *W, const float *b, int cin, int cout, int H, int Wd) {
    long hw = (long)H * Wd;
    for (int o = 0; o < cout; o++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < Wd; w++) {
                float acc = b[o];
                for (int i = 0; i < cin; i++) {
                    const float *wk = W + ((long)o * cin + i) * 9; /* W[o][i][.][.] */
                    const float *xi = x + i * hw;
                    for (int ky = 0; ky < 3; ky++) {
                        int hh = h + ky - 1;
                        if (hh < 0 || hh >= H) continue; /* zero padding */
                        for (int kx = 0; kx < 3; kx++) {
                            int ww = w + kx - 1;
                            if (ww < 0 || ww >= Wd) continue;
                            acc += wk[ky * 3 + kx] * xi[(long)hh * Wd + ww];
                        }
                    }
                }
                y[o * hw + (long)h * Wd + w] = acc;
            }
        }
    }
}

void mdx_conv2x2_s2(float *y, const float *x, const float *W, const float *b, int cin, int cout, int H, int Wd) {
    int Ho = H / 2, Wo = Wd / 2;
    long hw = (long)H * Wd;
    for (int o = 0; o < cout; o++) {
        for (int h = 0; h < Ho; h++) {
            for (int w = 0; w < Wo; w++) {
                float acc = b[o];
                for (int i = 0; i < cin; i++) {
                    const float *wk = W + ((long)o * cin + i) * 4; /* W[o][i][.][.] */
                    const float *xi = x + i * hw;
                    for (int ky = 0; ky < 2; ky++)
                        for (int kx = 0; kx < 2; kx++)
                            acc += wk[ky * 2 + kx] * xi[(long)(2 * h + ky) * Wd + (2 * w + kx)];
                }
                y[((long)o * Ho + h) * Wo + w] = acc;
            }
        }
    }
}

void mdx_convT2x2_s2(float *y, const float *x, const float *W, const float *b, int cin, int cout, int H, int Wd) {
    /* stride == kernel size, so every output pixel (2h+ky, 2w+kx) receives exactly
     * one kernel tap (ky,kx) from exactly one input pixel (h,w) per input channel */
    int Ho = 2 * H, Wo = 2 * Wd;
    long hw = (long)H * Wd;
    for (int o = 0; o < cout; o++) {
        for (int yy = 0; yy < Ho; yy++) {
            for (int xx = 0; xx < Wo; xx++) {
                int h = yy / 2, ky = yy % 2, w = xx / 2, kx = xx % 2;
                float acc = b[o];
                for (int i = 0; i < cin; i++)
                    acc += W[(((long)i * cout + o) * 2 + ky) * 2 + kx] * x[i * hw + (long)h * Wd + w]; /* W[i][o][ky][kx] */
                y[((long)o * Ho + yy) * Wo + xx] = acc;
            }
        }
    }
}

void mdx_matmul_lastdim(float *y, const float *x, const float *W, int rows, int fin, int fout) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (long)r * fin;
        for (int j = 0; j < fout; j++) {
            float acc = 0.0f;
            for (int f = 0; f < fin; f++) acc += xr[f] * W[(long)f * fout + j];
            y[(long)r * fout + j] = acc;
        }
    }
}

void mdx_batchnorm(float *x, const MdxBN *bn, float eps, int C, int hw) {
    for (int c = 0; c < C; c++) {
        float inv_std = 1.0f / sqrtf(bn->var[c] + eps);
        float *xc = x + (long)c * hw;
        for (int i = 0; i < hw; i++) xc[i] = (xc[i] - bn->mean[c]) * inv_std * bn->scale[c] + bn->bias[c];
    }
}

void mdx_relu(float *x, long n) {
    for (long i = 0; i < n; i++) x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

void mdx_add(float *y, const float *a, long n) {
    for (long i = 0; i < n; i++) y[i] += a[i];
}

void mdx_mul(float *y, const float *a, long n) {
    for (long i = 0; i < n; i++) y[i] *= a[i];
}

void mdx_transpose_last2(float *y, const float *x, int C, int H, int Wd) {
    for (int c = 0; c < C; c++) {
        const float *xc = x + (long)c * H * Wd;
        float *yc = y + (long)c * H * Wd;
        for (int i = 0; i < H; i++)
            for (int j = 0; j < Wd; j++) yc[(long)j * H + i] = xc[(long)i * Wd + j];
    }
}

/* ------------------------------------------------------------------------- */
/* forward pass */

int mdx_state_init(MdxState *s, const MdxConfig *cfg, int T) {
    memset(s, 0, sizeof(*s));
    if (T <= 0 || T % (1 << cfg->n_scales)) {
        fprintf(stderr, "mdx_state_init: T=%d must be a positive multiple of %d\n", T, 1 << cfg->n_scales);
        return -1;
    }
    s->T = T;
    size_t level0 = (size_t)cfg->growth * T * cfg->dim_f;
    s->a = malloc(level0 * sizeof(float));
    s->b = malloc(level0 * sizeof(float));
    s->h = malloc(level0 / cfg->bn_factor * sizeof(float));
    int ok = s->a && s->b && s->h;
    for (uint32_t i = 0; i < cfg->n_scales; i++) {
        size_t n = (size_t)cfg->growth * (i + 1) * (T >> i) * (cfg->dim_f >> i);
        s->skip[i] = malloc(n * sizeof(float));
        ok = ok && s->skip[i];
    }
    if (!ok) {
        fprintf(stderr, "mdx_state_init: out of memory\n");
        mdx_state_free(s);
        return -1;
    }
    return 0;
}

void mdx_state_free(MdxState *s) {
    free(s->a);
    free(s->b);
    free(s->h);
    for (int i = 0; i < MDX_MAX_SCALES; i++) free(s->skip[i]);
    memset(s, 0, sizeof(*s));
}

static void tap(MdxState *s, const char *name, const float *t, int c, int h, int w) {
    if (s->tap) s->tap(s->tap_ctx, name, t, c, h, w);
}

/* TFC-TDF block on x [C][T][F]. Uses x and tmp as ping-pong buffers and h for
 * the TDF hidden layer. Returns whichever of x / tmp holds the output; the
 * other one is free afterwards.
 *
 *   repeat n_tfc:  x = relu(conv3x3(x))
 *   h   = relu(BN(x @ tdf1))          [C][T][F/bn]
 *   out = relu(BN(h @ tdf2)) + x      [C][T][F]
 */
static float *tfc_tdf(const MdxBlock *blk, const MdxConfig *cfg, float *x, float *tmp, float *h, int C, int T,
                      int F) {
    long n = (long)C * T * F;
    int Fh = F / (int)cfg->bn_factor;
    float *cur = x, *nxt = tmp, *t;

    for (uint32_t i = 0; i < cfg->n_tfc; i++) {
        mdx_conv3x3(nxt, cur, blk->tfc[i].w, blk->tfc[i].b, C, C, T, F);
        mdx_relu(nxt, n);
        t = cur, cur = nxt, nxt = t;
    }
    /* cur = TFC output; nxt is free */
    mdx_matmul_lastdim(h, cur, blk->tdf1, C * T, F, Fh);
    mdx_batchnorm(h, &blk->tdf1_bn, cfg->bn_eps, C, T * Fh);
    mdx_relu(h, (long)C * T * Fh);

    mdx_matmul_lastdim(nxt, h, blk->tdf2, C * T, Fh, F);
    mdx_batchnorm(nxt, &blk->tdf2_bn, cfg->bn_eps, C, T * F);
    mdx_relu(nxt, n);

    mdx_add(nxt, cur, n); /* residual */
    return nxt;
}

void mdx_forward(const MdxModel *m, MdxState *s, const float *in, float *out) {
    const MdxConfig *cfg = &m->config;
    const MdxWeights *w = &m->weights;
    int n_scales = (int)cfg->n_scales, g = (int)cfg->growth;
    int C = g, T = s->T, F = (int)cfg->dim_f;
    float *cur, *other;

    /* first conv: [dim_c][F][T] -> [g][F][T], then to [g][T][F] so F is contiguous */
    mdx_conv1x1(s->a, in, w->first.w, w->first.b, (int)cfg->dim_c, g, F, T);
    mdx_relu(s->a, (long)g * F * T);
    tap(s, "first", s->a, g, F, T);
    mdx_transpose_last2(s->skip[0], s->a, g, F, T);

    /* encoder: block i runs on skip[i] (its input was written there), so its
     * output stays in skip[i] for the decoder; downsample into the next level */
    char name[16];
    for (int i = 0; i < n_scales; i++) {
        cur = tfc_tdf(&w->enc[i], cfg, s->skip[i], s->a, s->h, C, T, F);
        if (cur != s->skip[i]) memcpy(s->skip[i], cur, (size_t)C * T * F * sizeof(float)); /* even n_tfc */
        snprintf(name, sizeof(name), "enc%d", i);
        tap(s, name, s->skip[i], C, T, F);

        float *dst = i + 1 < n_scales ? s->skip[i + 1] : s->a;
        mdx_conv2x2_s2(dst, s->skip[i], w->down[i].w, w->down[i].b, C, C + g, T, F);
        C += g, T /= 2, F /= 2;
        mdx_relu(dst, (long)C * T * F);
    }

    /* bottleneck */
    cur = tfc_tdf(&w->mid, cfg, s->a, s->b, s->h, C, T, F);
    tap(s, "mid", cur, C, T, F);

    /* decoder */
    for (int i = 0; i < n_scales; i++) {
        int lvl = n_scales - 1 - i;
        other = cur == s->a ? s->b : s->a;
        mdx_convT2x2_s2(other, cur, w->up[i].w, w->up[i].b, C, C - g, T, F);
        C -= g, T *= 2, F *= 2;
        long n = (long)C * T * F;
        mdx_batchnorm(other, &w->up_bn[i], cfg->bn_eps, C, T * F);
        mdx_relu(other, n);
        mdx_mul(other, s->skip[lvl], n); /* multiplicative skip connection */
        cur = tfc_tdf(&w->dec[i], cfg, other, cur, s->h, C, T, F);
        snprintf(name, sizeof(name), "dec%d", i);
        tap(s, name, cur, C, T, F);
    }

    /* back to [g][F][T], final 1x1 conv to dim_c channels, no activation */
    other = cur == s->a ? s->b : s->a;
    mdx_transpose_last2(other, cur, C, T, F);
    mdx_conv1x1(out, other, w->final.w, w->final.b, g, (int)cfg->dim_c, F, T);
    tap(s, "output", out, (int)cfg->dim_c, F, T);
}
