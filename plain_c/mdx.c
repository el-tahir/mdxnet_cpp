#include "mdx.h"

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
