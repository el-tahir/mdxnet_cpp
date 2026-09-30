/* MDX-Net (UVR_MDXNET_KARA_2) forward pass in plain C99. */
#ifndef MDX_H
#define MDX_H

#include <stdint.h>

#define MDX_MAGIC 0x4E58444Du /* "MDXN" */
#define MDX_VERSION 1
#define MDX_HEADER_SIZE 64
#define MDX_MAX_SCALES 8
#define MDX_MAX_TFC 4

typedef struct {
    uint32_t dim_c;     /* input/output channels: L.re, L.im, R.re, R.im */
    uint32_t dim_f;     /* frequency bins seen by the model */
    uint32_t dim_t;     /* time frames per chunk */
    uint32_t n_scales;  /* encoder/decoder levels */
    uint32_t growth;    /* channels added per level */
    uint32_t n_tfc;     /* conv3x3 layers per TFC */
    uint32_t bn_factor; /* TDF bottleneck: F -> F / bn_factor */
    float bn_eps;
} MdxConfig;

/* convolution: w is [Cout][Cin][k][k] (ConvTranspose: [Cin][Cout][k][k]), b is [Cout] */
typedef struct { const float *w, *b; } MdxConv;

/* batch norm, each [C] */
typedef struct { const float *scale, *bias, *mean, *var; } MdxBN;

/* TFC-TDF block: n_tfc x (conv3x3 + relu), then two Linear+BN+ReLU along F, residual */
typedef struct {
    MdxConv tfc[MDX_MAX_TFC];
    const float *tdf1; /* [F][F/bn_factor] */
    MdxBN tdf1_bn;
    const float *tdf2; /* [F/bn_factor][F] */
    MdxBN tdf2_bn;
} MdxBlock;

/* field order below == order of tensors in the file == order the forward pass uses them */
typedef struct {
    MdxConv first;                   /* 1x1, dim_c -> growth */
    MdxBlock enc[MDX_MAX_SCALES];
    MdxConv down[MDX_MAX_SCALES];    /* 2x2 stride 2, C -> C + growth */
    MdxBlock mid;
    MdxConv up[MDX_MAX_SCALES];      /* ConvTranspose 2x2 stride 2, C -> C - growth */
    MdxBN up_bn[MDX_MAX_SCALES];
    MdxBlock dec[MDX_MAX_SCALES];
    MdxConv final;                   /* 1x1, growth -> dim_c */
} MdxWeights;

typedef struct {
    MdxConfig config;
    MdxWeights weights;
    float *data;          /* all weights, one allocation; MdxWeights points into it */
    uint64_t n_floats;
} MdxModel;

/* Returns 0 on success; on failure prints the reason to stderr and returns -1. */
int mdx_load(MdxModel *m, const char *path);
void mdx_free(MdxModel *m);

/* ------------------------------------------------------------------------- */
/* forward pass */

/* Called after each block with the block's output. Names and layouts:
 *   "first"            [growth][F][T]       after first conv + relu (ONNX 447)
 *   "enc0".."enc{n-1}" [C][T][F]            encoder block outputs = skips
 *   "mid"              [C][T][F]            bottleneck block output (ONNX 571)
 *   "dec0".."dec{n-1}" [C][T][F]            decoder block outputs
 *   "output"           [dim_c][F][T]        final output
 * C, T, F are the sizes at that level. */
typedef void (*MdxTapFn)(void *ctx, const char *name, const float *t, int c, int h, int w);

/* wall-clock seconds spent per kernel type, accumulated over mdx_forward calls */
enum { MDX_PROF_CONV3X3, MDX_PROF_TDF, MDX_PROF_DOWN, MDX_PROF_UP, MDX_PROF_CONV1X1, MDX_PROF_ELEMWISE, MDX_PROF_N };
extern const char *const mdx_prof_names[MDX_PROF_N];

/* Working memory for one forward pass, allocated once for a given T. */
typedef struct {
    int T;                       /* time frames; multiple of 2^n_scales */
    float *a, *b;                /* ping-pong buffers, level-0 size: growth * T * F */
    float *h;                    /* TDF hidden, level-0 size: growth * T * F / bn_factor */
    float *skip[MDX_MAX_SCALES]; /* encoder outputs, level i: C_i * (T >> i) * (F >> i) */
    MdxTapFn tap;                /* optional, NULL to disable */
    void *tap_ctx;
    int reference;               /* 1: use the naive *_ref kernels */
    double prof[MDX_PROF_N];     /* seconds per kernel type */
} MdxState;

/* Returns 0 on success, -1 on bad T or out of memory. */
int mdx_state_init(MdxState *s, const MdxConfig *cfg, int T);
void mdx_state_free(MdxState *s);

/* in:  [dim_c][dim_f][T]  (L.re, L.im, R.re, R.im) x freq x time
 * out: [dim_c][dim_f][T]  may not alias in */
void mdx_forward(const MdxModel *m, MdxState *s, const float *in, float *out);

/* ------------------------------------------------------------------------- */
/* kernels
 *
 * Every tensor is a dense row-major float array [C][H][W] (batch 1).
 * y is the output, x the input; y and x never alias unless noted "in place".
 * Formulas are in PLAN.md section 2 and in the docstrings of tools/reference.py.
 *
 * The *_ref kernels (mdx.c) are the readable reference: one loop nest per
 * formula, one output element at a time. The same kernels without _ref
 * (kernels.c) compute the same thing, reordered and blocked for speed; the
 * tests check them against each other and against numpy.
 */

/* y[o,h,w] = b[o] + sum_i W[o,i] x[i,h,w]                        W [cout][cin][1][1] */
void mdx_conv1x1_ref(float *y, const float *x, const float *W, const float *b, int cin, int cout, int H, int Wd);

/* y[o,h,w] = b[o] + sum_i,ky,kx W[o,i,ky,kx] x[i,h+ky-1,w+kx-1], zero outside   W [cout][cin][3][3] */
void mdx_conv3x3_ref(float *y, const float *x, const float *W, const float *b, int cin, int cout, int H, int Wd);

/* y[o,h,w] = b[o] + sum_i,ky,kx W[o,i,ky,kx] x[i,2h+ky,2w+kx]; x is [cin][H][Wd], y [cout][H/2][Wd/2]   W [cout][cin][2][2] */
void mdx_conv2x2_s2_ref(float *y, const float *x, const float *W, const float *b, int cin, int cout, int H, int Wd);

/* y[o,2h+ky,2w+kx] = b[o] + sum_i W[i,o,ky,kx] x[i,h,w]; x is [cin][H][Wd], y [cout][2H][2Wd]   W [cin][cout][2][2] */
void mdx_convT2x2_s2_ref(float *y, const float *x, const float *W, const float *b, int cin, int cout, int H, int Wd);

/* y[r,j] = sum_f x[r,f] W[f,j]; x is [rows][fin], y [rows][fout]   W [fin][fout]
 * (rows = C*T: the Linear acts on the last axis only) */
void mdx_matmul_lastdim_ref(float *y, const float *x, const float *W, int rows, int fin, int fout);

/* fast versions, same signatures and results (up to float summation order) */
void mdx_conv1x1(float *y, const float *x, const float *W, const float *b, int cin, int cout, int H, int Wd);
void mdx_conv3x3(float *y, const float *x, const float *W, const float *b, int cin, int cout, int H, int Wd);
void mdx_conv2x2_s2(float *y, const float *x, const float *W, const float *b, int cin, int cout, int H, int Wd);
void mdx_convT2x2_s2(float *y, const float *x, const float *W, const float *b, int cin, int cout, int H, int Wd);
void mdx_matmul_lastdim(float *y, const float *x, const float *W, int rows, int fin, int fout);

/* in place: x[c,i] = (x[c,i] - mean[c]) / sqrt(var[c] + eps) * scale[c] + bias[c], i over hw */
void mdx_batchnorm(float *x, const MdxBN *bn, float eps, int C, int hw);

/* in place: x = max(x, 0) */
void mdx_relu(float *x, long n);

/* in place: y += a */
void mdx_add(float *y, const float *a, long n);

/* in place: y *= a */
void mdx_mul(float *y, const float *a, long n);

/* y[c,j,i] = x[c,i,j]; x is [C][H][Wd], y [C][Wd][H] */
void mdx_transpose_last2(float *y, const float *x, int C, int H, int Wd);

#endif
